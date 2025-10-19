# task_dispatcher.py

import uuid
import threading
from datetime import datetime, timezone
import flet as ft
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from state_manager import MeetingState
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# A közös komponensek importálása
from shared_components import ATOM_DATA, PROMPTS, message_to_document

class TaskDispatcher:
    """
    Ez az osztály felelős a komplex, több ágenst igénylő feladatok
    fogadásáért és az ATOMOD vezérlési lánc elindításáért.
    """
    def __init__(self, page: ft.Page, chat_history_view: ft.ListView, firestore_history, config: dict, vector_store, search_memory_tool):
        self.page = page
        self.chat_history_view = chat_history_view
        self.firestore_history = firestore_history
        self.config = config
        self.vector_store = vector_store
        self.search_memory_tool = search_memory_tool
        self.atomod_graph = None  # A gráfot csak az első használatkor hozzuk létre
        self._graph_lock = threading.Lock() # Lock a versenyhelyzetek elkerülésére
        print("Task Dispatcher (Forgalomirányító) inicializálva, a rendszerkomponensekhez bekötve.")

    def _update_ui_and_memory(self, message: BaseMessage):
        """Segédfüggvény, ami a UI-t és a memóriákat is frissíti a háttérszálból."""
        # A MessageBubble-t a fő fájlból importáljuk a körkörös hivatkozás elkerülésére
        # Ellenőrizve: Az import helyes és a körkörös hivatkozás elkerülése miatt van a függvényen belül.
        from main_aito import MessageBubble
        
        self.firestore_history.add_message(message)
        # The dispatcher's own messages are not chunked, so we create a single document.
        doc = message_to_document(
            content=message.content,
            speaker=getattr(message, 'name', 'ATOMOD'),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.config.get('session_id', 'unknown_session')
        )
        self.vector_store.add_documents([doc])
        print(f"ATOMOD Ciklus: '{message.name}' üzenete rögzítve a memóriákban.")

        if self.chat_history_view.controls and "gondolkodik..." in self.chat_history_view.controls[-1].controls[0].content.value:
             self.chat_history_view.controls.pop()
        
        self.chat_history_view.controls.append(MessageBubble(message))
        self.page.update()

    def _select_speaker(self, state: MeetingState) -> dict:
        print("--- ATOMOD: Felszólaló kiválasztása... ---")
        participants = state['participants']
        # Exclude the user from the agent message count
        agent_messages = [msg for msg in state['messages'] if msg.name != self.config.get('user_id')]
        agent_messages_count = len(agent_messages)
        current_round = agent_messages_count // len(participants) + 1
        speaker_index = agent_messages_count % len(participants)
        max_rounds = 1 
        if current_round > max_rounds:
            next_speaker = None
        else:
            next_speaker = participants[speaker_index]
        return {"next_speaker": next_speaker, "current_round": current_round}

    def _run_agent_turn(self, state: MeetingState) -> dict:
        agent_id = state['next_speaker']
        moderator_message = AIMessage(content=f"ATOMOD: A következő felszólaló: {agent_id} (Kör: {state['current_round']}). Kérem a hozzászólását.", name="ATOMOD")
        
        self.page.run_thread(target=self._update_ui_and_memory, args=(moderator_message,))
        
        print(f"--- ATOMOD: {agent_id} aktiválása... ---")
        current_atom_config = ATOM_DATA[agent_id]
        final_system_prompt = PROMPTS['team_simulation_template'].format(
            active_atom_role=agent_id,
            personality_description=current_atom_config['personality'],
            grounding_instructions=PROMPTS['grounding_instructions']
        )
        llm = ChatVertexAI(
            model_name=current_atom_config["model_name"],
            project=self.config.get('project_id'),
            location=self.config.get('conversation_location'),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        tools = [self.search_memory_tool]
        llm_with_tools = llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm_with_tools
        response = chain.invoke({"messages": state['messages']})
        response.name = agent_id
        
        self.page.run_thread(target=self._update_ui_and_memory, args=(response,))
        return {"messages": [moderator_message, response]}

    def _should_continue(self, state: MeetingState) -> str:
        print("--- ATOMOD: Befejezési feltétel ellenőrzése... ---")
        if not state['next_speaker']:
            print("--- ATOMOD: Mindenki hozzászólt. Megbeszélés lezárva. ---")
            return "end"
        else:
            print("--- ATOMOD: Megbeszélés folytatódik. ---")
            return "continue"

    def _build_graph(self):
        workflow = StateGraph(MeetingState)
        workflow.add_node("select_speaker", self._select_speaker)
        workflow.add_node("run_agent_turn", self._run_agent_turn)
        workflow.set_entry_point("select_speaker")
        workflow.add_conditional_edges(
            "select_speaker", self._should_continue,
            {"continue": "run_agent_turn", "end": END}
        )
        workflow.add_edge("run_agent_turn", "select_speaker")
        # A .compile() hívás időigényes, ezért ezt csak akkor végezzük el, amikor tényleg kell.
        print("--- ATOMOD: LangGraph workflow definíció elkészült. Fordítás (compile) folyamatban... ---")
        graph = workflow.compile()
        print("--- ATOMOD: LangGraph workflow sikeresen lefordítva. ---")
        return graph

    def _get_or_build_graph(self):
        """
        Ellenőrzi, hogy a gráf már le van-e fordítva. Ha nem, akkor lock-olja
        a szálat és lefordítja. Ez biztosítja, hogy a fordítás csak egyszer
        történjen meg és a fő szálat nem blokkolja.
        """
        with self._graph_lock:
            if self.atomod_graph is None:
                print("--- ATOMOD: A gráf még nincs lefordítva. A fordítás elindítása a háttérben... ---")
                self.atomod_graph = self._build_graph()
        return self.atomod_graph

    def start_new_task(self, task_description: str, initiated_by: str) -> str:
        task_id = str(uuid.uuid4())
        initial_state = MeetingState(
            task_description=task_description,
            participants=["ATOM1", "ATOM5"],
            messages=[HumanMessage(content=task_description, name=initiated_by)],
            current_round=0,
            next_speaker=""
        )
        # A gráf futtatását egy külön szálba szervezzük, hogy ne fagyjon a UI.
        thread = threading.Thread(target=self._run_graph_in_background, args=(initial_state,))
        thread.start()
        return task_id

    def _run_graph_in_background(self, initial_state: MeetingState):
        """
        Ez a függvény egy háttérszálon fut. Először megszerzi (vagy lefordítja)
        a gráfot, majd futtatja a megadott állapottal.
        """
        print("\n--- ATOMOD MUNKAfolyamat a háttérben elindult ---")
        try:
            # A gráf megszerzése (vagy első futás esetén a fordítás kivárása)
            graph_to_run = self._get_or_build_graph()

            # A gráf futtatása
            final_state = graph_to_run.invoke(initial_state)

            final_report_message = AIMessage(
                content=f"ATOMOD JELENTÉS: A '{final_state['task_description']}' feladat megbeszélése befejeződött.",
                name="ATOMOD"
            )
            self.page.run_thread(target=self._update_ui_and_memory, args=(final_report_message,))
            print("\n--- ATOMOD JELENTÉS: MEGBESZÉLÉS BEFEJEZVE ---")
        except Exception as e:
            print(f"\n---!!! ATOMOD HIBA: A munkafolyamat megszakadt: {e} !!!---")
            error_message = AIMessage(content=f"ATOMOD HIBA: A feladat végrehajtása közben hiba történt: {e}", name="ATOMOD_ERROR")
            self.page.run_thread(target=self._update_ui_and_memory, args=(error_message,))


print("Forgalomirányító modul (task_dispatcher.py) sikeresen betöltve, LangGraph motorral.")