import flet as ft
import os
import threading
import time
import yaml
import logging
import sqlite3
from datetime import datetime, timezone

# --- LangChain Importok (Most már az AI motorhoz is kellenek) ---
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold, VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic.v1 import BaseModel, Field



# --- Saját modulok importálása ---
# Most már szükségünk van az összes eszközre is!
from shared_components import (
    ATOM_DATA, PROMPTS, message_to_document, chunk_text,
    search_memory_tool, search_knowledge_base_tool, list_uploaded_files_tool,
    set_registry_value, get_registry_value, list_registry_keys,
    read_full_document_tool,
    set_meeting_status, get_meeting_status, read_agent_notebook, update_agent_notebook
)
# from task_dispatcher import TaskDispatcher # Ezt még mindig nem
from document_processor import process_and_store_document # Erre most már szükség van

# --- Konfiguráció betöltése a YAML fájlból ---
try:
    with open('config_aito.yaml', 'r', encoding='utf-8') as file:
        CONFIG = yaml.safe_load(file)
    print("Az AITO konfiguráció (config_aito.yaml) sikeresen betöltve.")
except FileNotFoundError:
    print("HIBA: A 'config_aito.yaml' fájl nem található!")
    CONFIG = {}

# --- Globális Beállítások ---
INITIAL_ATOM_ID = "ATOM1"

# --- Hitelesítés (TESZTELVE, OK) ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CONFIG.get('credentials_file', '')

# --- Globális Változók (ideiglenesen None) ---
google_embeddings = None
vector_store = None
docs_vector_store = None
firestore_history = None

# --- Lokális Memória és Adatbázis Inicializálása (TESZTELVE, OK) ---
try:
    LOCAL_DB_PATH = "./aito_local_data"
    CHROMA_CONVERSATION_PATH = f"{LOCAL_DB_PATH}/chroma_conversations"
    CHROMA_DOCS_PATH = f"{LOCAL_DB_PATH}/chroma_documents"
    SQLITE_HISTORY_FILE = f"{LOCAL_DB_PATH}/aito_chat_history.db"
    os.makedirs(LOCAL_DB_PATH, exist_ok=True)

    print("Google Cloud embedding kliens inicializálása...")
    google_embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=CONFIG['project_id'],
    )
    print("Embedding kliens inicializálva.")

    vector_store = Chroma(
        persist_directory=CHROMA_CONVERSATION_PATH,
        embedding_function=google_embeddings
    )
    print(f"Beszélgetés-vektorok sikeresen csatlakoztatva: {CHROMA_CONVERSATION_PATH}")
    docs_vector_store = Chroma(
        persist_directory=CHROMA_DOCS_PATH,
        embedding_function=google_embeddings
    )
    print(f"Dokumentum-vektorok sikeresen csatlakoztatva: {CHROMA_DOCS_PATH}")

    connection_string = f"sqlite:///{SQLITE_HISTORY_FILE}"
    firestore_history = SQLChatMessageHistory(
        session_id=CONFIG['session_id'],
        connection=connection_string
    )
    print(f"Beszélgetés-napló sikeresen csatlakoztatva: {SQLITE_HISTORY_FILE}")
    try:
        initial_messages = firestore_history.messages
        print(f"{len(initial_messages)} üzenet betöltve a helyi adatbázisból.")
    except Exception as hist_err:
        print(f"Figyelmeztetés: Nem sikerült betölteni az előzményeket az adatbázisból: {hist_err}")
        print("Lehet, hogy az adatbázis még üres vagy a séma nem megfelelő.")

except Exception as e:
    print(f"!!! KRITIKUS HIBA az inicializálás közben: {e} !!!")
    import traceback
    traceback.print_exc()

class MessageBubble(ft.Row):
    def __init__(self, message: HumanMessage or AIMessage):
        super().__init__()
        speaker = getattr(message, 'name', None) or (CONFIG.get('user_id') if message.type == 'human' else "Ismeretlen")
        display_speaker = "Te" if speaker == CONFIG.get('user_id') else speaker

        color_name = ATOM_DATA.get(speaker, {}).get("color", "BLACK")
        bubble_color = getattr(ft.Colors, color_name, ft.Colors.BLACK)

        bubble_container = ft.Container(
            content=ft.SelectionArea( # <-- A kijelölhetőség kulcsa
                content=ft.Markdown(
                    f"**{display_speaker}:** {message.content}" if display_speaker != "Te" else message.content,
                    extension_set="gitHubWeb", # Ez egy robusztus, általános Markdown értelmező
                    code_theme="atom-one-dark" # Kódrészletekhez szép sötét téma
                )
            ),
            padding=12, border_radius=ft.border_radius.all(15), expand=True,
        )
        if speaker == CONFIG.get('user_id'):
            self.alignment = ft.MainAxisAlignment.END
            bubble_container.bgcolor = ft.Colors.WHITE10
        else:
            self.alignment = ft.MainAxisAlignment.START
            # Az eredeti színhez hozzáadunk egy kis áttetszőséget
            bubble_container.bgcolor = ft.Colors.with_opacity(0.5, bubble_color) # <-- JAVÍTVA
        self.controls = [bubble_container]


class ImageBubble(ft.Row):
    def __init__(self, base64_image: str, speaker: str):
        super().__init__()
        color_name = ATOM_DATA.get(speaker, {}).get("color", "BLACK")
        bubble_color = getattr(ft.Colors, color_name, ft.Colors.BLACK)

        image_container = ft.Container(
            content=ft.Image(src_base64=base64_image),
            padding=12,
            border_radius=ft.border_radius.all(15),
            bgcolor=bubble_color
        )
        self.alignment = ft.MainAxisAlignment.START
        self.controls = [image_container]


def main(page: ft.Page):
    start_time = time.monotonic()
    print(f"{start_time:.4f}: Main function started.")

    page.title = "AITO Vezérlőpult"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        text_theme=ft.TextTheme(
            body_medium=ft.TextStyle(size=16)
        )
    )

    print(f"{time.monotonic():.4f}: --- STARTING INITIALIZATION ---")

    print(f"{time.monotonic():.4f}: Checking critical components...")
    if not all(comp is not None for comp in [ATOM_DATA, PROMPTS, CONFIG, vector_store, docs_vector_store, firestore_history]): # <-- FIGYELEM: firestore_history is hozzáadva!
        page.add(ft.Text("Hiba: Az alkalmazás kritikus komponenseinek betöltése sikertelen!", color=ft.Colors.RED))
        return
    print(f"{time.monotonic():.4f}: Critical components OK.")


    # --- FÁJLKEZELŐ ÉS FELTÖLTÉS LOGIKA ---
    def on_document_upload(e: ft.FilePickerResultEvent):
        if e.files:
            uploaded_file_path = e.files[0].path
            print(f"Fájl kiválasztva: {uploaded_file_path}")

            # A feldolgozás elindítása egy külön szálon, hogy a UI ne fagyjon le
            thread = threading.Thread(
                target=process_and_store_document,
                args=(uploaded_file_path, docs_vector_store, CONFIG, page) # <-- page hozzáadva az args-hoz
            )
            thread.start()

            status_text = f"'{e.files[0].name}' feltöltése és feldolgozása megkezdődött a háttérben."

            page.snack_bar = ft.SnackBar(
                content=ft.Text(status_text),
                show_close_icon=True
            )
            page.snack_bar.open = True

            # Rendszerüzenet küldése a chat ablakba is
            system_feedback_message = AIMessage(content=status_text, name="SYSTEM")
            chat_history_view.controls.append(MessageBubble(system_feedback_message))
            page.update()

    file_picker = ft.FilePicker(on_result=on_document_upload)
    page.overlay.append(file_picker)
    # --- UI Komponensek (adatbázis és logika nélkül) ---
    chat_history_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    input_field = ft.TextField(hint_text="Írj ide...", expand=True, border_color="white", multiline=True, min_lines=3, max_lines=5, shift_enter=True)

    # === VALÓDI ESZKÖZ CSOMAGOLÓK ===
    # Ezek kellenek a valódi switch_atom-hoz
    def wrapped_search_memory_tool(query: str) -> str:
        return search_memory_tool(query=query, config=CONFIG, vector_store=vector_store)
    def wrapped_search_knowledge_base_tool(query: str) -> str:
        return search_knowledge_base_tool(query=query, config=CONFIG, docs_vector_store=docs_vector_store)
    def wrapped_list_uploaded_files_tool(filter: str = "ALL") -> str:
        return list_uploaded_files_tool(config=CONFIG, docs_vector_store=docs_vector_store, filter=filter)
    def wrapped_set_registry_value(key: str, value: str) -> str:
        return set_registry_value(key=key, value=value, config=CONFIG)
    def wrapped_get_registry_value(key: str) -> str:
        return get_registry_value(key=key, config=CONFIG)
    def wrapped_list_registry_keys() -> str:
        return list_registry_keys(config=CONFIG)
    def wrapped_read_full_document_tool(filename: str) -> str:
        return read_full_document_tool(filename=filename, docs_vector_store=docs_vector_store) # <- FIGYELEM: Ezt ki kellett egészítenem a docs_vector_store-ral
    def wrapped_set_meeting_status(active: bool, meeting_id: str = "") -> str:
        return set_meeting_status(active=active, meeting_id=meeting_id, config=CONFIG)
    def wrapped_get_meeting_status() -> dict:
        return get_meeting_status(config=CONFIG)

    def wrapped_read_notebook() -> str:
        """Beolvassa az aktuálisan aktív ATOM privát jegyzetfüzetét."""
        active_atom = app_state.get("active_atom_id")
        if not active_atom: return "Hiba: Nincs aktív ATOM a jegyzetfüzet olvasásához."
        return read_agent_notebook(agent_id=active_atom, config=CONFIG)

    def wrapped_update_notebook(new_content: str) -> str:
        """Frissíti az aktuálisan aktív ATOM privát jegyzetfüzetét."""
        active_atom = app_state.get("active_atom_id")
        if not active_atom: return "Hiba: Nincs aktív ATOM a jegyzetfüzet írásához."
        return update_agent_notebook(agent_id=active_atom, new_content=new_content, config=CONFIG)

    # === ÁLLAPOT ===
    app_state = {"active_atom_id": INITIAL_ATOM_ID, "atom_chain": None, "tool_registry": {}, "base_system_prompt": ""}

    # === AZ IGAZI `switch_atom` FÜGGVÉNY (main_aito.py-ból másolva) ===
    atom_buttons = {} # Ezt előre kell definiálni, hogy a switch_atom lássa

    def switch_atom(selected_atom_id: str):
        logging.info(f"--- ATOM VÁLTÁS: {selected_atom_id} ---")
        app_state["active_atom_id"] = selected_atom_id
        current_atom_config = ATOM_DATA[app_state["active_atom_id"]]
        logging.info(f"'{selected_atom_id}' konfigurációja betöltve. Modell: {current_atom_config.get('model_name')}")

        final_system_prompt = PROMPTS['team_simulation_template'].format(
            active_atom_role=selected_atom_id,
            personality_description=current_atom_config['personality'],
            grounding_instructions=PROMPTS['grounding_instructions']
        )
        logging.info("System prompt sikeresen összeállítva.")

        llm = ChatVertexAI(
            model_name=current_atom_config["model_name"],
            project=CONFIG['project_id'],
            location=CONFIG['conversation_location'],
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        logging.info(f"ChatVertexAI kliens inicializálva a '{current_atom_config['model_name']}' modellel, a '{CONFIG['conversation_location']}' régióban.")

        # A modellnek már a becsomagolt, egyszerűsített eszközt adjuk át.
        tool_registry = {
            "wrapped_search_memory_tool": wrapped_search_memory_tool,
            "wrapped_search_knowledge_base_tool": wrapped_search_knowledge_base_tool,
            "wrapped_list_uploaded_files_tool": wrapped_list_uploaded_files_tool,
            "wrapped_set_registry_value": wrapped_set_registry_value,
            "wrapped_get_registry_value": wrapped_get_registry_value,
            "wrapped_list_registry_keys": wrapped_list_registry_keys,
            "wrapped_read_full_document_tool": wrapped_read_full_document_tool,
            "wrapped_set_meeting_status": wrapped_set_meeting_status,
            "wrapped_get_meeting_status": wrapped_get_meeting_status,
            "wrapped_read_notebook": wrapped_read_notebook,
            "wrapped_update_notebook": wrapped_update_notebook,
        }

        tools = list(tool_registry.values())
        llm_with_tools = llm.bind_tools(tools)

        app_state["tool_registry"] = tool_registry
        app_state["base_system_prompt"] = final_system_prompt

        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # --- METADATOK HOZZÁADÁSA A VÁLASZHOZ ---
        # Ez a kis függvény veszi a modell válaszát (AIMessage) és hozzáadja
        # a 'name' és 'timestamp' mezőket, MIELŐTT a history-ba kerülne.
        def add_metadata_to_response(response_message: AIMessage):
            response_message.name = selected_atom_id
            response_message.additional_kwargs = {"timestamp": datetime.now(timezone.utc).isoformat()}
            return response_message

        # A lánc végére fűzzük a metadatokat hozzáadó függvényt
        chain_with_metadata = prompt | llm_with_tools | RunnableLambda(add_metadata_to_response)


        chain_with_history = RunnableWithMessageHistory(
            chain_with_metadata,
            lambda session_id: firestore_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        app_state["atom_chain"] = chain_with_history
        print(f"Motor átkonfigurálva: {app_state['active_atom_id']} aktív.")

        page.title = f"AITO Vezérlőpult - {app_state['active_atom_id']} Aktív"

        active_button_style = ft.ButtonStyle(side=ft.border.BorderSide(2, ft.Colors.WHITE))
        def get_inactive_style(color): return ft.ButtonStyle(bgcolor=color)

        for atom_id, button in atom_buttons.items():
            original_color_name = ATOM_DATA[atom_id]["color"]
            original_color = getattr(ft.Colors, original_color_name, ft.Colors.BLACK)

            if atom_id == selected_atom_id:
                button.style = active_button_style
                button.style.bgcolor = ft.Colors.with_opacity(0.3, original_color)
            else:
                button.style = get_inactive_style(original_color)
        page.update()

    # --- UI Elemek Létrehozása ---
    for atom_id, data in ATOM_DATA.items():
        button = ft.ElevatedButton(
            text=data["label"],
            on_click=lambda e: switch_atom(e.control.data),
            data=atom_id
        )
        atom_buttons[atom_id] = button

    upload_button = ft.IconButton(
        icon=ft.Icons.UPLOAD_FILE,
         tooltip="Dokumentum feltöltése a Tudásbázisba",
         on_click=lambda _: file_picker.pick_files(
             allow_multiple=False,
             allowed_extensions=["pdf", "txt", "md"]
         ),
        icon_color="white"
    )

    atom_selector = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[upload_button] + list(atom_buttons.values())
    )

    def get_ai_response(user_message: HumanMessage, chain_for_request, atom_id_for_request: str, tool_registry: dict):
        try:
            config = {"configurable": {"session_id": CONFIG['session_id']}}

            # === DINAMIKUS RENDSZERÜZENET ÖSSZEÁLLÍTÁSA ===
            # Aktuális idő lekérdezése
            current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
            time_prompt_addition = f"Current Timestamp: {current_time_str}\n"

            # Aktuális megbeszélés állapotának lekérdezése (ez egy gyors, helyi DB hívás)
            meeting_status = wrapped_get_meeting_status()
            meeting_id_str = "None (INACTIVE)"
            if meeting_status.get('is_active'):
                meeting_id_str = f"{meeting_status.get('meeting_id')} (ACTIVE)"
            status_prompt_addition = f"Current Meeting ID: {meeting_id_str}\n\n" # Két sortörés a jobb tagolásért

            # A végleges prompt összeállítása
            final_system_prompt = time_prompt_addition + status_prompt_addition + app_state["base_system_prompt"]
            # ============================================

            # A bemenet összeállítása a lánc számára (ez a sor már létezik, csak ellenőrizd)
            current_input = {
                "input": user_message.content, # vagy 'tool_messages' a ciklusban
                "active_atom_role": atom_id_for_request,
                "system_prompt": final_system_prompt
            }
            response = chain_for_request.invoke(current_input, config=config)

            while response.tool_calls:
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    logging.info(f"--- {atom_id_for_request} Eszközt Használ: {tool_name}, Argumentumok: {tool_call['args']} ---")

                    if tool_name not in tool_registry:
                        tool_messages.append(ToolMessage(content=f"Ismeretlen eszköz: {tool_name}", tool_call_id=tool_call['id'], name=tool_name))
                        continue

                    tool_to_call = tool_registry[tool_name]
                    tool_output = tool_to_call(**tool_call['args'])
                    logging.debug(f"Nyers eszköz-kimenet a '{tool_name}' eszköztől: {tool_output}")

                    tool_messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call['id'], name=tool_name))

                # A modell újrahívása az eszközök kimenetével
                # A rendszerüzenet frissítése itt is megtörténik
                # === DINAMIKUS RENDSZERÜZENET ÖSSZEÁLLÍTÁSA ===
                # Aktuális idő lekérdezése
                current_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                time_prompt_addition = f"Current Timestamp: {current_time_str}\n"

                # Aktuális megbeszélés állapotának lekérdezése (ez egy gyors, helyi DB hívás)
                meeting_status = wrapped_get_meeting_status()
                meeting_id_str = "None (INACTIVE)"
                if meeting_status.get('is_active'):
                    meeting_id_str = f"{meeting_status.get('meeting_id')} (ACTIVE)"
                status_prompt_addition = f"Current Meeting ID: {meeting_id_str}\n\n" # Két sortörés a jobb tagolásért

                # A végleges prompt összeállítása
                final_system_prompt = time_prompt_addition + status_prompt_addition + app_state["base_system_prompt"]
                # ============================================
                current_input = {
                    "input": tool_messages, # vagy 'tool_messages' a ciklusban
                    "active_atom_role": atom_id_for_request,
                    "system_prompt": final_system_prompt
                }
                response = chain_for_request.invoke(current_input, config=config)

            # Ha a ciklus lefutott (vagy nem is volt benne tool_calls), a `response` a végleges szöveges válasz.
            final_response = response
            print(f"AI üzenet ({final_response.name}) a láncon keresztül automatikusan mentve (SQLite).")

            # Szöveges válasz feldolgozása és megjelenítése
            text_chunks = chunk_text(final_response.content)
            documents_to_add = []
            meeting_status = wrapped_get_meeting_status()
            meeting_id = meeting_status.get('meeting_id') if meeting_status.get('is_active') else None
            for i, chunk in enumerate(text_chunks):
                doc = message_to_document(
                    content=chunk,
                    speaker=final_response.name,
                    timestamp=final_response.additional_kwargs.get("timestamp"),
                    session_id=CONFIG['session_id'],
                    chunk_num=i + 1,
                    total_chunks=len(text_chunks),
                    meeting_id=meeting_id
                )
                documents_to_add.append(doc)
            if documents_to_add:
                vector_store.add_documents(documents_to_add)
                print(f"AI üzenet {len(documents_to_add)} darabra vágva és a memóriába mentve.")

            page.run_thread(update_ui_with_ai_message, final_response)

        except Exception as ex:
            logging.error(f"Hiba a get_ai_response függvényben: {ex}", exc_info=True)
            error_message = AIMessage(content=f"Hiba történt: {ex}", name="SYSTEM_ERROR")
            page.run_thread(update_ui_with_ai_message, error_message)

    def update_ui_with_ai_message(ai_message: AIMessage):
        if chat_history_view.controls:
            chat_history_view.controls.pop()
        chat_history_view.controls.append(MessageBubble(ai_message))
        page.update()

    def send_click(e):
        user_input_text = input_field.value
        if not user_input_text.strip(): return

        human_message = HumanMessage(
            content=user_input_text,
            name=CONFIG['user_id'],
            additional_kwargs={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        chat_history_view.controls.append(MessageBubble(human_message))

        text_chunks = chunk_text(human_message.content)
        documents_to_add = []
        meeting_status = wrapped_get_meeting_status()
        meeting_id = meeting_status.get('meeting_id') if meeting_status.get('is_active') else None
        for i, chunk in enumerate(text_chunks):
            doc = message_to_document(
                content=chunk,
                speaker=human_message.name,
                timestamp=human_message.additional_kwargs.get("timestamp"),
                session_id=CONFIG['session_id'],
                chunk_num=i + 1,
                total_chunks=len(text_chunks),
                meeting_id=meeting_id
            )
            documents_to_add.append(doc)

        if documents_to_add:
            vector_store.add_documents(documents_to_add)
            print(f"Üzenet {len(documents_to_add)} darabra vágva és a memóriába mentve.")

        if user_input_text.strip().lower() == "exitchatnow":
            firestore_history.add_message(human_message)
            os._exit(0)
            return

        input_field.value = ""
        thinking_bubble = MessageBubble(AIMessage(content="gondolkodik...", name=app_state["active_atom_id"]))
        chat_history_view.controls.append(thinking_bubble)
        page.update()

        thread = threading.Thread(target=get_ai_response, args=(human_message, app_state["atom_chain"], app_state["active_atom_id"], app_state["tool_registry"]))
        thread.start()

    def on_keyboard(e: ft.KeyboardEvent):
        if e.key == "Enter" and not e.shift:
            send_click(None)
    page.on_keyboard_event = on_keyboard

    # --- Oldal Elrendezés Összeállítása ---
    page.add(
        ft.Column([
            atom_selector,
            ft.Container(content=chat_history_view, border=ft.border.all(1, "white"), border_radius=ft.border_radius.all(5), padding=10, expand=True),
            ft.Row([input_field, ft.IconButton(icon=ft.Icons.SEND_ROUNDED, tooltip="Küldés", on_click=send_click, icon_color="white")]),
        ], expand=True)
    )

    page.padding = 20
    page.update()
    print(f"{time.monotonic():.4f}: Első Flet UI kirajzolás (page.update) elküldve.")

    try:
        logging.info("Az első ATOM motorjának beállítása (fő szál)...")
        switch_atom(INITIAL_ATOM_ID)
        logging.info("Az első ATOM motorja sikeresen beállítva.")
    except Exception as e:
         logging.error(f"HIBA az első switch_atom hívásakor: {e}", exc_info=True)
         page.add(ft.Text(f"Indítási hiba: {e}", color=ft.Colors.RED))
         page.update()

    def initialize_app_in_background():
        """CSAK az előzményeket tölti be a háttérben, THREAD-SAFE módon."""
        logging.info("--- Háttér-előzmény betöltés elindult ---")
        try:
            logging.info("Előzmények betöltése az SQLite adatbázisból...")
            messages_to_load = firestore_history.messages # Ez lehet lassú
            logging.info(f"{len(messages_to_load)} üzenet sikeresen betöltve az adatbázisból.")

            if messages_to_load:
                logging.info("Előzmény buborékok létrehozása (memóriában)...")
                # A MessageBubble objektumok létrehozása biztonságos a háttérszálon
                bubbles_to_add = [MessageBubble(msg) for msg in messages_to_load]

                logging.info("Előzmények hozzáadása a UI-hoz (thread-safe)...")
                chat_history_view.controls.clear() # Töröljük a (valószínűleg üres) listát
                chat_history_view.controls.extend(bubbles_to_add) # Adjuk hozzá az összes új elemet
                chat_history_view.scroll_to(offset=-1, duration=0) # Görgessünk az aljára (azonnal)

                # Most hívjuk az EGYETLEN, thread-safe frissítést
                page.update()

                logging.info("Előzmények megjelenítve és aljára görgetve.")
            else:
                    logging.info("Nincsenek előzmények a megjelenítéshez.")

            logging.info("--- Háttér-előzmény betöltés sikeresen befejeződött. ---")

        except Exception as e:
            logging.error(f"KRITIKUS HIBA a háttér-előzmény betöltés során: {e}", exc_info=True)
            # A hibaüzenet hozzáadása is legyen thread-safe
            error_bubble = MessageBubble(AIMessage(content=f"Indítási hiba: {e}", name="SYSTEM_ERROR"))
            chat_history_view.controls.append(error_bubble)
            page.update() # page.update() thread-safe

    # Csak az előzmények betöltését indítjuk a háttérben
    logging.info("Háttér-előzmény betöltési szál indítása...")
    page.run_thread(initialize_app_in_background)
    logging.info("Háttérszál elindítva. A main függvény véget ért.")

if __name__ == "__main__":
    ft.app(target=main)
