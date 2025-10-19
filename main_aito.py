# aito_ui_v2.py

import flet as ft
import os
import threading
import time
import yaml
import logging
from datetime import datetime, timezone

# --- LangChain Importok ---
from langchain_google_vertexai import ChatVertexAI, HarmCategory, HarmBlockThreshold, VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document

# --- Saját modulok importálása ---
from shared_components import ATOM_DATA, PROMPTS, message_to_document, chunk_text, search_memory_tool, search_knowledge_base_tool, list_uploaded_files_tool, set_registry_value, get_registry_value, list_registry_keys, generate_diagram_tool, read_full_document_tool, display_image_tool, set_meeting_status, get_meeting_status
from task_dispatcher import TaskDispatcher
from document_processor import process_and_store_document


# --- Konfiguráció betöltése a YAML fájlból ---
try:
    with open('config_aito.yaml', 'r', encoding='utf-8') as file:
        CONFIG = yaml.safe_load(file)
    print("Az AITO konfiguráció (config_aito.yaml) sikeresen betöltve.")
except FileNotFoundError:
    print("HIBA: A 'config_aito.yaml' fájl nem található!")
    CONFIG = {}

# --- Globális Beállítások (az UI-hoz) ---
INITIAL_ATOM_ID = "ATOM1"
CHAT_FONT_FAMILY = "Verdana"
CHAT_FONT_SIZE = 14

# --- Hitelesítés ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CONFIG.get('credentials_file', '')


# --- Lokális Memória és Adatbázis Inicializálása (ChromaDB + SQLite) ---
import sqlite3
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import SQLiteChatMessageHistory

try:
    # === HELYI ADATBÁZIS FÁJLOK ÉS MAPPÁK ===
    LOCAL_DB_PATH = "./aito_local_data"
    CHROMA_CONVERSATION_PATH = f"{LOCAL_DB_PATH}/chroma_conversations"
    CHROMA_DOCS_PATH = f"{LOCAL_DB_PATH}/chroma_documents"
    SQLITE_HISTORY_FILE = f"{LOCAL_DB_PATH}/aito_chat_history.db"
    os.makedirs(LOCAL_DB_PATH, exist_ok=True)

    # === GOOGLE CLOUD EMBEDDING (A MINŐSÉGÉRT) ===
    print("Google Cloud embedding kliens inicializálása...")
    google_embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=CONFIG['project_id']
    )
    print("Embedding kliens inicializálva.")

    # === LOKÁLIS VEKTOR TÁROLÓ (CHROMA DB) ===
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

    # === LOKÁLIS BESZÉLGETÉS NAPLÓ (SQLITE) ===
    firestore_history = SQLiteChatMessageHistory(
        session_id=CONFIG['session_id'],
        db_path=SQLITE_HISTORY_FILE
    )
    print(f"Beszélgetés-napló sikeresen csatlakoztatva: {SQLITE_HISTORY_FILE}")
    print(f"{len(firestore_history.messages)} üzenet betöltve a helyi adatbázisból.")

except Exception as e:
    print(f"HIBA a lokális komponensek inicializálása közben: {e}")
    google_embeddings = None
    vector_store = None
    docs_vector_store = None
    firestore_history = None


# --- FLET ALKALMAZÁS ---
class MessageBubble(ft.Row):
    def __init__(self, message: HumanMessage or AIMessage):
        super().__init__()
        speaker = getattr(message, 'name', None) or (CONFIG.get('user_id') if message.type == 'human' else "Ismeretlen")
        display_speaker = "Te" if speaker == CONFIG.get('user_id') else speaker

        color_name = ATOM_DATA.get(speaker, {}).get("color", "BLACK")
        bubble_color = getattr(ft.Colors, color_name, ft.Colors.BLACK)

        bubble_container = ft.Container(
            content=ft.Markdown(
                f"**{display_speaker}:** {message.content}" if display_speaker != "Te" else message.content,
                selectable=True,
                extension_set="gitHubWeb", # Ez egy robusztus, általános Markdown értelmező
                code_theme="atom-one-dark" # Kódrészletekhez szép sötét téma
            ),
            padding=12, border_radius=ft.border_radius.all(15), expand=True,
        )
        if speaker == CONFIG.get('user_id'):
            self.alignment = ft.MainAxisAlignment.END
            bubble_container.bgcolor = ft.Colors.WHITE10
        else:
            self.alignment = ft.MainAxisAlignment.START
            bubble_container.bgcolor = bubble_color
        self.controls = [bubble_container]

def main(page: ft.Page):
    start_time = time.monotonic()
    print(f"{start_time:.4f}: Main function started.")

    page.title = "AITO Vezérlőpult"
    page.theme_mode = ft.ThemeMode.DARK

    print(f"{time.monotonic():.4f}: --- STARTING INITIALIZATION ---")

    print(f"{time.monotonic():.4f}: Checking critical components...")
    if not all(comp is not None for comp in [ATOM_DATA, PROMPTS, CONFIG, vector_store, docs_vector_store]):
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
                args=(uploaded_file_path, docs_vector_store, CONFIG) # <--- A CONFIG hozzáadása itt
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

    print(f"{time.monotonic():.4f}: Initializing FilePicker...")
    file_picker = ft.FilePicker(on_result=on_document_upload)
    page.overlay.append(file_picker)
    print(f"{time.monotonic():.4f}: FilePicker OK.")

    # --- ESZKÖZ CSOMAGOLÓ FÜGGVÉNYEK LÉTREHOZÁSA ---
    def wrapped_search_memory_tool(query: str) -> str:
        """A teljes SYNERGAQUA memóriában (beszélgetésekben) keres releváns, teljes beszélgetések után."""
        return search_memory_tool(query=query, config=CONFIG, vector_store=vector_store)

    def wrapped_search_knowledge_base_tool(query: str) -> str:
        """Kizárólag a feltöltött dokumentumok (PDF, MD, TXT) tudásbázisában keres releváns információk után."""
        return search_knowledge_base_tool(query=query, config=CONFIG, docs_vector_store=docs_vector_store)

    def wrapped_list_uploaded_files_tool() -> str:
        """Kilistázza a Tudásbázisba feltöltött összes dokumentum egyedi fájlnevét."""
        return list_uploaded_files_tool(config=CONFIG)

    def wrapped_set_registry_value(key: str, value: str) -> str:
        """Beállít vagy frissít egy kulcs-érték párt a rendszer-nyilvántartásban."""
        return set_registry_value(key=key, value=value, config=CONFIG)

    def wrapped_get_registry_value(key: str) -> str:
        """Lekérdez egy értéket a rendszer-nyilvántartásból a kulcsa alapján."""
        return get_registry_value(key=key, config=CONFIG)

    def wrapped_list_registry_keys() -> str:
        """Kilistázza az összes kulcsot a rendszer-nyilvántartásból."""
        return list_registry_keys(config=CONFIG)

    def wrapped_generate_diagram_tool(definition: str, filename: str) -> str:
        """Egy szöveges definíció (pl. Graphviz DOT nyelv) alapján legenerál egy képfájlt."""
        return generate_diagram_tool(definition=definition, filename=filename, config=CONFIG)

    def wrapped_read_full_document_tool(filename: str) -> str:
        """Beolvassa egy adott nevű, korábban feltöltött dokumentum teljes, rekonstruált tartalmát."""
        return read_full_document_tool(filename=filename, config=CONFIG)

    def wrapped_display_image_tool(filename: str) -> str:
        """Megjelenít egy, a 'diagrams' mappában található képfájlt a chat ablakban."""
        return display_image_tool(filename=filename)

    def wrapped_set_meeting_status(active: bool, meeting_id: str = "") -> str:
        """Beállítja a megbeszélés állapotát a rendszer-nyilvántartásban."""
        return set_meeting_status(active=active, meeting_id=meeting_id, config=CONFIG)

    def wrapped_get_meeting_status() -> dict:
        """Lekérdezi a megbeszélés aktuális állapotát."""
        return get_meeting_status(config=CONFIG)


    print(f"{time.monotonic():.4f}: Initializing UI components (ListView)...")
    chat_history_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    print(f"{time.monotonic():.4f}: UI components (ListView) OK.")


    print(f"{time.monotonic():.4f}: Initializing TaskDispatcher...")
    task_dispatcher = TaskDispatcher(
        page=page,
        chat_history_view=chat_history_view,
        firestore_history=firestore_history,
        config=CONFIG,
        vector_store=vector_store,
        search_memory_tool=wrapped_search_memory_tool
    )
    print(f"{time.monotonic():.4f}: TaskDispatcher OK.")

    app_state = {"active_atom_id": INITIAL_ATOM_ID, "atom_chain": None}

    print(f"{time.monotonic():.4f}: Initializing UI components (TextField)...")
    input_field = ft.TextField(hint_text="Írj ide...", expand=True, border_color="white", multiline=True, min_lines=3, max_lines=5, shift_enter=True)
    print(f"{time.monotonic():.4f}: UI components (TextField) OK.")

def get_ai_response(user_message: HumanMessage, chain_for_request, atom_id_for_request: str, tool_registry: dict):
    try:
        config = {"configurable": {"session_id": CONFIG['session_id']}}

        current_input = {"input": user_message.content, "active_atom_role": atom_id_for_request}
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

                # --- JAVÍTOTT BLOKK: Kliens-oldali UI parancsok kezelése ---
                if isinstance(tool_output, str) and tool_output.startswith("UI_COMMAND:DISPLAY_IMAGE:"):
                    logging.info(f"UI parancs észlelve: {tool_output}")
                    image_path = tool_output.split(":", 2)[2]

                    page.run_thread(add_image_bubble_to_chat, image_path.strip())

                    # KRITIKUS JAVÍTÁS: Mivel a feladat (a kép megjelenítése) egy VÉGSŐ művelet,
                    # itt azonnal, erőszakosan kilépünk a teljes get_ai_response függvényből.
                    # Nincs több modellhívás, nincs több ciklus.
                    logging.info("UI parancs végrehajtva, a kör befejeződött.")
                    return  # <-- EZ A LÉNYEG!
                elif tool_name == "wrapped_generate_diagram_tool" and "IMAGE_PATH:" in tool_output:
                    logging.info(f"Diagram generálás utáni azonnali megjelenítés észlelve.")
                    # A szöveges üzenetet és az elérési utat szétválasztjuk
                    message_part, path_part = tool_output.split("IMAGE_PATH:")
                    image_path = path_part.strip()

                    # Először kiírjuk a szöveges üzenetet, hogy a diagram elkészült
                    tool_messages.append(ToolMessage(content=message_part.strip(), tool_call_id=tool_call['id'], name=tool_name))

                    # Majd azonnal meghívjuk a képmegjelenítőt
                    page.run_thread(add_image_bubble_to_chat, image_path)

                    # A biztonság kedvéért itt is kiléphetünk.
                    logging.info("A diagram azonnal megjelenítve, a kör befejeződött.")
                    return # Kilépés a teljes függvényből
                tool_messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call['id'], name=tool_name))

            current_input = {"input": tool_messages, "active_atom_role": atom_id_for_request}
            response = chain_for_request.invoke(current_input, config=config)

        # A ciklus végén a 'response' már a végleges, emberi válasz
        final_response = response
        final_response.name = atom_id_for_request
        final_response.additional_kwargs = {"timestamp": datetime.now(timezone.utc).isoformat()}

        # ... (a mentési és UI frissítési kód innentől változatlan) ...
        # ...
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

    def add_image_bubble_to_chat(image_path: str):
        """Biztonságosan hozzáad egy kép-buborékot a chat ablakhoz."""
        image_bubble = ft.Row(
            [ft.Container(
                content=ft.Image(src=image_path),
                padding=12, border_radius=ft.border_radius.all(15)
            )],
            alignment=ft.MainAxisAlignment.START
        )
        chat_history_view.controls.append(image_bubble)
        page.update()
        logging.info(f"Kép-buborék sikeresen hozzáadva a chathez: {image_path}")

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

        if user_input_text.strip().lower().startswith("/task"):
            task_content = user_input_text.strip()[5:].strip()
            firestore_history.add_message(human_message)
            task_id = task_dispatcher.start_new_task(task_content, initiated_by=CONFIG['user_id'])
            task_init_message = AIMessage(
                content=f"Rendben, a(z) '{task_content}' feladat fogadva (Azonosító: {task_id[:8]}...). ATOMOD átvette az irányítást.",
                name="SYSTEM"
            )
            chat_history_view.controls.append(MessageBubble(task_init_message))
            input_field.value = ""
            page.update()
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

    def initialize_app_in_background():
        """Minden lassú, blokkoló indítási feladatot elvégez egy háttérszálon."""
        logging.info("--- Háttér-inicializálás elindult ---")
        try:
            # 1. LASSÚ LÉPÉS: Az első ATOM motorjának beállítása
            logging.info("1/2: Motor-konfiguráció indítása...")
            switch_atom(INITIAL_ATOM_ID)
            logging.info("1/2: Motor-konfiguráció befejezve.")

            # 2. LASSÚ LÉPÉS: Előzmények letöltése és megjelenítése
            logging.info("2/2: Előzmények betöltése...")
            messages_to_load = firestore_history.messages

            if messages_to_load:
                for msg in messages_to_load:
                    chat_history_view.controls.append(MessageBubble(msg))

                page.update()
                time.sleep(0.5)
                chat_history_view.scroll_to(offset=-1, duration=300)

            page.update()
            logging.info("--- Háttér-inicializálás befejeződött, a rendszer készen áll. ---")
        except Exception as e:
            logging.error(f"KRITIKUS HIBA a háttér-inicializálás során: {e}", exc_info=True)
            chat_history_view.controls.append(
                MessageBubble(AIMessage(content=f"Indítási hiba: {e}", name="SYSTEM_ERROR"))
            )
            page.update()

    print(f"{time.monotonic():.4f}: Initializing UI components (Atom Buttons)...")
    atom_buttons = {}

    def switch_atom(selected_atom_id: str):
        app_state["active_atom_id"] = selected_atom_id
        current_atom_config = ATOM_DATA[app_state["active_atom_id"]]

        final_system_prompt = PROMPTS['team_simulation_template'].format(
            active_atom_role=selected_atom_id,
            personality_description=current_atom_config['personality'],
            grounding_instructions=PROMPTS['grounding_instructions']
        )

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

        # A modellnek már a becsomagolt, egyszerűsített eszközt adjuk át.
        tool_registry = {
            "wrapped_search_memory_tool": wrapped_search_memory_tool,
            "wrapped_search_knowledge_base_tool": wrapped_search_knowledge_base_tool,
            "wrapped_list_uploaded_files_tool": wrapped_list_uploaded_files_tool,
            "wrapped_set_registry_value": wrapped_set_registry_value,
            "wrapped_get_registry_value": wrapped_get_registry_value,
            "wrapped_list_registry_keys": wrapped_list_registry_keys,
            "wrapped_generate_diagram_tool": wrapped_generate_diagram_tool,
            "wrapped_read_full_document_tool": wrapped_read_full_document_tool,
            "wrapped_display_image_tool": wrapped_display_image_tool,
            "wrapped_set_meeting_status": wrapped_set_meeting_status,
            "wrapped_get_meeting_status": wrapped_get_meeting_status,
        }
        tools = list(tool_registry.values())
        llm_with_tools = llm.bind_tools(tools)

        app_state["tool_registry"] = tool_registry

        prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        chain_with_history = RunnableWithMessageHistory(
            prompt | llm_with_tools,
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

    for atom_id, data in ATOM_DATA.items():
        button = ft.ElevatedButton(
            text=data["label"],
            on_click=lambda e: switch_atom(e.control.data),
            data=atom_id
        )
        atom_buttons[atom_id] = button
    print(f"{time.monotonic():.4f}: UI components (Atom Buttons) OK.")


    # --- DOKUMENTUM FELTÖLTŐ GOMB ---
    print(f"{time.monotonic():.4f}: Initializing UI components (Upload Button)...")
    upload_button = ft.IconButton(
        icon=ft.Icons.UPLOAD_FILE,
        tooltip="Dokumentum feltöltése a Tudásbázisba",
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["pdf", "txt", "md"]
        ),
        icon_color="white"
    )
    print(f"{time.monotonic():.4f}: UI components (Upload Button) OK.")

    print(f"{time.monotonic():.4f}: Building page layout...")
    atom_selector = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[upload_button] + list(atom_buttons.values())
    )

    page.add(
        ft.Column([
            atom_selector,
            ft.Container(content=chat_history_view, border=ft.border.all(1, "white"), border_radius=ft.border_radius.all(5), padding=10, expand=True),
            ft.Row([input_field, ft.IconButton(icon=ft.Icons.SEND_ROUNDED, tooltip="Küldés", on_click=send_click, icon_color="white")]),
        ], expand=True)
    )
    print(f"{time.monotonic():.4f}: Page layout OK.")

    page.padding = 20
    page.update() # Ez azonnal kirajzolja a statikus, üres UI-t

    # Elindítjuk a teljes lassú inicializálást a háttérben
    page.run_thread(initialize_app_in_background)

if __name__ == "__main__":
    ft.app(target=main)