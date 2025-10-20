# debug_main.py (6. LÉPÉS: A valódi switch_atom tesztelése)

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
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- Saját modulok importálása ---
# Most már szükségünk van az összes eszközre is!
from shared_components import (
    ATOM_DATA, PROMPTS, message_to_document, chunk_text, 
    search_memory_tool, search_knowledge_base_tool, list_uploaded_files_tool, 
    set_registry_value, get_registry_value, list_registry_keys, 
    generate_diagram_tool, read_full_document_tool, display_image_tool, 
    set_meeting_status, get_meeting_status
)
# from task_dispatcher import TaskDispatcher # Ezt még mindig nem
# from document_processor import process_and_store_document # Ezt még mindig nem

# --- Konfiguráció betöltése (TESZTELVE, OK) ---
try:
    with open('config_aito.yaml', 'r', encoding='utf-8') as file:
        CONFIG = yaml.safe_load(file)
    print("DEBUG: Az AITO konfiguráció (config_aito.yaml) sikeresen betöltve.")
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
    print("DEBUG: Adatbázis útvonalak beállítva.")

    print("DEBUG: Google Cloud embedding kliens inicializálása...")
    google_embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=CONFIG['project_id'],
    )
    print("DEBUG: Embedding kliens inicializálva. OK.")

    print("DEBUG: Chroma (beszélgetések) csatlakoztatása...")
    vector_store = Chroma(
        persist_directory=CHROMA_CONVERSATION_PATH,
        embedding_function=google_embeddings
    )
    print("DEBUG: Chroma (beszélgetések) OK.")

    print("DEBUG: Chroma (dokumentumok) csatlakoztatása...")
    docs_vector_store = Chroma(
        persist_directory=CHROMA_DOCS_PATH,
        embedding_function=google_embeddings
    )
    print("DEBUG: Chroma (dokumentumok) OK.")

    print("DEBUG: SQLite (napló) csatlakoztatása...")
    connection_string = f"sqlite:///{SQLITE_HISTORY_FILE}"
    firestore_history = SQLChatMessageHistory(
        session_id=CONFIG['session_id'],
        connection=connection_string
    )
    print("DEBUG: SQLite (napló) OK.")
    
    print("DEBUG: Próbaolvasás az SQLite naplóból...")
    try:
        initial_messages = firestore_history.messages
        print(f"DEBUG: Próbaolvasás OK. ({len(initial_messages)} üzenet az adatbázisban)")
    except Exception as hist_err:
        print(f"DEBUG: Figyelmeztetés: Nem sikerült betölteni az előzményeket: {hist_err}")
        
except Exception as e:
    print(f"!!! KRITIKUS HIBA az inicializálás közben: {e} !!!")
    import traceback
    traceback.print_exc()

# --- Hamis MessageBubble osztály (az eredeti main_aito.py-ból) ---
# Erre csak azért van szükség, hogy a send_click működjön.
class MessageBubble(ft.Row):
    def __init__(self, message: HumanMessage or AIMessage):
        super().__init__()
        # Ez egy egyszerűsített verzió a debugoláshoz
        speaker = getattr(message, 'name', None) or (CONFIG.get('user_id') if message.type == 'human' else "Ismeretlen")
        display_speaker = "Te" if speaker == CONFIG.get('user_id') else speaker
        bubble_color = ft.Colors.BLACK
        if speaker != CONFIG.get('user_id'):
            color_name = ATOM_DATA.get(speaker, {}).get("color", "BLACK")
            bubble_color = getattr(ft.Colors, color_name, ft.Colors.BLACK)

        bubble_container = ft.Container(
            content=ft.Markdown(
                f"**{display_speaker}:** {message.content}" if display_speaker != "Te" else message.content,
                selectable=True,
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


# --- A FŐ FÜGGVÉNY (leegyszerűsítve) ---
def main(page: ft.Page):
    start_time = time.monotonic()
    print(f"{start_time:.4f}: DEBUG Main function started.")

    page.title = "AITO Vezérlőpult (DEBUG MÓD)"
    page.theme_mode = ft.ThemeMode.DARK
    logging.info("--- DEBUG MÓD: UI Inicializálás ---")

    # === Kritikus ellenőrzés ===
    if not all(comp is not None for comp in [ATOM_DATA, PROMPTS, CONFIG, vector_store, docs_vector_store, firestore_history]):
        page.add(ft.Text("Hiba: Az adatbázis-komponensek betöltése sikertelen!", color=ft.Colors.RED))
        return
    print(f"{time.monotonic():.4f}: DEBUG: Kritikus komponensek (DB-k) ellenőrizve, OK.")

    
    # --- UI Komponensek (adatbázis és logika nélkül) ---
    chat_history_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    input_field = ft.TextField(hint_text="Írj ide...", expand=True, border_color="white", multiline=True, min_lines=3, max_lines=5, shift_enter=True)
    
    # === VALÓDI ESZKÖZ CSOMAGOLÓK ===
    # Ezek kellenek a valódi switch_atom-hoz
    def wrapped_search_memory_tool(query: str) -> str:
        return search_memory_tool(query=query, config=CONFIG, vector_store=vector_store)
    def wrapped_search_knowledge_base_tool(query: str) -> str:
        return search_knowledge_base_tool(query=query, config=CONFIG, docs_vector_store=docs_vector_store)
    def wrapped_list_uploaded_files_tool() -> str:
        return list_uploaded_files_tool(config=CONFIG, docs_vector_store=docs_vector_store) # <- FIGYELEM: Ezt ki kellett egészítenem a docs_vector_store-ral
    def wrapped_set_registry_value(key: str, value: str) -> str:
        return set_registry_value(key=key, value=value, config=CONFIG)
    def wrapped_get_registry_value(key: str) -> str:
        return get_registry_value(key=key, config=CONFIG)
    def wrapped_list_registry_keys() -> str:
        return list_registry_keys(config=CONFIG)
    def wrapped_generate_diagram_tool(definition: str, filename: str) -> str:
        return generate_diagram_tool(definition=definition, filename=filename, config=CONFIG)
    def wrapped_read_full_document_tool(filename: str) -> str:
        return read_full_document_tool(filename=filename, docs_vector_store=docs_vector_store) # <- FIGYELEM: Ezt ki kellett egészítenem a docs_vector_store-ral
    def wrapped_display_image_tool(filename: str) -> str:
        return display_image_tool(filename=filename)
    def wrapped_set_meeting_status(active: bool, meeting_id: str = "") -> str:
        return set_meeting_status(active=active, meeting_id=meeting_id, config=CONFIG)
    def wrapped_get_meeting_status() -> dict:
        return get_meeting_status(config=CONFIG)

    print(f"{time.monotonic():.4f}: DEBUG: Eszköz csomagolók (wrapperek) létrehozva.")

    # === ÁLLAPOT ===
    app_state = {"active_atom_id": INITIAL_ATOM_ID, "atom_chain": None, "tool_registry": {}}

    # === AZ IGAZI `switch_atom` FÜGGVÉNY (main_aito.py-ból másolva) ===
    atom_buttons = {} # Ezt előre kell definiálni, hogy a switch_atom lássa

    def switch_atom(selected_atom_id: str):
        print(f"--- DEBUG: VALÓDI ATOM VÁLTÁS INDUL: {selected_atom_id} ---")
        app_state["active_atom_id"] = selected_atom_id
        current_atom_config = ATOM_DATA[app_state["active_atom_id"]]
        print(f"DEBUG: '{selected_atom_id}' konfigurációja betöltve. Modell: {current_atom_config.get('model_name')}")

        final_system_prompt = PROMPTS['team_simulation_template'].format(
            active_atom_role=selected_atom_id,
            personality_description=current_atom_config['personality'],
            grounding_instructions=PROMPTS['grounding_instructions']
        )
        print("DEBUG: System prompt sikeresen összeállítva.")

        # === EZ A GYANÚSÍTOTT BLOKK ===
        print(f"DEBUG: ChatVertexAI kliens inicializálása... (Modell: {current_atom_config['model_name']})")
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
        print(f"DEBUG: ChatVertexAI kliens inicializálva. OK.")
        # ================================

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
        
        print("DEBUG: Eszközök hozzákötése a modellhez (bind_tools)...")
        llm_with_tools = llm.bind_tools(tools)
        print("DEBUG: Eszközök hozzákötve. OK.")

        app_state["tool_registry"] = tool_registry

        prompt = ChatPromptTemplate.from_messages([
            ("system", final_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        print("DEBUG: RunnableWithMessageHistory lánc összeállítása...")
        chain_with_history = RunnableWithMessageHistory(
            prompt | llm_with_tools,
            lambda session_id: firestore_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        app_state["atom_chain"] = chain_with_history
        print(f"DEBUG: Motor átkonfigurálva: {app_state['active_atom_id']} aktív.")

        page.title = f"AITO Vezérlőpult - {app_state['active_atom_id']} Aktív (DEBUG)"

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
        
        print("DEBUG: Gomb stílusok beállítva.")
        page.update()
        print("DEBUG: Page.update() (switch_atom-ból) lefutott.")

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
        tooltip="Dokumentum feltöltése (DEBUG - Inaktív)",
        on_click=None,
        icon_color="white"
    )
    
    atom_selector = ft.Row(
        alignment=ft.MainAxisAlignment.CENTER,
        controls=[upload_button] + list(atom_buttons.values())
    )
    
    # --- Hamis send_click (nincs AI hívás) ---
    def send_click(e):
        # Ez a függvény egyelőre maradjon buta, 
        # ne hívja az AI-t, csak jelenítsen meg valamit
        user_input_text = input_field.value
        if not user_input_text.strip(): 
            return
        
        # Ez a sor az eredeti MessageBubble-t használja
        human_bubble = MessageBubble(HumanMessage(content=user_input_text, name=CONFIG['user_id']))
        chat_history_view.controls.append(human_bubble)
        input_field.value = ""
        
        ai_response_bubble = MessageBubble(AIMessage(content="Ez egy HAMIS válasz. A switch_atom tesztelése folyik.", name=app_state["active_atom_id"]))
        chat_history_view.controls.append(ai_response_bubble)
        page.update()

    # --- Oldal Elrendezés Összeállítása ---
    page.add(
        ft.Column([
            atom_selector,
            ft.Container(content=chat_history_view, border=ft.border.all(1, "white"), border_radius=ft.border_radius.all(5), padding=10, expand=True),
            ft.Row([input_field, ft.IconButton(icon=ft.Icons.SEND_ROUNDED, tooltip="Küldés", on_click=send_click, icon_color="white")]),
        ], expand=True)
    )
    
    page.padding = 20
    
    # === AZONNALI PAGE.UPDATE() ===
    print(f"{time.monotonic():.4f}: Azonnali page.update() hívása (a switch_atom ELŐTT)...")
    page.update()
    print(f"{time.monotonic():.4f}: Első page.update() lefutott.")

    # === AZ ELSŐ, IGAZI SWITCH_ATOM HÍVÁSA (A VÁRT HIBA HELYE) ===
    try:
        print(f"{time.monotonic():.4f}: A VALÓDI switch_atom({INITIAL_ATOM_ID}) hívása...")
        switch_atom(INITIAL_ATOM_ID) # EZ FOG MOST FAGYNI?
        print(f"{time.monotonic():.4f}: A VALÓDI switch_atom() sikeresen lefutott.")
    except Exception as e:
         print(f"{time.monotonic():.4f}: HIBA a switch_atom hívásakor: {e}", exc_info=True)
         page.add(ft.Text(f"HIBA a switch_atom hívásakor: {e}", color=ft.Colors.RED))
         page.update()
    
    print(f"{time.monotonic():.4f}: DEBUG main függvény véget ért. A Flet fut (ft.app).")


if __name__ == "__main__":
    ft.app(target=main)