# shared_components.py

import yaml
import os
import graphviz
import tiktoken
from collections import Counter
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import sqlite3

# --- YAML KONFIGURÁCIÓK BETÖLTÉSE ---
try:
    with open('atoms.yaml', 'r', encoding='utf-8') as file:
        ATOM_DATA = yaml.safe_load(file)
    print("ATOM konfiguráció sikeresen betöltve az 'atoms.yaml' fájlból.")
    
    with open('prompts.yaml', 'r', encoding='utf-8') as file:
        PROMPTS = yaml.safe_load(file)
    print("Prompt sablonok sikeresen betöltve a 'prompts.yaml' fájlból.")

except Exception as e:
    print(f"HIBA a YAML fájlok olvasása közben: {e}")
    ATOM_DATA, PROMPTS = {}, {}

# --- FÜGGVÉNYEK ---

def chunk_text(text: str) -> list[str]:
    """Feloszt egy hosszabb szöveget tokenek alapján, kb. 1000 tokenes darabokra, 100 tokenes átfedéssel."""
    # A 'cl100k_base' kódolás a legtöbb modern OpenAI modellhez (pl. GPT-4) megfelelő.
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    chunks = []
    chunk_size = 1000
    overlap = 100

    i = 0
    while i < len(tokens):
        # Meghatározzuk a darab végét
        end = min(i + chunk_size, len(tokens))

        # A token darabot visszaalakítjuk szöveggé
        chunk_tokens = tokens[i:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # A következő darab kezdete az átfedés figyelembevételével
        i += chunk_size - overlap

    return chunks

def message_to_document(content: str, speaker: str, timestamp: str, session_id: str, chunk_num: int = 1, total_chunks: int = 1, meeting_id: str = None) -> Document:
    """Létrehoz egy LangChain Document objektumot a megadott adatokból és metaadatokból."""
    metadata = {
        "speaker": speaker,
        "timestamp": timestamp,
        "session_id": session_id,
        "chunk_number": chunk_num,
        "total_chunks": total_chunks
    }
    if meeting_id:
        metadata["meeting_id"] = meeting_id

    return Document(
        page_content=content,
        metadata=metadata
    )

def search_memory_tool(query: str, config: dict, vector_store: VectorStore) -> str:
    """A teljes AITO memóriában keres releváns, teljes beszélgetések után."""
    print(f"--- ESZKÖZHÍVÁS: Kontextus-alapú Memória Keresés, Keresőkifejezés: '{query}' ---")
    if not vector_store:
        return "Hiba: A Vector Store nincs inicializálva."

    try:
        # 1. FÁZIS: Vektoros keresés a legrelevánsabb DARABOKért
        scored_chunks = vector_store.similarity_search_with_score(query, k=5)
        if not scored_chunks:
            return "A memóriában nem található releváns információ."

        # 2. FÁZIS: A fő kontextus (beszélgetés) azonosítása
        session_ids = [chunk.metadata.get('session_id') for chunk, score in scored_chunks if chunk.metadata.get('session_id')]
        if not session_ids:
            # Visszaeső logika: ha nincsenek session_id-k, adjuk vissza a régi stílusú találatot.
            return "A memóriában talált releváns darabok (kontextus nélkül):\n\n" + "\n---\n".join([chunk.page_content for chunk, score in scored_chunks])

        most_common_session_id = Counter(session_ids).most_common(1)[0][0]
        print(f"Legrelevánsabb beszélgetés azonosítva: {most_common_session_id}")

        # 3. FÁZIS: A teljes kontextus rekonstruálása SQLite-ból
        LOCAL_DB_PATH = "./aito_local_data"
        SQLITE_HISTORY_FILE = f"{LOCAL_DB_PATH}/aito_chat_history.db"
        connection_string = f"sqlite:///{SQLITE_HISTORY_FILE}"
        history = SQLChatMessageHistory(
            session_id=most_common_session_id,
            connection=connection_string
        )
        
        full_conversation_messages = history.messages
        print(f"Teljes beszélgetés ({len(full_conversation_messages)} üzenet) sikeresen lekérve az SQLite adatbázisból.")

        formatted_context = ""
        for msg in full_conversation_messages:
            speaker = getattr(msg, 'name', 'Ismeretlen')
            display_speaker = "Te" if speaker == config.get('user_id', 'user') else speaker
            formatted_context += f"{display_speaker}: {msg.content}\n"

        # 4. FÁZIS: Végső válasz összeállítása
        response = f"A keresés a '{most_common_session_id}' azonosítójú beszélgetést találta a legrelevánsabbnak. A teljes rekonstruált beszélgetés:\n\n---\n{formatted_context.strip()}\n---"
        return response

    except Exception as e:
        # A traceback importálása csak itt, a hibakezeléshez
        import traceback
        print("\n!!! RÉSZLETES HIBAJELENTÉS A KERESÉSI ESZKÖZBŐL !!!")
        traceback.print_exc()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return f"Hiba történt a memória keresése közben: {e}"

def search_knowledge_base_tool(query: str, config: dict, docs_vector_store: VectorStore) -> str:
    """Kizárólag a feltöltött dokumentumok (PDF, MD, TXT) tudásbázisában keres releváns információk után."""
    print(f"--- ESZKÖZHÍVÁS: Tudásbázis_Keresése, Kifejezés: '{query}' ---")
    try:
        # Itt elég egy egyszerűbb keresés, ami a legjobb darabokat adja vissza
        scored_documents = docs_vector_store.similarity_search_with_score(query, k=3) # Kérjünk le 3 releváns darabot

        if not scored_documents:
            return "A tudásbázisban nem található releváns dokumentumrészlet."

        results_text = "A tudásbázisból a következő releváns információk kerültek elő:\n\n"
        for (doc, score) in scored_documents:
            results_text += f"- Forrás: {doc.metadata.get('source_document', 'Ismeretlen')}\n"
            results_text += f"  Relevancia: {1-score:.2f}\n" # A koszinusz-hasonlóságot jelenítjük meg (1 - távolság)
            results_text += f"  Részlet: {doc.page_content}\n---\n"
        return results_text

    except Exception as e:
        return f"Hiba történt a tudásbázis keresése közben: {e}"

def list_uploaded_files_tool(config: dict, docs_vector_store: VectorStore) -> str:
    """Kilistázza a Tudásbázisba feltöltött összes dokumentum egyedi fájlnevét."""
    print("--- ESZKÖZHÍVÁS: Feltöltött_Fájlok_Listázása ---")
    try:
        # ChromaDB-ben a dokumentumokat a get() metódussal tudjuk lekérni.
        # A `where` feltétel nélküli get() az összes dokumentumot visszaadja.
        all_docs = docs_vector_store.get()

        # A 'metadatas' listából kinyerjük a 'source_document' értékeket
        unique_files = set(
            metadata.get('source_document')
            for metadata in all_docs.get('metadatas', [])
            if metadata and metadata.get('source_document')
        )

        if not unique_files:
            return "A Tudásbázis jelenleg üres."
        return "A Tudásbázisban a következő dokumentumok találhatók:\n- " + "\n- ".join(sorted(list(unique_files)))
    except Exception as e:
        return f"Hiba történt a fájlok listázása közben: {e}"

def _get_registry_db_path():
    # Ez a központi függvény, amely meghatározza a rendszer-nyilvántartás adatbázisának helyét.
    # Minden registry-művelet, beleértve az ágensek jegyzetfüzeteit is, ezt az adatbázist használja.
    return os.path.join("./aito_local_data", "system_registry.db")

def _initialize_registry_db():
    db_path = _get_registry_db_path()
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE registry (
                key TEXT PRIMARY KEY,
                value TEXT,
                last_updated TEXT
            )
        ''')
        conn.commit()
        conn.close()

def set_registry_value(key: str, value: str, config: dict) -> str:
    """Beállít vagy frissít egy kulcs-érték párt a helyi SQLite rendszer-nyilvántartásban."""
    try:
        db_path = _get_registry_db_path()
        _initialize_registry_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO registry (key, value, last_updated)
            VALUES (?, ?, ?)
        ''', (key, value, datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        return f"A '{key}' kulcs sikeresen beállítva a következőre: '{value}'."
    except Exception as e:
        return f"Hiba a registry írása közben: {e}"

def get_registry_value(key: str, config: dict) -> str:
    """Lekérdez egy értéket a helyi SQLite rendszer-nyilvántartásból a kulcsa alapján."""
    try:
        db_path = _get_registry_db_path()
        _initialize_registry_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM registry WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return f"A '{key}' kulcs értéke: '{row[0]}'."
        else:
            return f"A '{key}' kulcs nem található a nyilvántartásban."
    except Exception as e:
        return f"Hiba a registry olvasása közben: {e}"

def list_registry_keys(config: dict) -> str:
    """Kilistázza az összes kulcsot a helyi SQLite rendszer-nyilvántartásból."""
    try:
        db_path = _get_registry_db_path()
        _initialize_registry_db()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key FROM registry")
        keys = [row[0] for row in cursor.fetchall()]
        conn.close()
        if not keys:
            return "A rendszer-nyilvántartás üres."
        return "A rendszer-nyilvántartásban a következő kulcsok találhatók: " + ", ".join(keys)
    except Exception as e:
        return f"Hiba a kulcsok listázása közben: {e}"

def set_meeting_status(active: bool, config: dict, meeting_id: str = "") -> str:
    """Beállítja a megbeszélés állapotát a rendszer-nyilvántartásban."""
    try:
        set_registry_value('is_meeting_active', str(active), config)
        if active:
            set_registry_value('current_meeting_id', meeting_id, config)
            return f"A megbeszélés '{meeting_id}' azonosítóval aktív állapotba került."
        else:
            set_registry_value('current_meeting_id', "", config)
            return "A megbeszélés inaktív állapotba került."
    except Exception as e:
        return f"Hiba a megbeszélés állapotának beállítása közben: {e}"

def get_meeting_status(config: dict) -> dict:
    """Lekérdezi a megbeszélés aktuális állapotát a rendszer-nyilvántartásból."""
    try:
        is_active_str = get_registry_value('is_meeting_active', config)
        is_active = is_active_str.split("'")[3].lower() == 'true' if "kulcs értéke" in is_active_str else False

        meeting_id_str = get_registry_value('current_meeting_id', config)
        meeting_id = meeting_id_str.split("'")[3] if "kulcs értéke" in meeting_id_str else ""

        return {'is_active': is_active, 'meeting_id': meeting_id}
    except Exception as e:
        print(f"Hiba a megbeszélés állapotának lekérdezése közben: {e}")
        return {'is_active': False, 'meeting_id': ""}

def generate_diagram_tool(definition: str, filename: str, config: dict) -> str:
    """Egy szöveges definíció (pl. Graphviz DOT nyelv) alapján legenerál egy képfájlt."""
    print(f"--- ESZKÖZHÍVÁS: Diagram_Generálása, Fájlnév: '{filename}' ---")
    try:
        # Biztonsági okokból ellenőrizzük a fájlnevet
        if not filename.endswith(('.png', '.svg')):
            return "Hiba: A fájlnévnek .png vagy .svg végződésűnek kell lennie."

        # Létrehozzuk a diagrams mappa abszolút elérési útját
        output_dir = os.path.join(os.getcwd(), "diagrams")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Létrehozzuk a Graphviz objektumot a szöveges definícióból
        source = graphviz.Source(definition)
        # Rendereljük a fájlt (a formátumot a kiterjesztésből veszi)
        source.render(filepath, format=filepath.split('.')[-1], cleanup=True)

        return f"A '{filename}' diagram sikeresen legenerálva és elmentve a '{output_dir}' mappába. IMAGE_PATH:{filepath}"
    except Exception as e:
        return f"Hiba történt a diagram generálása közben: {e}"


def display_image_tool(filename: str) -> str:
    """Megjelenít egy, a 'diagrams' mappában található képfájlt a chat ablakban."""
    print(f"--- ESZKÖZHÍVÁS: Kép_Megjelenítése, Fájlnév: '{filename}' ---")
    # Létrehozzuk a keresett fájl abszolút elérési útját
    filepath = os.path.join(os.getcwd(), "diagrams", filename)
    if os.path.exists(filepath):
        # Visszaadjuk a speciális parancsot a UI számára
        return f"UI_COMMAND:DISPLAY_IMAGE:{filepath}"
    else:
        return f"Hiba: A '{filename}' nem található a 'diagrams' mappában."

def read_full_document_tool(filename: str, docs_vector_store: VectorStore) -> str:
    """
    Beolvassa egy adott nevű, korábban feltöltött dokumentum teljes, rekonstruált tartalmát
    a ChromaDB-ből.
    Ellenőrizve: A függvény a ChromaDB get metódusát használja a where={"source_document": filename} szűrővel.
    """
    print(f"--- ESZKÖZHÍVÁS: Teljes_Dokumentum_Olvasása, Fájlnév: '{filename}' ---")
    try:
        # A ChromaDB-ben a 'where' szűrővel tudunk metaadatokra keresni.
        retrieved_docs = docs_vector_store.get(
            where={"source_document": filename}
        )

        if not retrieved_docs or not retrieved_docs.get('documents'):
            return f"Hiba: A '{filename}' nevű dokumentum nem található a tudásbázisban."

        # A get() metódus a dokumentumokat és a metaadatokat külön listákban adja vissza.
        # Párosítanunk kell őket, majd rendezni a 'chunk_number' alapján.
        docs_with_meta = zip(retrieved_docs['documents'], retrieved_docs['metadatas'])

        # Rendezés a 'chunk_number' alapján, ami a metaadatokban van.
        sorted_docs = sorted(docs_with_meta, key=lambda item: item[1].get('chunk_number', 0))

        # A rendezett dokumentumok tartalmának összefűzése.
        full_text = "".join(doc[0] for doc in sorted_docs)

        return f"DOCUMENT_CONTENT:\n{full_text}"

    except Exception as e:
        return f"Hiba történt a(z) '{filename}' dokumentum olvasása közben: {e}"

print("Közös komponensek modul (shared_components.py) sikeresen betöltve.")

def read_agent_notebook(agent_id: str, config: dict) -> str:
    """Beolvassa egy adott ágens privát jegyzetfüzetének tartalmát."""
    notebook_key = f"notebook_{agent_id.lower()}" # Pl. notebook_atom1
    value_str = get_registry_value(notebook_key, config) # A meglévő registry olvasót használjuk
    if f"'{notebook_key}' kulcs nem található" in value_str:
        return f"A(z) {agent_id} jegyzetfüzete még üres."
    else:
        # Kivesszük a körítést a get_registry_value válaszából
        try:
            content = value_str.split(f"A '{notebook_key}' kulcs értéke: '")[1].rstrip("'.")
            return f"A(z) {agent_id} jegyzetfüzetének tartalma:\n---\n{content}\n---"
        except IndexError:
            return f"Hiba a(z) {agent_id} jegyzetfüzet tartalmának értelmezésekor."

def update_agent_notebook(agent_id: str, new_content: str, config: dict) -> str:
    """Felülírja egy adott ágens privát jegyzetfüzetének tartalmát."""
    notebook_key = f"notebook_{agent_id.lower()}"
    result = set_registry_value(notebook_key, new_content, config) # A meglévő registry írót használjuk
    if "sikeresen beállítva" in result:
        return f"A(z) {agent_id} jegyzetfüzete sikeresen frissítve."
    else:
        return f"Hiba a(z) {agent_id} jegyzetfüzetének frissítése közben: {result}"
