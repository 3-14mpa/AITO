# document_processor.py

import os
import time
import flet as ft
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from shared_components import chunk_text, summarize_document

def create_document_chunk(content: str, source: str, chunk_num: int, total_chunks: int) -> Document:
    """Creates a LangChain Document object for a document chunk."""
    return Document(
        page_content=content,
        metadata={
            "source_document": source,
            "chunk_number": chunk_num,
            "total_chunks": total_chunks
        }
    )

def process_and_store_document(filepath: str, docs_vector_store, config: dict, page: ft.Page):
    """Loads, processes, chunks, and stores a document in the specified vector store."""
    print(f"--- Dokumentum feldolgozása: {filepath} ---")
    file_name = os.path.basename(filepath) # Fájlnév kinyerése

    # Segédfüggvény a biztonságos UI frissítéshez, már itt definiáljuk, hogy a `except` blokk is elérje
    def _add_msg_to_chat(msg):
        try:
            from aito_main_rebuild import MessageBubble # Importálás a használat helyén
            chat_history_view = page.controls[0].controls[1].content
            chat_history_view.controls.append(MessageBubble(msg))
            page.update()
        except Exception as ui_update_err:
            print(f"HIBA a chat UI frissítése közben: {ui_update_err}")


    try:
        # === KORÁBBI VERZIÓ TÖRLÉSE ===
        print(f"Korábbi '{file_name}' darabok keresése és törlése...")
        try:
            # Lekérdezzük az összes ID-t, ami ehhez a fájlhoz tartozik
            existing_ids = docs_vector_store.get(where={"source_document": file_name}).get("ids", [])
            if existing_ids:
                print(f"  {len(existing_ids)} korábbi darab törlése...")
                docs_vector_store.delete(ids=existing_ids)
                print(f"  Korábbi darabok sikeresen törölve.")
            else:
                print(f"  Nincsenek korábbi darabok ehhez a fájlhoz.")
        except Exception as delete_err:
            # Logoljuk a hibát, de folytatjuk a feltöltéssel
            print(f"!!! FIGYELMEZTETÉS: Hiba történt a korábbi darabok törlése közben: {delete_err}")
        # =============================

        # Determine loader based on file extension
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.lower().endswith(".txt") or filepath.lower().endswith(".md"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            print(f"HIBA: Nem támogatott fájltípus: {os.path.basename(filepath)}")
            # Optionally send a message to the chat about the unsupported file type
            error_message = AIMessage(content=f"'{os.path.basename(filepath)}' fájltípus nem támogatott.", name="SYSTEM_ERROR")
            page.run_thread(_add_msg_to_chat, error_message)
            return

        # Load the document content
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])

        # Chunk the text using the shared function
        text_chunks = chunk_text(full_text)

        # Create Document objects for each chunk AND ADD THEM IMMEDIATELY
        total_chunks_processed = len(text_chunks)
        added_chunks_count = 0
        for i, chunk in enumerate(text_chunks):
            doc = create_document_chunk(
                content=chunk,
                source=os.path.basename(filepath),
                chunk_num=i + 1,
                total_chunks=total_chunks_processed
            )
            # === AZONNALI HOZZÁADÁS DARABONKÉNT, ÚJRAPRÓBÁLKOZÁSSAL ===
            max_retries = 5
            added_successfully = False
            for attempt in range(max_retries + 1):
                try:
                    docs_vector_store.add_documents([doc])
                    print(f"  Darab #{i + 1}/{total_chunks_processed} sikeresen hozzáadva (próbálkozás: {attempt + 1}).")
                    added_chunks_count += 1
                    added_successfully = True
                    time.sleep(5) # Consider making this sleep configurable or removing it if not necessary for rate limiting
                    break # Kilépés az újrapróbálkozási ciklusból

                except Exception as add_err:
                    is_quota_error = "429" in str(add_err)
                    if is_quota_error and attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        print(f"!!! KVÓTA HIBA a(z) {i + 1}. darabnál ({attempt + 1}. próbálkozás). Várakozás {wait_time} mp...")
                        time.sleep(wait_time)
                    else:
                        print(f"!!! VÉGLEGES HIBA a(z) {i + 1}. darab hozzáadása közben ({attempt + 1}. próbálkozás): {add_err}")
                        break # Kilépés az újrapróbálkozási ciklusból, a darab kimarad

        # A ciklus után már csak az összefoglaló kiírás marad
        file_name = os.path.basename(filepath)
        if added_chunks_count == total_chunks_processed:
            print(f"--- '{file_name}' sikeresen feldolgozva: {added_chunks_count} darab mentve a memóriába. ---")

            # === ÖSSZEFOGLALÓ KÉSZÍTÉSE ÉS TÁROLÁSA ===
            try:
                print(f"Összefoglaló készítése a(z) '{file_name}' dokumentumhoz...")
                summary_content = summarize_document(full_text, config)
                summary_filename = f"SUM_{file_name}"

                # Mentsük az összefoglalót egy külön fájlba is (opcionális, de jó gyakorlat)
                summary_file_path = os.path.join(os.path.dirname(filepath), summary_filename)
                with open(summary_file_path, "w", encoding="utf-8") as f:
                    f.write(summary_content)
                print(f"Összefoglaló sikeresen elmentve a(z) '{summary_file_path}' fájlba.")

                # Töröljük a korábbi összefoglaló-darabokat is
                summary_existing_ids = docs_vector_store.get(where={"source_document": summary_filename}).get("ids", [])
                if summary_existing_ids:
                    print(f"  {len(summary_existing_ids)} korábbi összefoglaló-darab törlése...")
                    docs_vector_store.delete(ids=summary_existing_ids)

                # Daraboljuk és tároljuk az összefoglalót a vektoradatbázisban
                summary_chunks = chunk_text(summary_content)
                summary_chunks_added = 0
                for i, chunk in enumerate(summary_chunks):
                    doc = create_document_chunk(
                        content=chunk,
                        source=summary_filename,
                        chunk_num=i + 1,
                        total_chunks=len(summary_chunks)
                    )
                    # === AZONNALI HOZZÁADÁS DARABONKÉNT, ÚJRAPRÓBÁLKOZÁSSAL (ÖSSZEFOGLALÓHOZ) ===
                    max_retries = 5
                    for attempt in range(max_retries + 1):
                        try:
                            docs_vector_store.add_documents([doc])
                            print(f"  Összefoglaló darab #{i + 1}/{len(summary_chunks)} sikeresen hozzáadva.")
                            summary_chunks_added += 1
                            time.sleep(5)  # API hívások közötti szünet
                            break  # Sikeres hozzáadás után kilépünk a ciklusból
                        except Exception as add_err:
                            is_quota_error = "429" in str(add_err) or "RESOURCE_EXHAUSTED" in str(add_err)
                            if is_quota_error and attempt < max_retries:
                                wait_time = (attempt + 1) * 5
                                print(f"!!! KVÓTA HIBA az összefoglaló darabnál. Várakozás {wait_time} mp...")
                                time.sleep(wait_time)
                            else:
                                print(f"!!! VÉGLEGES HIBA az összefoglaló darab hozzáadása közben: {add_err}")
                                break # A darab kimarad

                print(f"Összefoglaló ({summary_chunks_added}/{len(summary_chunks)} darab) sikeresen hozzáadva a tudásbázishoz '{summary_filename}' néven.")

                if summary_chunks_added == len(summary_chunks):
                    completion_message_content = f"'{file_name}' feldolgozása és automatikus összefoglalása sikeresen befejeződött."
                else:
                    completion_message_content = f"'{file_name}' feldolgozása sikeres, de az összefoglaló mentése közben hibák léptek fel."

            except Exception as summary_err:
                print(f"!!! HIBA az összefoglaló készítése vagy tárolása közben: {summary_err}")
                completion_message_content = f"'{file_name}' feldolgozása sikeres, de az automatikus összefoglalás hibára futott."
            # =============================================

        else:
            print(f"--- '{file_name}' feldolgozása BEFEJEZVE HIBÁKKAL: {added_chunks_count}/{total_chunks_processed} darab mentve. Kérlek, ellenőrizd a naplót. ---")
            completion_message_content = f"'{file_name}' feldolgozása hibákkal fejeződött be ({added_chunks_count}/{total_chunks_processed} darab mentve)."

        # === BEFEJEZŐ ÜZENET KÜLDÉSE A CHATBE ===
        completion_message = AIMessage(content=completion_message_content, name="SYSTEM")
        page.run_thread(_add_msg_to_chat, completion_message)
        # ==========================================

    except Exception as e:
        print(f"HIBA a dokumentum feldolgozása közben: {e}")
        import traceback
        traceback.print_exc()
        # Ide is betehetnénk egy hibaüzenet küldést a chatbe, ha a teljes feldolgozás elhasal
        error_message = AIMessage(content=f"Kritikus hiba '{os.path.basename(filepath)}' feldolgozása közben: {e}", name="SYSTEM_ERROR")
        page.run_thread(_add_msg_to_chat, error_message)

print("Dokumentum feldolgozó modul (document_processor.py) betöltve.")
