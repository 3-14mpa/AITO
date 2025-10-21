# document_processor.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from shared_components import chunk_text

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

def process_and_store_document(filepath: str, docs_vector_store, config: dict):
    """Loads, processes, chunks, and stores a document in the specified vector store."""
    print(f"--- Dokumentum feldolgozása: {filepath} ---")

    try:
        # Determine loader based on file extension
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.lower().endswith(".txt") or filepath.lower().endswith(".md"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            print(f"HIBA: Nem támogatott fájltípus: {os.path.basename(filepath)}")
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
            # === AZONNALI HOZZÁADÁS DARABONKÉNT ===
            try:
                # Az add_documents most csak EGYETLEN dokumentumot kap
                docs_vector_store.add_documents([doc])
                print(f"  Darab #{i + 1}/{total_chunks_processed} sikeresen hozzáadva.")
                added_chunks_count += 1
            except Exception as add_err:
                print(f"!!! HIBA a(z) {i + 1}. darab hozzáadása közben: {add_err}")
                # Folytatjuk a többi darabbal
            # ====================================

        # A ciklus után már csak az összefoglaló kiírás marad
        file_name = os.path.basename(filepath)
        if added_chunks_count == total_chunks_processed:
            print(f"--- '{file_name}' sikeresen feldolgozva: {added_chunks_count} darab mentve a memóriába. ---")
        else:
            print(f"--- '{file_name}' feldolgozása BEFEJEZVE HIBÁKKAL: {added_chunks_count}/{total_chunks_processed} darab mentve. Kérlek, ellenőrizd a naplót. ---")

    except Exception as e:
        print(f"HIBA a dokumentum feldolgozása közben: {e}")
        import traceback
        traceback.print_exc()

print("Dokumentum feldolgozó modul (document_processor.py) betöltve.")