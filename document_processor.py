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

        # Create Document objects for each chunk
        docs_to_add = []
        for i, chunk in enumerate(text_chunks):
            doc = create_document_chunk(
                content=chunk,
                source=os.path.basename(filepath),
                chunk_num=i + 1,
                total_chunks=len(text_chunks)
            )
            docs_to_add.append(doc)

        # --- ÚJ BLOKK: Mentés a Firestore-ba a nyilvántartáshoz ---
        try:
            from google.cloud import firestore
            client = firestore.Client(project=config['project_id'])
            collection_ref = client.collection(config['docs_firestore_collection_name'])

            # Egy 'batch' írást használunk a hatékonyságért
            batch = client.batch()
            for doc_obj in docs_to_add:
                # Létrehozunk egy egyedi azonosítót minden darabnak
                doc_id = f"{doc_obj.metadata['source_document']}_chunk_{doc_obj.metadata['chunk_number']}"
                doc_ref = collection_ref.document(doc_id)

                # A LangChain Document objektumot egy egyszerű szótárrá alakítjuk
                firestore_data = {
                    "page_content": doc_obj.page_content,
                    "metadata": doc_obj.metadata
                }
                batch.set(doc_ref, firestore_data)

            batch.commit()
            print(f"--- Nyilvántartás frissítve: {len(docs_to_add)} darab mentve a Firestore '{config['docs_firestore_collection_name']}' kollekciójába. ---")

        except Exception as e:
            print(f"HIBA a Firestore-ba mentés során: {e}")
        # --- ÚJ BLOKK VÉGE ---


        # Store the documents in the provided vector store
        if docs_to_add:
            docs_vector_store.add_documents(docs_to_add)
            file_name = os.path.basename(filepath)
            print(f"--- '{file_name}' sikeresen feldolgozva: {len(docs_to_add)} darab mentve a memóriába. ---")

    except Exception as e:
        print(f"HIBA a dokumentum feldolgozása közben: {e}")
        import traceback
        traceback.print_exc()

print("Dokumentum feldolgozó modul (document_processor.py) betöltve.")