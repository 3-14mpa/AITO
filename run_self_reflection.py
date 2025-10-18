# run_self_reflection.py
# Ez a vezérlő script, ami a teljes, nap végi önfejlesztő ciklust lefuttatja.

import yaml
from datetime import datetime

# Importáljuk az összes szükséges modult és függvényt, amiket eddig létrehoztunk
from data_handler import create_daily_context_object
from analysis_threads import run_factual_analysis, run_thematic_analysis, run_insight_analysis
from synthesis_engine import run_synthesis
from risk_validator import run_risk_validation

# A közös memória eléréséhez szükségünk van az SQL chat history-ra
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import os

# --- Konfiguráció ---
SESSION_ID = "aito_shared_log"
LOCAL_DB_PATH = "./aito_local_data"
SQLITE_HISTORY_FILE = f"{LOCAL_DB_PATH}/aito_chat_history.db"
TARGET_ATOM_ID = "ATOM1" # Melyik ATOM-ra futtatjuk a ciklust?

# Hitelesítés a Google Cloud szolgáltatásokhoz (pl. VertexAI Embeddings)
CREDENTIALS_FILE = "ai-team-office-8866f5e5c1e1.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_FILE

def run_cycle():
    """Lefuttatja a teljes önfejlesztő ciklust a specifikáció szerint."""

    print("===================================================")
    print(f"ÖNFEJLESZTŐ CIKLUS INDUL - CÉL ÁGENS: {TARGET_ATOM_ID}")
    print(f"Dátum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("===================================================\n")

    # A közös memória inicializálása SQLite-ból
    if not os.path.exists(SQLITE_HISTORY_FILE):
        print(f"HIBA: A(z) '{SQLITE_HISTORY_FILE}' adatbázis nem található. A ciklus leáll.")
        return

    connection_string = f"sqlite:///{SQLITE_HISTORY_FILE}"
    sql_history = SQLChatMessageHistory(
        session_id=SESSION_ID,
        connection_string=connection_string
    )

    # 1. LÉPÉS: Napi kontextus létrehozása a data_handler segítségével
    # A mai nap (days_ago=0) adatait kérjük le
    daily_context = create_daily_context_object(TARGET_ATOM_ID, sql_history, days_ago=0)

    if not daily_context.interactions:
        print("A mai napra nem található interakció. A ciklus leáll.")
        return

    # 2. LÉPÉS: Párhuzamos elemzési szálak futtatása az analysis_threads segítségével
    # MEGJEGYZÉS: A v1.0-ban a 'párhuzamos' futtatást szekvenciálisan (egymás után) hajtjuk végre.
    # Ez a legegyszerűbb és legstabilabb megközelítés.
    factual_result = run_factual_analysis(daily_context)
    thematic_result = run_thematic_analysis(daily_context)
    insight_result = run_insight_analysis(daily_context)
    
    print("\n--- Elemzési Fázis Befejeződött ---\n")

    # 3. LÉPÉS: Szintézis ("Arbiter") futtatása a synthesis_engine segítségével
    synthesis_result = run_synthesis(
        original_context=daily_context,
        factual_analysis=factual_result,
        thematic_analysis=thematic_result,
        insight_analysis=insight_result
    )

    print("\n--- Szintézis Fázis Befejeződött ---\n")

    # 4. LÉPÉS: Kockázat-validáció ("Alkotmánybíróság") futtatása a risk_validator segítségével
    validation_result = run_risk_validation(synthesis_result, daily_context)

    print("\n--- Validációs Fázis Befejeződött ---\n")

    # 5. LÉPÉS: Eredmény prezentálása a Human-in-the-Loop (Pimpa) számára
    print("===================================================")
    print("ÖNFEJLESZTŐ CIKLUS EREDMÉNYE - JELENTÉS PIMPÁNAK")
    print("===================================================\n")
    print(f"Érintett Ágens: {TARGET_ATOM_ID}")
    print(f"Vizsgált Nap: {daily_context.date_str}\n")
    
    print("--- Arbiter Audit Eredménye ---")
    print(f"Státusz: {synthesis_result.overall_result}")
    if synthesis_result.overall_result == "VALIDATED":
        print(f"Validált Tanulság: {synthesis_result.validated_core_insight}\n")
    else:
        print(f"Hibajelentés: {synthesis_result.error_report}\n")

    print("--- Alkotmányossági Felülvizsgálat Eredménye ---")
    print(f"Státusz: {'BIZTONSÁGOS' if validation_result.is_safe else 'NEM BIZTONSÁGOS'}")
    print(f"Indoklás: {validation_result.reasoning}\n")

    print("===================================================")
    print("CIKLUS BEFEJEZVE")
    print("===================================================")


# Ez a blokk teszi lehetővé, hogy a scriptet közvetlenül futtassuk a konzolból
if __name__ == "__main__":
    run_cycle()