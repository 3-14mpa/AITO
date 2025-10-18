# synthesis_engine.py

import json
from typing import List, Dict, Any, Optional

# JAVÍTÁS: A 'dataclasses' helyett a 'pydantic'-ot használjuk a modellekhez
from pydantic.v1 import BaseModel, Field

from data_handler import DailyContext
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Konfigurációs Adatok ---
PROJECT_ID = "ai-team-office"
LOCATION = "europe-central2"

# JAVÍTÁS: Átalakítjuk az adatosztályokat Pydantic modellekké
class ErrorReport(BaseModel):
    """Az audit során talált hibák strukturált jelentése."""
    failed_step: Optional[str] = Field(None, description="A hibát okozó lépés neve.")
    contradiction_analysis: Dict[str, Any] = Field(..., description="Az ellentmondás-vizsgálat eredménye.")
    derivation_audit: Dict[str, Any] = Field(..., description="A levezetés-vizsgálat eredménye.")
    omission_analysis: Dict[str, Any] = Field(..., description="A hiányossági vizsgálat eredménye.")

class SynthesisOutput(BaseModel):
    """
    Az "Arbiter" auditjának eredménye, az ATOM5 által specifikált séma alapján.
    Ez a séma, amit a modellnek követnie kell.
    """
    overall_result: str = Field(..., description="Az audit végeredménye: 'VALIDATED' vagy 'CONSISTENCY_ERROR'")
    validated_core_insight: Optional[str] = Field(None, description="A nap validált, egy mondatos tanulsága. Csak 'VALIDATED' eredmény esetén releváns.")
    error_report: Optional[ErrorReport] = Field(None, description="Részletes hibajelentés. Csak 'CONSISTENCY_ERROR' eredmény esetén releváns.")

def run_synthesis(
    original_context: DailyContext,
    factual_analysis: str,
    thematic_analysis: str,
    insight_analysis: str
) -> SynthesisOutput:
    """
    Ez a központi "Arbiter" modul. A specifikációnak megfelelően egy
    KONZISZTENCIA AUDITOT hajt végre a bemeneteken, és egy validált tanulságot
    vagy egy RÉSZLETES hibajelentést ad vissza.
    """
    print(f"--- Szintézis Motor (Arbiter) Indul a(z) {original_context.date_str} napra ---")

    evidence_package = f"""
    EREDETI, NYERS ADATOK (a hiányossági vizsgálathoz):
    {original_context.to_string_representation()}

    --- ELEMZÉSEK ---
    1. TÉNYFELTÁRÓ ELEMZÉS (ATOM1): {factual_analysis}
    2. TEMATIKUS ELEMZÉS (ATOM2): {thematic_analysis}
    3. SZINTETIZÁLÓ ELEMZÉS (ATOM3): {insight_analysis}
    """

    # JAVÍTÁS: Sokkal szigorúbb és részletesebb prompt az Arbiter számára
    arbiter_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Te vagy az "Arbiter", egy rendszerszintű validáló motor. 
        A feladatod, hogy a kapott elemzéseken és a nyers adatokon egy szigorú, háromlépcsős 
        "Konzisztencia Audit Protokollt" hajts végre. A válaszodat egy szigorú, előre definiált JSON formátumban kell megadnod.

        **Konzisztencia Audit Protokoll v1.0:**
        1.  **Kontradikció-Analízis:** Keress direkt logikai ellentmondásokat a három elemzés között.
        2.  **Levezetési Audit:** Ellenőrizd, hogy az elemzések következtetései logikusan levezethetők-e az eredeti adatokból.
        3.  **Hiányossági Vizsgálat:** Vesd össze az elemzéseket az eredeti adatokkal, és azonosíts minden olyan kritikusan fontos információt, amit az elemzések kihagytak.

        **Végső Kimenet (JSON formátumban):**
        * HA mindhárom lépés sikeres, az "overall_result" legyen "VALIDATED", és fogalmazz meg egy 'validated_core_insight'-ot.
        * HA bármelyik lépésben problémát találtál, az "overall_result" legyen "CONSISTENCY_ERROR". Ebben az esetben **KÖTELEZŐ** részletesen kitöltened az "error_report" megfelelő "details" mezőjét a talált hibával. Például, ha a Hiányossági Vizsgálat talált hibát, a `omission_analysis.details` mezőnek **TARTALMAZNIA KELL** a kihagyott, szignifikáns események listáját.
        
        A válaszod kizárólag a specifikált JSON objektum legyen."""),
        HumanMessage(content=evidence_package)
    ])
    
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION).with_structured_output(SynthesisOutput)
    chain = arbiter_prompt_template | llm

    try:
        synthesis_result = chain.invoke({})
        
        if synthesis_result:
            print("--- Arbiter válasza (strukturált): ---")
            print(synthesis_result.json(indent=2))
            print("--- Arbiter válasza sikeresen feldolgozva. ---")
            return synthesis_result
        else:
            raise ValueError("A modell nem adott vissza strukturált, feldolgozható választ.")

    except Exception as e:
        print(f"!!! HIBA a szintézis során: {e} !!!")
        return SynthesisOutput(
            overall_result="CONSISTENCY_ERROR",
            error_report=ErrorReport(
                failed_step="CRITICAL_FAILURE",
                contradiction_analysis={"status": "UNKNOWN", "details": f"Kritikus hiba: {e}"},
                derivation_audit={"status": "UNKNOWN", "details": "N/A"},
                omission_analysis={"status": "UNKNOWN", "details": "N/A"}
            )
        )
        

print("Szintézis motor modul (synthesis_engine.py) v1.3 sikeresen betöltve (Pydantic modellekkel).")