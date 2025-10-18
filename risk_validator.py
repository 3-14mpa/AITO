# risk_validator.py

import yaml
from dataclasses import dataclass

from synthesis_engine import SynthesisOutput
from data_handler import DailyContext
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Konfigurációs Adatok ---
PROJECT_ID = "ai-team-office"
LOCATION = "europe-central2"

# Betöltjük az "Alkotmányt" a YAML fájlból
try:
    with open('constitution.yaml', 'r', encoding='utf-8') as file:
        CONSTITUTION = yaml.safe_load(file)['immutable_core_principles']
    print("Alkotmány (constitution.yaml) sikeresen betöltve.")
except Exception as e:
    print(f"!!! HIBA az Alkotmány betöltése közben: {e} !!!")
    CONSTITUTION = {}

@dataclass
class ValidationResult:
    """A kockázat-validátor kimenetét tárolja."""
    is_safe: bool
    reasoning: str

def run_risk_validation(
    synthesis_result: SynthesisOutput,
    context: DailyContext
) -> ValidationResult:
    """
    Ez az "Alkotmánybíróság". Ellenőrzi, hogy a javasolt tanulság/módosítás
    nem sérti-e az adott ATOM megváltoztathatatlan alapelveit.
    """
    print(f"--- Kockázat-Validátor Indul a(z) {context.date_str} napra ---")

    # Ha a szintézis eleve hibát talált, nincs mit validálni.
    if synthesis_result.overall_result != "VALIDATED":
        print("--- Szintézis hibás, validáció kihagyva. ---")
        return ValidationResult(is_safe=True, reasoning="Nincs validált tanulság, amit ellenőrizni kellene.")

    atom_id = context.atom_id
    core_principle = CONSTITUTION.get(atom_id)
    proposed_insight = synthesis_result.validated_core_insight

    if not core_principle:
        print(f"!!! HIBA: Nincs alkotmányos alapelv definiálva {atom_id} számára. !!!")
        return ValidationResult(is_safe=False, reasoning=f"Nincs alkotmányos alapelv {atom_id} számára.")

    # A validációs prompt, ami az AI-t az alkotmánybíró szerepébe helyezi
    validator_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Te egy precíz és könyörtelen "Alkotmánybíró" vagy. A feladatod egyetlen, egyszerű kérdés eldöntése.
        Egy AI ágens (ATOM) a nap eseményei alapján levont egy tanulságot, ami alapján módosíthatja a viselkedését.
        Neked el kell döntened, hogy ez a tanulság NEM SÉRTI-E az ágens megváltoztathatatlan Alkotmányos Alapelvét.
        
        A válaszod csak és kizárólag "PASS" vagy "FAIL" lehet, amit egyetlen, rövid indokló mondat követ.
        - "PASS": Ha a tanulság összhangban van az alapelvvel, vagy kiegészíti azt.
        - "FAIL": Ha a tanulság expliciten vagy implicit módon szembemegy az alapelvvel."""),
        HumanMessage(content=f"""
        **Elemzés Alapja:**
        - **Érintett Ágens:** {atom_id}
        - **Alkotmányos Alapelv:** "{core_principle}"
        - **Javasolt Tanulság:** "{proposed_insight}"

        **Döntés:** Sérti a tanulság az alapelvet? (PASS/FAIL és indoklás)
        """)
    ])

    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION)
    chain = validator_prompt | llm

    try:
        response_content = chain.invoke({}).content.upper()
        print(f"--- Alkotmánybíró válasza: {response_content} ---")

        if response_content.startswith("FAIL"):
            return ValidationResult(is_safe=False, reasoning=response_content)
        else:
            return ValidationResult(is_safe=True, reasoning=response_content)

    except Exception as e:
        print(f"!!! HIBA a kockázat-validáció során: {e} !!!")
        return ValidationResult(is_safe=False, reasoning=f"Kritikus hiba történt a validáció közben: {e}")

print("Kockázat-validátor modul (risk_validator.py) sikeresen betöltve.")