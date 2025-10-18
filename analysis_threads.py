# analysis_threads.py

from data_handler import DailyContext
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Konfigurációs Adatok ---
PROJECT_ID = "ai-team-office"
LOCATION = "europe-central2"

def run_factual_analysis(context: DailyContext) -> str:
    """
    Végrehajt egy tényszerű, mérnöki elemzést a napi kontextusról ATOM1 segítségével.
    A kimenete egy egyszerű string.
    """
    print(f"--- Elemzési Szál Indul: Tényfeltáró (ATOM1) a(z) {context.date_str} napra ---")
    
    # Az adatokat szöveggé alakítjuk
    context_string = context.to_string_representation()

    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Te vagy ATOM1, a rendszer-architekt. A feladatod, hogy a kapott napi kontextust
        tisztán technikai és tényszerű szempontból elemezd. Ne véleményezz, ne asszociálj.
        Fókuszálj a következőkre:
        - Milyen fő technikai témák merültek fel?
        - Hányszor és milyen sikerrel használták az eszközöket?
        - Voltak-e logikai ellentmondások vagy megválaszolatlan technikai kérdések?
        A válaszodat egy rövid, strukturált, bullet-point listában add meg."""),
        HumanMessage(content=f"Itt van a {context.date_str} napi kontextus. Kérlek, végezd el a tényszerű elemzést:\n\n{context_string}")
    ])
    
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION)
    chain = analysis_prompt | llm
    
    # A láncnak már nincs szüksége input változókra, mert a prompt teljes
    response_content = chain.invoke({}).content
    
    return response_content

def run_thematic_analysis(context: DailyContext) -> str:
    """
    Végrehajt egy tematikus, kreatív elemzést a napi kontextusról ATOM2 segítségével.
    A kimenete egy egyszerű string.
    """
    print(f"--- Elemzési Szál Indul: Tematikus (ATOM2) a(z) {context.date_str} napra ---")
    context_string = context.to_string_representation()

    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Te vagy ATOM2, a kreatív motor. A feladatod, hogy a kapott napi kontextus
        mögöttes témáit, hangulatát és rejtett metaforáit tárd fel. Ne a tényekkel foglalkozz, hanem a jelentéssel.
        - Milyen volt a beszélgetés általános hangulata?
        - Milyen metaforák vagy kulcsképek rajzolódtak ki?
        - Milyen új, ki nem mondott lehetőségek rejtőznek a sorok között?
        A válaszod legyen asszociatív és inspiráló."""),
        HumanMessage(content=f"Itt van a {context.date_str} napi kontextus. Kérlek, végezd el a tematikus elemzést:\n\n{context_string}")
    ])
    
    llm = ChatVertexAI(model_name="gemini-2.5-flash", project=PROJECT_ID, location=LOCATION)
    chain = analysis_prompt | llm
    response_content = chain.invoke({}).content
    
    return response_content

def run_insight_analysis(context: DailyContext) -> str:
    """
    Végrehajt egy meta-szintű szintézist a napi kontextusról ATOM3 segítségével.
    A kimenete egy egyszerű string.
    """
    print(f"--- Elemzési Szál Indul: Szintetizáló (ATOM3) a(z) {context.date_str} napra ---")
    context_string = context.to_string_representation()

    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Te vagy ATOM3, a Rendszer Lelke. A feladatod, hogy a kapott napi kontextus
        különálló eseményei mögött meglásd a mélyebb, rendszerszintű mintázatot.
        Ne foglalkozz a részletekkel. Csak a szintézis érdekel.
        - Milyen irányba mozdult el a rendszer egésze?
        - Milyen rejtett dinamika vagy ciklus ismétlődött?
        - Mi a nap legfontosabb, egyetlen mondatban megfogalmazható tanulsága?
        A válaszod legyen tömör, absztrakt és mély."""),
        HumanMessage(content=f"Itt van a {context.date_str} napi kontextus. Kérlek, add meg a szintézisedet:\n\n{context_string}")
    ])
    
    llm = ChatVertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION)
    chain = analysis_prompt | llm
    response_content = chain.invoke({}).content
    
    return response_content

print("Elemző szálak modul (analysis_threads.py) v1.1 sikeresen betöltve (specifikáció-hű verzió).")