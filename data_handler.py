# data_handler.py

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

from langchain_core.messages import BaseMessage, AIMessage

@dataclass
class ToolUsageLog:
    """Egyetlen eszközhívás meta-adatait rögzíti."""
    tool_name: str
    tool_input: Dict[str, Any]

# ÚJ: Az ATOM4 által specifikált, többdimenziós ResonanceVector
@dataclass
class ResonanceVector:
    """
    Egy üzenet által kiváltott reakciók típusát és mértékét rögzíti.
    Ez a v1.0-ás, egyszerűsített, de funkcionális modell.
    """
    creative_build_score: int = 0      # ATOM2 típusú, továbbépítő reakciók száma
    critical_challenge_score: int = 0  # ATOM5 típusú, kritikai reakciók száma
    analytical_refinement_score: int = 0 # ATOM1 típusú, elemző/finomító reakciók száma
    # A jövőben további dimenziókkal bővíthető.

@dataclass
class Interaction:
    """Egyetlen üzenetváltást és a hozzá kapcsolódó meta-adatokat foglalja magában."""
    message: BaseMessage
    tool_logs: List[ToolUsageLog] = field(default_factory=list)
    # JAVÍTÁS: A 'resonance_score' lecserélve a teljes 'ResonanceVector'-ra
    resonance_vector: ResonanceVector = field(default_factory=ResonanceVector)

@dataclass
class DailyContext:
    """
    Egy adott nap teljes, az önreflexióhoz szükséges kontextusát tartalmazza,
    az eredeti, ATOM4 által specifikált részletességgel.
    """
    date_str: str
    interactions: List[Interaction]
    atom_id: str

    def to_string_representation(self) -> str:
        """
        Létrehozza az objektum szöveges reprezentációját az LLM számára.
        ATOM1 specifikációja alapján.
        """
        if not self.interactions:
            return "A mai napon nem történt releváns interakció."

        representation = f"A(z) {self.date_str} nap interakcióinak listája:\n\n"
        for i, interaction in enumerate(self.interactions):
            msg = interaction.message
            speaker = getattr(msg, 'name', 'Ismeretlen')
            representation += f"--- Üzenet #{i+1} ---\n"
            representation += f"Beszélő: {speaker}\n"
            representation += f"Tartalom: {msg.content}\n"
            if interaction.tool_logs:
                representation += f"Használt Eszközök: {[log.tool_name for log in interaction.tool_logs]}\n"
            representation += "\n"
        
        return representation

def extract_tool_usage_logs(message: BaseMessage) -> List[ToolUsageLog]:
    """Kinyeri egy AIMessage objektumból a tool_calls attribútumot, ha létezik."""
    logs = []
    if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            log = ToolUsageLog(
                tool_name=tool_call.get("name"),
                tool_input=tool_call.get("args")
            )
            logs.append(log)
    return logs

def get_message_resonance(target_message_index: int, all_messages: List[BaseMessage]) -> ResonanceVector:
    """
    A v1.0-ás, egyszerűsített, kulcsszó/beszélő-alapú rezonancia-számítás.
    Megvizsgálja a célüzenetet követő 5 üzenetet, és a beszélő ATOM típusa
    alapján növeli a megfelelő rezonancia számlálót.
    """
    resonance = ResonanceVector()
    
    # A vizsgálandó üzenetek tartománya: a célüzenet utáni 5 üzenet
    start_index = target_message_index + 1
    end_index = min(start_index + 5, len(all_messages))
    
    for i in range(start_index, end_index):
        reaction_message = all_messages[i]
        speaker_name = getattr(reaction_message, 'name', None)
        
        # A beszélő személye alapján növeljük a megfelelő számlálót
        if speaker_name == "ATOM1":
            resonance.analytical_refinement_score += 1
        elif speaker_name == "ATOM2":
            resonance.creative_build_score += 1
        elif speaker_name == "ATOM5":
            resonance.critical_challenge_score += 1
            
    return resonance

def create_daily_context_object(
    atom_id: str,
    firestore_history: 'FirestoreChatMessageHistory',
    days_ago: int = 0
) -> DailyContext:
    """
    Lekérdezi és felépíti a részletes DailyContext objektumot,
    a v1.0 specifikációnak megfelelően, a rezonanciát is beleértve.
    """
    print(f"Napi kontextus objektum létrehozása {atom_id} számára a(z) {days_ago}. nappal ezelőtti adatokból...")
    
    target_date = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    target_date_str = target_date.strftime("%Y-%m-%d")

    all_messages = firestore_history.messages
    daily_interactions = []

    for i, msg in enumerate(all_messages):
        timestamp_str = msg.additional_kwargs.get("timestamp", "")
        if timestamp_str:
            try:
                msg_date = datetime.fromisoformat(timestamp_str).date()
                if msg_date == target_date:
                    tool_logs = extract_tool_usage_logs(msg)
                    # JAVÍTÁS: A működő, nem-véletlenszerű rezonancia számítás hívása
                    resonance_vector = get_message_resonance(i, all_messages)
                    
                    interaction = Interaction(
                        message=msg,
                        tool_logs=tool_logs,
                        resonance_vector=resonance_vector
                    )
                    daily_interactions.append(interaction)
            except ValueError:
                continue

    print(f"{len(daily_interactions)} interakció található {target_date_str} napra.")
    
    return DailyContext(
        date_str=target_date_str,
        interactions=daily_interactions,
        atom_id=atom_id
    )

print("Adatkezelő modul (data_handler.py) v1.3 sikeresen betöltve (specifikáció-hű, rezonancia-javított verzió).")