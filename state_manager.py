# state_manager.py

from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator

# A TypedDict segítségével definiáljuk a "jegyzőkönyv" szerkezetét.
# Ez olyan, mint egy tervrajz egy adat-objektumhoz.
class MeetingState(TypedDict):
    """
    Ez az adatstruktúra tárolja egy moderált megbeszélés teljes állapotát.
    Ez ATOMOD "jegyzőkönyve".
    """
    # A feladat leírása, amit a felhasználó adott
    task_description: str
    
    # A megbeszélésben résztvevő ATOM-ok listája
    participants: List[str]
    
    # A megbeszélés üzeneteinek listája
    messages: Annotated[List[BaseMessage], operator.add]
    
    # A megbeszélés jelenlegi körének sorszáma
    current_round: int
    
    # A következőnek felszólaló ATOM neve
    next_speaker: str

print("Állapotkezelő modul (state_manager.py) sikeresen betöltve.")