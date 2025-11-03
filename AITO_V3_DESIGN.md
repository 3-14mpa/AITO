# AITO V3 Rendszerterv

Ez a dokumentum az AITO V3 architektúrájának és működésének magas szintű tervét rögzíti.

## 1. Alapelvek és Célok

Az AITO V3 fő célja, hogy a jelenlegi, reaktív chatbot modellt egy proaktív, autonóm, párhuzamos működésre képes multi-ágens rendszerré alakítsa.

- **Architekturális Szétválasztás:** A komponensek (UI, agens-logika, ütemezés) legyenek lazán csatoltak.
- **Autonóm Működés:** Az ágensek emberi beavatkozás nélkül, a "Digitális Szívverés" által vezérelve is képesek legyenek a feladataikat végezni.
- **Párhuzamosság és Skálázhatóság:** Minden ágens külön Google Cloud projektet és dedikált futtatói szálat használ a rate limit hibák elkerülése és a párhuzamos feladatvégzés érdekében.
- **Robusztusság és Hibakezelés:** A rendszer legyen képes felismerni és kezelni a "zombie" (beragadt) ágenseket.

## 2. Magas Szintű Architektúra

A rendszer a következő fő komponensekből áll:

- **Chat UI (Frontend):** Kizárólag az üzenetek megjelenítéséért és a felhasználói input fogadásáért felelős.
- **Message Bus (Központi Üzenetkezelő):** Egy FIFO (First-In, First-Out) elven működő üzenetsor, ami a rendszer komponensei közötti összes kommunikációt kezeli.
- **Heartbeat Modul (Digitális Szívverés):** Egy független, külső ütemező (scheduler), ami periodikusan "ébreszti" az ágenseket.
- **ATOM Workerek:** Külön szálakon futó, dedikált processzek, amelyek az egyes ATOM-ok logikáját hajtják végre.
- **Adatbázisok:**
    - **Üzenet Adatbázis:** A Message Bus perzisztens tárolója.
    - **System Registry:** Az ágensek állapotát (pl. `READY`/`BUSY`), jegyzetfüzeteit és egyéb rendszer-szintű beállításokat (pl. `meeting_id`) tárolja.
- **Konfiguráció (`config_aito_v3.yaml`):** A multi-projektes beállításokat (projektek, credentials) kezeli.

## 3. Komponensek Részletes Terve

### 3.1. Heartbeat Modul

- **Feladat:** Az ATOM workerek periodikus meghívása ("tik-tak").
- **Dinamikus Ütemezés:** A "szívverés" frekvenciája a rendszer állapotától függ:
    - **Meeting Mód:** 1 perces időköz (ha `meeting_id` aktív).
    - **Task Mód:** 5 perces időköz (ha van aktív feladat, de nincs megbeszélés).
    - **Idle Mód:** 30 perces időköz (ha nincs aktív feladat).
- **Állapot-észlelés:** A modul aktívan figyeli a `meeting_id` állapotát a System Registry-ben. Ha nincs aktív meeting, körkérdéssel ellenőrzi az ATOM-ok állapotát (a jegyzetfüzeteik alapján), hogy eldöntse, Task vagy Idle módba kapcsoljon.

### 3.2. ATOM Worker és Állapotkezelés

- **Állapot Flag:** Minden ágenshez tartozik egy állapotjelző a System Registry-ben (pl. `agent_status_atom1`), ami lehet:
    - `READY`: Az ágens szabad, fogadhat hívást.
    - `BUSY`: Az ágens éppen egy modellhívásban van.
- **"Ébresztési" Ciklus:** Amikor a Heartbeat meghív egy ATOM-ot (és az `READY` állapotú):
    1. Az ágens állapota `BUSY`-ra vált.
    2. Beolvassa a saját jegyzetfüzetét, hogy megértse a jelenlegi feladatát.
    3. Beolvassa a neki címzett (`@ATOM_ID`) új üzeneteket a Message Bus-ról.
    4. Végrehajtja a logikai lépését (gondolkodás, eszközhasználat, stb.).
    5. A válaszát (ha van) elküldi a Message Bus-ra.
    6. Frissíti a jegyzetfüzetét az új állapottal.
    7. Az ágens állapota visszavált `READY`-re.
- **Zombie Detektálás:** Ha egy ágens állapota 5-10 Heartbeat cikluson keresztül `BUSY` marad, a Heartbeat modul "zombie"-nak tekinti, és hibajelzést küld a rendszernek és/vagy megpróbálja a worker szálat újraindítani.

### 3.3. Message Bus és Kommunikációs Protokoll

- **Működés:** Egy központi, FIFO elvű üzenetpuffer, ami garantálja az üzenetek sorrendiségét és atomi írását.
- **Címzés:** Az üzenetek kezdődhetnek `@CÍMZETT_ID` prefix-szel. Ez esetben az üzenet egy konkrét ágensnek szól. Címzés nélkül az üzenet "broadcast", amit minden ágens lát.
- **ATOMOD Hozzáférése:** ATOMOD, mint koordinátor, teljes hozzáféréssel rendelkezik a Message Bus-hoz, és képes a megbeszéléseket a `meeting_id` beállításával menedzselni.
