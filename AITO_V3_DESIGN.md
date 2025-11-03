# AITO V3 Rendszerterv (Revideált)

Ez a dokumentum az AITO V3 architektúrájának és működésének **revideált, végrehajtásra jóváhagyott** tervét rögzíti. A terv a lokális, többprocesszes futtatási környezetet és az egyedi, ágens-szintű ütemezést veszi alapul.

## 1. Alapelvek és Célok

Az AITO V3 fő célja egy proaktív, autonóm, párhuzamos működésre képes multi-ágens rendszer létrehozása.

- **Architekturális Szétválasztás:** A komponensek (UI, ágens-logika, ütemezés) lazán csatoltak.
- **Autonóm Működés:** Az ágensek a "Digitális Szívverés" által vezérelve, emberi beavatkozás nélkül is végzik a feladataikat.
- **Párhuzamosság Lokális Környezetben:** Minden ágens külön szálon/processzben fut, de az erőforrásokat és a kommunikációt a helyi gép menedzseli.
- **Robusztusság és Hibakezelés:** A rendszer felismeri és kezeli a "zombie" (beragadt) ágenseket.

## 2. Magas Szintű Architektúra

A rendszer a következő fő komponensekből áll:

- **Chat UI (Frontend):** Kizárólag az üzenetek megjelenítéséért és a felhasználói input fogadásáért felelős.
- **Helyi Message Bus (Producer-Consumer Queue):** Egy memóriában futó, szál-biztos sor (`queue`), ami a belső komponensek közötti aszinkron kommunikációt kezeli.
- **Adatbázis Író Worker (Consumer):** Egyetlen, dedikált szál, amely az üzeneteket a `queue`-ból kivéve szekvenciálisan írja az adatbázisba.
- **Heartbeat Modul (Digitális Szívverés):** Egy független ütemező, ami periodikusan "ébreszti" az ágenseket az egyedi állapotuk alapján.
- **ATOM Workerek (Producerek):** Külön szálakon futó processzek, amelyek az egyes ATOM-ok logikáját hajtják végre és üzeneteket helyeznek a `queue`-ba.
- **Adatbázisok (Lokális):**
    - **Üzenet Adatbázis:** A chat kommunikáció perzisztens tárolója.
    - **System Registry:** Az ágensek állapotát (`READY`/`BUSY`), feladatait (`has_task`), jegyzetfüzeteit és egyéb rendszer-szintű beállításokat (`meeting_id`) tárolja.
- **Konfiguráció (`config_aito_v3.yaml`):** A multi-projektes beállításokat (projektek, credentials) kezeli. A Google Secret Manager integrációja egy jövőbeli, hibrid felhős működés esetén megfontolandó.

## 3. Komponensek Részletes Terve

### 3.1. Heartbeat Modul

- **Feladat:** Az ATOM workerek egyedi, állapot-alapú ébresztése.
- **Működési Ciklus:** A modul egy központi ciklusban fut, ahol minden ATOM esetében külön-külön lekérdezi a `System Registry`-ből a következő állapotokat: `agent_status` (READY/BUSY), `has_task`, `last_heartbeat`.
- **Egyedi, Állapot-alapú Ütemezés:** Ezen adatok alapján, minden ATOM-ra egyénileg dönti el, hogy az adott ágenst az aktuális ütemnek megfelelően ébreszteni kell-e:
    - **Meeting Mód:** 1 perces időköz (ha a globális `meeting_id` aktív).
    - **Task Mód:** 5 perces időköz (ha az ágensnek `has_task` státusza van).
    - **Idle Mód:** 30 perces időköz (alapértelmezett, ha nincs feladat).
- **Mérnöki Értékelés:** Ez a modell a leghatékonyabb a célzott lokális környezetben. A "kérdezés" egy gyors, helyi adatbázis-lekérdezés, ami minimális terhelést jelent, és elkerüli az inaktív ágensek felesleges API hívásait.

### 3.2. ATOM Worker és Állapotkezelés

- **Állapot Flag:** Minden ágenshez tartozik egy `agent_status` kulcs a System Registry-ben, ami lehet: `READY` vagy `BUSY`.
- **"Ébresztési" Ciklus:** Amikor a Heartbeat egy `READY` állapotú ATOM-ot ébreszt:
    1. Az ágens állapota azonnal `BUSY`-ra vált.
    2. Beolvassa a jegyzetfüzetét és a neki címzett új üzeneteket.
    3. Végrehajtja a logikai lépését.
    4. A válaszát behelyezi a központi, memóriában lévő `queue`-ba (nem ír közvetlenül adatbázisba!).
    5. Frissíti a jegyzetfüzetét és a `has_task` állapotát.
    6. Az ágens állapota visszavált `READY`-re.
- **Zombie Detektálás:** Ha egy ágens állapota több cikluson át `BUSY` marad, a Heartbeat modul hibát jelez.

### 3.3. Helyi Message Bus (Producer-Consumer Architektúra)

- **Működés:** A rendszer kommunikációs gerince egy **memóriában futó, szál-biztos sor (pl. Python `queue.Queue`)**.
- **Producer-Consumer Minta:**
    1. **Producerek (ATOM Workerek):** Amikor egy ATOM üzenetet akar küldeni (pl. a chatre), azt nem írja be közvetlenül az adatbázisba. Ehelyett az üzenet-objektumot beleteszi a központi `queue`-ba. Ez egy rendkívül gyors, nem-blokkoló memóriaművelet.
    2. **Consumer (Adatbázis Író Worker):** Egyetlen, dedikált szál folyamatosan figyeli a `queue`-t. Amikor új elem érkezik, kiveszi azt, és **egyetlen pontként, szekvenciálisan** beírja a perzisztens Üzenet Adatbázisba.
- **Indoklás:** Ez az architektúra tökéletesen és egyszerűen oldja meg a lokális konkurrens írási problémát. Garantálja az üzenetek sorrendiségét, kiküszöböli az adatbázis-zárolás szükségességét és a versenyhelyzeteket (race conditions), és rendkívül nagy teljesítményt nyújt.
