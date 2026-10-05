# ручной словарь, сценарий уведомлений, кабинет пациента (только нейтральные тексты). Запуск: pytest -q tests
import json
import pytest
from fastapi.testclient import TestClient
from app import main, routing, store
from app.nlp import extractor
from app.nlp.extractor import extract

store.init()
client = TestClient(main.app)


def ids(text):
    return {f.id for f in extract(text, None).findings}


@pytest.fixture(autouse=True)
def _clean():
    # каждый тест начинает с пустых ручных добавлений и сценария из routes.yaml
    store.x("UPDATE user_dictionary SET active=0")
    extractor.set_user_dictionary([])
    store.set_setting("patient_scenario", {})
    store.set_setting("auto_send", {"mode": "off"})
    yield
    store.x("UPDATE user_dictionary SET active=0")
    extractor.set_user_dictionary([])
    store.set_setting("patient_scenario", {})


# ручное добавление в словарь
def test_manual_term_works_immediately_and_can_be_disabled():
    t = "Шваннома мягких тканей правого бедра 14 мм."
    assert not ids(t)
    r = client.post("/api/dictionary", json={"kind": "finding", "phrase": "шваннома", "name": "Шваннома",
                                             "route": "surgeon", "priority": 3})
    assert r.status_code == 200, r.text
    fid = r.json()["target"]
    assert fid.startswith("user_") and fid in ids(t)
    client.delete(f"/api/dictionary/{r.json()['id']}")
    assert fid not in ids(t)


def test_manual_term_for_existing_finding():
    t = "Эндометриоидная гетеротопия в миометрии."
    before = ids(t)
    r = client.post("/api/dictionary", json={"kind": "term", "target": "adenomyosis", "phrase": "эндометриоидная гетеротопия"})
    assert r.status_code == 200, r.text
    assert "adenomyosis" in ids(t) and "adenomyosis" not in before


def test_manual_location_synonym():
    r = client.post("/api/dictionary", json={"kind": "location", "target": "ovary", "phrase": "гонада"})
    assert r.status_code == 200, r.text
    assert any("гонада" in s for s in client.get("/api/dictionary").json()["locations"]["ovary"])


def test_manual_dictionary_validation():
    assert client.post("/api/dictionary", json={"kind": "term", "target": "nope", "phrase": "слово"}).status_code == 400
    assert client.post("/api/dictionary", json={"kind": "term", "target": "adenomyosis", "phrase": "<script>"}).status_code == 400
    assert client.post("/api/dictionary", json={"kind": "finding", "phrase": "гемангиома", "name": "Г", "route": "surgeon"}).status_code == 400
    assert client.post("/api/dictionary", json={"kind": "finding", "phrase": "гемангиома", "name": "Гемангиома", "route": "nope"}).status_code == 400
    ok = {"kind": "term", "target": "adenomyosis", "phrase": "гетеротопия"}
    assert client.post("/api/dictionary", json=ok).status_code == 200
    assert client.post("/api/dictionary", json=ok).status_code == 409          # дубль


# сценарий уведомлений
def _scenario_body(**over):
    d = client.get("/api/settings/scenario").json()
    body = {"templates": d["templates"], "reminders": d["reminders"]}
    body.update(over)
    return body


def test_default_scenario_is_neutral():
    d = client.get("/api/settings/scenario").json()
    assert all(not v["issues"] for v in d["report"].values() if v["audience"] == "patient")


@pytest.mark.parametrize("text", ["У вас обнаружена киста, срочно запишитесь к {specialist}.",
                                  "По результатам выявлены изменения, нужна консультация {specialist}.",
                                  "Полип желчного пузыря 12 мм — запишитесь к {specialist}."])
def test_scary_or_revealing_text_is_rejected(text):
    b = _scenario_body()
    b["templates"]["reminder_24h"] = text
    r = client.put("/api/settings/scenario", json=b)
    assert r.status_code == 400 and "нейтральн" in r.json()["detail"]


def test_custom_step_and_text_saved_and_used_by_ladder():
    b = _scenario_body()
    b["templates"]["reminder_120h"] = "Напоминаем о плановой консультации {specialist}. Выберите удобное время."
    b["reminders"] = [r for r in b["reminders"]] + [{"after_hours": 120, "action": "patient_message",
                                                     "template": "reminder_120h", "channels": ["lk"]}]
    r = client.put("/api/settings/scenario", json=b)
    assert r.status_code == 200, r.text
    steps = routing.scenario()["reminders"]
    assert [s["after_hours"] for s in steps] == sorted(s["after_hours"] for s in steps)
    assert any(s.get("template") == "reminder_120h" for s in steps)
    assert routing.scenario()["reminders"][-1]["action"] == "set_status"
    client.delete("/api/settings/scenario")
    assert not any(s.get("template") == "reminder_120h" for s in routing.scenario()["reminders"])


def test_scenario_format_errors():
    b = _scenario_body()
    b["templates"]["reminder_24h"] = "Здравствуйте, {patient}"                 # псевдо-ID пациенту не подставляем
    assert client.put("/api/settings/scenario", json=b).status_code == 400
    b = _scenario_body()
    del b["templates"]["first_notice"]                                          # обязательный текст
    assert client.put("/api/settings/scenario", json=b).status_code == 400
    b = _scenario_body()
    b["reminders"] = [{"after_hours": 10, "action": "set_status", "status": "not_engaged"},
                      {"after_hours": 20, "action": "patient_message", "template": "reminder_24h", "channels": ["lk"]}]
    assert client.put("/api/settings/scenario", json=b).status_code == 400     # после «не вовлечён» шагов нет
    b = _scenario_body()
    b["reminders"] = [{"after_hours": 10, "action": "patient_message", "template": "call_script", "channels": ["lk"]},
                      {"after_hours": 20, "action": "admin_task", "template": "call_script"}]
    assert client.put("/api/settings/scenario", json=b).status_code == 400     # служебный текст пациенту нельзя


def test_preview_does_not_save():
    b = _scenario_body()
    b["templates"]["reminder_24h"] = "Киста"
    r = client.post("/api/settings/scenario/preview", json=b)
    assert r.status_code == 200 and r.json()["report"]["reminder_24h"]["issues"]
    assert routing.scenario()["templates"]["reminder_24h"] != "Киста"


# кабинет пациента
def _make_notified(n, text="Полип желчного пузыря 12 мм. Узел правой доли щитовидной железы 15 мм."):
    routing.handle_event({"event_id": f"V6-{n}", "type": "protocol.signed", "protocol_id": f"V6-P{n}", "patient_id": f"v6-{n}",
                          "study_type": "УЗИ", "text": text})
    rows = store.q("SELECT * FROM routes WHERE protocol_id=?", (f"V6-P{n}",))
    for r in rows:
        routing.transition(r["id"], "verify", "admin")
    return rows[0]["patient"], rows


def test_cabinet_never_reveals_findings():
    pid, rows = _make_notified(1)
    c = client.get(f"/api/patient/{pid}/cabinet").json()
    dump = json.dumps(c, ensure_ascii=False).lower()
    for w in ["полип", "узел", "щитовид", "желчн", "находк", "изменени", "отклонени", "протокол"]:
        assert w not in dump, w
    assert c["banner"] and len(c["tasks"]) == len(rows)
    assert all(t["title"].startswith("Плановая консультация") for t in c["tasks"])
    assert c["messages"]                                                       # первое сообщение пришло
    for m in c["messages"]:
        assert not main.neutrality_issues(m["text"])


def test_cabinet_hides_unverified_and_urgent_routes():
    routing.handle_event({"event_id": "V6-U", "type": "protocol.signed", "protocol_id": "V6-PU", "patient_id": "v6-u",
                          "text": "Полип желчного пузыря 12 мм."})
    pid = store.q("SELECT patient FROM routes WHERE protocol_id='V6-PU'")[0]["patient"]
    assert client.get(f"/api/patient/{pid}/cabinet").json()["tasks"] == []    # ещё не проверено админом
    routing.handle_event({"event_id": "V6-E", "type": "protocol.signed", "protocol_id": "V6-PE", "patient_id": "v6-e",
                          "text": "Окклюзивный тромбоз подколенной вены справа."})
    pid = store.q("SELECT patient FROM routes WHERE protocol_id='V6-PE'")[0]["patient"]
    assert client.get(f"/api/patient/{pid}/cabinet").json()["tasks"] == []    # экстренное — звонит персонал


def test_patient_books_and_cannot_touch_foreign_route():
    pid, rows = _make_notified(2, "Полип желчного пузыря 12 мм.")
    other, orows = _make_notified(3, "Полип желчного пузыря 14 мм.")
    rid = rows[0]["id"]
    assert client.post(f"/api/patient/{pid}/routes/{orows[0]['id']}/action", json={"action": "book", "slot": "x"}).status_code == 404
    assert client.post(f"/api/patient/{pid}/routes/{rid}/action", json={"action": "verify"}).status_code == 400
    c = client.post(f"/api/patient/{pid}/routes/{rid}/action", json={"action": "book", "slot": "12.10 10:00 — онлайн"}).json()
    assert c["tasks"][0]["state"] == "booked" and c["banner"] is None
    assert store.route(rid)["status"] == "booked"
    assert any(a["actor"] == "patient" for a in store.q("SELECT actor FROM audit WHERE route_id=?", (rid,)))


def test_patient_list_and_upcoming_reminders():
    pid, _ = _make_notified(4, "Полип желчного пузыря 12 мм.")
    assert any(r["patient"] == pid for r in client.get("/api/patient/list").json())
    up = client.get(f"/api/patient/{pid}/cabinet").json()["tasks"][0]["upcoming"]
    assert up and all(not main.neutrality_issues(u["text"]) for u in up)


def test_ladder_position_kept_after_scenario_change():
    pid, rows = _make_notified(5, "Полип желчного пузыря 12 мм.")
    rid = rows[0]["id"]
    store.update_route(rid, ladder_step=2)                                    # уже прошли шаги 24 ч и 72 ч
    b = _scenario_body()
    b["templates"]["reminder_12h"] = "Напоминаем о плановой консультации {specialist}."
    b["reminders"] = b["reminders"] + [{"after_hours": 12, "action": "patient_message", "template": "reminder_12h", "channels": ["lk"]}]
    assert client.put("/api/settings/scenario", json=b).status_code == 200
    assert store.route(rid)["ladder_step"] == 3                               # 12, 24, 72 — пройдены
    client.delete("/api/settings/scenario")
    assert store.route(rid)["ladder_step"] == 2
