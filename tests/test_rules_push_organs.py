# правила автоотправки, пуш пациенту, расширенный словарь
import pytest
from fastapi.testclient import TestClient
from app import main, routing, store
from app.nlp.extractor import extract

store.init()
client = TestClient(main.app)


def ids(text):
    return {f.id for f in extract(text, None).findings}


def _routes(pid):
    return {r["route_key"]: r for r in store.q("SELECT * FROM routes WHERE protocol_id=?", (pid,))}


def _event(n, text, study="УЗИ"):
    routing.handle_event({"event_id": f"V7-{n}", "type": "protocol.signed", "protocol_id": f"V7-P{n}",
                          "patient_id": f"v7-{n}", "study_type": study, "text": text})
    return _routes(f"V7-P{n}")


@pytest.fixture(autouse=True)
def _off():
    store.set_setting("auto_send", {"mode": "off"})
    yield
    store.set_setting("auto_send", {"mode": "off"})


# правила автоотправки
RULE = {"name": "Щитовидка планово", "active": True, "routes": ["endocrinologist"], "findings": [], "priorities": [3],
        "min_confidence": 0.5, "require_clinician": False, "study_contains": "щитовид"}


def _put(rules, mode="rules"):
    return client.put("/api/settings/auto_send", json={"mode": mode, "routes": [], "skip_needs_review": True, "rules": rules})


def test_rule_sends_matching_route_and_skips_others():
    assert _put([RULE]).status_code == 200
    rs = _event(1, "Узел правой доли щитовидной железы 15 мм. Полип желчного пузыря 12 мм.", "УЗИ щитовидной железы")
    assert rs["endocrinologist"]["status"] == "notified"
    assert rs["surgeon"]["status"] == "pending_verification"
    assert any("Щитовидка планово" in (a["details"] or "") for a in store.q("SELECT details FROM audit WHERE route_id=?", (rs["endocrinologist"]["id"],)))


def test_rule_conditions_study_and_clinician():
    assert _put([RULE]).status_code == 200
    rs = _event(2, "Узел правой доли щитовидной железы 15 мм.", "УЗИ шеи")
    assert rs["endocrinologist"]["status"] == "pending_verification"          # тип исследования не подошёл
    assert _put([{**RULE, "study_contains": "", "require_clinician": True}]).status_code == 200
    rs = _event(3, "Узел правой доли щитовидной железы 15 мм. Рекомендована консультация эндокринолога.")
    assert rs["endocrinologist"]["status"] == "notified"
    rs = _event(4, "Узел правой доли щитовидной железы 15 мм.")
    assert rs["endocrinologist"]["status"] == "pending_verification"          # врач не рекомендовал


def test_disabled_rule_and_validation():
    assert _put([{**RULE, "active": False}]).status_code == 400                # режим правил без включённых правил
    assert _put([{**RULE, "routes": ["nope"]}]).status_code == 400
    assert _put([{**RULE, "name": ""}]).status_code == 400
    assert _put([{**RULE, "min_confidence": 2}]).status_code == 400


def test_rule_never_sends_urgent_or_needs_review():
    assert _put([{**RULE, "routes": [], "priorities": [], "study_contains": "", "min_confidence": 0}]).status_code == 200
    rs = _event(5, "Окклюзивный тромбоз подколенной вены справа.")
    assert all(r["status"] == "escalated" for r in rs.values())
    rs = _event(6, "Полип желчного пузыря.")                                   # нет размера -> «проверить»
    assert rs["surgeon"]["status"] == "pending_verification"


def test_simulate_does_not_change_anything():
    _event(7, "Узел правой доли щитовидной железы 15 мм.", "УЗИ щитовидной железы")
    before = store.q("SELECT id, status FROM routes ORDER BY id")
    r = client.post("/api/settings/auto_send/simulate", json={"mode": "rules", "routes": [], "skip_needs_review": True, "rules": [RULE]})
    assert r.status_code == 200 and r.json()["auto"] >= 1
    assert store.q("SELECT id, status FROM routes ORDER BY id") == before


# пуш пациенту (заглушка телефона)
def test_push_view_on_phone_and_generic_mode(monkeypatch):
    rs = _event(8, "Полип желчного пузыря 12 мм.")
    rid = rs["surgeon"]["id"]
    routing.transition(rid, "verify", "admin")
    pid = rs["surgeon"]["patient"]
    c = client.get(f"/api/patient/{pid}/cabinet").json()
    assert c["push"]["items"] and c["push"]["items"][0]["body"] == c["messages"][0]["text"]
    assert all(not main.neutrality_issues(p["body"]) for p in c["push"]["items"])
    from app import config
    real = config.routes
    monkeypatch.setattr(config, "routes", lambda: {**real(), "behavior": {**real()["behavior"], "push_on_lock_screen": "generic"}})
    c = client.get(f"/api/patient/{pid}/cabinet").json()
    assert all(p["body"] == main.PUSH_GENERIC for p in c["push"]["items"])


# расширенный словарь (0.6): популярные органы
@pytest.mark.parametrize("text,fid", [
    ("В S6 печени гипоэхогенное образование 18 мм.", "liver_focal_lesion"),
    ("Поджелудочная железа: в хвосте кистозное образование 15 мм.", "pancreatic_lesion"),
    ("Правая почка: в среднем сегменте солидное образование 25 мм.", "kidney_mass"),
    ("Надпочечники: в области левого надпочечника образование 22 мм.", "adrenal_mass"),
    ("Мочевой пузырь: по задней стенке пристеночное образование 9 мм.", "bladder_mass"),
    ("Правое яичко: гипоэхогенное образование 11 мм.", "testicular_mass"),
    ("Левостороннее варикоцеле.", "scrotal_veins_fluid"),
    ("ВСА справа: стеноз 65% за счёт бляшки.", "carotid_stenosis"),
    ("Инфраренальный отдел брюшной аорты расширен до 34 мм.", "aortic_aneurysm"),
    ("Определяются увеличенные лимфатические узлы шеи до 25 мм.", "lymphadenopathy"),
    ("Спленомегалия, селезёнка 160 мм.", "splenomegaly"),
    ("Асцит.", "free_fluid"),
    ("Липома передней брюшной стенки 30 мм.", "soft_tissue_mass"),
])
def test_new_organs_found(text, fid):
    assert fid in ids(text)


@pytest.mark.parametrize("text", [
    "Печень: контуры ровные, очаговых образований не выявлено.",
    "Левая почка: простая киста 12 мм.",
    "Почки: конкременты не определяются, объёмных образований нет.",
    "Атеросклероз БЦА без гемодинамически значимых стенозов, стеноз ОСА 30%.",
    "Брюшная аорта не расширена, 18 мм.",
    "Абдоминальный отдел аорты не расширен, стенки его не изменены.",
    "Регионарные лимфатические узлы не увеличены.",
    "Реактивные лимфоузлы паховой области.",
    "Свободная жидкость в малом тазу в незначительном количестве.",
    "Объемные образования: на доступных визуализации участках не выявлены.",
    "Наличие объемных образований с учетом ЦДК :не",
    "Киста придатка левого яичка 5 мм.",
    "Эхопризнаки гепатомегалии.",                                   # выключено в словаре
])
def test_new_organs_norms_not_found(text):
    assert not ids(text)
