# веб-сервис (FastAPI). Запуск: uvicorn app.main:app --port 8000 -> http://localhost:8000
import hashlib                                              # event_id файла = хэш содержимого (повторный файл не дублируется)
import hmac                                                 # сравнение токенов за постоянное время
import io                                                   # чтение загруженного файла из памяти
import json                                                 # JSON-поля
import os                                                   # пути
import shutil                                               # перемещение файлов из inbox
import threading                                            # фоновый поток для папки inbox
import time                                                 # пауза между проверками папки
import uuid                                                 # генерация ID событий
from contextlib import asynccontextmanager                  # запуск и остановка сервера
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile   # веб-фреймворк
from fastapi.responses import FileResponse, PlainTextResponse        # ответы: файл, текст
from pydantic import BaseModel                              # описание входных данных
from app import analytics, config, routing, schedule_mock, store     # модули проекта
from app.config import BASE_DIR                             # корень проекта
import re                                                   # проверка вводимых исключений
from app.nlp import extractor as _extractor                 # подгрузка исключений от администратора
from app.nlp.extractor import extract                       # извлечение находок
from app.nlp.ml_layer import get_ml_hook                    # необязательная модель

DEMO = os.environ.get("CM_ENV", "production") == "demo"    # демо-функции только в демо-режиме
API_TOKEN = os.environ.get("CM_API_TOKEN", "")              # токен для админки/API
MIS_TOKEN = os.environ.get("CM_MIS_TOKEN", "")              # отдельный токен для вебхука МИС
MAX_UPLOAD = 10 * 1024 * 1024                               # лимит загрузки 10 МБ
if not DEMO and (len(API_TOKEN) < 32 or len(MIS_TOKEN) < 32):
    raise RuntimeError("В бою нужны CM_API_TOKEN и CM_MIS_TOKEN (≥32 символов). Для демо: CM_ENV=demo")


def _check(token: str, expected: str):
    if DEMO and not expected:                               # демо без токена — разрешаем (только синтетика!)
        return
    if not token or not hmac.compare_digest(token, f"Bearer {expected}"):
        raise HTTPException(401, "Нужна авторизация")


def admin_auth(authorization: str = Header(None)):
    # токен администратора для всех эндпоинтов интерфейса
    _check(authorization, API_TOKEN)


def mis_auth(authorization: str = Header(None)):
    _check(authorization, MIS_TOKEN)


def demo_only():
    if not DEMO:
        raise HTTPException(404, "Недоступно в рабочем режиме")


@asynccontextmanager
async def lifespan(_app):
    # запуск сервера: таблицы, правки администратора, фоновые потоки
    startup()
    yield


app = FastAPI(lifespan=lifespan, title="CM Route — не потерять пациента (прототип)",
              docs_url="/docs" if DEMO else None, redoc_url=None, openapi_url="/openapi.json" if DEMO else None)
INBOX = os.path.join(BASE_DIR, "data", "inbox")             # папка автозагрузки
STATIC = os.path.join(BASE_DIR, "static")                   # папка интерфейса


# входные модели
class Event(BaseModel):
    # событие от МИС. Формат описан в README (раздел «Контракт событий»)
    event_id: str | None = None                             # уникальный ID события (для защиты от дублей)
    type: str                                               # protocol.signed | protocol.corrected | protocol.cancelled | appointment.* | patient.visit
    protocol_id: str | None = None                          # ID протокола в МИС
    patient_id: str | None = None                           # ID пациента в МИС (у нас превратится в псевдо-ID)
    study_type: str | None = None                           # тип исследования («УЗИ органов малого таза»)
    study_date: str | None = None                           # дата исследования
    text: str | None = None                                 # текст протокола
    specialty: str | None = None                            # для appointment.*: ключ маршрута (gyn_surgeon ...)
    route_id: int | None = None                             # для appointment.*: ID маршрута, если МИС его знает
    slot: str | None = None                                 # для appointment.booked: время/место


class Action(BaseModel):
    # действие по маршруту из интерфейса
    action: str                                             # verify | reject | book | cancel | no_show | visit | external | decline
    actor: str = "admin"                                    # кто нажал: admin / patient
    slot: str | None = None                                 # выбранный слот (для book)
    comment: str = ""                                       # комментарий (например, почему ложное)


class TextIn(BaseModel):
    # текст для песочницы
    text: str                                               # произвольный текст протокола


class ExclusionIn(BaseModel):
    # исключение, добавляемое администратором
    finding_id: str                                         # ключ находки из словаря или "*" (для всех находок)
    phrase: str                                             # слово/фраза: если оно рядом с находкой — находку не считаем
    scope: str = "sentence"                                 # sentence — во всём предложении; clause — только между запятыми


class AutoRuleIn(BaseModel):
    # правило автоотправки: все условия должны выполниться, пустой список — любое значение
    name: str
    active: bool = True
    routes: list[str] = []                                  # маршруты (специалисты)
    findings: list[str] = []                                # все находки маршрута должны быть из этого списка
    priorities: list[int] = []                              # 1 — высокий, 3 — плановый
    min_confidence: float = 0.0                             # уверенность каждой находки не ниже
    require_clinician: bool = False                         # врач УЗИ сам рекомендовал этого специалиста
    study_contains: str = ""                                # тип исследования содержит текст


class AutoSendIn(BaseModel):
    # настройка автоотправки пациенту без проверки администратором
    mode: str                                               # off | all | clinician_agrees | selected
    routes: list[str] = []                                  # для mode=selected: ключи маршрутов (routes.yaml)
    skip_needs_review: bool = True                          # находки с пометкой «проверить» всё равно ждут админа
    rules: list[AutoRuleIn] = []                            # для mode=rules: свои сценарии администратора


class Missed(BaseModel):
    # админ отмечает пропущенную находку (ложноотрицательный случай)
    protocol_id: str                                        # протокол
    finding_id: str                                         # какая находка пропущена
    quote: str = ""                                         # фрагмент текста
    comment: str = ""                                       # комментарий


# служебное
def startup():
    # при старте: создать таблицы и запустить наблюдение за папкой inbox
    store.init()                                            # таблицы
    _extractor.set_user_exclusions(store.user_exclusions()) # исключения, добавленные админом
    _extractor.set_user_dictionary(store.user_dictionary()) # ручные добавления в словарь
    health()                                                # битый/пустой конфиг — не стартуем
    os.makedirs(os.path.join(INBOX, "processed"), exist_ok=True)   # папки автозагрузки
    threading.Thread(target=_watch_inbox, daemon=True).start()     # фоновый поток
    if not DEMO:                                            # напоминания по реальным часам (в демо — кнопками)
        threading.Thread(target=_scheduler, daemon=True).start()


def _scheduler():
    # фоновый планировщик напоминаний по реальным часам (в демо время двигают кнопками)
    while True:
        try:
            routing.tick()
        except Exception as e:
            print("[scheduler] ошибка:", repr(e), flush=True)
        time.sleep(60)


def _pdf_text(data: bytes) -> str:
    # текст из PDF (текстовый слой). Сканы потребуют OCR — см. README
    import pdfplumber                                       # локальная библиотека чтения PDF
    with pdfplumber.open(io.BytesIO(data)) as pdf:          # открываем PDF из памяти
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)   # текст всех страниц
    if not text.strip():                                    # текста нет — это скан
        raise HTTPException(422, "В PDF нет текстового слоя (скан). Нужен OCR (Tesseract, локально) — заглушка.")   # сообщаем
    return text                                             # текст


def _docx_text(data: bytes) -> str:
    # текст из .docx (выданные протоколы МИС 1С — именно .docx, весь текст внутри таблицы)
    import docx                                             # python-docx, локально
    from docx.oxml.ns import qn
    d = docx.Document(io.BytesIO(data))
    lines = []
    for p in d.element.body.iter(qn("w:p")):                # все абзацы, в т.ч. внутри таблиц, по порядку
        t = "".join(x.text or "" for x in p.iter(qn("w:t"))).strip()
        if t and (not lines or lines[-1] != t):
            lines.append(t)
    text = "\n".join(lines)
    if not text.strip():
        raise ValueError("В .docx нет текста")
    return text


def _ingest_file(name: str, data: bytes, patient_id: str = None, study_type: str = None) -> dict:
    # превратить файл в событие protocol.signed и обработать
    low_name = name.lower()
    if low_name.endswith(".pdf"):
        text = _pdf_text(data)
    elif low_name.endswith(".docx"):                        # .docx: текст из абзацев и таблиц
        text = _docx_text(data)
    else:
        text = data.decode("utf-8")                         # без errors="ignore" — битая кодировка не «съедает» текст молча
    stem = os.path.splitext(os.path.basename(name))[0]      # имя без расширения
    return routing.handle_event({                           # обычное событие, как будто от МИС
        "event_id": "file-" + hashlib.sha256(data).hexdigest(),   # тот же файл повторно = дубль, а не новая версия протокола
        "type": "protocol.signed",                          # протокол подписан
        "protocol_id": stem,                                # ID протокола = имя файла
        "patient_id": patient_id or stem.split("__")[0],    # ID пациента: из формы или из имени «ПАЦИЕНТ__что-угодно.pdf»
        "study_type": study_type or "Исследование (файл)",  # тип исследования
        "text": text,                                       # текст
    })


def _watch_inbox():
    # «Файл загружается автоматически»: каждые 3 секунды забираем новые файлы из data/inbox
    for sub in ("processed", "failed"):
        os.makedirs(os.path.join(INBOX, sub), exist_ok=True)
    sizes = {}
    while True:
        try:
            names = sorted(os.listdir(INBOX))
        except OSError as e:
            print("[inbox] папка недоступна:", repr(e), flush=True)
            names = []
        for fn in names:
            p = os.path.join(INBOX, fn)
            if not (os.path.isfile(p) and fn.lower().endswith((".pdf", ".txt", ".docx"))):   # pdf, docx, txt
                continue
            size = os.path.getsize(p)
            if sizes.get(fn) != size:                   # файл ещё копируется — ждём следующего прохода
                sizes[fn] = size
                continue
            sizes.pop(fn, None)
            try:
                with open(p, "rb") as f:
                    _ingest_file(fn, f.read())
                shutil.move(p, os.path.join(INBOX, "processed", fn))
            except Exception as e:
                detail = e.detail if isinstance(e, HTTPException) else repr(e)
                shutil.move(p, os.path.join(INBOX, "failed", fn))
                from app import notify
                notify.to_admin(f"Файл протокола {fn} не обработан: {detail}. Нужна ручная обработка.", None, "admin_task")
                print("[inbox] ошибка:", fn, detail, flush=True)
        time.sleep(3)


# API
@app.get("/health")
def health():
    # проверка для HEALTHCHECK — сервис жив И словарь/матрица загружены и согласованы
    try:
        f, r = config.findings(), config.routes()
        missing = {x["route"] for x in f["findings"].values()} - set(r["routes"])
        if missing:
            raise ValueError(f"маршрутов нет в routes.yaml: {missing}")
        store.q("SELECT 1")
    except Exception as e:
        raise HTTPException(503, f"не готов: {e}")
    return {"ok": True, "dictionary": f.get("version"), "matrix": r.get("version")}


@app.get("/")   # адрес API /: GET — получить данные
def index():    # страница статична; все данные она берёт из API с токеном
    # интерфейс администратора
    return FileResponse(os.path.join(STATIC, "index.html")) # отдаём HTML-страницу


@app.post("/api/events", dependencies=[Depends(mis_auth)])
def events(ev: Event):
    # вебхук для МИС. Повторная доставка того же event_id не создаёт дублей
    try:
        return routing.handle_event(ev.model_dump())
    except ValueError as e:                                 # BadEvent — подкласс ValueError
        raise HTTPException(400, str(e))


@app.post("/api/upload", dependencies=[Depends(admin_auth)])
async def upload(file: UploadFile = File(...), patient_id: str = Form(...), study_type: str = Form(None)):
    # загрузить файл протокола вручную, patient_id обязателен
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл больше 10 МБ")
    try:
        return _ingest_file(file.filename or "upload.txt", data, patient_id, study_type)
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:                                  # битый .docx/.pdf — понятная ошибка вместо 500
        raise HTTPException(400, f"Не удалось прочитать файл: {e.__class__.__name__}")


@app.post("/api/analyze", dependencies=[Depends(admin_auth)])   # адрес API /api/analyze: POST — выполнить действие
def analyze(t: TextIn):
    # песочница: что найдёт система и почему (ничего не сохраняет)
    return extract(t.text, get_ml_hook()).to_dict()        # результат извлечения


@app.get("/api/routes", dependencies=[Depends(admin_auth)])   # адрес API /api/routes: GET — получить данные
def list_routes(status: str = None):
    # очередь маршрутов: экстренные и требующие верификации — сверху
    rows = store.q("SELECT r.*, p.study, p.study_date FROM routes r JOIN protocols p ON p.id=r.protocol_id ORDER BY r.id DESC")   # все маршруты
    order = {"escalated": 0, "pending_verification": 1}    # что показывать первым
    out = []                                                # результат
    for r in rows:                                          # каждый маршрут
        if status and r["status"] != status:                # фильтр по статусу
            continue                                        # пропускаем
        r["findings"] = json.loads(r["findings"])           # находки из JSON
        r["status_label"] = routing.STATUS.get(r["status"], r["status"])   # подпись статуса
        r["specialist"] = config.routes()["routes"].get(r["route_key"], {}).get("specialist", r["route_key"])   # специалист
        out.append(r)                                       # добавляем
    return sorted(out, key=lambda r: (order.get(r["status"], 2), r["priority"], -r["id"]))   # сортировка очереди


@app.get("/api/routes/{rid}", dependencies=[Depends(admin_auth)])   # адрес API /api/routes/{rid}: GET — получить данные
def get_route(rid: int):
    # карточка маршрута: всё для решения админа и объяснения «почему так сработала система»
    r = store.route(rid)                                    # маршрут
    if not r:                                               # нет такого
        raise HTTPException(404, "Маршрут не найден")       # 404
    p = store.q("SELECT * FROM protocols WHERE id=?", (r["protocol_id"],))[0]   # протокол
    p["extraction"] = json.loads(p["extraction"])           # результат извлечения
    rc = config.routes()["routes"].get(r["route_key"], {})  # настройки маршрута
    r["status_label"] = routing.STATUS.get(r["status"], r["status"])   # подпись статуса
    return {
        "route": r, "route_cfg": rc, "protocol": p,         # маршрут, его настройки, протокол
        "audit": store.q("SELECT * FROM audit WHERE route_id=? OR (protocol_id=? AND route_id IS NULL) ORDER BY id", (rid, r["protocol_id"])),   # журнал
        "notifications": store.q("SELECT * FROM notifications WHERE route_id=? ORDER BY id", (rid,)),   # сообщения
        "slots": schedule_mock.slots(r["route_key"]),       # ближайшие слоты (заглушка расписания)
        "open_routes_banner": routing.open_routes(r["patient"]),   # баннер для МИС
    }


@app.post("/api/routes/{rid}/action", dependencies=[Depends(admin_auth)])   # адрес API /api/routes/{rid}/action: POST — выполнить действие
def route_action(rid: int, a: Action):
    # действие по маршруту (кнопки администратора; пациент действует через /api/patient/...)
    actor = a.actor if (DEMO and a.actor in ("admin", "patient")) else "admin"
    try:
        return routing.transition(rid, a.action, actor, a.slot, a.comment)
    except ValueError as e:                                 # неизвестное действие / нет маршрута
        raise HTTPException(400, str(e))                    # ошибка 400


@app.get("/api/notifications", dependencies=[Depends(admin_auth)])   # адрес API /api/notifications: GET — получить данные
def notifications(kind: str = None, limit: int = 50):
    limit = max(1, min(limit, 500))                         # без верхней границы limit=10**9 выгружал всю таблицу
    # лента «отправленных» сообщений (заглушка): kind=admin | patient
    sql = "SELECT * FROM notifications" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC LIMIT ?"   # запрос
    return store.q(sql, ((kind, limit) if kind else (limit,)))   # результат


@app.get("/api/dashboard", dependencies=[Depends(admin_auth)])   # адрес API /api/dashboard: GET — получить данные
def dashboard():
    # воронка и экономика
    return analytics.dashboard()                            # считаем


@app.get("/api/audit", dependencies=[Depends(admin_auth)])   # адрес API /api/audit: GET — получить данные
def audit(limit: int = 200):
    limit = max(1, min(limit, 1000))
    # журнал действий
    return store.q("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))   # последние записи


@app.get("/api/clock", dependencies=[Depends(admin_auth)])   # адрес API /api/clock: GET — получить данные
def clock():
    # текущее модельное время
    return {"now": store.ts()}                              # время


@app.post("/api/clock/advance", dependencies=[Depends(admin_auth), Depends(demo_only)])   # адрес API /api/clock/advance: POST — выполнить действие
def clock_advance(hours: float = 24):
    # промотать модельное время и выполнить созревшие напоминания (только демо)
    if not 0 < hours <= 24 * 60:
        raise HTTPException(400, "hours: от 0 до 1440")
    store.advance(hours)                                    # двигаем часы
    return {"now": store.ts(), **routing.tick()}            # и запускаем планировщик


@app.get("/api/patients/{patient}/open_routes", dependencies=[Depends(admin_auth)])   # адрес API /api/patients/{patient}/open_routes: GET — получить данные
def patient_open_routes(patient: str):
    # для МИС: незавершённые маршруты пациента (баннер при любом визите). patient — псевдо-ID
    return routing.open_routes(patient)                     # список баннеров


@app.post("/api/feedback/missed", dependencies=[Depends(admin_auth)])   # адрес API /api/feedback/missed: POST — выполнить действие
def feedback_missed(m: Missed):
    # админ отметил пропущенную находку — это материал для словаря и дообучения
    store.x("INSERT INTO feedback (ts, protocol_id, finding_id, kind, quote, comment) VALUES (?,?,?,?,?,?)",
            (store.ts(), m.protocol_id, m.finding_id, "missed", m.quote, m.comment))   # сохраняем
    store.audit("admin", "feedback_missed", protocol_id=m.protocol_id, finding=m.finding_id)   # журнал
    return {"ok": True}                                     # ответ


@app.get("/api/feedback/export", response_class=PlainTextResponse, dependencies=[Depends(admin_auth)])   # адрес API /api/feedback/export: GET — получить данные
def feedback_export():
    # выгрузка обратной связи в JSONL (для scripts/train.py и пополнения словаря)
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in store.q("SELECT * FROM feedback"))   # построчно JSON


@app.get("/api/config", dependencies=[Depends(admin_auth)])   # адрес API /api/config: GET — получить данные
def get_config():
    # текущие словарь и матрица (только просмотр; редактируются YAML-файлы)
    return {"findings": {**config.findings(), "findings": _extractor.all_findings()}, "routes": config.routes()}   # вместе с находками, добавленными администратором


@app.post("/api/demo/reset", dependencies=[Depends(admin_auth), Depends(demo_only)])   # очистка базы: только демо и только с токеном
def demo_reset():
    # очистить базу (демо)
    store.reset()                                           # удаляем и создаём заново
    return {"ok": True}                                     # ответ


@app.post("/api/demo/load", dependencies=[Depends(admin_auth), Depends(demo_only)])   # адрес API /api/demo/load: POST — выполнить действие
def demo_load():
    # загрузить синтетические протоколы как события МИС (демо)
    path = os.path.join(BASE_DIR, "data", "synthetic", "protocols.jsonl")   # файл с синтетикой
    n = 0                                                   # счётчик
    with open(path, encoding="utf-8") as f:                 # читаем
        for line in f:                                      # построчно
            if not line.strip():                            # пустые строки пропускаем
                continue                                    # дальше
            d = json.loads(line)                            # запись
            routing.handle_event({"event_id": f"demo-{d['id']}", "type": "protocol.signed", "protocol_id": d["id"],
                                  "patient_id": d.get("patient_id", d["id"]), "study_type": d.get("study", "УЗИ"),
                                  "text": d["text"]})       # как событие от МИС
            n += 1                                          # считаем
    return {"loaded": n}                                    # ответ


# исключения от администратора
_PHRASE_RE = re.compile(r"^[А-Яа-яЁёA-Za-z0-9][А-Яа-яЁёA-Za-z0-9 \-]{1,59}$")


def _reload_exclusions():
    _extractor.set_user_exclusions(store.user_exclusions())


@app.get("/api/exclusions", dependencies=[Depends(admin_auth)])
def list_exclusions():
    # исключения: из словаря (только чтение) и добавленные администратором (можно выключить)
    f = _extractor.all_findings()
    return {"dictionary": {k: {"name": v["name"], "exclude": v.get("exclude", []), "scope": v.get("exclude_scope", "clause"),
                               "context_sensitive": bool(v.get("context_sensitive"))} for k, v in f.items()},
            "user": store.user_exclusions(active_only=False)}


@app.post("/api/exclusions", dependencies=[Depends(admin_auth)])
def add_exclusion(e: ExclusionIn):
    # добавить исключение: если слово стоит рядом с находкой, находка не засчитывается
    phrase = " ".join(e.phrase.split())
    if e.finding_id != "*" and e.finding_id not in _extractor.all_findings():
        raise HTTPException(400, "Нет такой находки в словаре")
    if e.scope not in ("sentence", "clause"):
        raise HTTPException(400, "scope: sentence или clause")
    if not _PHRASE_RE.match(phrase):
        raise HTTPException(400, "Фраза: 2–60 символов, буквы, цифры, пробел, дефис")
    ex_id = store.add_user_exclusion(e.finding_id, phrase, e.scope)
    store.audit("admin", "exclusion_added", finding=e.finding_id, phrase=phrase, scope=e.scope, exclusion_id=ex_id)
    _reload_exclusions()
    return {"ok": True, "id": ex_id}


@app.delete("/api/exclusions/{ex_id}", dependencies=[Depends(admin_auth)])
def delete_exclusion(ex_id: int):
    # выключить исключение, добавленное администратором (словарные — только правкой YAML)
    store.disable_user_exclusion(ex_id)
    store.audit("admin", "exclusion_disabled", exclusion_id=ex_id)
    _reload_exclusions()
    return {"ok": True}


# автоотправка без проверки
@app.get("/api/settings/auto_send", dependencies=[Depends(admin_auth)])
def get_auto_send():
    # текущая настройка автоотправки + варианты для интерфейса
    rr = config.routes()["routes"]
    fs = _extractor.all_findings()
    return {"current": routing.auto_send_settings(), "modes": routing.AUTO_MODES,
            "routes": {k: v.get("specialist", k) for k, v in rr.items() if not v.get("escalation_only")},
            "findings": {k: v["name"] for k, v in fs.items() if not v.get("urgent")}}


def _check_rules(rules: list) -> list:
    known_r, known_f = set(config.routes()["routes"]), set(_extractor.all_findings())
    out = []
    if len(rules) > 30:
        raise HTTPException(400, "Не больше 30 правил")
    for r in rules:
        d = r.model_dump()
        d["name"] = " ".join(d["name"].split())[:80]
        d["study_contains"] = " ".join(d["study_contains"].split())[:60]
        if len(d["name"]) < 2:
            raise HTTPException(400, "У правила должно быть название")
        bad = [x for x in d["routes"] if x not in known_r] + [x for x in d["findings"] if x not in known_f]
        if bad:
            raise HTTPException(400, f"Правило «{d['name']}»: неизвестные маршруты или находки {bad}")
        if any(p not in (1, 2, 3) for p in d["priorities"]) or not 0 <= d["min_confidence"] <= 1:
            raise HTTPException(400, f"Правило «{d['name']}»: приоритет 1–3, уверенность 0–1")
        out.append(d)
    return out


@app.post("/api/settings/auto_send/simulate", dependencies=[Depends(admin_auth)])
def simulate_auto_send(a: AutoSendIn):
    # что произошло бы с уже созданными маршрутами при такой настройке (ничего не меняет)
    if a.mode not in routing.AUTO_MODES:
        raise HTTPException(400, f"mode: {', '.join(routing.AUTO_MODES)}")
    cfg = {"mode": a.mode, "routes": a.routes, "skip_needs_review": a.skip_needs_review, "rules": _check_rules(a.rules)}
    rows = store.q("SELECT r.id, r.route_key, r.priority, r.urgent, r.findings, r.status, p.study, p.extraction "
                   "FROM routes r JOIN protocols p ON p.id=r.protocol_id ORDER BY r.id DESC LIMIT 500")
    out = {"total": 0, "auto": 0, "blocked": 0, "manual": 0, "examples": []}
    for r in rows:
        if r["urgent"]:
            continue
        out["total"] += 1
        rec = json.loads(r["extraction"] or "{}").get("clinician_recommended", [])
        fs = json.loads(r["findings"] or "[]")
        ok, why = routing.auto_send_decision(r["route_key"], fs, rec, r["study"], r["priority"], cfg)
        kind = "auto" if ok else ("blocked" if why else "manual")
        out[kind] += 1
        if kind != "manual" and len(out["examples"]) < 15:
            out["examples"].append({"route_id": r["id"], "result": kind, "why": why,
                                    "findings": [f["name"] for f in fs], "route": r["route_key"]})
    return out


@app.put("/api/settings/auto_send", dependencies=[Depends(admin_auth)])
def put_auto_send(a: AutoSendIn):
    # изменить настройку. Экстренные находки НИКОГДА не уходят пациенту автоматически — только персоналу
    if a.mode not in routing.AUTO_MODES:
        raise HTTPException(400, f"mode: {', '.join(routing.AUTO_MODES)}")
    known = set(config.routes()["routes"])
    bad = [r for r in a.routes if r not in known]
    if bad:
        raise HTTPException(400, f"Неизвестные маршруты: {bad}")
    old = routing.auto_send_settings()
    new = {"mode": a.mode, "routes": a.routes, "skip_needs_review": a.skip_needs_review, "rules": _check_rules(a.rules),
           "changed_by": "admin", "changed_at": store.ts()}
    if a.mode == "rules" and not any(r["active"] for r in new["rules"]):
        raise HTTPException(400, "Режим «по правилам»: добавьте хотя бы одно включённое правило")
    store.set_setting("auto_send", new)
    keys = ("mode", "routes", "skip_needs_review", "rules")
    store.audit("admin", "auto_send_changed", old={k: old.get(k) for k in keys}, new={k: new[k] for k in keys})
    return {"ok": True, "current": new}


# ручное добавление в словарь
# kind: term — самостоятельный термин находки («аденомиоз»); what — «что» для «ЧТО + ГДЕ» (нужен орган в where);
# location — синоним органа («придатки», «ЛЖВ»); finding — новая находка со своим маршрутом
DICT_KINDS = {"term": "термин находки", "what": "«что» (ищется рядом с органом)", "location": "синоним органа",
              "finding": "новая находка"}


class DictIn(BaseModel):
    # ручное добавление в словарь из интерфейса
    kind: str                                               # term | what | location | finding
    target: str = ""                                        # ключ находки или органа (для finding — пусто, ключ создаём сами)
    phrase: str                                             # слово/фраза (для finding — первый термин)
    name: str = ""                                          # finding: название для администратора
    route: str = ""                                         # finding: маршрут из routes.yaml
    priority: int = 3                                       # finding: 1 — высокий, 3 — плановый
    urgent: bool = False                                    # finding: экстренная (сразу персоналу, пациенту не пишем)
    where: list[str] = []                                   # finding: органы (нужны, чтобы потом добавлять «что»)


def _reload_dictionary():
    _extractor.set_user_dictionary(store.user_dictionary())


def _locations_view() -> dict:
    locs = {k: list(v) for k, v in config.findings().get("locations", {}).items()}
    for it in store.user_dictionary():
        if it["kind"] == "location" and it["target"] in locs:
            locs[it["target"]].append(it["phrase"] + " (админ)")
    return locs


@app.get("/api/dictionary", dependencies=[Depends(admin_auth)])
def list_dictionary():
    # словарь для просмотра и ручные добавления администратора
    fs = _extractor.all_findings()
    rr = config.routes()["routes"]
    return {"kinds": DICT_KINDS,
            "findings": {k: {"name": v["name"], "route": v.get("route"), "where": v.get("where", []),
                             "terms": v.get("terms", []), "what": v.get("what", []),
                             "user_added": bool(v.get("user_added"))} for k, v in fs.items()},
            "locations": _locations_view(),
            "routes": {k: v.get("specialist", k) for k, v in rr.items()},
            "user": store.user_dictionary(active_only=False)}


@app.post("/api/dictionary", dependencies=[Depends(admin_auth)])
def add_dictionary(d: DictIn):
    # добавить термин / «что» / синоним органа / новую находку; действует сразу, без перезапуска
    phrase = " ".join(d.phrase.split())
    if d.kind not in DICT_KINDS:
        raise HTTPException(400, f"kind: {', '.join(DICT_KINDS)}")
    if not _PHRASE_RE.match(phrase) or sum(ch.isalpha() for ch in phrase) < 2:
        raise HTTPException(400, "Фраза: 2–60 символов, буквы, цифры, пробел, дефис")
    fs = _extractor.all_findings()
    locs = config.findings().get("locations", {})
    extra, target = {}, d.target
    if d.kind in ("term", "what"):
        if target not in fs:
            raise HTTPException(400, "Нет такой находки")
        if d.kind == "what" and not fs[target].get("where"):
            raise HTTPException(400, "У этой находки не задан орган — добавьте фразу как «термин»")
    elif d.kind == "location":
        if target not in locs:
            raise HTTPException(400, "Нет такого органа в словаре")
    else:                                                   # новая находка
        name = " ".join(d.name.split())
        rr = config.routes()["routes"]
        if not 2 <= len(name) <= 80:
            raise HTTPException(400, "Название находки: 2–80 символов")
        if d.route not in rr:
            raise HTTPException(400, "Нет такого маршрута в routes.yaml")
        if d.priority not in (1, 2, 3):
            raise HTTPException(400, "Приоритет: 1, 2 или 3")
        bad = [w for w in d.where if w not in locs]
        if bad:
            raise HTTPException(400, f"Неизвестные органы: {bad}")
        target = "user_" + uuid.uuid4().hex[:8]
        extra = {"name": name, "route": d.route, "priority": d.priority, "urgent": d.urgent, "where": d.where}
    dup = [it for it in store.user_dictionary() if it["kind"] == d.kind and it["target"] == target
           and it["phrase"].lower() == phrase.lower()]
    if dup:
        raise HTTPException(409, "Такая запись уже есть")
    item_id = store.add_user_dictionary(d.kind, target, phrase, extra)
    store.audit("admin", "dictionary_added", kind=d.kind, target=target, phrase=phrase, item_id=item_id, **extra)
    _reload_dictionary()
    return {"ok": True, "id": item_id, "target": target}


@app.delete("/api/dictionary/{item_id}", dependencies=[Depends(admin_auth)])
def delete_dictionary(item_id: int):
    # выключить ручное добавление (записи YAML меняются только правкой файла)
    store.disable_user_dictionary(item_id)
    store.audit("admin", "dictionary_disabled", item_id=item_id)
    _reload_dictionary()
    return {"ok": True}


# сценарий уведомлений пациента
ADMIN_TEMPLATES = {"admin_new", "admin_urgent"}                         # служебные, в редакторе не показываем
PATIENT_FIELDS = {"specialist", "slot"}                                 # что можно подставлять в текст пациенту
ADMIN_FIELDS = {"specialist", "patient"}                                # в задачу админу (скрипт звонка)
_TPL_KEY_RE = re.compile(r"^[a-z0-9_]{2,40}$")
# слова, которые пугают или раскрывают находку. Сравниваем по началу слова: «кист» ловит «киста», «кисты»
FORBIDDEN_STEMS = ["рак", "онко", "опухол", "злокачеств", "доброкачеств", "метастаз", "патолог", "новообразов",
                   "образовани", "узел", "узл", "кист", "полип", "миом", "тромб", "камн", "камен", "конкремент",
                   "диагноз", "подозрен", "изменени", "отклонени", "находк", "опасн", "срочн", "экстренн", "немедленн",
                   "биопси", "операци", "болезн", "заболевани", "серьезн", "серьёзн", "тревож", "угроз"]
_EXACT = {"рак": r"рак(?:а|у|ом|е|и|ов)?(?![а-яё])", "камен": r"камень(?![а-яё])|каменист[а-яё]*"}   # короткие основы — без «ракета», «каменщик»
_FORBIDDEN_RE = re.compile(r"(?<![а-яё])(?:" + "|".join(_EXACT.get(x, x + r"[а-яё]*") for x in FORBIDDEN_STEMS) + ")", re.IGNORECASE)
SAMPLE = {"specialist": "врача-гинеколога", "slot": "12.10 10:00 — Текстильщики", "patient": "P-1a2b3c4d"}


class ReminderIn(BaseModel):
    after_hours: int                                        # через сколько часов после первого сообщения
    action: str                                             # patient_message | admin_task | set_status
    template: str = ""                                      # ключ текста (для patient_message / admin_task)
    channels: list[str] = []                                # lk | push | sms
    status: str = ""                                        # для set_status: not_engaged


class ScenarioIn(BaseModel):
    templates: dict[str, str]                               # ключ -> текст
    reminders: list[ReminderIn]                             # лестница напоминаний


def _fields(text: str) -> set:
    import string
    try:
        return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}
    except ValueError:
        raise HTTPException(400, f"Незакрытая фигурная скобка в тексте: «{text[:40]}…»")


def neutrality_issues(text: str) -> list:
    # проверка текста пациенту: нет пугающих/медицинских слов и словарь не находит в нём ни одной находки
    issues = [f"слово «{m.group(0)}»" for m in _FORBIDDEN_RE.finditer(text)]
    found = extract(text, None).findings
    issues += [f"текст похож на находку «{f.name}»" for f in found]
    return issues


def _admin_facing(templates_used_by_admin: set, key: str) -> bool:
    return key in ADMIN_TEMPLATES or key in templates_used_by_admin


def _check_scenario(templates: dict, reminders: list) -> dict:
    # возвращает отчёт по каждому тексту; ошибки формата -> HTTP 400, проблемы нейтральности -> в отчёте
    if len(reminders) > 12:
        raise HTTPException(400, "Не больше 12 шагов напоминаний")
    admin_used = {r["template"] for r in reminders if r["action"] == "admin_task"}
    hours = [r["after_hours"] for r in reminders]
    if len(set(hours)) != len(hours):
        raise HTTPException(400, "У двух шагов одинаковое время")
    for i, r in enumerate(reminders):
        if not 1 <= r["after_hours"] <= 2160:
            raise HTTPException(400, "Время шага: от 1 до 2160 часов (90 дней)")
        if r["action"] not in ("patient_message", "admin_task", "set_status"):
            raise HTTPException(400, "Действие шага: patient_message, admin_task или set_status")
        if r["action"] == "set_status":
            if r.get("status") != "not_engaged":
                raise HTTPException(400, "set_status: только not_engaged")
            if i != len(reminders) - 1:
                raise HTTPException(400, "«Маршрут не реализован» должен быть последним шагом — после него напоминаний нет")
        else:
            if r["template"] not in templates:
                raise HTTPException(400, f"Шаг через {r['after_hours']} ч ссылается на несуществующий текст «{r['template']}»")
            if r["action"] == "patient_message":
                bad = set(r.get("channels") or []) - {"lk", "push", "sms"}
                if bad or not r.get("channels"):
                    raise HTTPException(400, "Каналы: lk, push, sms (хотя бы один)")
                if r["template"] in admin_used or r["template"] in ADMIN_TEMPLATES:
                    raise HTTPException(400, f"Текст «{r['template']}» — служебный, пациенту его отправлять нельзя")
    report = {}
    for key, text in templates.items():
        if not _TPL_KEY_RE.match(key):
            raise HTTPException(400, f"Ключ текста «{key}»: латиница, цифры и _ (2–40 символов)")
        if not 5 <= len(text.strip()) <= 600:
            raise HTTPException(400, f"Текст «{key}»: от 5 до 600 символов")
        admin = _admin_facing(admin_used, key)
        allowed = (ADMIN_FIELDS | {"study", "route_specialist", "priority"}) if admin else PATIENT_FIELDS
        unknown = _fields(text) - allowed
        if unknown:
            raise HTTPException(400, f"Текст «{key}»: неизвестные поля {sorted(unknown)}; можно {sorted(allowed)}")
        class _Safe(dict):
            def __missing__(self, k): return ""
        rendered = text.format_map(_Safe(SAMPLE))
        report[key] = {"audience": "admin" if admin else "patient", "preview": rendered,
                       "issues": [] if admin else neutrality_issues(rendered)}
    for k in routing.PATIENT_TEMPLATES:                     # обязательные тексты пациенту
        if k not in templates:
            raise HTTPException(400, f"Нельзя удалить обязательный текст «{k}»")
    return report


def _scenario_payload() -> dict:
    sc = routing.scenario()
    tpl = {k: v for k, v in sc["templates"].items() if k not in ADMIN_TEMPLATES}
    rem = [dict(r, channels=r.get("channels") or ["lk", "push"]) if r["action"] == "patient_message" else dict(r)
           for r in sc["reminders"]]                        # в YAML каналы можно не писать: по умолчанию ЛК + пуш
    report = _check_scenario(tpl, rem)
    return {"templates": tpl, "reminders": rem, "report": report, "required": list(routing.PATIENT_TEMPLATES),
            "changed_at": sc.get("changed_at"), "changed_by": sc.get("changed_by"), "forbidden": FORBIDDEN_STEMS,
            "fields": {"patient": sorted(PATIENT_FIELDS), "admin": sorted(ADMIN_FIELDS)}}


def _remap_ladder(old_steps: list, new_steps: list):
    # маршрут помнит номер шага лестницы. После правки лестницы номер пересчитываем по времени уже
    # выполненного шага: пройденные шаги не повторяются, новые шаги в прошлом не «догоняют» пациента
    for r in store.q("SELECT id, ladder_step FROM routes WHERE ladder_step > 0"):
        done_h = old_steps[min(r["ladder_step"], len(old_steps)) - 1]["after_hours"] if old_steps else 0
        store.update_route(r["id"], ladder_step=sum(1 for st in new_steps if st["after_hours"] <= done_h))


@app.get("/api/settings/scenario", dependencies=[Depends(admin_auth)])
def get_scenario():
    # тексты пациенту + лестница напоминаний + отчёт о нейтральности каждого текста
    return _scenario_payload()


@app.post("/api/settings/scenario/preview", dependencies=[Depends(admin_auth)])
def preview_scenario(s: ScenarioIn):
    # проверить черновик, ничего не сохраняя
    rem = sorted([r.model_dump() for r in s.reminders], key=lambda r: r["after_hours"])
    return {"report": _check_scenario(s.templates, rem)}


@app.put("/api/settings/scenario", dependencies=[Depends(admin_auth)])
def put_scenario(s: ScenarioIn):
    # сохранить сценарий. Текст с пугающими словами или похожий на находку НЕ сохраняется
    rem = sorted([r.model_dump() for r in s.reminders], key=lambda r: r["after_hours"])
    for r in rem:                                           # лишние поля не храним
        if r["action"] != "set_status":
            r.pop("status", None)
        if r["action"] != "patient_message":
            r.pop("channels", None)
        if r["action"] == "set_status":
            r.pop("template", None)
    report = _check_scenario(s.templates, rem)
    bad = {k: v["issues"] for k, v in report.items() if v["issues"]}
    if bad:
        raise HTTPException(400, "Тексты для пациента должны быть нейтральными. " +
                            "; ".join(f"«{k}»: {', '.join(v)}" for k, v in bad.items()))
    old = routing.scenario()
    store.set_setting("patient_scenario", {"templates": s.templates, "reminders": rem,
                                           "changed_at": store.ts(), "changed_by": "admin"})
    _remap_ladder(old["reminders"], rem)
    changed = sorted(k for k in s.templates if old["templates"].get(k) != s.templates[k])
    store.audit("admin", "scenario_changed", templates_changed=changed, reminders_changed=old["reminders"] != rem)
    return {"ok": True, **_scenario_payload()}


@app.delete("/api/settings/scenario", dependencies=[Depends(admin_auth)])
def reset_scenario():
    # вернуть тексты и лестницу из routes.yaml
    old = routing.scenario()
    store.set_setting("patient_scenario", {})
    _remap_ladder(old["reminders"], routing.scenario()["reminders"])
    store.audit("admin", "scenario_reset")
    return {"ok": True, **_scenario_payload()}


# личный кабинет пациента (эмулятор)
# что видит пациент. ГЛАВНОЕ ПРАВИЛО: ни названий находок, ни текста протокола, ни слов «отклонение»/«изменения»
# только: «плановая консультация специалиста», статус записи, слоты. В бою этот экран — личный кабинет МИС
PATIENT_STATUS = {
    "notified": ("todo", "Ждёт записи"), "rebook_needed": ("todo", "Ждёт записи"), "no_show": ("todo", "Ждёт новой записи"),
    "not_engaged": ("todo", "Ждёт записи"), "booked": ("booked", "Вы записаны"), "visited": ("done", "Завершено"),
    "external": ("done", "Отмечено: консультация в другой клинике"), "declined": ("done", "Вы отказались от записи"),
}
PATIENT_ACTIONS = {"book", "cancel", "external", "decline"}
BANNER = ("У вас есть незавершённый маршрут обследования. Рекомендуем записаться на плановую консультацию — "
          "это обычный следующий шаг, он займёт пару минут.")


def _patient_routes(pid: str) -> list:
    rows = store.q("SELECT * FROM routes WHERE patient=? ORDER BY id", (pid,))
    return [r for r in rows if r["status"] in PATIENT_STATUS and (r["notified_at"] or r["status"] in ("booked", "visited"))]


@app.get("/api/patient/list", dependencies=[Depends(admin_auth)])
def patient_list():
    # пациенты, которым уже что-то отправлено (для выбора в эмуляторе кабинета). Только псевдо-ID
    rows = store.q("SELECT recipient AS patient, COUNT(DISTINCT batch) AS messages, MAX(ts) AS last FROM notifications "
                   "WHERE kind='patient' AND status='sent' GROUP BY recipient ORDER BY last DESC")
    for r in rows:
        r["short"] = r["patient"][:10]
        r["open"] = sum(1 for x in _patient_routes(r["patient"]) if PATIENT_STATUS[x["status"]][0] == "todo")
    return rows


def _upcoming(r: dict, rc: dict) -> list:
    # какие напоминания ещё придут по этому маршруту, если пациент не запишется (для демонстрации)
    if r["status"] not in routing.LADDER or not r["ladder_start"]:
        return []
    from datetime import datetime, timedelta
    sc = routing.scenario()
    start = datetime.fromisoformat(r["ladder_start"])
    out = []
    for i, s in enumerate(sc["reminders"]):
        if i < (r["ladder_step"] or 0) or s["action"] != "patient_message":
            continue
        out.append({"at": (start + timedelta(hours=s["after_hours"])).isoformat(), "channels": s.get("channels", ["lk", "push"]),
                    "text": routing._tpl(s["template"], specialist=rc.get("specialist_for_patient", ""))})
    return out


@app.get("/api/patient/{pid}/cabinet", dependencies=[Depends(admin_auth)])
def patient_cabinet(pid: str):
    # экран пациента: нейтральный баннер, лента сообщений, шаги маршрута с кнопками записи
    msgs = store.q("SELECT batch, MAX(id) AS id, MIN(ts) AS ts, text, GROUP_CONCAT(channel) AS channels FROM notifications "
                   "WHERE recipient=? AND kind='patient' AND status='sent' GROUP BY batch ORDER BY ts DESC", (pid,))
    rr = config.routes()["routes"]
    tasks = []
    for r in _patient_routes(pid):
        rc = rr.get(r["route_key"], {})
        kind, label = PATIENT_STATUS[r["status"]]
        spec = rc.get("specialist_for_patient") or "врача"
        tasks.append({"route_id": r["id"], "title": f"Плановая консультация {spec}", "state": kind, "status": label,
                      "slot": r["booked_slot"] if kind == "booked" else None,
                      "slots": schedule_mock.slots(r["route_key"]) if kind == "todo" else [],
                      "upcoming": _upcoming(r, rc) if DEMO else []})
    todo = any(t["state"] == "todo" for t in tasks)
    return {"patient": pid[:10], "now": store.ts(), "banner": BANNER if todo else None,
            "messages": [{"id": m["id"], "ts": m["ts"], "text": m["text"], "channels": sorted(set(m["channels"].split(",")))}
                         for m in msgs],
            "push": _push_view(msgs), "tasks": tasks, "can_act": DEMO}


PUSH_GENERIC = "У вас новое сообщение в личном кабинете клиники."


def _push_view(msgs: list) -> dict:
    # что пациент видит на экране блокировки телефона (заглушка пуш-сервиса)
    beh = config.routes().get("behavior", {})
    generic = beh.get("push_on_lock_screen", "full") == "generic"
    return {"title": beh.get("push_title", "Клиника"),
            "items": [{"id": m["id"], "ts": m["ts"], "body": PUSH_GENERIC if generic else m["text"]}
                      for m in msgs if "push" in m["channels"].split(",")],
            "sms": [{"id": m["id"], "ts": m["ts"], "body": m["text"]} for m in msgs if "sms" in m["channels"].split(",")]}


class PatientAction(BaseModel):
    action: str                                             # book | cancel | external | decline
    slot: str | None = None


@app.post("/api/patient/{pid}/routes/{rid}/action", dependencies=[Depends(admin_auth)])
def patient_action(pid: str, rid: int, a: PatientAction):
    # кнопки в кабинете пациента. В бою пациент действует в ЛК МИС, поэтому здесь — только демо
    if not DEMO:
        raise HTTPException(403, "В боевом режиме кабинет пациента — только просмотр (действия выполняет ЛК МИС)")
    if a.action not in PATIENT_ACTIONS:
        raise HTTPException(400, "Действие: book, cancel, external, decline")
    if not any(x["id"] == rid for x in _patient_routes(pid)):     # маршрут чужой или пациенту ещё не показан
        raise HTTPException(404, "Нет такого шага в маршруте пациента")
    try:
        routing.transition(rid, a.action, "patient", a.slot, "")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return patient_cabinet(pid)
