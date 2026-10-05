# хранилище: локальная SQLite-база (один файл). Ничего не уходит наружу
import json                                        # хранение сложных полей как JSON
import os                                          # пути
import sqlite3                                     # встроенная база данных
import threading                                   # блокировка для многопоточного веб-сервера
from contextlib import contextmanager              # транзакция «всё или ничего» для обработки события
from datetime import datetime, timedelta           # время
from app.config import BASE_DIR                    # корень проекта

DB_PATH = os.environ.get("CM_DB", os.path.join(BASE_DIR, "data", "cm_route.db"))   # путь к файлу базы
# модельное время — только для демо. В бою часы реальные, иначе напоминания никогда не наступят
MODEL_CLOCK = os.environ.get("CM_ENV", "production") == "demo"
_lock = threading.RLock()                          # одна блокировка на запись (в пределах ОДНОГО процесса — запускать с 1 воркером)
_local = threading.local()                         # соединение текущей транзакции (для вложенных вызовов)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS protocols (id TEXT PRIMARY KEY NOT NULL, patient TEXT NOT NULL, study TEXT, study_date TEXT, text TEXT,
    version INTEGER, status TEXT, extraction TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS routes (id INTEGER PRIMARY KEY AUTOINCREMENT, protocol_id TEXT NOT NULL, patient TEXT NOT NULL, route_key TEXT NOT NULL,
    status TEXT NOT NULL, findings TEXT, verification TEXT, priority INTEGER, urgent INTEGER, due_date TEXT, created_at TEXT,
    notified_at TEXT, ladder_start TEXT, ladder_step INTEGER DEFAULT 0, booked_slot TEXT, updated_at TEXT,
    acked_by TEXT, acked_at TEXT, last_escalation TEXT);
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, received_at TEXT);
CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, channel TEXT, recipient TEXT,
    text TEXT, route_id INTEGER, status TEXT, kind TEXT, batch TEXT);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, route_id INTEGER,
    protocol_id TEXT, details TEXT);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, protocol_id TEXT, finding_id TEXT,
    kind TEXT, quote TEXT, comment TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
-- НОВОЕ v5: исключения, которые администратор добавил через интерфейс (словарь YAML при этом не меняется)
CREATE TABLE IF NOT EXISTS user_exclusions (id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id TEXT NOT NULL, phrase TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'sentence', added_by TEXT, ts TEXT, active INTEGER NOT NULL DEFAULT 1);
-- НОВОЕ v6: ручные добавления в словарь (термины, «что», синонимы органов, новые находки)
CREATE TABLE IF NOT EXISTS user_dictionary (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, target TEXT NOT NULL,
    phrase TEXT NOT NULL, extra TEXT, added_by TEXT, ts TEXT, active INTEGER NOT NULL DEFAULT 1);
-- FIX: индексов не было — очередь, баннер и планировщик делали полный перебор таблиц
CREATE INDEX IF NOT EXISTS ix_routes_protocol ON routes(protocol_id);
CREATE INDEX IF NOT EXISTS ix_routes_patient ON routes(patient);
CREATE INDEX IF NOT EXISTS ix_routes_status ON routes(status);
CREATE INDEX IF NOT EXISTS ix_notif_sched ON notifications(status, ts);
CREATE INDEX IF NOT EXISTS ix_notif_route ON notifications(route_id);
CREATE INDEX IF NOT EXISTS ix_audit_route ON audit(route_id);
"""


def _connect():
    # новое соединение (строки как словари, ожидание блокировки до 10 с вместо мгновенной ошибки)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)   # папка базы
    c = sqlite3.connect(DB_PATH, timeout=10)       # timeout — фоновые потоки и веб не падают с "database is locked"
    c.row_factory = sqlite3.Row                    # строки как словари
    return c


@contextmanager
def tx():
    # одна транзакция на всю обработку события. Ошибка в середине -> откат всего,
    if getattr(_local, "conn", None) is not None:  # уже внутри транзакции — просто участвуем в ней
        yield _local.conn
        return
    with _lock:                                    # сериализуем запись внутри процесса
        c = _connect()
        _local.conn = c                            # все store.q/x в этом потоке пойдут через это соединение
        try:
            c.execute("BEGIN IMMEDIATE")           # сразу берём блокировку записи
            yield c
            c.commit()                             # всё прошло — фиксируем
        except BaseException:
            c.rollback()                           # любая ошибка — откат целиком
            raise
        finally:
            _local.conn = None
            c.close()                              # соединение закрываем явно


def _run(sql, args, fetch):
    c = getattr(_local, "conn", None)              # внутри транзакции?
    if c is not None:
        cur = c.execute(sql, args)
        return [dict(r) for r in cur.fetchall()] if fetch else cur.lastrowid
    with _lock if not fetch else _nolock():        # одиночная запись — под блокировкой
        c = _connect()
        try:
            cur = c.execute(sql, args)
            out = [dict(r) for r in cur.fetchall()] if fetch else cur.lastrowid
            c.commit()
            return out
        finally:
            c.close()


@contextmanager
def _nolock():
    yield


def init():
    # создать таблицы и индексы, если их ещё нет
    with _lock:
        c = _connect()
        try:
            c.executescript(SCHEMA)
            for col in ("acked_by TEXT", "acked_at TEXT", "last_escalation TEXT"):   # миграция старых баз
                try:
                    c.execute(f"ALTER TABLE routes ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass                           # колонка уже есть
            try:
                c.execute("ALTER TABLE notifications ADD COLUMN batch TEXT")
            except sqlite3.OperationalError:
                pass
            c.commit()
        finally:
            c.close()


def reset():
    # полностью очистить базу (ТОЛЬКО демо — эндпоинт закрыт в бою)
    with _lock:
        for suffix in ("", "-wal", "-shm"):        # WAL-файлы тоже
            if os.path.exists(DB_PATH + suffix):
                os.remove(DB_PATH + suffix)
        init()


def q(sql, args=()):
    # SELECT -> список словарей
    return _run(sql, args, True)


def x(sql, args=()):
    # INSERT/UPDATE/DELETE -> ID вставленной строки
    return _run(sql, args, False)


def claim_event(event_id: str) -> bool:
    # атомарно «занять» event_id: один INSERT OR IGNORE, без гонки двух одновременных запросов
    c = getattr(_local, "conn", None)
    assert c is not None, "claim_event вызывается только внутри store.tx()"
    cur = c.execute("INSERT OR IGNORE INTO events VALUES (?,?)", (event_id, ts()))
    return cur.rowcount == 1


# время
def now() -> datetime:
    # текущее время. В бою — реальные часы; в демо — модельные (сдвигаются кнопками)
    if not MODEL_CLOCK:
        return datetime.now().replace(microsecond=0)
    r = q("SELECT value FROM settings WHERE key='clock'")
    if r:
        return datetime.fromisoformat(r[0]["value"])
    t = datetime.now().replace(microsecond=0)
    # INSERT OR IGNORE: одновременные первые запросы не падают с IntegrityError
    x("INSERT OR IGNORE INTO settings VALUES ('clock', ?)", (t.isoformat(),))
    return datetime.fromisoformat(q("SELECT value FROM settings WHERE key='clock'")[0]["value"])


def advance(hours: float) -> datetime:
    # сдвинуть модельное время (только демо)
    if not MODEL_CLOCK:
        raise RuntimeError("модельное время доступно только при CM_ENV=demo")
    t = now() + timedelta(hours=hours)
    x("UPDATE settings SET value=? WHERE key='clock'", (t.isoformat(),))
    return t


def ts() -> str:
    # время строкой (ISO) для записи в таблицы
    return now().isoformat()


# журнал
def audit(actor, action, route_id=None, protocol_id=None, **details):
    # записать действие в журнал
    x("INSERT INTO audit (ts, actor, action, route_id, protocol_id, details) VALUES (?,?,?,?,?,?)",
      (ts(), actor, action, route_id, protocol_id, json.dumps(details, ensure_ascii=False)))


# удобные выборки
def route(rid: int) -> dict:
    # один маршрут (JSON-поля разобраны) или None
    r = q("SELECT * FROM routes WHERE id=?", (rid,))
    if not r:
        return None
    r = r[0]
    r["findings"] = json.loads(r["findings"] or "[]")
    return r


_ROUTE_COLS = {"status", "findings", "verification", "priority", "urgent", "due_date", "notified_at", "ladder_start",
               "ladder_step", "booked_slot", "acked_by", "acked_at", "last_escalation"}


def update_route(rid: int, **fields):
    # обновить поля маршрута. белый список колонок — имена полей подставляются в SQL строкой
    bad = set(fields) - _ROUTE_COLS
    if bad:
        raise ValueError(f"недопустимые поля маршрута: {bad}")
    fields["updated_at"] = ts()
    cols = ", ".join(f"{k}=?" for k in fields)
    x(f"UPDATE routes SET {cols} WHERE id=?", (*fields.values(), rid))


# исключения от администратора
def user_exclusions(active_only: bool = True) -> list:
    # список исключений, добавленных через интерфейс
    return q("SELECT * FROM user_exclusions" + (" WHERE active=1" if active_only else "") + " ORDER BY id")


def add_user_exclusion(finding_id: str, phrase: str, scope: str, added_by: str = "admin") -> int:
    # добавить исключение. Возвращает ID
    return x("INSERT INTO user_exclusions (finding_id, phrase, scope, added_by, ts, active) VALUES (?,?,?,?,?,1)",
             (finding_id, phrase, scope, added_by, ts()))


def disable_user_exclusion(ex_id: int):
    # выключить исключение (не удаляем — остаётся след в журнале)
    x("UPDATE user_exclusions SET active=0 WHERE id=?", (ex_id,))


# настройки, которые админ меняет из интерфейса
def get_setting(key: str, default=None):
    # прочитать настройку (JSON) из таблицы settings
    r = q("SELECT value FROM settings WHERE key=?", (key,))
    if not r:
        return default
    try:
        return json.loads(r[0]["value"])
    except ValueError:
        return default


def set_setting(key: str, value):
    # сохранить настройку (JSON)
    x("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
      (key, json.dumps(value, ensure_ascii=False)))


# ручные добавления в словарь
def user_dictionary(active_only: bool = True) -> list:
    # список ручных добавлений в словарь (extra — JSON с полями новой находки)
    rows = q("SELECT * FROM user_dictionary" + (" WHERE active=1" if active_only else "") + " ORDER BY id")
    for r in rows:
        r["extra"] = json.loads(r["extra"]) if r.get("extra") else {}
    return rows


def add_user_dictionary(kind: str, target: str, phrase: str, extra: dict = None, added_by: str = "admin") -> int:
    # добавить запись в словарь. Возвращает ID
    return x("INSERT INTO user_dictionary (kind, target, phrase, extra, added_by, ts, active) VALUES (?,?,?,?,?,?,1)",
             (kind, target, phrase, json.dumps(extra or {}, ensure_ascii=False), added_by, ts()))


def disable_user_dictionary(item_id: int):
    # выключить запись словаря (не удаляем — остаётся след)
    x("UPDATE user_dictionary SET active=0 WHERE id=?", (item_id,))
