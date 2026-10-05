# заглушка уведомлений. Реально ничего не отправляет — только записывает в таблицу notifications
import uuid                                       # ID «отправки» (одно сообщение во все каналы)
from datetime import timedelta                    # расчёт «перенести на завтра»
from app import config, store                     # настройки и база


def _sent_today(recipient: str) -> int:
    # сколько ОТПРАВОК (не каналов) пациенту уже ушло за текущие сутки, включая запланированные на сегодня
    day = store.now().date().isoformat()
    r = store.q("SELECT COUNT(DISTINCT batch) AS n FROM notifications WHERE recipient=? AND kind='patient' "
                "AND status IN ('sent','scheduled') AND substr(ts,1,10)=?", (recipient, day))
    return r[0]["n"]


def to_patient(patient: str, text: str, route_id: int, channels=("lk", "push"), delay_hours: float = 0):
    # сообщение пациенту (заглушка). Учитывает антиспам и отложенную отправку
    beh = config.routes().get("behavior", {})
    when = store.now() + timedelta(hours=delay_hours)
    status = "scheduled" if delay_hours > 0 else "sent"
    limit = beh.get("max_patient_messages_per_day", 2)
    if when.date() == store.now().date() and _sent_today(patient) >= limit:   # отложенные на сегодня тоже под лимитом
        when = (store.now() + timedelta(days=1)).replace(hour=beh.get("quiet_send_hour", 10), minute=0, second=0)
        status = "scheduled"
        # время сервера. В контейнере это UTC — задайте TZ=Europe/Moscow (или храните часовой пояс пациента)
        store.audit("system", "antispam_deferred", route_id, until=when.isoformat())
    batch = uuid.uuid4().hex                      # одна отправка = один batch на все каналы
    for ch in channels:
        store.x("INSERT INTO notifications (ts, channel, recipient, text, route_id, status, kind, batch) VALUES (?,?,?,?,?,?,?,?)",
                (when.isoformat(), ch, patient, text, route_id, status, "patient", batch))
    return status


def to_admin(text: str, route_id, channel: str = "admin_push"):
    # пуш/задача администратору (заглушка). channel: admin_push | admin_task | staff_escalation
    store.x("INSERT INTO notifications (ts, channel, recipient, text, route_id, status, kind, batch) VALUES (?,?,?,?,?,?,?,?)",
            (store.ts(), channel, "admin", text, route_id, "sent", "admin", uuid.uuid4().hex))
    print(f"[PUSH {channel}] route={route_id}", flush=True)   # в лог только ID, без текста (в тексте псевдо-ID и профиль)


def flush_scheduled(final_statuses=()):
    # отправить созревшие отложенные сообщения. сообщения маршрутов, уже завершённых
    due = store.q("SELECT n.id, r.status AS rstatus FROM notifications n LEFT JOIN routes r ON r.id=n.route_id "
                  "WHERE n.status='scheduled' AND n.ts<=?", (store.ts(),))
    sent = 0
    for n in due:
        if n["rstatus"] in final_statuses:
            store.x("UPDATE notifications SET status='cancelled' WHERE id=?", (n["id"],))
        else:
            store.x("UPDATE notifications SET status='sent' WHERE id=?", (n["id"],))
            sent += 1
    return sent


def cancel_for_route(route_id: int):
    # отменить все ещё не отправленные сообщения маршрута
    store.x("UPDATE notifications SET status='cancelled' WHERE route_id=? AND status='scheduled'", (route_id,))
