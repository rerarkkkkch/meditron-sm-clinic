# маршрутизация: детерминированные правила по матрице routes.yaml. Здесь нет «ИИ» — только логика
import json
from datetime import date, datetime, timedelta
from app import config, notify, store
from app.nlp import anonymize
from app.nlp.extractor import extract
from app.nlp.ml_layer import get_ml_hook

STATUS = {
    "pending_verification": "Ожидает верификации админом",
    "escalated": "ЭКСТРЕННО: передано персоналу",
    "notified": "Пациент уведомлён, ждём записи",
    "booked": "Записан к врачу",
    "rebook_needed": "Отменил запись — нужна повторная",
    "no_show": "Неявка — сценарий возврата",
    "visited": "Пациент дошёл до врача ✔",
    "external": "Обратился в другую клинику",
    "declined": "Пациент отказался",
    "rejected": "Отклонено админом (ложное срабатывание)",
    "not_engaged": "Маршрут не реализован / пациент не вовлечён",
    "closed": "Закрыт (протокол исправлен/аннулирован)",
}
LADDER = {"notified", "rebook_needed", "no_show"}
FINAL = {"visited", "external", "declined", "rejected", "not_engaged", "closed"}

# машина состояний: какое действие из каких статусов допустимо
# («приём состоялся» на неподтверждённом маршруте или «записать» после отклонения — ошибка)
ALLOWED = {
    "verify":   {"pending_verification"},
    "reject":   {"pending_verification"},
    "ack":      {"escalated"},
    "book":     {"notified", "rebook_needed", "no_show", "escalated"},
    "cancel":   {"booked"},
    "no_show":  {"booked"},
    "visit":    {"booked", "notified", "rebook_needed", "no_show", "escalated"},   # МИС может прислать факт приёма без записи через нас
    "external": {"notified", "rebook_needed", "no_show", "booked", "not_engaged"},
    "decline":  {"notified", "rebook_needed", "no_show", "booked", "not_engaged"},
}
APPT = {"booked": "book", "cancelled": "cancel", "completed": "visit", "no_show": "no_show"}

# автоотправка пациенту БЕЗ проверки администратором. Включается в интерфейсе (вкладка «Настройки»)
AUTO_MODES = {
    "off": "выключена — все маршруты проверяет администратор",
    "all": "все плановые маршруты отправляются пациенту сразу",
    "clinician_agrees": "только если врач УЗИ сам рекомендовал этого же специалиста",
    "selected": "только для выбранных маршрутов (специалистов)",
    "rules": "по своим правилам администратора (сценарии без проверки)",
}

# правило автоотправки: все условия должны выполниться; пустой список = «любое значение»
RULE_FIELDS = {"name", "active", "routes", "findings", "priorities", "min_confidence", "require_clinician", "study_contains"}


def _plain(fs) -> list:
    # находки маршрута в одном виде: объекты Finding из извлечения или словари из базы
    out = []
    for f in fs:
        d = f if isinstance(f, dict) else _finding_brief(f)
        out.append({"id": d["id"], "confidence": d.get("confidence", 0), "needs_review": d.get("needs_review", False),
                    "ml": any(str(e.get("rule", "")).startswith("ml:") for e in d.get("evidence", []))})
    return out


def rule_matches(rule: dict, key: str, fs, recommended, study: str = "", priority: int = 3) -> bool:
    # подходит ли маршрут под правило администратора
    if not rule.get("active", True):
        return False
    fl = _plain(fs)
    if rule.get("routes") and key not in rule["routes"]:
        return False
    if rule.get("findings") and any(f["id"] not in rule["findings"] for f in fl):
        return False
    if rule.get("priorities") and priority not in rule["priorities"]:
        return False
    if any(f["confidence"] < float(rule.get("min_confidence") or 0) for f in fl):
        return False
    if rule.get("require_clinician") and key not in (recommended or []):
        return False
    if rule.get("study_contains") and rule["study_contains"].lower() not in (study or "").lower():
        return False
    return True


def auto_send_settings() -> dict:
    # текущая настройка автоотправки: из интерфейса (БД), а если её не меняли — из routes.yaml (behavior)
    beh = config.routes().get("behavior", {})
    if not beh.get("require_admin_verification", True):
        mode = "all"
    elif beh.get("auto_verify_if_clinician_agrees"):
        mode = "clinician_agrees"
    else:
        mode = "off"
    cur = {"mode": mode, "routes": [], "rules": [], "skip_needs_review": True, "changed_by": None, "changed_at": None}
    cur.update(store.get_setting("auto_send", {}) or {})
    return cur


def auto_send_decision(key: str, fs, recommended, study: str = "", priority: int = 3, settings: dict = None) -> tuple:
    # можно ли отправить пациенту без админа -> (решение, причина или название правила); экстренные сюда не попадают
    a = settings or auto_send_settings()
    mode = a.get("mode", "off")
    rule = None
    if mode == "rules":
        rule = next((r for r in a.get("rules") or [] if rule_matches(r, key, fs, recommended, study, priority)), None)
    allowed = mode == "all" or (mode == "clinician_agrees" and key in recommended) or \
              (mode == "selected" and key in (a.get("routes") or [])) or rule is not None
    if not allowed:
        return False, None
    fl = _plain(fs)
    if any(f["ml"] for f in fl):
        return False, "находку предложил второй контур — такие всегда проверяет человек"
    if a.get("skip_needs_review", True) and any(f["needs_review"] for f in fl):
        return False, "у находок есть пометки «проверить» (опечатка, нет размера, неуверенная формулировка)"
    return True, (f"правило «{rule.get('name') or 'без названия'}»" if rule else f"настройка «{AUTO_MODES[mode]}»")


class BadEvent(ValueError):
    # некорректное событие — API вернёт 400, а отметка event_id откатится
    pass


def _route_cfg(key: str) -> dict:
    return config.routes()["routes"].get(key, {})


PATIENT_TEMPLATES = ("first_notice", "reminder_24h", "reminder_72h", "final_soft", "no_show", "booked")   # что видит пациент


def scenario() -> dict:
    # сценарий уведомлений = routes.yaml (templates, reminders) + правки администратора из «Настроек»
    rc = config.routes()
    over = store.get_setting("patient_scenario", {}) or {}
    return {"templates": {**rc.get("templates", {}), **(over.get("templates") or {})},
            "reminders": over.get("reminders") or rc.get("reminders", []),
            "changed_at": over.get("changed_at"), "changed_by": over.get("changed_by")}


def _tpl(name: str, **kw) -> str:
    # текст сообщения по шаблону (с учётом правок администратора). Неизвестные поля подставляются пустыми
    class _Safe(dict):
        def __missing__(self, k):
            return ""
    return scenario()["templates"][name].format_map(_Safe(kw))


def _req(ev: dict, *fields):
    # проверка обязательных полей: пустое значение — ошибка 400
    for f in fields:
        if not ev.get(f) or (isinstance(ev[f], str) and not ev[f].strip()):
            raise BadEvent(f"не хватает поля: {f}")


# 1. обработка событий от МИС
def handle_event(ev: dict) -> dict:
    # единая точка входа. проверка, отметка event_id и вся обработка — в ОДНОЙ транзакции
    t = ev.get("type")
    if t in ("protocol.signed", "protocol.corrected"):
        _req(ev, "protocol_id", "patient_id", "text")
    elif t == "protocol.cancelled":
        _req(ev, "protocol_id")
    elif t and t.startswith("appointment."):
        _req(ev, "patient_id")
        if t.split(".", 1)[1] not in APPT:
            raise BadEvent(f"неизвестное событие записи: {t}")
    elif t == "patient.visit":
        _req(ev, "patient_id")
    else:
        raise BadEvent(f"неизвестный тип события: {t}")
    if t in ("protocol.signed", "protocol.corrected", "appointment.booked", "appointment.cancelled",
             "appointment.completed", "appointment.no_show", "protocol.cancelled"):
        _req(ev, "event_id")                       # без event_id нет идемпотентности — для изменяющих событий обязателен

    with store.tx():
        if ev.get("event_id") and not store.claim_event(ev["event_id"]):
            return {"status": "duplicate", "event_id": ev["event_id"]}
        if t in ("protocol.signed", "protocol.corrected"):
            return process_protocol(ev)
        if t == "protocol.cancelled":
            return cancel_protocol(ev["protocol_id"])
        if t.startswith("appointment."):
            return appointment_event(ev)
        return {"open_routes": open_routes(anonymize.pseudo_id(ev["patient_id"]))}


def _ids(findings) -> set:
    return {f["id"] for f in findings}


def process_protocol(ev: dict) -> dict:
    # протокол подписан/исправлен: извлечь находки, создать/обновить маршруты без дублей
    pid = ev["protocol_id"]
    patient = anonymize.pseudo_id(ev["patient_id"])
    text = anonymize.scrub(ev["text"])
    exists = store.q("SELECT version, study, study_date, patient, status FROM protocols WHERE id=?", (pid,))
    if exists and exists[0]["patient"] != patient:  # исправление пришло с другим пациентом — не смешиваем молча
        raise BadEvent("protocol_id уже принадлежит другому пациенту — нужна ручная проверка")
    study = ev.get("study_type") or (exists[0]["study"] if exists else "Исследование")
    study_date = ev.get("study_date") or (exists[0]["study_date"] if exists else store.now().date().isoformat())
    try:
        date.fromisoformat(study_date[:10])         # неверная дата — ошибка 400 до создания маршрута
    except ValueError:
        raise BadEvent(f"study_date не в формате ГГГГ-ММ-ДД: {study_date}")
    res = extract(text, get_ml_hook())
    version = (exists[0]["version"] + 1) if exists else 1
    store.x("INSERT OR REPLACE INTO protocols VALUES (?,?,?,?,?,?,?,?,COALESCE((SELECT created_at FROM protocols WHERE id=?),?),?)",
            (pid, patient, study, study_date, text, version, "active", json.dumps(res.to_dict(), ensure_ascii=False),
             pid, store.ts(), store.ts()))
    store.audit("mis", ev["type"], protocol_id=pid, version=version, summary=res.summary,
                dictionary=res.dictionary_version)

    groups = {}
    for f in res.findings:
        groups.setdefault(f.route, []).append(f)
    current = {r["route_key"]: r for r in store.q("SELECT * FROM routes WHERE protocol_id=? AND status!='closed' ORDER BY id", (pid,))}
    created, updated, closed, reopened = [], [], [], []

    for key, fs in groups.items():
        briefs = [_finding_brief(f) for f in fs]
        fjson = json.dumps(briefs, ensure_ascii=False)
        prio = min(f.priority for f in fs)
        urgent = any(f.urgent for f in fs) or bool(_route_cfg(key).get("escalation_only"))
        if key not in current:
            created.append(_create_route(pid, patient, study, study_date, key, fs, fjson, prio, urgent, res.clinician_recommended))
            continue
        r = current[key]
        old = json.loads(r["findings"] or "[]")
        changed = _ids(old) != _ids(briefs) or _attrs(old) != _attrs(briefs)
        store.update_route(r["id"], findings=fjson, priority=prio)
        store.audit("system", "route_updated", r["id"], pid, reason="исправление протокола", changed=changed)
        updated.append(r["id"])
        if urgent and not r["urgent"] and r["status"] not in FINAL:
            # экстренность, появившаяся при исправлении протокола, — сразу эскалация
            store.update_route(r["id"], urgent=1, status="escalated")
            notify.cancel_for_route(r["id"])
            notify.to_admin(_tpl("admin_urgent", patient=anonymize.short(patient), study=study), r["id"], "staff_escalation")
        elif changed and r["status"] == "rejected":
            # отклонённый маршрут молча «поглощал» новые находки исправленного протокола
            store.update_route(r["id"], status="pending_verification")
            notify.to_admin(f"Протокол {pid} исправлен: по ранее отклонённому маршруту изменились находки — проверьте снова.", r["id"])
            reopened.append(r["id"])
        elif changed and r["status"] != "pending_verification":
            notify.to_admin(f"Протокол {pid} исправлен: изменились находки маршрута ({STATUS[r['status']]}) — проверьте.", r["id"])

    for key, r in current.items():
        if key not in groups and r["status"] not in FINAL:
            store.update_route(r["id"], status="closed")
            notify.cancel_for_route(r["id"])
            store.audit("system", "route_closed", r["id"], pid, reason="после исправления находка не описана")
            if r["status"] in LADDER | {"booked"}:   # пациент уже получил сообщение — сообщаем админу, чтобы снять запись/объяснить
                notify.to_admin(f"Протокол {pid} исправлен: маршрут закрыт, пациент уже был уведомлён — свяжитесь с ним.", r["id"])
            closed.append(r["id"])

    missed = [k for k in res.clinician_recommended if k not in groups]
    if missed:
        store.audit("system", "clinician_recommends_but_no_trigger", protocol_id=pid, routes=missed)
        notify.to_admin(f"Проверьте протокол {pid}: врач рекомендовал {', '.join(_route_cfg(k).get('specialist', k) for k in missed)}, "
                        f"а система находок не нашла.", None)
    if res.suppressed_in_recommendation and not res.findings:
        # находка описана в предложении-рекомендации («…полип 12 мм, рекомендовано наблюдение»)
        notify.to_admin(f"Проверьте протокол {pid}: находка упомянута только в рекомендации врача.", None)
    return {"protocol_id": pid, "patient": patient, "version": version, "summary": res.summary,
            "created": created, "updated": updated, "closed": closed, "reopened": reopened, "extraction": res.to_dict()}


def _attrs(findings) -> list:
    return sorted(json.dumps([e.get("attributes") for e in f.get("evidence", [])], sort_keys=True) for f in findings)


def _finding_brief(f) -> dict:
    return {"id": f.id, "name": f.name, "confidence": f.confidence, "needs_review": f.needs_review,
            "evidence": [e.__dict__ for e in f.evidence]}


def _create_route(pid, patient, study, study_date, key, fs, fjson, prio, urgent, recommended) -> int:
    rc = _route_cfg(key)
    if key in recommended:
        verification = "совпадает с рекомендацией врача"
    elif recommended:
        verification = "врач рекомендовал другого специалиста — проверить"
    else:
        verification = "рекомендации врача нет — найдено системой"
    due = (date.fromisoformat(study_date[:10]) + timedelta(days=rc.get("target_days", 7))).isoformat()
    status = "escalated" if urgent else "pending_verification"
    rid = store.x("INSERT INTO routes (protocol_id, patient, route_key, status, findings, verification, priority, urgent, due_date, created_at, updated_at, last_escalation) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                  (pid, patient, key, status, fjson, verification, prio, int(urgent), due, store.ts(), store.ts(),
                   store.ts() if urgent else None))
    store.audit("system", "route_created", rid, pid, route=key, matrix=config.routes().get("version"),
                findings=[f.id for f in fs], verification=verification)
    if urgent:
        notify.to_admin(_tpl("admin_urgent", patient=anonymize.short(patient), study=study), rid, "staff_escalation")
        return rid
    beh = config.routes()["behavior"]
    if beh.get("admin_push_on_new_route", True):
        notify.to_admin(_tpl("admin_new", patient=anonymize.short(patient), study=study,
                             route_specialist=rc.get("specialist", key), priority=prio), rid)
    ok, why = auto_send_decision(key, fs, recommended, study, prio)
    if ok:
        transition(rid, "verify", actor="system:auto", comment=f"автоотправка без проверки ({why})")
        notify.to_admin(f"Маршрут {rid}: пациенту отправлено автоматически, без проверки ({why}).", rid)
    elif why:
        store.audit("system", "auto_send_skipped", rid, pid, reason=why)
        notify.to_admin(f"Маршрут {rid}: автоотправка не выполнена — {why}. Нужна ваша проверка.", rid)
    return rid


def cancel_protocol(pid: str) -> dict:
    store.x("UPDATE protocols SET status='cancelled', updated_at=? WHERE id=?", (store.ts(), pid))
    rows = store.q("SELECT id, status FROM routes WHERE protocol_id=?", (pid,))
    ids = [r["id"] for r in rows if r["status"] not in FINAL]
    for r in rows:
        if r["status"] in FINAL:
            continue
        store.update_route(r["id"], status="closed")
        notify.cancel_for_route(r["id"])
        store.audit("mis", "route_closed", r["id"], pid, reason="протокол аннулирован")
        if r["status"] in LADDER | {"booked"}:
            notify.to_admin(f"Протокол {pid} аннулирован, а пациент уже уведомлён — свяжитесь с ним.", r["id"])
    return {"protocol_id": pid, "closed": ids}


def appointment_event(ev: dict) -> dict:
    patient = anonymize.pseudo_id(ev["patient_id"])
    rid = ev.get("route_id")
    if rid:
        r = store.route(rid)
        if not r or r["patient"] != patient:        # route_id должен принадлежать этому пациенту
            raise BadEvent("route_id не найден у этого пациента")
    else:
        sql = "SELECT id FROM routes WHERE patient=? AND status NOT IN (%s)" % ",".join("?" * len(FINAL))
        args = [patient, *FINAL]
        if ev.get("specialty"):
            sql += " AND route_key=?"
            args.append(ev["specialty"])
        rows = store.q(sql + " ORDER BY id DESC", tuple(args))
        if len(rows) != 1:
            # без specialty или при нескольких открытых маршрутах событие не привязываем — задача админу
            store.audit("mis", "appointment_unmatched", details=ev.get("type"), candidates=[r["id"] for r in rows])
            if rows:
                notify.to_admin(f"Событие {ev['type']} не удалось однозначно привязать к маршруту ({len(rows)} кандидата) — привяжите вручную.", None)
            return {"status": "no_route" if not rows else "ambiguous", "candidates": [r["id"] for r in rows]}
        rid = rows[0]["id"]
    try:
        return {"route": transition(rid, APPT[ev["type"].split(".", 1)[1]], actor="mis", slot=ev.get("slot"))}
    except ValueError as e:
        store.audit("mis", "appointment_rejected", rid, reason=str(e))
        return {"status": "ignored", "reason": str(e)}


# 2. действия по маршруту (админ / пациент / МИС)
def transition(rid: int, action: str, actor: str = "admin", slot: str = None, comment: str = "") -> dict:
    # все переходы статусов в одном месте, с проверкой допустимости. атомарно (одна транзакция)
    with store.tx():
        return _transition(rid, action, actor, slot, comment)


def _transition(rid, action, actor, slot, comment):
    r = store.route(rid)
    if r is None:
        raise ValueError("маршрут не найден")
    if action not in ALLOWED:
        raise ValueError(f"неизвестное действие {action}")
    mis_fact = actor == "mis" and action in ("book", "visit") and r["status"] == "pending_verification"   # МИС сообщает факт: пациент записался/пришёл сам
    if r["status"] not in ALLOWED[action] and not mis_fact:
        raise ValueError(f"действие «{action}» недопустимо в статусе «{STATUS.get(r['status'], r['status'])}»")
    rc = _route_cfg(r["route_key"])
    beh = config.routes()["behavior"]
    if action == "verify":
        store.update_route(rid, status="notified", notified_at=store.ts(), ladder_start=store.ts(), ladder_step=0)
        notify.to_patient(r["patient"], _tpl("first_notice", specialist=rc.get("specialist_for_patient", "профильного специалиста")), rid)
    elif action == "reject":
        store.update_route(rid, status="rejected")
        for f in r["findings"]:
            store.x("INSERT INTO feedback (ts, protocol_id, finding_id, kind, quote, comment) VALUES (?,?,?,?,?,?)",
                    (store.ts(), r["protocol_id"], f["id"], "false_positive", (f["evidence"] or [{}])[0].get("quote", ""), comment))
    elif action == "ack":
        store.update_route(rid, acked_by=actor, acked_at=store.ts())   # статус остаётся escalated; дальше — запись/приём
    elif action == "book":
        store.update_route(rid, status="booked", booked_slot=slot)
        notify.cancel_for_route(rid)                # отложенные «запишитесь» после записи не нужны
        notify.to_patient(r["patient"], _tpl("booked", slot=slot or "время уточняется"), rid, channels=("lk",))
    elif action == "cancel":
        store.update_route(rid, status="rebook_needed", ladder_start=store.ts(), ladder_step=0)
    elif action == "no_show":
        store.update_route(rid, status="no_show", ladder_start=store.ts(), ladder_step=0)
        notify.to_patient(r["patient"], _tpl("no_show"), rid, delay_hours=beh.get("no_show_message_delay_hours", 1))
    elif action in ("visit", "external", "decline"):
        store.update_route(rid, status={"visit": "visited", "external": "external", "decline": "declined"}[action])
        notify.cancel_for_route(rid)                # отложенные сообщения после приёма или отказа не нужны
        if action == "decline" and r["priority"] == 1:
            # отказ при высоком приоритете не должен закрываться молча
            notify.to_admin(_tpl("call_script", patient=anonymize.short(r["patient"]), specialist=rc.get("specialist_for_patient", "")) +
                            " (пациент отказался — зафиксируйте информированный отказ)", rid, "admin_task")
    store.audit(actor, action, rid, r["protocol_id"], slot=slot, comment=comment)
    return store.route(rid)


def verify(rid: int, actor: str = "admin") -> dict:
    # совместимость со старым кодом
    return transition(rid, "verify", actor)


def open_routes(patient: str) -> list:
    rows = store.q("SELECT r.*, p.study, p.study_date FROM routes r JOIN protocols p ON p.id=r.protocol_id WHERE r.patient=?", (patient,))
    out = []
    for r in rows:
        if r["status"] in FINAL - {"not_engaged"}:
            continue
        names = ", ".join(f["name"] for f in json.loads(r["findings"]))
        out.append(f"Незавершённый клинический маршрут: {r['study_date']} {r['study']} — {names}. "
                   f"Консультация «{_route_cfg(r['route_key']).get('specialist')}» не зафиксирована.")
    return out


# планировщик: в бою вызывается фоновым потоком по реальным часам (main.py), в демо — кнопками времени
def tick() -> dict:
    with store.tx():
        sent = notify.flush_scheduled(FINAL)
        done = _ladder() + _sla()
    return {"flushed": sent, "steps": done}


def _ladder() -> list:
    done = []
    steps = scenario()["reminders"]                 # лестница может быть изменена администратором
    now = store.now()
    marks = ",".join("?" * len(LADDER))
    for r in store.q(f"SELECT * FROM routes WHERE status IN ({marks}) AND ladder_start IS NOT NULL", tuple(LADDER)):
        hours = (now - datetime.fromisoformat(r["ladder_start"])).total_seconds() / 3600
        step = r["ladder_step"] or 0
        due = [i for i in range(step, len(steps)) if hours >= steps[i]["after_hours"]]
        if not due:
            continue                                # нечего делать — маршрут не трогаем
        last = due[-1]
        # после простоя созревших шагов может быть несколько: выполняем только последний,
        # чтобы пациент не получил несколько сообщений сразу, но задачу админу (звонок) не пропускаем
        to_run = [i for i in due[:-1] if steps[i]["action"] == "admin_task"] + [last]
        for i in due:
            if i not in to_run:
                store.audit("scheduler", f"reminder_{steps[i]['after_hours']}h_skipped", r["id"], r["protocol_id"])
        rc = _route_cfg(r["route_key"])
        for i in to_run:
            s = steps[i]
            if s["action"] == "patient_message":
                notify.to_patient(r["patient"], _tpl(s["template"], specialist=rc.get("specialist_for_patient", "")), r["id"],
                                  channels=tuple(s.get("channels", ["lk", "push"])))
            elif s["action"] == "admin_task":
                notify.to_admin(_tpl(s["template"], patient=anonymize.short(r["patient"]), specialist=rc.get("specialist_for_patient", "")),
                                r["id"], "admin_task")
            elif s["action"] == "set_status":
                store.update_route(r["id"], status=s["status"])
                notify.cancel_for_route(r["id"])
                if r["priority"] == 1:              # высокий приоритет не уходит в «не вовлечён» без человека
                    notify.to_admin(f"Маршрут {r['id']} (приоритет 1) переведён в «не вовлечён» — нужно решение врача.", r["id"], "admin_task")
            store.audit("scheduler", f"reminder_{s['after_hours']}h", r["id"], r["protocol_id"], step=s["action"])
            done.append((r["id"], s["after_hours"]))
        store.update_route(r["id"], ladder_step=last + 1)
    return done


def _sla() -> list:
    # срок на верификацию и подтверждение экстренных: просроченные снова уходят персоналу
    beh = config.routes()["behavior"]
    now = store.now()
    done = []
    v_hours = beh.get("verification_sla_hours", 24)
    for r in store.q("SELECT * FROM routes WHERE status='pending_verification'"):
        base = datetime.fromisoformat(r["last_escalation"] or r["created_at"])
        if (now - base).total_seconds() / 3600 >= v_hours:
            notify.to_admin(f"Маршрут {r['id']} ждёт верификации дольше {v_hours} ч.", r["id"], "staff_escalation")
            store.update_route(r["id"], last_escalation=now.isoformat())
            done.append((r["id"], "verification_overdue"))
    u_min = beh.get("urgent_ack_minutes", 15)
    for r in store.q("SELECT * FROM routes WHERE status='escalated' AND acked_at IS NULL"):
        base = datetime.fromisoformat(r["last_escalation"] or r["created_at"])
        if (now - base).total_seconds() / 60 >= u_min:
            notify.to_admin(f"ПОВТОРНО ЭКСТРЕННО: маршрут {r['id']} не принят в работу {u_min}+ мин.", r["id"], "staff_escalation")
            store.update_route(r["id"], last_escalation=now.isoformat())
            done.append((r["id"], "urgent_reescalated"))
    return done
