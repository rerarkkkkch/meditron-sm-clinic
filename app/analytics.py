# аналитика: воронка «находка -> уведомление -> запись -> приём» и экономический эффект
import json                                            # разбор JSON-полей
from collections import Counter                        # подсчёт находок по типам
from app import config, store                          # настройки и база
from app.routing import FINAL, STATUS                  # статусы


def _pct(a, b):
    # процент a от b (0, если b = 0)
    return round(100 * a / b) if b else 0              # защита от деления на ноль


def dashboard() -> dict:
    # воронка по данным прототипа (как в ТЗ: УЗИ с триггерами -> уведомлены -> записались -> приём)
    protos = store.q("SELECT extraction FROM protocols WHERE status='active'")   # активные протоколы
    routes = store.q("SELECT * FROM routes")           # все маршруты
    with_find = sum(1 for p in protos if json.loads(p["extraction"])["findings"])   # протоколы с находками
    normal = [r for r in routes if not r["urgent"]]    # плановые маршруты
    notified = [r for r in normal if r["notified_at"]] # уведомлённые
    booked = [r for r in normal if r["booked_slot"] or r["status"] in ("booked", "visited")]   # записавшиеся
    visited = [r for r in normal if r["status"] == "visited"]   # дошедшие до врача
    now = store.now().date().isoformat()               # модельная дата
    overdue = [r for r in normal if r["status"] not in FINAL and r["due_date"] < now]   # просрочен целевой срок
    by_finding = Counter(f["name"] for r in routes for f in json.loads(r["findings"]))   # находки по типам
    funnel = [                                         # шаги воронки: (название, значение, % от предыдущего)
        ("Протоколов обработано", len(protos), None),
        ("С находками", with_find, _pct(with_find, len(protos))),
        ("Плановых маршрутов создано", len(normal), None),
        ("Подтверждено и пациент уведомлён", len(notified), _pct(len(notified), len(normal))),
        ("Записались к специалисту", len(booked), _pct(len(booked), len(notified))),
        ("Дошли до врача", len(visited), _pct(len(visited), len(booked))),
    ]
    return {
        "funnel": [{"step": s, "value": v, "pct": p} for s, v, p in funnel],   # воронка
        "by_status": {STATUS[k]: sum(1 for r in routes if r["status"] == k) for k in STATUS},   # маршруты по статусам
        "by_finding": dict(by_finding),                 # находки по типам
        "overdue": len(overdue),                        # просроченные
        "escalated": sum(1 for r in routes if r["urgent"]),   # экстренные
        "rejected_false_positive": sum(1 for r in routes if r["status"] == "rejected"),   # отклонённые как ложные
        "economics": economics(),                       # экономика
    }


def economics() -> dict:
    # экономический эффект на N исследований. Считаем ТОЛЬКО прирост от доходимости, клинические доли не трогаем
    e = config.economics()                              # допущения
    triggers = e["studies"] * e["trigger_rate"]         # исследований с находкой (одинаково «до» и «после»)
    delta = triggers * (e["reach_specialist_with"] - e["reach_specialist_without"])   # дополнительно дошедших до врача
    extra_surg = delta * e["surgery_indicated_share"] * e["surgery_accept_share"]   # дополнительные операции
    revenue = delta * e["consult_price"] + extra_surg * e["surgery_price"]   # дополнительная выручка
    margin = revenue * e["margin"]                      # дополнительная маржа
    saved = triggers * e["admin_minutes_saved_per_case"] / 60 * e["admin_cost_per_hour"]   # экономия времени координаторов
    return {
        "assumptions": e,                               # допущения показываем рядом с результатом
        "triggers": round(triggers),                    # находок
        "extra_patients_reached": round(delta, 1),      # +пациентов дошло
        "extra_surgeries": round(extra_surg, 1),        # +операций
        "extra_revenue_rub": round(revenue),            # +выручка
        "extra_margin_rub": round(margin),              # +маржа
        "admin_savings_rub": round(saved),              # экономия на ручном контроле
        "total_effect_rub": round(margin + saved),      # итоговый эффект
    }
