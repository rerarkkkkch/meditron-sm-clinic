# заглушка сервиса расписания (у организаторов API нет — описываем контракт и имитируем)
from datetime import timedelta                     # сдвиги по времени
from app import store                              # модельное время

BRANCHES = ["Текстильщики", "Сенежская", "ВДНХ"]   # филиалы (пример из ТЗ)


def _work_hours(t):
    # слоты только с 9:00 до 19:00, иначе пациенту предлагались «06.10 00:00»
    if t.hour < 9:
        return t.replace(hour=9 + t.hour % 3)
    if t.hour >= 19:
        return (t + timedelta(days=1)).replace(hour=10 + t.hour % 3)
    return t


def slots(route_key: str, limit: int = 4) -> list:
    # ближайшие слоты для профиля. Генерируются от модельного «сейчас»
    base = store.now().replace(minute=0, second=0)   # округляем до часа
    plan = [                                       # (через сколько часов, филиал или онлайн)
        (3, None),                                 # сегодня онлайн
        (18, BRANCHES[0]),                         # завтра утром — Текстильщики
        (26, BRANCHES[1]),                         # завтра вечером — Сенежская
        (45, None),                                # послезавтра онлайн
    ]
    out = []                                       # результат
    for i, (h, br) in enumerate(plan[:limit]):     # каждый слот
        t = _work_hours(base + timedelta(hours=h)) # время слота (только в часы работы клиники)
        out.append({                               # описание слота
            "slot_id": f"{route_key}-{t:%Y%m%d%H}-{i}",   # уникальный ID слота
            "start": t.isoformat(),                # время начала
            "branch": br or "Онлайн",              # филиал
            "online": br is None,                  # онлайн-консультация?
            "label": f"{t:%d.%m %H:%M} — {br or 'онлайн'}",   # подпись для кнопки
        })
    return out                                     # список слотов
