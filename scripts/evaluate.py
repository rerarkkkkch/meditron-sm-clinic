# оценка КАЧЕСТВА: сколько находок найдено правильно, сколько лишних, сколько пропущено
import csv                                                     # запись таблиц
import json                                                    # чтение разметки
import os                                                      # пути
import sys                                                     # аргументы командной строки
from collections import defaultdict                            # счётчики

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # чтобы импортировать app/ из корня
from app import config                                         # словарь и маршруты
from app.nlp.extractor import extract                          # то, что оцениваем
from app.nlp.ml_layer import get_ml_hook                       # модель (если включена)


def main():
    argv = sys.argv[1:]                                        # аргументы командной строки
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--split")]   # позиционные (без значения --split)
    path = args[0] if args else "data/synthetic/protocols.jsonl"   # файл разметки
    split = "all"                                              # какую часть оценивать
    if "--split" in sys.argv:                                  # указан --split
        split = sys.argv[sys.argv.index("--split") + 1]        # значение после флага
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]   # читаем разметку
    if split != "all":                                         # фильтр по части
        rows = [r for r in rows if r.get("split", "dev") == split]   # только нужная часть
    fmap = config.findings()["findings"]                       # словарь находок (для маршрутов и названий)
    stat = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})    # счётчики по каждой находке
    errors = []                                                # список ошибок с объяснением
    norm_total = norm_fp = 0                                   # протоколы-нормы и ложные срабатывания на них
    route_ok = 0                                               # протоколы с правильным набором маршрутов
    hook = None if "--no-ml" in sys.argv else get_ml_hook()   # --no-ml: только правила, без второго контура

    for r in rows:                                             # каждый протокол
        res = extract(r["text"], hook)                         # прогоняем систему
        got = {f.id for f in res.findings}                     # что нашла система
        exp = {e["id"] for e in r["expected"]}                 # что должно быть
        for fid in got & exp:                                  # совпало
            stat[fid]["TP"] += 1                               # правильно
        for fid in got - exp:                                  # лишнее
            stat[fid]["FP"] += 1                               # ошибка 1-го рода
            q = next(f.evidence[0].quote for f in res.findings if f.id == fid)   # что именно сработало
            errors.append([r["id"], "FP (лишнее)", fid, q, "система нашла, в разметке нет"])   # в отчёт
        for fid in exp - got:                                  # пропущено
            stat[fid]["FN"] += 1                               # ошибка 2-го рода
            why = next((s.reason for s in res.suppressed if s.id == fid), "нет совпадения со словарём")   # почему не сработало
            q = next(e["quote"] for e in r["expected"] if e["id"] == fid)   # что ожидали
            errors.append([r["id"], "FN (пропуск)", fid, q, why])   # в отчёт
        if not exp:                                            # протокол-норма
            norm_total += 1                                    # считаем нормы
            norm_fp += bool(got)                               # норма с любым срабатыванием = ошибка
        exp_routes = {fmap[e]["route"] for e in exp if e in fmap}   # ожидаемые маршруты
        got_routes = {f.route for f in res.findings}           # маршруты системы
        route_ok += exp_routes == got_routes                   # совпал ли набор маршрутов

    # таблица
    table = []                                                 # строки отчёта
    tot = {"TP": 0, "FP": 0, "FN": 0}                          # итого
    for fid in sorted(set(stat) | set(fmap)):                  # все находки словаря и все встреченные
        s = stat[fid]                                          # счётчики
        for k in tot:                                          # копим итог
            tot[k] += s[k]                                     # суммируем
        p = s["TP"] / (s["TP"] + s["FP"]) if s["TP"] + s["FP"] else None   # точность
        rc = s["TP"] / (s["TP"] + s["FN"]) if s["TP"] + s["FN"] else None  # полнота
        table.append([fid, fmap.get(fid, {}).get("name", fid), s["TP"], s["FP"], s["FN"], p, rc])   # строка
    P = tot["TP"] / (tot["TP"] + tot["FP"]) if tot["TP"] + tot["FP"] else 0   # общая точность
    R = tot["TP"] / (tot["TP"] + tot["FN"]) if tot["TP"] + tot["FN"] else 0   # общая полнота
    F1 = 2 * P * R / (P + R) if P + R else 0                   # F1 — среднее гармоническое

    fmt = lambda v: "—" if v is None else f"{v:.2f}"           # формат чисел
    print(f"\nПротоколов: {len(rows)}  (часть: {split})\n")    # заголовок
    print(f"{'Находка':42} {'TP':>4} {'FP':>4} {'FN':>4} {'Точн.':>6} {'Полн.':>6}")   # шапка таблицы
    for fid, name, tp, fp, fn, p, rc in table:                 # строки
        print(f"{name[:42]:42} {tp:>4} {fp:>4} {fn:>4} {fmt(p):>6} {fmt(rc):>6}")   # печать
    print("-" * 70)                                            # разделитель
    print(f"{'ИТОГО':42} {tot['TP']:>4} {tot['FP']:>4} {tot['FN']:>4} {P:>6.2f} {R:>6.2f}   F1={F1:.2f}")   # итог
    print(f"\nЛожные срабатывания на нормах: {norm_fp} из {norm_total}")   # нормы
    print(f"Правильный набор маршрутов: {route_ok} из {len(rows)}")   # маршруты
    if errors:                                                 # есть ошибки
        print("\nОшибки:")                                     # заголовок
        for e in errors:                                       # каждая
            print(f"  [{e[0]}] {e[1]} {e[2]}: «{e[3]}» — {e[4]}")   # печать

    os.makedirs("reports", exist_ok=True)                      # папка отчётов
    with open("reports/metrics.csv", "w", newline="", encoding="utf-8") as f:   # таблица метрик
        w = csv.writer(f)                                      # писатель CSV
        w.writerow(["finding_id", "name", "TP", "FP", "FN", "precision", "recall"])   # шапка
        w.writerows(table + [["TOTAL", "", tot["TP"], tot["FP"], tot["FN"], P, R]])  # строки + итог
    with open("reports/errors.csv", "w", newline="", encoding="utf-8") as f:    # список ошибок
        w = csv.writer(f)                                      # писатель CSV
        w.writerow(["protocol", "type", "finding", "quote", "reason"])   # шапка
        w.writerows(errors)                                    # строки
    print("\nСохранено: reports/metrics.csv, reports/errors.csv")   # куда сохранили


if __name__ == "__main__":                                     # запуск как скрипта
    main()                                                     # поехали
