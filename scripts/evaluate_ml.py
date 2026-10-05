# оценка второго контура честной кросс-валидацией по протоколам
import argparse, csv, json, math, os, sys, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CM_ENV", "demo")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from app import config
from app.nlp import anonymize
from app.nlp.extractor import extract
from app.nlp.ml_layer import SecondLayer, training_rows

GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/real/protocols.jsonl")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--extra-train", default=None, help="доп. разметка ТОЛЬКО для обучения (напр. data/web/protocols.jsonl); в тест не попадает")
    ap.add_argument("--report", default="reports/second_layer_report.md")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    extra = [json.loads(l) for l in open(a.extra_train, encoding="utf-8") if l.strip()] if a.extra_train else []
    fcfg = config.findings()["findings"]
    dict_types = {k for k, v in fcfg.items() if not v.get("ml_only")}
    t0 = time.time()
    cache = {}

    def ex(text):                                   # первый слой без модели (кэш: правила детерминированы)
        if text not in cache:
            cache[text] = extract(text, None)                 # text уже обезличен
        return cache[text]

    for r in rows:
        r["scrubbed"] = anonymize.scrub(r["text"])
        r["res"] = ex(r["scrubbed"])
        r["gold"] = {e["id"] for e in r["expected"]}
        r["rules"] = {f.id for f in r["res"].findings}
    tr = lambda r: {"text": r["text"], "expected": r["expected"]}

    # первый слой
    def counts(pred_of):
        tp = fp = fn = 0
        for r in rows:
            p = pred_of(r); tp += len(p & r["gold"]); fp += len(p - r["gold"]); fn += len(r["gold"] - p)
        return tp, fp, fn
    base = counts(lambda r: r["rules"])

    # кросс-валидация
    per_repeat = []
    oof_ver, oof_sent = [], []                      # для AUC (первый повтор)
    sweep = defaultdict(lambda: [0, 0, 0, 0])       # (miss_thr) -> tp, fp, fn, rules_fn_caught ; на всех повторах
    sweep_fp = defaultdict(lambda: [0, 0, 0])       # (fp_thr) -> flagged_true_fp, flagged_tp, total_rule_fp
    strata = [r["category"] for r in rows]; groups = [r["group"] for r in rows]
    by_type = defaultdict(lambda: [0, 0, 0, 0])     # тип -> [в разметке, нашли правила, нашли правила+контур, лишние подсказки]
    for rep in range(a.repeats):
        skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=rep)
        dec = {}
        for tr_idx, te_idx in skf.split(rows, strata, groups):
            m = SecondLayer().fit([tr(rows[i]) for i in tr_idx] + [tr(x) for x in extra], ex)
            for i in te_idx:
                r = rows[i]
                dec[i] = {}
                for thr in GRID:
                    m.miss_threshold = m.fp_threshold = thr
                    dec[i][thr] = m.decide(r["scrubbed"], r["res"], allowed_types=set(fcfg))
                if rep == 0:
                    d = dec[i][0.5]
                    for fid, p in d["verifier"].items():
                        oof_ver.append((p, int(fid in r["gold"])))
                    from app.nlp.ml_features import sentence_records, sentence_input
                    sx, sy, _, _ = training_rows(tr(r), r["res"], r["scrubbed"])
                    recs = sentence_records(r["scrubbed"], r["res"])
                    for pt, y in zip(m.miss.predict(sx), sy):
                        oof_sent.append((max(pt.values(), default=0.0), int(bool(y))))
        # метрики повтора при пороге 0.5
        route_of = lambda ts: {fcfg[t]["route"] for t in ts}

        def at(thr_miss, thr_fp, drop_fp):
            rtp = rfp = rfn_ = 0
            tp = fp = fn = 0; caught = caught_dict = rfn = rfn_dict = 0; alarms = 0; norm_alarm = 0
            fl_true_fp = fl_tp = rule_fp = rule_tp = 0
            for i, r in enumerate(rows):
                d_m = dec[i][thr_miss]; d_f = dec[i][thr_fp]
                new = set(d_m["rescued"]) | set(d_m["missed"])
                flagged = {t for t, p in d_f["verifier"].items() if p < thr_fp}
                pred = (r["rules"] - (flagged if drop_fp else set())) | new
                g = r["gold"]
                tp += len(pred & g); fp += len(pred - g); fn += len(g - pred)
                rp, rg = route_of(pred), route_of(g)
                rtp += len(rp & rg); rfp += len(rp - rg); rfn_ += len(rg - rp)
                miss = g - r["rules"]
                rfn += len(miss); rfn_dict += len(miss & dict_types)
                caught += len(miss & new); caught_dict += len(miss & dict_types & new)
                alarms += len(new - g); norm_alarm += int(not g and bool(new))
                rule_fp += len(r["rules"] - g); rule_tp += len(r["rules"] & g)
                fl_true_fp += len(flagged - g); fl_tp += len(flagged & g)
            return dict(rtp=rtp, rfp=rfp, rfn_=rfn_, tp=tp, fp=fp, fn=fn, rfn=rfn, caught=caught, rfn_dict=rfn_dict, caught_dict=caught_dict,
                        alarms=alarms, norm_alarm=norm_alarm, rule_fp=rule_fp, rule_tp=rule_tp,
                        fl_true_fp=fl_true_fp, fl_tp=fl_tp)
        per_repeat.append({"flag": at(0.5, 0.5, False), "drop": at(0.5, 0.5, True)})
        for i, r in enumerate(rows):                 # разбивка по типам (порог 0.5, верификатор помечает)
            d = dec[i][0.5]; new = set(d["missed"]) | set(d["rescued"]); pred = r["rules"] | new
            for t in r["gold"]:
                by_type[t][0] += 1; by_type[t][1] += t in r["rules"]; by_type[t][2] += t in pred
            for t in new - r["gold"]:
                by_type[t][3] += 1
        for thr in GRID:
            x = at(thr, 0.5, False); s = sweep[thr]; s[0] += x["tp"]; s[1] += x["fp"]; s[2] += x["fn"]; s[3] += x["caught"]
            y = at(0.5, thr, True); f = sweep_fp[thr]; f[0] += y["fl_true_fp"]; f[1] += y["fl_tp"]; f[2] += y["rule_fp"]
        print(f"  повтор {rep + 1}/{a.repeats} готов ({time.time() - t0:.0f} с)", flush=True)

    n_norm = sum(1 for r in rows if not r["gold"])

    def agg(key, mode="flag"):
        v = [p[mode][key] for p in per_repeat]
        return np.mean(v), min(v), max(v)

    def rate(num, den, mode="flag"):
        v = [p[mode][num] / p[mode][den] if p[mode][den] else 0 for p in per_repeat]
        return 100 * np.mean(v), 100 * min(v), 100 * max(v)

    full = SecondLayer().fit([tr(r) for r in rows] + [tr(x) for x in extra], ex)
    lines = []
    P = lines.append
    tpb, fpb, fnb = base
    P(f"# Второй контур: оценка на реальных протоколах\n")
    P(f"Протоколов: {len(rows)} (с находками: {sum(1 for r in rows if r['gold'])}, норм: {n_norm}); "
      f"находок в разметке (протокол × тип): {sum(len(r['gold']) for r in rows)}.")
    P(f"Кросс-валидация: {a.folds} фолдов по протоколам (дубли — в одной группе) × {a.repeats} повторов; "
      f"в скобках — разброс между повторами." + (f" Дополнительно в обучении (не в тесте): {len(extra)} протоколов из {a.extra_train}." if extra else "") + "\n")
    P("## 1. Первый слой (правила), как есть\n")
    lo, hi = wilson(fnb, tpb + fnb)
    P(f"- Пропуски (FN): **{fnb} из {tpb + fnb} = {pct(fnb, tpb + fnb)}** (95% ДИ {100*lo:.0f}–{100*hi:.0f}%)")
    lo, hi = wilson(fpb, tpb + fpb)
    P(f"- Ложные (FP): **{fpb} из {tpb + fpb} показанных = {pct(fpb, tpb + fpb)}** (95% ДИ {100*lo:.0f}–{100*hi:.0f}%)")
    rfn_dict = per_repeat[0]["flag"]["rfn_dict"]
    P(f"- Из пропусков {fnb - rfn_dict} — типы, которых в словаре нет вообще; {rfn_dict} — типы словаря.\n")

    P("## 2. Детектор пропусков (FN)\n")
    m, mn, mx = rate("caught", "rfn")
    P(f"- Ловит пропуски правил: **{m:.1f}%** ({mn:.0f}–{mx:.0f}%) из {per_repeat[0]['flag']['rfn']}")
    m2, mn2, mx2 = agg("alarms")
    P(f"- Лишние подсказки (находки нет в разметке): **{m2:.1f}** на {len(rows)} протоколов ({mn2}–{mx2}); "
      f"протоколов-норм с подсказкой: {agg('norm_alarm')[0]:.1f} из {n_norm}")
    if oof_sent:
        P(f"- Качество по предложениям (ROC AUC, вне обучения): {roc_auc_score([y for _, y in oof_sent], [p for p, _ in oof_sent]):.3f}\n")

    P("## 3. Верификатор (FP)\n")
    m, mn, mx = rate("fl_true_fp", "rule_fp")
    P(f"- Помечает ложные срабатывания правил: **{m:.1f}%** ({mn:.0f}–{mx:.0f}%) из {per_repeat[0]['flag']['rule_fp']}")
    m, mn, mx = rate("fl_tp", "rule_tp")
    P(f"- Ошибочно помечает настоящие находки: **{m:.1f}%** ({mn:.0f}–{mx:.0f}%) из {per_repeat[0]['flag']['rule_tp']}")
    if len({y for _, y in oof_ver}) > 1:
        P(f"- ROC AUC по находкам правил (вне обучения): {roc_auc_score([y for _, y in oof_ver], [p for p, _ in oof_ver]):.3f}\n")

    P("## 4. Итог: доля ошибок системы\n")
    P("| Конфигурация | Пропуски (FN-rate) | Ложные (FDR) | TP / FP / FN |")
    P("|---|---|---|---|")
    P(f"| Только правила | {pct(fnb, tpb + fnb)} | {pct(fpb, tpb + fpb)} | {tpb} / {fpb} / {fnb} |")
    for mode, title in (("flag", "Правила + детектор пропусков (верификатор только помечает)"),
                        ("drop", "Правила + детектор + верификатор убирает «вероятно ложные»")):
        v_fn = [p[mode]["fn"] / (p[mode]["tp"] + p[mode]["fn"]) for p in per_repeat]
        v_fd = [p[mode]["fp"] / (p[mode]["tp"] + p[mode]["fp"]) for p in per_repeat]
        P(f"| {title} | {100*np.mean(v_fn):.1f}% ({100*min(v_fn):.0f}–{100*max(v_fn):.0f}) | "
          f"{100*np.mean(v_fd):.1f}% ({100*min(v_fd):.0f}–{100*max(v_fd):.0f}) | "
          f"{agg('tp', mode)[0]:.1f} / {agg('fp', mode)[0]:.1f} / {agg('fn', mode)[0]:.1f} |")
    P("\n**На уровне маршрутов** (протокол × профильный специалист — именно это решает, дойдёт ли пациент):\n")
    P("| Конфигурация | Пропущенные маршруты | Лишние маршруты | TP / FP / FN |")
    P("|---|---|---|---|")
    rb = [0, 0, 0]
    for r in rows:
        rp, rg = route_of(r["rules"]), route_of(r["gold"])
        rb[0] += len(rp & rg); rb[1] += len(rp - rg); rb[2] += len(rg - rp)
    P(f"| Только правила | {pct(rb[2], rb[0] + rb[2])} | {pct(rb[1], rb[0] + rb[1])} | {rb[0]} / {rb[1]} / {rb[2]} |")
    for mode, title in (("flag", "Правила + второй контур (верификатор помечает)"), ("drop", "Правила + второй контур (верификатор убирает)")):
        v_fn = [p[mode]["rfn_"] / (p[mode]["rtp"] + p[mode]["rfn_"]) for p in per_repeat]
        v_fd = [p[mode]["rfp"] / (p[mode]["rtp"] + p[mode]["rfp"]) for p in per_repeat]
        P(f"| {title} | {100*np.mean(v_fn):.1f}% ({100*min(v_fn):.0f}–{100*max(v_fn):.0f}) | "
          f"{100*np.mean(v_fd):.1f}% ({100*min(v_fd):.0f}–{100*max(v_fd):.0f}) | "
          f"{agg('rtp', mode)[0]:.1f} / {agg('rfp', mode)[0]:.1f} / {agg('rfn_', mode)[0]:.1f} |")
    P("\n## 5. Пороги (сумма по всем повторам)\n")
    P("| Порог детектора | Пропуски | Ложные | Поймано пропусков правил |")
    P("|---|---|---|---|")
    for thr in GRID:
        tp, fp, fn, c = sweep[thr]
        P(f"| {thr} | {pct(fn, tp + fn)} | {pct(fp, tp + fp)} | {pct(c, per_repeat[0]['flag']['rfn'] * a.repeats)} |")
    P("\n| Порог верификатора | Помечено ложных правил | Ошибочно помечено настоящих |")
    P("|---|---|---|")
    for thr in GRID:
        t_fp, t_tp, rfp = sweep_fp[thr]
        P(f"| {thr} | {pct(t_fp, rfp)} | {pct(t_tp, per_repeat[0]['flag']['rule_tp'] * a.repeats)} |")
    P("\n## 6. По типам находок (порог 0.5, сумма по повторам ÷ число повторов)\n")
    P("| Тип | В словаре? | В разметке | Нашли правила | Правила + 2-й контур | Лишние подсказки |")
    P("|---|---|---|---|---|---|")
    R = a.repeats
    for t, (n, ru, both, extra) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        P(f"| {fcfg[t]['name']} | {'да' if t in dict_types else 'нет'} | {n // R} | {ru / R:.0f} | {both / R:.1f} | {extra / R:.1f} |")
    P(f"\nОбучающая выборка полной модели: {full.meta}")
    os.makedirs("reports", exist_ok=True)
    open(a.report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open("reports/second_layer_cv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["repeat", "mode", *per_repeat[0]["flag"].keys()])
        for i, p in enumerate(per_repeat):
            for mode in ("flag", "drop"):
                w.writerow([i, mode, *p[mode].values()])
    print("\n".join(lines))
    print(f"\nСохранено: {a.report}, reports/second_layer_cv.csv ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
