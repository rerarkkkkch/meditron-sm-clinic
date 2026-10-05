# перенос: обучить второй контур на одном наборе, проверить на ДРУГОМ (без кросс-валидации)
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CM_ENV", "demo")
from app import config
from app.nlp import anonymize
from app.nlp.extractor import extract
from app.nlp.ml_layer import SecondLayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/real/protocols.jsonl")
    ap.add_argument("--test", default="data/web/protocols.jsonl")
    ap.add_argument("--miss", type=float, default=0.5, help="порог детектора пропусков")
    ap.add_argument("--fp", type=float, default=0.5, help="порог верификатора")
    a = ap.parse_args()
    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    tr, te = load(a.train), load(a.test)
    fcfg = config.findings()["findings"]
    m = SecondLayer(miss_threshold=a.miss, fp_threshold=a.fp).fit(tr, lambda t: extract(t, None))
    route = lambda ts: {fcfg[t]["route"] for t in ts}
    c = {k: [0, 0, 0] for k in ("rules", "flag", "drop", "r_rules", "r_flag", "r_drop")}
    for r in te:
        t = anonymize.scrub(r["text"]); res = extract(t, None)
        g = {e["id"] for e in r["expected"]}; rules = {f.id for f in res.findings}
        d = m.decide(t, res, allowed_types=set(fcfg))
        new = set(d["missed"]) | set(d["rescued"]); low = {k for k, p in d["verifier"].items() if p < a.fp}
        for name, pred in (("rules", rules), ("flag", rules | new), ("drop", (rules - low) | new)):
            for key, P, G in ((name, pred, g), ("r_" + name, route(pred), route(g))):
                c[key][0] += len(P & G); c[key][1] += len(P - G); c[key][2] += len(G - P)
    pct = lambda x, y: f"{100 * x / y:.1f}%" if y else "—"
    print(f"Обучение: {a.train} ({len(tr)}), проверка: {a.test} ({len(te)}), пороги: детектор {a.miss}, верификатор {a.fp}")
    print("| Конфигурация | Пропуски находок | Ложные находки | Пропущенные маршруты | Лишние маршруты |")
    print("|---|---|---|---|---|")
    for name, title in (("rules", "Только правила"), ("flag", "+ второй контур (помечает)"), ("drop", "+ второй контур (убирает «вероятно ложные»)")):
        tp, fp, fn = c[name]; rtp, rfp, rfn = c["r_" + name]
        print(f"| {title} | {pct(fn, tp + fn)} ({fn}) | {pct(fp, tp + fp)} ({fp}) | {pct(rfn, rtp + rfn)} ({rfn}) | {pct(rfp, rtp + rfp)} ({rfp}) |")


if __name__ == "__main__":
    main()
