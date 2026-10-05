# обучение второго контура (детектор пропусков + верификатор). Секунды на CPU, GPU не нужен
import argparse, json, os, re, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CM_ENV", "demo")
import joblib
from app.nlp.extractor import extract
from app.nlp.ml_layer import SecondLayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["data/real/protocols.jsonl", "data/web/protocols.jsonl"])
    ap.add_argument("--miss", type=float, default=0.9, help="порог детектора пропусков")
    ap.add_argument("--fp", type=float, default=0.4, help="порог верификатора")
    ap.add_argument("--out", default="models/second_layer.joblib")
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.paths if os.path.exists(p) for l in open(p, encoding="utf-8") if l.strip()]
    m = SecondLayer(miss_threshold=a.miss, fp_threshold=a.fp, rescue_threshold=a.miss)
    m.fit(rows, lambda t: extract(t, None))
    m.meta.update(trained_at=datetime.now().isoformat(timespec="seconds"), data=[os.path.basename(p) for p in a.paths])

    # проверка: в словаре модели не должно быть персональных данных (ФИО врачей/пациентов из шапок)
    vocab = set()
    for v in m.miss.vecs[1:]:
        vocab |= set(v.vocabulary_)
    leaked = set()
    for r in rows:
        for mm in re.finditer(r"(?im)^(?:врач|подпись\s+врача|пациент)\s*:[ \t]*\n?([^\n:]+)$", r["text"].replace("\xa0", " ")):
            leaked |= {w.lower() for w in re.findall(r"[А-ЯЁ][а-яё]{2,}", mm.group(1))} & vocab
    if leaked:
        sys.exit(f"СТОП: в словаре модели остались персональные данные {sorted(leaked)[:10]} — дополните anonymize.scrub")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump(m, a.out)
    print(f"Модель сохранена: {a.out}")
    print(f"  протоколов {m.meta['n_protocols']}, предложений {m.meta['n_sentences']} (с находками {m.meta['n_pos_sentences']}), "
          f"кандидатов правил {m.meta['n_candidates']} (настоящих {m.meta['n_true_candidates']})")
    print(f"  типы детектора: {', '.join(m.miss.types)}")
    if m.miss.skipped:
        print(f"  мало примеров (<{m.miss.MIN_POS}), не обучены: {', '.join(m.miss.skipped)}")
    print("Включите: ml.enabled: true в config/findings.yaml")


if __name__ == "__main__":
    main()
