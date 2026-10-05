# второй контур — обучаемые модели поверх правил. Первый слой (extractor.py) — это фильтр
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app import config
from app.config import BASE_DIR
from app.nlp.ml_features import candidate_records, context_tags, sentence_input, sentence_records

FORMAT_VERSION = 2


def _text_vectorizers():
    # буквосочетания (устойчивы к опечаткам и окончаниям) + слова/пары слов
    return [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")]


class _TextModel:
    # TF-IDF -> логистическая регрессия (бинарная или многоклассовая)

    def __init__(self, C=4.0):
        self.C = C

    def fit(self, texts, y):
        self.vecs = _text_vectorizers()
        X = hstack([v.fit_transform(texts) for v in self.vecs]).tocsr()
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self.clf.fit(X, y)
        return self

    def proba(self, texts):
        X = hstack([v.transform(texts) for v in self.vecs]).tocsr()
        return self.clf.predict_proba(X)


class MissDetector:
    # FN: предложение -> P(в нём описана находка типа T) для каждого типа (один-против-всех)
    MIN_POS = 2

    def fit(self, sent_texts, sent_types):
        self.vecs = _text_vectorizers()
        X = hstack([v.fit_transform(sent_texts) for v in self.vecs]).tocsr()
        counts = {}
        for ts in sent_types:
            for t in ts:
                counts[t] = counts.get(t, 0) + 1
        self.types = sorted(t for t, n in counts.items() if n >= self.MIN_POS)
        self.skipped = sorted(t for t, n in counts.items() if n < self.MIN_POS)
        self.clfs = {}
        for t in self.types:
            y = [int(t in ts) for ts in sent_types]
            self.clfs[t] = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000).fit(X, y)
        return self

    def predict(self, sent_texts):
        # -> [{тип: P}] по каждому предложению
        if not sent_texts or not self.clfs:
            return [{} for _ in sent_texts]
        X = hstack([v.transform(sent_texts) for v in self.vecs]).tocsr()
        P = {t: c.predict_proba(X)[:, 1] for t, c in self.clfs.items()}
        return [{t: float(P[t][i]) for t in self.types} for i in range(len(sent_texts))]


class FPVerifier:
    # FP (и спасение отклонённых): кандидат правил -> P(находка настоящая)

    def fit(self, cand_rows, y):
        self.dv = DictVectorizer()
        self.tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=40000)
        X = hstack([self.dv.fit_transform([r["feats"] for r in cand_rows]) * 2.0,   # признаки правил весят вдвое больше текста
                    self.tv.fit_transform([r["text"] for r in cand_rows])]).tocsr()
        self.constant = None
        if len(set(y)) < 2:                      # в выборке один класс — модель вырождена
            self.constant = float(y[0]) if y else 0.5
            return self
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(X, y)
        return self

    def predict(self, cand_rows):
        if not cand_rows:
            return np.array([])
        if self.constant is not None:
            return np.full(len(cand_rows), self.constant)
        X = hstack([self.dv.transform([r["feats"] for r in cand_rows]) * 2.0,
                    self.tv.transform([r["text"] for r in cand_rows])]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


class SecondLayer:
    # оба модуля + пороги. Сохраняется целиком в models/second_layer.joblib (scripts/train.py)
    name = "second_layer"

    # спасение отклонённых правилами кандидатов выключено: на реальных протоколах (3×5 CV) оно ни разу
    # не было правым и дважды «отменило» верное отрицание. Включается порогом rescue_threshold < 1
    def __init__(self, miss_threshold=0.5, fp_threshold=0.5, rescue_threshold=1.01):
        self.miss_threshold, self.fp_threshold, self.rescue_threshold = miss_threshold, fp_threshold, rescue_threshold
        self.format_version = FORMAT_VERSION
        self.meta = {}

    # обучение
    def fit(self, protocols, extract_fn):
        # protocols: [{"text", "expected": [{"id", "quotes": [...]}]}]; extract_fn(text) -> ExtractionResult без модели
        from app.nlp.anonymize import scrub
        sent_x, sent_y, cand_rows, cand_y = [], [], [], []
        for p in protocols:
            text = scrub(p["text"])                  # учимся ТОЛЬКО на обезличенном тексте (как в бою)
            res = extract_fn(text)
            sx, sy, cr, cy = training_rows(p, res, text)
            sent_x += sx; sent_y += sy; cand_rows += cr; cand_y += cy
        self.miss = MissDetector().fit(sent_x, sent_y)
        # тип находки ↔ исследуемый орган: в каких контекстах тип встречался в обучении
        # не даёт «полипу ЖП» стать «образованием яичника», а доле печени — «узлом щитовидной»
        self.type_ctx = {}
        for p in protocols:
            tags = set(context_tags(scrub(p["text"])).split())
            for e in p["expected"]:
                self.type_ctx.setdefault(e["id"], set()).update(tags)
        self.verifier = FPVerifier().fit(cand_rows, cand_y)
        self.meta.update(n_protocols=len(protocols), n_sentences=len(sent_x), n_pos_sentences=sum(1 for t in sent_y if t),
                         n_candidates=len(cand_rows), n_true_candidates=int(sum(cand_y)), types=self.miss.types)
        return self

    # решение по одному протоколу: сравнение с пороговыми значениями
    def decide(self, text, res, allowed_types=None):
        # что скажет второй контур. Ничего не меняет; возвращает:
        accepted = {c["id"] for c in res.candidates if c["status"] == "accepted"}
        sents = {s: (a, b) for s, a, b in res.sentences}
        cands = candidate_records(text, res, lambda sid: text[sents[sid][0]:sents[sid][1]] if sid in sents else "")
        probs = self.verifier.predict(cands)
        verifier, rescued = {}, {}
        for r, p in zip(cands, probs):
            if r["status"] == "accepted":
                verifier[r["id"]] = max(verifier.get(r["id"], 0.0), float(p))
            elif r["id"] not in accepted and p >= self.rescue_threshold:
                if p > rescued.get(r["id"], (0,))[0]:
                    rescued[r["id"]] = (float(p), r["cand"])
        missed = {}
        # лингвистику (отрицание, анамнез, исключение, рекомендация) правила разбирают точнее, чем мешок n-грамм:
        # если правило в ЭТОМ предложении увидело тип и явно отклонило его — модель не спорит
        LING = ("отрицание", "упоминание в анамнезе", "исключение", "упоминание в рекомендации")
        vetoed = {(c["sid"], c["id"]) for c in res.candidates if c["status"] == "suppressed" and c["reason"].startswith(LING)}
        recs = sentence_records(text, res)
        ctx = context_tags(text)
        tags = set(ctx.split())
        for rec, p_typ in zip(recs, self.miss.predict([sentence_input(r, ctx) for r in recs])):
            for t, score in p_typ.items():                                   # p(в предложении находка типа t)
                if score < self.miss_threshold:
                    continue
                if allowed_types is not None and t not in allowed_types:
                    continue
                if (rec["sid"], t) in vetoed:
                    continue                                                 # правило явно отклонило в этом предложении
                if tags and self.type_ctx.get(t) and not (tags & self.type_ctx[t]):
                    continue                                                 # тип не про этот орган
                if t in accepted or t in rescued:
                    continue
                if score > missed.get(t, (0,))[0]:
                    missed[t] = (float(score), rec)
        return {"verifier": verifier, "rescued": rescued, "missed": missed}

    # применение в extractor
    def apply(self, text, res, findings_cfg):
        # встроить решения в результат извлечения: пометки, новые находки «на проверку», сортировка
        from app.nlp.extractor import Evidence, Finding     # поздний импорт: избегаем циклической зависимости
        d = self.decide(text, res, allowed_types=set(findings_cfg))
        log = {"model": self.name, "trained": self.meta.get("trained_at"), "likely_fp": [], "rescued": [], "missed": []}
        ml_cfg = config.findings().get("ml", {})
        trusted = ml_cfg.get("verifier_trusted_types")         # типы, на которых верификатор видел достаточно примеров
        COND_ATTR = {"size_mm_min": "size_mm", "birads_min": "birads", "stenosis_min": "stenosis_pct"}
        for f in res.findings:
            p = d["verifier"].get(f.id)
            if p is None:
                continue
            if trusted is not None and f.id not in trusted:
                # модель почти не видела настоящих примеров этого типа (полип ЖП ≥10 мм, камень почки,
                # грыжа — 0 штук в обучении) и «выучила», что такая находка всегда ложная: «Полип ЖП 12 мм» -> p=0.01
                # её мнение по таким типам не учитываем, пока не переобучим на данных с настоящими примерами
                continue
            cond = (findings_cfg.get(f.id) or {}).get("condition") or {}
            if cond and all(any(COND_ATTR.get(k) in (e.attributes or {}) for e in f.evidence) for k in cond):
                # правило само измерило признак, и порог выполнен (размер ≥ 10 мм, BI-RADS ≥ 3) — модель не спорит
                f.evidence[0].attributes = {**f.evidence[0].attributes, "p_true": round(p, 2)}
                continue
            f.evidence[0].attributes = {**f.evidence[0].attributes, "p_true": round(p, 2)}
            if p < self.fp_threshold:
                f.needs_review = True
                f.evidence[0].flags.append(f"второй контур: вероятно ложное срабатывание (p={p:.2f}) — проверить")
                log["likely_fp"].append({"id": f.id, "p": round(p, 2)})

        def add(fid, p, quote, sentence, start, end, flag):
            cfg = findings_cfg[fid]
            fd = Finding(fid, cfg["name"], cfg["route"], cfg.get("priority", 3), bool(cfg.get("urgent")), round(p * 0.6, 2), True)
            fd.evidence.append(Evidence(quote, sentence, start, end, f"ml:{self.name}", {"p_model": round(p, 2)}, round(p * 0.6, 2), [flag]))
            res.findings.append(fd)

        sents = {s: (a, b) for s, a, b in res.sentences}
        for fid, (p, c) in d["rescued"].items():
            a, b = sents.get(c["sid"], (0, 0))
            add(fid, p, c["quote"], text[a:b], a, b, f"правило отклонило ({c['reason']}), второй контур считает находкой (p={p:.2f}) — проверить")
            log["rescued"].append({"id": fid, "p": round(p, 2), "reason": c["reason"]})
        for fid, (p, rec) in d["missed"].items():
            add(fid, p, rec["text"], rec["text"], rec["start"], rec["end"], f"правила не сработали, второй контур нашёл (p={p:.2f}) — проверить")
            log["missed"].append({"id": fid, "p": round(p, 2)})
        # сортировка: экстренные -> приоритет -> «вероятно ложные» в конец своей группы -> уверенность
        res.findings.sort(key=lambda x: (not x.urgent, x.priority, any("вероятно ложное" in fl for e in x.evidence for fl in e.flags), -x.confidence))
        res.second_layer = log
        if log["rescued"] or log["missed"]:
            res.summary = f"Найдено находок: {len(res.findings)} (из них второй контур: {len(log['rescued']) + len(log['missed'])})"
        return res


def training_rows(p, res, text=None):
    # разметка протокола -> обучающие строки обоих модулей
    from app.nlp.ml_features import norm
    gold = {e["id"]: [norm(q) for q in (e.get("quotes") or [e.get("quote", "")]) if q] for e in p["expected"]}
    text = text if text is not None else p["text"]          # должен совпадать с текстом, на котором считан res
    recs = sentence_records(text, res)
    ctx = context_tags(text)
    sx, sy = [], []
    for r in recs:
        st = norm(r["text"])
        types = {t for t, qs in gold.items() if any(q and (q in st or (len(st) > 15 and st in q)) for q in qs)}
        sx.append(sentence_input(r, ctx)); sy.append(types)
    sents = {s: (a, b) for s, a, b in res.sentences}
    cr = candidate_records(text, res, lambda sid: text[sents[sid][0]:sents[sid][1]] if sid in sents else "")
    cy = [int(r["id"] in gold) for r in cr]
    return sx, sy, cr, cy


_cached = {"key": None, "obj": None}


def get_ml_hook():
    # второй контур, если он включён в findings.yaml (ml.enabled) и модель обучена. Иначе None
    ml = config.findings().get("ml", {})
    if not ml.get("enabled"):
        return None
    path = os.path.join(BASE_DIR, ml.get("model_path", "models/second_layer.joblib"))
    if not os.path.exists(path):
        if _cached.get("warned") != path:                  # предупреждаем один раз, а не на каждый протокол
            print(f"[ml] включено, но модели нет ({path}) — запустите scripts/train.py; работаю только на правилах", flush=True)
            _cached["warned"] = path
        return None
    key = (path, os.path.getmtime(path), ml.get("miss_threshold"), ml.get("fp_threshold"))
    if _cached["key"] != key:
        import joblib                              # joblib = pickle. Грузим ТОЛЬКО модель из образа, не из пользовательской папки
        obj = joblib.load(path)
        if getattr(obj, "format_version", None) != FORMAT_VERSION:
            print("[ml] модель старого формата — переобучите scripts/train.py", flush=True)
            return None
        obj.miss_threshold = ml.get("miss_threshold", obj.miss_threshold)
        obj.rescue_threshold = ml.get("rescue_threshold", 1.01)
        obj.fp_threshold = ml.get("fp_threshold", obj.fp_threshold)
        _cached.update(key=key, obj=obj)
    return _cached["obj"]
