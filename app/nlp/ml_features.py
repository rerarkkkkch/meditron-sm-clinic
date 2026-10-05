# признаки для ВТОРОГО КОНТУРА. Общие для обучения (scripts/train.py) и работы (ml_layer.py),
import re

_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
_SPACES = re.compile(r"\s+")
_CONCL = re.compile(r"заключени|диагноз\s*:|основной\s*:")


def norm(s: str) -> str:
    # нижний регистр, ё->е, все числа -> 0 (модель учит слова, а не конкретные размеры)
    s = s.lower().replace("ё", "е")
    s = _DIGITS.sub("0", s)
    return _SPACES.sub(" ", s).strip()


def conclusion_start(text: str) -> int:
    # позиция начала раздела «Заключение» (или конец текста, если раздела нет)
    m = _CONCL.search(text.lower())
    return m.start() if m else len(text)


def sentence_records(text: str, res) -> list:
    # предложения протокола (без частей-рекомендаций) с контекстом для детектора пропусков
    c0 = conclusion_start(text)
    acc = {c["sid"] for c in res.candidates if c["status"] == "accepted"}
    sup = {c["sid"] for c in res.candidates if c["status"] == "suppressed"}
    out = []
    for sid, a, b in res.sentences:
        out.append({"sid": sid, "start": a, "end": b, "text": text[a:b], "concl": a >= c0,
                    "rule_acc": sid in acc, "rule_sup": sid in sup})
    return out


# контекст протокола: какой орган исследуют. В бою это поле study_type из МИС; здесь — по ключевым словам
# во всём тексте, чтобы «узел 16 мм» в предложении без органа модель отнесла к нужному профилю
_CTX = {
    "gyn": r"матк|яичник|эндометр|миометр",
    "gb": r"желчн|холедох",
    "breast": r"молочн|маммо",
    "prostate": r"предстательн|простат",
    "thyroid": r"щитовидн|тирео",
    "legs": r"бедренн|подколенн|берцов|сафено|бпв|мпв",
}


def context_tags(text: str) -> str:
    low = text.lower().replace("ё", "е")
    return " ".join(f"zzctx{k}" for k, rx in _CTX.items() if re.search(rx, low))


def sentence_input(rec: dict, ctx: str = "") -> str:
    # строка для векторизатора: нормализованный текст + метки «заключение» и «орган исследования»
    return norm(rec["text"]) + (" zzconcl" if rec["concl"] else "") + (" " + ctx if ctx else "")


def _reason_kind(reason: str) -> str:
    return reason.split(":")[0].strip() if reason else "accepted"


def candidate_records(text: str, res, sentence_text) -> list:
    # кандидаты правил с признаками для верификатора. sentence_text(sid) -> текст предложения
    c0 = conclusion_start(text)
    by_type = {}
    for c in res.candidates:
        by_type.setdefault(c["id"], []).append(c)
    sent_start = {sid: a for sid, a, _ in res.sentences}
    out = []
    for c in res.candidates:
        sib = by_type[c["id"]]
        rule_kind = c["rule"].split("[")[0].split(".")[-1] if "." in c["rule"] else c["rule"]
        if "+section" in c["rule"]:
            rule_kind += "_section"
        feats = {
            "fid=" + c["id"]: 1, "status=" + c["status"]: 1, "reason=" + _reason_kind(c["reason"]): 1,
            "rule=" + rule_kind: 1, "fid_status=" + c["id"] + "_" + c["status"]: 1,
            "concl": int(sent_start.get(c["sid"], 0) >= c0),
            "conf": c["conf"],
            "flag_missing_attr": int(any("в тексте нет" in x for x in c["flags"])),
            "flag_fuzzy": int(any("нечёткое" in x for x in c["flags"])),
            "flag_uncertain": int(any("неуверенная" in x for x in c["flags"])),
            "flag_birads_multi": int(any("несколько категорий" in x for x in c["flags"])),
            "has_size": int("size_mm" in c["attrs"]),
            # что правила решили по ЭТОМУ ЖЕ типу в других местах протокола — сильный контекст:
            "sib_accepted": int(any(s["status"] == "accepted" and s is not c for s in sib)),
            "sib_cond_failed": int(any(s["reason"].startswith("условие не выполнено") for s in sib)),
            "sib_negated": int(any(s["reason"].startswith("отрицание") for s in sib)),
            "sib_history": int(any(s["reason"].startswith("упоминание в анамнезе") for s in sib)),
            "sib_excluded": int(any(s["reason"].startswith("исключение") for s in sib)),
            "n_sib": min(len(sib), 5) / 5,
        }
        out.append({"id": c["id"], "status": c["status"], "feats": feats,
                    "text": norm(sentence_text(c["sid"])) + " || " + norm(c["quote"]), "cand": c})
    return out
