# извлечение находок из текста протокола. Полностью локально, без внешних LLM
import re                                            # регулярные выражения для размеров и BI-RADS
from dataclasses import dataclass, field, asdict     # структуры результата
from app import config                               # доступ к YAML-настройкам
from app.nlp.normalize import tokenize, phrase_lemmas, similar, norm_text, ABBR   # нормализация текста
from app.nlp.normalize import _MORPH as _M
MORPH_OK = _M is not None                            # есть ли pymorphy3 (в бою — обязательно)

# размер: «12 мм», «1,5 см», «12х8х10 мм» (берём максимальное измерение)
SIZE_RE = re.compile(r"(\d+(?:[.,]\d+)?(?:\s*[xх×*]\s*\d+(?:[.,]\d+)?)*)\s*(мм|см)(?![а-я])")
# BI-RADS: «BI-RADS 4», «BIRADS: 3», «би-рэдс 5»
BIRADS_RE = re.compile(r"(?:bi\s*-?\s*rads|birads|би\s*-?\s*рэдс)\s*[:\-]?\s*(?:категория\s*)?([0-6])")
# «BI-RADS справа 2, слева 2», «Bi-RADS: справа - 3, слева - 2»
BIRADS_KW = re.compile(r"(?:bi\s*-?\s*rads|birads|би\s*-?\s*рэдс)")
BIRADS_VAL = re.compile(r"(?:^|[\s:\-–(,])([0-6])(?:\s*[абвabc](?![а-яa-z]))?(?=$|[\s.,;:)])(?!\s*(?:мм|см|%|х|x)\b)")


def birads_values(low: str, a: int = 0, b: int = None) -> list:
    # все категории BI-RADS в диапазоне [a, b): [(позиция_ключевого_слова, [значения])]
    b = len(low) if b is None else b
    out = []
    for m in BIRADS_KW.finditer(low, a, b):
        seg = low[m.end(): min(m.end() + 40, b + 3)]
        cut = re.search(r"\n|\.\s", seg)                    # до конца предложения/строки
        seg = seg[: cut.start()] if cut else seg
        vals = [int(v) for v in BIRADS_VAL.findall(seg)]
        if vals:
            out.append((m.start(), vals))
    return out


@dataclass
class Evidence:
    # доказательство одного срабатывания — то, что видит врач/админ
    quote: str             # найденный фрагмент текста
    sentence: str          # всё предложение целиком (контекст)
    start: int             # позиция начала фрагмента в тексте (для подсветки)
    end: int               # позиция конца фрагмента
    rule: str              # какое правило словаря сработало
    attributes: dict       # извлечённые атрибуты (размер, BI-RADS)
    confidence: float      # уверенность 0..1
    flags: list            # пометки: «неуверенная формулировка», «нет размера — проверить» и т.п


@dataclass
class Finding:
    # принятая находка (может иметь несколько доказательств в тексте)
    id: str                # ключ находки из словаря
    name: str              # человекочитаемое название
    route: str             # ключ маршрута из routes.yaml
    priority: int          # приоритет 1..3
    urgent: bool           # экстренная ли
    confidence: float      # итоговая уверенность (максимум по доказательствам)
    needs_review: bool     # требуется ли особое внимание админа
    evidence: list = field(default_factory=list)   # список Evidence


@dataclass
class Suppressed:
    # кандидат, который НЕ стал находкой, и почему (объяснимость для нормы)
    id: str                # ключ находки
    name: str              # название
    quote: str             # фрагмент текста
    reason: str            # причина отказа


@dataclass
class ExtractionResult:
    # всё, что извлекли из одного протокола
    findings: list                 # принятые находки
    suppressed: list               # отклонённые кандидаты с причинами
    clinician_recommended: list    # ключи маршрутов, которые рекомендовал сам врач УЗИ
    normal_quotes: list            # найденные фразы нормы («без патологии»)
    dictionary_version: str        # версия словаря
    summary: str                   # короткое объяснение итога одной строкой
    suppressed_in_recommendation: bool = False   # была находка внутри рекомендации — сигнал админу
    candidates: list = field(default_factory=list)   # журнал кандидатов правил (для второго контура; в БД не сохраняется)
    sentences: list = field(default_factory=list)    # предложения [(start, end, в_рекомендации)] (для второго контура)
    second_layer: dict = field(default_factory=dict) # что сделал второй контур (для объяснения)

    def to_dict(self):
        d = asdict(self)
        d.pop("candidates", None); d.pop("sentences", None)   # служебное, в БД не храним
        return d


# подготовка словаря: фразы из YAML превращаем в списки лемм один раз и кэшируем
_compiled = {"src": None, "data": None}
_KNOWN = set()   # кэш: исходный словарь -> скомпилированный
_OPTS = {"fuzzy_only_unknown": True}   # нечётко сравниваем только «незнакомые» слова (опечатки)

# исключения, добавленные администратором через интерфейс (хранятся в БД, см. store.user_exclusions)
# формат: {"ver": число, "items": [{"finding_id": "venous_thrombosis" | "*", "phrase": "аорта", "scope": "sentence"|"clause"}]}
_USER_EX = {"ver": 0, "items": []}


# ручные добавления в словарь из интерфейса (хранятся в БД, см. store.user_dictionary)
# kind: term (термин к находке) | what («что» к находке) | location (синоним органа) | finding (новая находка)
_USER_DICT = {"ver": 0, "items": []}


def set_user_dictionary(items):
    # подменить список ручных добавлений в словарь (main.py при старте и после правки в интерфейсе)
    _USER_DICT["items"] = list(items)
    _USER_DICT["ver"] += 1


def all_findings() -> dict:
    # находки словаря YAML + новые находки, добавленные администратором (только описания, без лемм)
    out = dict(config.findings().get("findings", {}))
    for it in _USER_DICT["items"]:
        if it["kind"] == "finding" and it["target"] not in out:
            ex = it.get("extra") or {}
            out[it["target"]] = {"name": ex.get("name", it["target"]), "route": ex.get("route"), "priority": ex.get("priority", 3),
                                 "urgent": bool(ex.get("urgent")), "terms": [it["phrase"]], "where": ex.get("where", []),
                                 "user_added": True}
    for it in _USER_DICT["items"]:
        if it["kind"] in ("term", "what") and it["target"] in out:
            f = dict(out[it["target"]])
            key = "terms" if it["kind"] == "term" else "what"
            f[key] = list(f.get(key, [])) + [it["phrase"] + " (админ)"]
            out[it["target"]] = f
    return out


def set_user_exclusions(items):
    # подменить список пользовательских исключений (вызывает main.py при старте и после правки в интерфейсе)
    _USER_EX["items"] = list(items)
    _USER_EX["ver"] += 1


def _compile(fcfg: dict, rcfg: dict) -> dict:
    # перевести словарь из YAML во внутренний вид (леммы). Пересобирается при изменении YAML
    if (_compiled["src"] is fcfg and _compiled.get("rsrc") is rcfg and _compiled.get("uver") == _USER_EX["ver"]
            and _compiled.get("dver") == _USER_DICT["ver"]):   # ничего не менялось — кэш
        return _compiled["data"]                                    # уже скомпилировано
    mods = fcfg.get("modifiers", {})                                # общие модификаторы
    data = {
        "matching": fcfg.get("matching", {}),                       # параметры поиска
        "locations": {k: [phrase_lemmas(p) for p in v] for k, v in fcfg.get("locations", {}).items()},   # органы
        "mods": {k: [phrase_lemmas(p, drop_stop=False) for p in v] for k, v in mods.items() if k != "scope_breakers"},  # модификаторы
        "breakers": set(mods.get("scope_breakers", [])),            # слова-разрыватели области отрицания
        "findings": {},                                             # находки
        "aliases": {rid: [phrase_lemmas(a) for a in r.get("aliases", [])] for rid, r in rcfg.get("routes", {}).items()},  # специалисты в рекомендациях
    }
    for it in _USER_DICT["items"]:                                  # синонимы органов от администратора
        if it["kind"] == "location" and it["target"] in data["locations"]:
            data["locations"][it["target"]].append(phrase_lemmas(it["phrase"]))
    src_findings = dict(fcfg.get("findings", {}))
    for it in _USER_DICT["items"]:                                  # новые находки от администратора
        if it["kind"] == "finding" and it["target"] not in src_findings:
            ex = it.get("extra") or {}
            src_findings[it["target"]] = {"name": ex.get("name", it["target"]), "route": ex.get("route"),
                                          "priority": ex.get("priority", 3), "urgent": bool(ex.get("urgent")), "terms": [it["phrase"]],
                                          "where": ex.get("where", [])}
    for fid, f in src_findings.items():                             # каждая находка словаря
        if f.get("enabled") is False:                               # выключена в словаре (решение медэксперта)
            continue
        data["findings"][fid] = {
            **f,                                                    # все поля как есть (name, route, condition...)
            "terms_l": [(t, phrase_lemmas(t)) for t in f.get("terms", [])],     # самодостаточные термины
            "what_l": [(t, phrase_lemmas(t)) for t in f.get("what", [])],       # «что»
            "exclude_l": [(t, phrase_lemmas(t)) for t in f.get("exclude", [])], # исключения (в своей части предложения)
            "exclude_sent_l": [],                                               # исключения на всё предложение
        }
        if f.get("exclude_scope") == "sentence":                    # для контекстных слов («тромб») — всё предложение
            data["findings"][fid]["exclude_sent_l"] = data["findings"][fid]["exclude_l"]
            data["findings"][fid]["exclude_l"] = []
    for it in _USER_DICT["items"]:                                  # термины и «что» от администратора
        fd = data["findings"].get(it["target"])
        if fd is not None and it["kind"] == "term":
            fd["terms_l"].append((it["phrase"] + " (админ)", phrase_lemmas(it["phrase"])))
        elif fd is not None and it["kind"] == "what":
            fd["what_l"].append((it["phrase"] + " (админ)", phrase_lemmas(it["phrase"])))
    for ux in _USER_EX["items"]:                                    # исключения от администратора
        targets = data["findings"] if ux["finding_id"] == "*" else [ux["finding_id"]]
        for fid in targets:
            if fid in data["findings"]:
                key = "exclude_sent_l" if ux.get("scope") == "sentence" else "exclude_l"
                data["findings"][fid][key].append((ux["phrase"] + " (админ)", phrase_lemmas(ux["phrase"])))
    _KNOWN.clear()                                                  # все слова словаря (для запрета нечётких подмен)
    for fd in data["findings"].values():
        for _, pl in fd["terms_l"] + fd["what_l"]:
            for w in pl:
                _KNOWN.update(w)
    for vs in data["locations"].values():
        for pl in vs:
            for w in pl:
                _KNOWN.update(w)
    _compiled.update(src=fcfg, rsrc=rcfg, uver=_USER_EX["ver"], dver=_USER_DICT["ver"], data=data)   # сохраняем в кэш
    return data                                                     # скомпилированный словарь


# поиск фразы в последовательности слов
def _tok_sim(tok, variants, thr):
    # похожесть слова текста на слово словаря (variants — все формы слова словаря)
    for v in variants:                                      # аббревиатура из словаря («ОБА», «ЖП»)
        if v.startswith(ABBR):                              # совпадает только буква в букву; если это ещё и обычное
            return 1.0 if tok.raw == v[len(ABBR):] and (tok.upper or not tok.known) else 0.0   # слово («оба») — только ЗАГЛАВНЫМИ
    if tok.alts & variants or tok.raw in variants:          # общая начальная форма («железы» и «железа»)
        return 1.0                                          # точное совпадение
    if tok.lemma in _KNOWN or tok.alts & _KNOWN:            # слово само есть в словаре («эндометриоз») — опечаткой
        return 0.0                                          # другого термина («эндометриома») его не считаем
    if MORPH_OK and _OPTS["fuzzy_only_unknown"] and getattr(tok, "known", False):
        return 0.0                                          # обычное слово («простой», «точка») — не опечатка органа
    if not MORPH_OK:                                        # запасной режим (стеммер): основу сравниваем только с основой,
        best = max(similar(tok.lemma, v, thr) for v in variants)   # иначе «эндометрий» ≈ «эндометриом» (85,7%)
        if not best:                                        # snowball непоследователен: «холецистит» -> «холецист»,
            for v in variants:                              # «холецистита» -> «холецистит»; «камня» -> «камн», «камень» -> «камен»
                a_, b_ = sorted((tok.lemma, v), key=len)
                d_ = len(b_) - len(a_)
                if (len(a_) >= 6 and d_ <= 2 and b_.startswith(a_)) or (len(a_) >= 4 and d_ <= 1 and a_[:-1] == b_[:len(a_) - 1]):
                    return 0.95
        return best
    return max(max(similar(tok.lemma, v, thr), similar(tok.raw, v, thr)) for v in variants)   # лучшая похожесть с учётом опечаток


def _match(tokens, plem, thr, max_gap):
    # найти все вхождения фразы plem (список лемм). Вернуть [(индекс_начала, индекс_конца, оценка)]
    out = []                                                        # найденные вхождения
    n = len(tokens)                                                 # число слов
    if not plem:                                                    # пустая фраза — нечего искать
        return out                                                  # ничего не нашли
    for i in range(n):                                              # пробуем начать с каждого слова
        score = _tok_sim(tokens[i], plem[0], thr)                   # совпадает ли первое слово фразы
        if not score:                                               # не совпало
            continue                                                # следующее слово
        j, ok = i, True                                             # j — позиция последнего совпавшего слова
        for p in plem[1:]:                                          # остальные слова фразы
            hit = None                                              # совпадение для текущего слова фразы
            for k in range(j + 1, min(j + 2 + max_gap, n)):         # смотрим вперёд с допуском «лишних» слов
                if tokens[k].sent != tokens[i].sent:                # фраза не может перескочить в другое предложение
                    break                                           # прекращаем поиск
                s = _tok_sim(tokens[k], p, thr)                     # похожесть
                if s:                                               # нашли
                    hit = (k, s)                                    # запоминаем позицию и оценку
                    break                                           # дальше не ищем
            if not hit:                                             # слово фразы не нашлось
                ok = False                                          # фраза не совпала
                break                                               # выходим
            j, score = hit[0], min(score, hit[1])                   # сдвигаемся; итоговая оценка — по худшему слову
        if ok:                                                      # вся фраза совпала
            out.append((i, j, score))                               # сохраняем вхождение
    return out                                                      # все вхождения


def _any_near(tokens, phrases, lo, hi, thr, same_sent, same_clause=None, breakers=frozenset()):
    # есть ли одна из фраз в окне слов [lo, hi]. Возвращает найденную фразу (текстом) или None
    lo, hi = max(lo, 0), min(hi, len(tokens) - 1)                   # обрезаем окно по границам текста
    window = [t for t in tokens[lo : hi + 1] if t.sent == same_sent and (same_clause is None or t.clause == same_clause)]  # слова окна в нужном предложении/части
    if not window:                                                  # окно пустое
        return None                                                 # ничего нет
    for plem in phrases:                                            # каждая фраза-модификатор
        for i, j, _ in _match(window, plem, thr, 0):                # ищем её в окне (без пропусков слов)
            return " ".join(t.raw for t in window[i : j + 1])       # нашли — возвращаем текст
    return None                                                     # не нашли


def _has_breaker(tokens, a, b, breakers):
    # есть ли между словами a и b слово-разрыватель («но», «однако»)
    return any(t.raw in breakers for t in tokens[min(a, b) + 1 : max(a, b)])   # проверяем промежуток


def _sizes(low, a, b):
    # все размеры в диапазоне символов [a, b): [(позиция, размер_в_мм)]
    res = []                                                        # результат
    for m in SIZE_RE.finditer(low, a, b):                           # ищем размеры в диапазоне
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", m.group(1))]   # все измерения
        mm = max(nums) * (10 if m.group(2) == "см" else 1)          # максимальное, сантиметры -> миллиметры
        res.append((m.start(), round(mm, 1)))                       # сохраняем позицию и размер
    return res                                                      # список размеров


# главная функция
def extract(text: str, ml_hook=None) -> ExtractionResult:
    # найти находки в тексте протокола. ml_hook — необязательная обученная модель (второй слой)
    fcfg, rcfg = config.findings(), config.routes()                 # актуальные настройки
    D = _compile(fcfg, rcfg)                                        # скомпилированный словарь
    M = D["matching"]                                               # параметры поиска
    thr = M.get("fuzzy_threshold", 85)                              # порог опечаток
    gap = M.get("max_gap", 1)                                       # допуск лишних слов
    nwin = M.get("negation_window", 4)                              # окно отрицания
    mwin = M.get("modifier_window", 5)                              # окно модификаторов
    low = norm_text(text)                                           # нормализованный текст (та же длина)
    _OPTS["fuzzy_only_unknown"] = M.get("fuzzy_only_unknown_words", True)
    tokens = tokenize(text, M.get("join_colon_lines", False))       # слова (опция склейки «Ключ:\nзначение»)
    if not tokens:                                                  # пустой протокол
        return ExtractionResult([], [], [], [], fcfg.get("version", "?"), "Пустой текст")   # нечего анализировать

    sents = {}                                                      # номер предложения -> его слова
    for t in tokens:                                                # раскладываем слова по предложениям
        sents.setdefault(t.sent, []).append(t)                      # добавляем слово в своё предложение

    def sent_text(sid):                                             # исходный текст предложения
        ts = sents[sid]                                             # слова предложения
        return text[ts[0].start : ts[-1].end]                       # от первого до последнего символа

    # шаг 2. Предложения-рекомендации и рекомендованные специалисты
    # исключаем не всё предложение, а только часть от слова «рекомендовано» до конца предложения:
    # в «Полип эндометрия 12 мм, рекомендована консультация гинеколога» полип остаётся находкой
    rec_from = {}                                                   # предложение -> индекс первого слова рекомендации
    for sid, ts in sents.items():
        for p in D["mods"].get("recommendation_markers", []):
            for i, _, _ in _match(ts, p, thr, 0):
                rec_from[sid] = min(rec_from.get(sid, 10**9), ts[i].idx)
    rec_sents = set(rec_from)                                       # для второго слоя (модель)

    def in_recommendation(tok_idx):
        t = tokens[tok_idx]
        return t.sent in rec_from and t.idx >= rec_from[t.sent]

    recommended = []
    for rid, aliases in D["aliases"].items():
        for sid, start in rec_from.items():
            part = [t for t in sents[sid] if t.idx >= start]        # только сама рекомендация
            if any(_match(part, a, thr, gap) for a in aliases):
                recommended.append(rid)
                break

    # упоминания органов (для правил «ЧТО + ГДЕ»)
    loc_mentions = []                                               # [(орган, начало, конец)]
    for loc, variants in D["locations"].items():                    # каждый орган
        for plem in variants:                                       # каждый вариант названия
            for i, j, _ in _match(tokens, plem, thr, gap):          # все вхождения
                loc_mentions.append((loc, i, j))                    # сохраняем

    # орган из заголовка раздела: протоколы УЗИ пишутся блоками
    # «Желчный пузырь: размеры… .\nВ просвете полип 12 мм.» — во втором предложении органа нет,
    # поэтому запоминаем орган, с которого начинается строка или предложение (первые 3 слова)
    section_of = {}                                                 # номер предложения -> (орган, a, b) раздела
    cur = None
    starts = {}
    used_locs = {w for fd in D["findings"].values() for w in (fd.get("where") or [])}   # органы, к которым привязаны находки
    for loc, a, b in sorted(loc_mentions, key=lambda m: m[1]):
        st_ = sents[tokens[a].sent]
        if tokens[a].idx - st_[0].idx >= 3:                         # орган не в начале предложения
            continue
        if loc not in used_locs and not (a == st_[0].idx and b == st_[-1].idx):
            # «справочный» орган (лимфоузлы, маточная труба…) открывает раздел, только если строка —
            # чистый заголовок («ЛИМФОУЗЛЫ»). Иначе «Фаллопиева труба: не визуализируется» внутри раздела яичника
            # обрывала раздел, и следующая строка «Параовариально … образование» теряла орган
            continue
        starts.setdefault(tokens[a].sent, (loc, a, b))
    for sid in sorted(sents):
        line_start = sents[sid][0].start == 0 or "\n" in text[max(0, sents[sid][0].start - 3):sents[sid][0].start]
        if sid in starts:
            cur = starts[sid]
        elif line_start and sid > 0 and "\n\n" in text[sents[sid - 1][-1].end:sents[sid][0].start]:
            cur = None                                              # пустая строка — раздел закончился
        section_of[sid] = cur

    def nearest_location(i, j):
        # ближайший орган к словам [i, j] в том же предложении. Своя часть предложения важнее
        best, best_d = None, 10**9                                  # лучший кандидат и его «расстояние»
        for loc, a, b in loc_mentions:                              # все упоминания органов
            if tokens[a].sent != tokens[i].sent:                    # другой предложение — не подходит
                continue                                            # пропускаем
            d = min(abs(a - j), abs(i - b))                         # расстояние в словах
            if tokens[a].clause != tokens[i].clause:                # орган в другой части предложения (за запятой)
                d += 10                                             # штраф: свой фрагмент приоритетнее
            if d < best_d:                                          # нашли ближе
                best, best_d = (loc, a, b), d                       # запоминаем
        if best is None and section_of.get(tokens[i].sent):         # в предложении органа нет — берём орган раздела
            return section_of[tokens[i].sent] + (True,)
        return best + (False,) if best else None                    # (орган, начало, конец, из_заголовка)

    heads = [plem[0] for f in D["findings"].values() for _, plem in f["terms_l"] + f["what_l"] if plem]   # первые слова всех находок

    def list_negation(anchor):
        # отрицание, «наследуемое» через перечисление: «Без признаков полипа, гиперплазии» —
        k = anchor                                                  # идём назад по частям предложения
        while True:                                                 # пока продолжается перечисление
            ta = tokens[k]                                          # текущее слово-начало
            if k == 0 or tokens[k - 1].clause == ta.clause or tokens[k - 1].sent != ta.sent:   # не начало части или начало предложения
                return None                                         # перечисления нет
            prev = [t for t in tokens if t.sent == ta.sent and t.clause == ta.clause - 1]   # предыдущая часть предложения
            if not prev:                                            # её нет
                return None                                         # перечисления нет
            for plem in D["mods"].get("negation_before", []):       # ищем отрицание в предыдущей части
                for ci, cj, _ in _match(prev, plem, thr, 0):        # нашли отрицание
                    if any(_tok_sim(t, h, thr) for t in prev[cj + 1 :] for h in heads):   # и после него идёт находка
                        return " ".join(t.raw for t in prev[ci : cj + 1])   # значит, это отрицательное перечисление
            if not any(_tok_sim(prev[0], h, thr) for h in heads):   # предыдущая часть тоже не начинается с находки
                return None                                         # перечисление закончилось
            k = prev[0].idx                                         # шагаем ещё на часть назад

    def negation_after(j):
        # отрицание после находки — только в своей части предложения или через перечисление находок
        tj = tokens[j]
        own = [t for t in sents[tj.sent] if t.clause == tj.clause and t.idx > j][:M.get("negation_after_window", nwin)]   # настраивается отдельно
        for plem in D["mods"].get("negation_after", []):
            for a, b, _ in _match(own, plem, thr, 0):
                if not _has_breaker(tokens, j, own[b].idx, D["breakers"]):
                    return " ".join(t.raw for t in own[a:b + 1])
        mine = [t for t in sents[tj.sent] if t.clause == tj.clause]
        nxt = [t for t in sents[tj.sent] if t.clause == tj.clause + 1]
        if mine and nxt and ":" in low[mine[-1].end:nxt[0].start]:
            # форма «ключ: значение» — «Объёмные образования: не выявлены», «Наличие образований: нет»
            if nxt[0].raw in ("не", "нет"):
                return " ".join(t.raw for t in nxt[:3])
            for plem in D["mods"].get("negation_after", []):
                for a, b, _ in _match(nxt[:M.get("negation_after_window", nwin)], plem, thr, 0):
                    return " ".join(t.raw for t in nxt[a:b + 1])
        c = tj.clause + 1
        while True:                                                 # следующие части: продолжаем, только пока это перечисление находок
            part = [t for t in sents[tj.sent] if t.clause == c]
            if not part or any(t.raw in D["breakers"] for t in part):
                return None
            for plem in D["mods"].get("negation_after", []):
                for a, b, _ in _match(part, plem, thr, 0):
                    if a == 0 or any(_tok_sim(part[0], h, thr) for h in heads):   # «…, не выявлено» / «…, полипов не выявлено»
                        return " ".join(t.raw for t in part[a:b + 1])
            if not any(_tok_sim(part[0], h, thr) for h in heads):
                return None
            c += 1

    def span_size(i, j, need):
        # размер для фрагмента [i, j]: сначала своя часть предложения, затем (если размер важен) дальше по предложению
        t0_, t1_ = tokens[i], tokens[j]
        sid_ = t0_.sent
        cl_ = [t for t in sents[sid_] if t.clause in (t0_.clause, t1_.clause)]
        sz_ = _sizes(low, cl_[0].start, cl_[-1].end + 3)
        if not sz_ and need:
            sz_ = [x for x in _sizes(low, sents[sid_][0].start, sents[sid_][-1].end + 3) if x[0] >= t1_.end]
        after_ = [mm for pos, mm in sz_ if pos >= t0_.start]
        return (after_[0] if after_ else sz_[-1][1]) if sz_ else None

    accepted, suppressed = {}, []                                   # итоговые находки и отклонённые кандидаты
    in_rec_flag = [False]
    cand_log = []                                                   # ВСЕ кандидаты правил (принятые и отклонённые) — вход второго контура
    seen_spans = set()                                              # чтобы одно место текста не засчитать дважды

    for fid, f in D["findings"].items():                            # шаг 3. Каждая находка словаря
        cands = []                                                  # кандидаты: (нач, кон, оценка, правило)
        for term, plem in f["terms_l"]:                             # а) самодостаточные термины
            for i, j, s in _match(tokens, plem, thr, gap):          # все вхождения
                cands.append((i, j, s * 0.95, f"{fid}.terms['{term}']", s, i))   # базовая уверенность 0.95; отрицание ищем перед термином
        for what, plem in f["what_l"]:                              # б) «ЧТО + ГДЕ»
            for i, j, s in _match(tokens, plem, thr, gap):          # все вхождения «что»
                loc = nearest_location(i, j)                        # ближайший орган
                if loc and loc[0] in f.get("where", []):            # орган подходит этой находке
                    if loc[3]:                                      # орган из заголовка раздела — фрагмент только «что»
                        cands.append((i, j, s * 0.8, f"{fid}.what['{what}']+section[{loc[0]}]", s, i))
                    else:
                        a, b = min(i, loc[1]), max(j, loc[2])       # общий фрагмент «что + где»
                        cands.append((a, b, s * 0.9, f"{fid}.what['{what}']+where[{loc[0]}]", s, i))

        for i, j, score, rule, raw_s, anchor in sorted(cands, key=lambda c: -c[2]):   # шаг 4. Проверяем кандидатов (сильные первыми)
            if (fid, i, j) in seen_spans or any(fid == s[0] and not (j < s[1] or i > s[2]) for s in seen_spans):  # пересекается с уже принятым
                continue                                            # дубль — пропускаем
            t0, t1 = tokens[i], tokens[j]                           # первое и последнее слово фрагмента
            sid = t0.sent                                           # номер предложения
            quote = text[t0.start : t1.end]                         # фрагмент текста
            flags, conf = [], score                                 # пометки и уверенность
            attrs = {}

            def _sup(reason, _fid=fid, _f=f, _q=quote, _sid=sid, _i=i, _j=j, _rule=rule, _anchor=anchor):
                # отклонить кандидата и записать его в журнал кандидатов (для второго контура)
                suppressed.append(Suppressed(_fid, _f["name"], _q, reason))
                cand_log.append({"id": _fid, "status": "suppressed", "reason": reason, "quote": _q, "sid": _sid,
                                 "i": _i, "j": _j, "anchor": _anchor, "rule": _rule, "flags": list(flags),
                                 "conf": round(conf, 3), "attrs": dict(attrs)})
            if raw_s < 1.0:                                         # совпадение нечёткое (опечатка)
                flags.append(f"нечёткое совпадение (возможна опечатка), сходство {raw_s:.2f}")   # пометка для админа

            if in_recommendation(anchor):                           # находка внутри самой рекомендации («рекомендовано исключить полип»)
                _sup("упоминание в рекомендации, а не в описании")
                in_rec_flag[0] = True
                continue

            # исключение действует только в своей части предложения:
            # в «Фолликулярная киста справа, слева образование яичника 35 мм» образование остаётся находкой
            ex = None
            for name_, pl_ in f["exclude_l"]:                       # исключения в своей части предложения
                for cl_ in {t0.clause, t1.clause, tokens[anchor].clause}:
                    hit_ = _any_near(tokens, [pl_], i - mwin, j + mwin, thr, sid, cl_)
                    if hit_:
                        ex = (hit_, name_)
                        break
                if ex:
                    break
            if not ex and f.get("exclude_sent_l"):                  # исключения на всё предложение («тромб» + «аорта»)
                s_tok = sents[sid]
                for name_, pl_ in f["exclude_sent_l"]:
                    hit_ = _any_near(tokens, [pl_], s_tok[0].idx, s_tok[-1].idx, thr, sid)
                    if hit_:
                        ex = (hit_, name_)
                        break
            if ex:                                                  # нашлось исключение
                src_ = "от администратора" if ex[1].endswith("(админ)") else "по словарю"
                _sup(f"исключение {src_}: «{ex[0]}»")
                continue                                            # не считаем

            ta = tokens[anchor]                                     # слово, перед которым ищем отрицание («что»)
            nbw = M.get("negation_before_window", nwin)             # окно «отрицание перед» настраивается отдельно
            neg = _any_near(tokens, D["mods"].get("negation_before", []), anchor - nbw, anchor - 1, thr, sid, ta.clause)   # отрицание перед (в той же части предложения)
            if not neg and anchor > 0 and tokens[anchor - 1].raw == "не" and tokens[anchor - 1].sent == ta.sent:
                neg = "не " + ta.raw                                # «аорта не расширена», «лимфоузлы не увеличены»
            if not neg:                                             # перед нет — может, это продолжение отрицательного перечисления
                neg = list_negation(anchor)                         # «Без полипа, гиперплазии...»
            if not neg:                                             # перед нет — смотрим после
                neg = negation_after(j)
            if neg and neg.split()[0].startswith(("исключит", "исключени")):
                # «исключить полип», «для исключения полипа» — это подозрение, а не отрицание
                # (у «исключить» и «исключён» общая начальная форма, поэтому разбираем по самому слову)
                flags.append(f"неуверенная формулировка: «{neg}»")
                conf *= 0.75
                neg = None
            unc = _any_near(tokens, D["mods"].get("uncertainty", []), i - mwin, j + mwin, thr, sid)   # неуверенность («не исключён»)
            if neg and not (unc and "исключ" in unc):               # отрицание есть и это не «не исключён»
                _sup(f"отрицание: «{neg}»")
                continue                                            # не считаем

            # анамнез — только в своей части предложения или в начале следующей («Полип, удалён в 2022»);
            # слова «вновь / рецидив / повторно» анамнез отменяют:
            # «Состояние после полипэктомии, в полости матки вновь полип 10 мм» — находка
            hist = None
            for cl_ in {t0.clause, t1.clause, tokens[anchor].clause}:
                hist = hist or _any_near(tokens, D["mods"].get("history", []), i - mwin, j + mwin, thr, sid, cl_)
            nxt = [t for t in sents[sid] if t.clause == t1.clause + 1][:3]
            if not hist and nxt:
                hist = _any_near(nxt, D["mods"].get("history", []), 0, len(nxt) - 1, thr, sid)
            if hist and _any_near(tokens, D["mods"].get("recurrence", []), i - mwin, j + mwin, thr, sid):
                hist = None
            if hist:                                                # находка в прошлом
                _sup(f"упоминание в анамнезе: «{hist}»")
                continue                                            # не считаем

            if unc:                                                 # формулировка неуверенная
                flags.append(f"неуверенная формулировка: «{unc}»")   # пометка для админа
                conf *= 0.75                                        # снижаем уверенность, но находку оставляем

            # атрибуты: размер (ближайший после фрагмента в предложении) и BI-RADS
            s_lo, s_hi = sents[sid][0].start, sents[sid][-1].end    # границы предложения в символах
            cl = [t for t in sents[sid] if t.clause in (t0.clause, t1.clause)]   # слова той же части предложения
            sizes = _sizes(low, cl[0].start, cl[-1].end + 3)        # сначала ищем размер в своей части (между запятыми)
            if not sizes and "size_mm_min" in (f.get("condition") or {}):   # не нашли, а размер важен для условия
                sizes = [x for x in _sizes(low, s_lo, s_hi + 3) if x[0] >= t1.end]   # тогда — дальше по предложению
            after = [mm for pos, mm in sizes if pos >= t0.start]    # размеры после начала фрагмента
            attrs.clear()                                           # извлечённые атрибуты
            if after or sizes:                                      # размер найден
                attrs["size_mm"] = after[0] if after else sizes[-1][1]   # ближайший после, иначе последний перед
            # BI-RADS берём из своего предложения (максимум, если несколько); из всего текста —
            # только если категория в протоколе одна (типично «Заключение: BI-RADS 4»), иначе «проверить»
            in_sent = [v for _, vs in birads_values(low, s_lo, s_hi + 3) for v in vs]   # все категории записи
            in_text = set()                                         # категории из «самостоятельных» предложений (заключение)
            mass_heads = [pl_[0] for _, pl_ in f["what_l"] if pl_] or heads   # «другое образование» — только объёмные слова ЭТОЙ находки
            for pos_, vals_ in birads_values(low):
                ms = next((t.sent for t in tokens if t.start >= pos_), None)
                if ms is None or ms == sid:
                    continue
                if any(_tok_sim(t, h, thr) for t in sents[ms] if t.raw not in ("bi", "rads", "би", "рэдс") for h in mass_heads):
                    in_text.add(-1)                                 # категория относится к другому образованию/стороне
                else:
                    in_text.update(vals_)
            if in_sent:
                attrs["birads"] = max(in_sent)
            elif in_text and -1 not in in_text and len(in_text) == 1:
                attrs["birads"] = next(iter(in_text))               # одна категория на всё исследование («справа 2, слева 2»)
            elif in_text:
                flags.append("в протоколе несколько категорий BI-RADS, сторона не определена — проверить")

            cond = f.get("condition") or {}                         # числовые условия находки
            if "stenosis_min" in cond:                              # степень стеноза, %: максимум в предложении; окклюзия = 100
                seg = low[s_lo:s_hi + 3]
                pcts = [float(x) for x in re.findall(r"(\d{1,3})(?:[.,]\d+)?\s*%", seg)]
                pcts += [float(b) for _, b in re.findall(r"(\d{1,3})\s*[-–]\s*(\d{1,3})\s*%", seg)]
                if "окклюз" in seg:
                    pcts.append(100.0)
                if pcts:
                    attrs["stenosis_pct"] = max(p for p in pcts if p <= 100)
            if "size_mm_min" in cond and "size_mm" not in attrs:    # размер из другого упоминания этой же находки
                sib_ = [span_size(ci, cj, True) for ci, cj, *_ in cands if (ci, cj) != (i, j) and not in_recommendation(ci)]
                sib_ = [x for x in sib_ if x is not None]
                if sib_:                                            # «Полип ЖП» в заключении без размера, а в описании —
                    attrs["size_mm"] = max(sib_)                    # «образование 3,6 мм — полип»: берём размер из описания
                    flags.append(f"размер взят из другого места протокола ({attrs['size_mm']} мм)")
            failed, missing = None, []                              # причина провала и недостающие атрибуты
            if "stenosis_min" in cond:
                if "stenosis_pct" not in attrs:
                    missing.append("степень стеноза")
                elif attrs["stenosis_pct"] < cond["stenosis_min"]:
                    failed = f"стеноз {attrs['stenosis_pct']:.0f}% < {cond['stenosis_min']}%"
            if "size_mm_min" in cond:                               # условие по размеру
                if "size_mm" not in attrs:                          # размера нет в тексте
                    missing.append("размер")                        # запоминаем
                elif attrs["size_mm"] < cond["size_mm_min"]:        # размер меньше порога
                    failed = f"размер {attrs['size_mm']} мм < {cond['size_mm_min']} мм"   # причина
            if "birads_min" in cond:                                # условие по BI-RADS
                if "birads" not in attrs:                           # категории нет
                    missing.append("BI-RADS")                       # запоминаем
                elif attrs["birads"] < cond["birads_min"]:          # категория ниже порога
                    failed = f"BI-RADS {attrs['birads']} < {cond['birads_min']}"   # причина
            if failed:                                              # условие не выполнено
                _sup(f"условие не выполнено: {failed}")
                continue                                            # не считаем
            if missing:                                             # атрибута нет в тексте
                if f.get("on_missing_attribute", "review") == "skip":   # по настройке — не считать
                    _sup(f"нет атрибута: {', '.join(missing)}")
                    continue                                        # не считаем
                flags.append(f"в тексте нет: {', '.join(missing)} — проверить вручную")   # иначе — показать с пометкой
                conf *= 0.7                                         # и снизить уверенность

            ev = Evidence(quote, sent_text(sid), t0.start, t1.end, f"{rule} @словарь v{fcfg.get('version')}", attrs, round(conf, 2), flags)   # доказательство
            fd = accepted.get(fid)                                  # уже есть такая находка?
            if not fd:                                              # первая — создаём
                fd = accepted[fid] = Finding(fid, f["name"], f["route"], f.get("priority", 3), bool(f.get("urgent")), 0.0, False)   # новая находка
            fd.evidence.append(ev)                                  # добавляем доказательство
            fd.confidence = max(fd.confidence, ev.confidence)       # итоговая уверенность — максимум
            fd.needs_review = fd.needs_review or bool(flags)        # есть пометки — нужна проверка
            seen_spans.add((fid, i, j))                             # фрагмент занят
            cand_log.append({"id": fid, "status": "accepted", "reason": "", "quote": quote, "sid": sid, "i": i, "j": j,
                             "anchor": anchor, "rule": rule, "flags": list(flags), "conf": round(conf, 3), "attrs": dict(attrs)})

    # «поглощение» общей находки более точной («патология эндометрия (полип)» -> только «полип»)
    for fid_, fd_ in list(accepted.items()):
        better = [b for b in (D["findings"][fid_].get("subsumed_by") or []) if b in accepted]
        if better:
            del accepted[fid_]
            suppressed.append(Suppressed(fid_, fd_.name, fd_.evidence[0].quote,
                                         f"поглощено более точной находкой: {accepted[better[0]].name}"))
            for c in cand_log:
                if c["id"] == fid_ and c["status"] == "accepted":
                    c["status"], c["reason"] = "suppressed", "поглощено более точной находкой"
    # фразы нормы (для объяснения)
    normal_quotes = []                                              # найденные «без патологии»
    for plem in D["mods"].get("normal_phrases", []):                # каждая фраза нормы
        for i, j, _ in _match(tokens, plem, thr, 0):                # вхождения
            normal_quotes.append(text[tokens[i].start : tokens[j].end])   # сохраняем текст

    uniq = {}                                                       # убираем дубли среди отклонённых
    for x in suppressed:                                            # каждый отклонённый кандидат
        if x.id not in accepted:                                    # если находка всё же принята по другому месту — не путаем админа
            uniq.setdefault((x.id, x.reason), x)                    # одна запись на (находка, причина)
    suppressed = list(uniq.values())                                # итоговый список

    found = sorted(accepted.values(), key=lambda x: (not x.urgent, x.priority, -x.confidence))   # сортировка: экстренные, приоритет, уверенность
    if found:                                                       # есть находки
        summary = f"Найдено находок: {len(found)}"                  # короткий итог
    elif suppressed:                                                # кандидаты были, но отклонены
        summary = "Находок нет: кандидаты отклонены (" + "; ".join(sorted({s.reason.split(':')[0] for s in suppressed})) + ")"   # почему
    elif normal_quotes:                                             # явная норма
        summary = "Находок нет: в протоколе описана норма"          # итог
    else:                                                           # ничего из словаря
        summary = "Находок нет: совпадений со словарём не найдено"  # итог
    sent_spans = []                                                 # предложения без части-рекомендации (вход детектора пропусков)
    for sid, ts in sents.items():
        keep = [t for t in ts if not in_recommendation(t.idx)]
        if keep:
            sent_spans.append((sid, keep[0].start, keep[-1].end))
    res = ExtractionResult(found, suppressed, recommended, normal_quotes, fcfg.get("version", "?"), summary, in_rec_flag[0],
                           candidates=cand_log, sentences=sent_spans)
    if ml_hook is not None:                                         # второй контур (модели): см. ml_layer.py
        ml_hook.apply(text, res, D["findings"])
    return res
