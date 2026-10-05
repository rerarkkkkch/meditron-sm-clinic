# нормализация текста: токены, леммы (начальные формы слов), предложения, части предложений
import re                                   # регулярные выражения
from dataclasses import dataclass           # удобные классы-структуры
from functools import lru_cache             # кэш для ускорения лемматизации

import os                                   # флаг разрешения работы без морфологии

try:                                        # пробуем подключить морфологический анализатор
    import pymorphy3                        # офлайн-лемматизатор русского языка
    _MORPH = pymorphy3.MorphAnalyzer()      # создаём один раз (загрузка словаря ~1 сек)
except Exception as _e:                     # без pymorphy3 не стартуем
    # без морфологии качество падает резко (9 ложных «образований яичника» на слове «эндометрий»),
    # поэтому стеммер — только при явном разрешении CM_ALLOW_NO_MORPH=1
    if os.environ.get("CM_ALLOW_NO_MORPH") != "1":
        raise RuntimeError(f"pymorphy3 недоступен ({_e}). Установите pymorphy3 и pymorphy3-dicts-ru "
                           "или явно задайте CM_ALLOW_NO_MORPH=1 (только для отладки).")
    _MORPH = None

try:                                        # пробуем подключить быстрый нечёткий поиск
    from rapidfuzz.fuzz import ratio as _ratio      # сходство строк 0..100
    from rapidfuzz.distance import Levenshtein as _lev
    from rapidfuzz.distance import OSA as _osa
    _edits = _lev.distance                  # число правок между строками
    _swaps = _osa.distance                  # то же, но перестановка двух соседних букв = 1 правка («пзуыря» -> «пузыря»)
except Exception:                           # запасной вариант из стандартной библиотеки
    from difflib import SequenceMatcher     # медленнее, но без зависимостей

    def _ratio(a, b):                       # та же сигнатура, что у rapidfuzz
        return 100 * SequenceMatcher(None, a, b).ratio()   # переводим 0..1 в 0..100

    def _edits(a, b):                       # расстояние Левенштейна (короткие слова — дёшево)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def _swaps(a, b):                       # запасной вариант: без учёта перестановок
        return _edits(a, b)

WORD_RE = re.compile(r"[a-zа-я0-9]+")       # слово = подряд идущие буквы/цифры (дефис делит слово на части)
SENT_BREAK_RE = re.compile(r"[.!?;]\s|\n")  # граница предложения: точка/!/?/; с пробелом после, или перевод строки
CLAUSE_BREAK_RE = re.compile(r"[,:()]|\s[-–—]\s")  # граница части предложения: запятая, двоеточие, скобка, тире

# простые окончания для запасного стемминга (если нет pymorphy3)
_ENDINGS = sorted("ами ями ого его ому ему ыми ими ой ей ий ый ая яя ое ее ые ие ов ев ах ях ом ем ую юю а я о е ы и у ю ь".split(), key=len, reverse=True)


def norm_text(text: str) -> str:
    # нижний регистр и ё->е. Длина строки НЕ меняется — позиции символов совпадают с оригиналом
    return text.lower().replace("ё", "е")   # посимвольная замена, длина сохраняется (важно для подсветки)


@lru_cache(maxsize=100_000)                 # одно и то же слово лемматизируем один раз
def lemma(word: str) -> str:
    # начальная форма слова: «полипов» -> «полип». Латиница и числа не меняются
    w = norm_text(word)                     # приводим к нижнему регистру, ё->е
    if not re.search("[а-я]", w):           # не кириллица (BI-RADS, числа) — оставляем как есть
        return w                            # возвращаем без изменений
    if _MORPH is not None:                  # если есть морфология
        return norm_text(_MORPH.parse(w)[0].normal_form)   # берём самый вероятный разбор и его лемму
    return _stem(w)                         # иначе — упрощённый стемминг


@lru_cache(maxsize=100_000)                 # кэш вариантов
def lemma_variants(word: str) -> frozenset:
    # ВСЕ возможные начальные формы слова. Нужно для омонимов: «железы» -> {железа, железо}
    w = norm_text(word)                     # нижний регистр
    if _MORPH is None or not re.search("[а-я]", w):   # нет морфологии или не кириллица
        return frozenset({lemma(w)})        # один вариант
    known = {norm_text(p.normal_form) for p in _MORPH.parse(w)[:4] if p.is_known}   # разборы только по словарю (без догадок)
    return frozenset(known or {lemma(w)})   # незнакомое слово (термин, опечатка) — одна, самая вероятная форма


@lru_cache(maxsize=100_000)
def is_known_word(word: str) -> bool:
    # это обычное слово русского языка из словаря pymorphy3 («простой», «точка», «узел»)?
    w = norm_text(word)
    if _MORPH is None or not re.search("[а-я]", w):
        return False                        # без морфологии не знаем — нечёткость разрешена (старое поведение)
    return any(p.is_known for p in _MORPH.parse(w)[:4])


_SNOW = None


def _stem(w: str) -> str:
    # запасной стемминг без словаря (только при CM_ALLOW_NO_MORPH=1): Snowball, а если его нет — отрезаем окончание
    global _SNOW
    if _SNOW is None:
        try:
            from app.nlp.snowball_ru import RussianStemmer
            _SNOW = RussianStemmer()
        except Exception:
            _SNOW = False
    if _SNOW:
        return _SNOW.stem(w)
    for end in _ENDINGS:                    # запасной путь: отрезаем окончание
        if w.endswith(end) and len(w) - len(end) >= 4:     # но оставляем основу не короче 4 букв
            return w[: -len(end)]           # основа слова
    return w                                # короткое слово — без изменений


def similar(a: str, b: str, threshold: int = 85) -> float:
    # похожесть двух слов 0..1 с учётом опечаток. Короткие слова (<5 букв) — только точное совпадение
    if a == b:                              # точное совпадение
        return 1.0                          # максимальная уверенность
    if len(a) < 5 or len(b) < 5:            # «узел», «жкб», «нет» — опечатки не допускаем (иначе ложные срабатывания)
        return 0.0                          # не совпало
    if abs(len(a) - len(b)) > 1:            # опечатка меняет длину максимум на 1 букву; иначе это разные слова («эндометрий» ≠ «эндометриома»)
        return 0.0                          # не совпало
    if _swaps(a, b) == 1:                   # одна опечатка, в т.ч. перестановка соседних букв
        return max(0.85, _ratio(a, b) / 100)
    if min(len(a), len(b)) >= 6 and _swaps(a[:-1], b[:-1]) == 1:   # опечатка + другое окончание («пзуыря» ~ «пузырь»)
        return 0.85
    if min(len(a), len(b)) <= 6:            # у слов из 5–6 букв одна замена = 80–83% < порога 85,
        # поэтому «палип», «кисто» не ловились никогда. Для коротких слов допускаем ровно одну правку
        return 0.85 if _edits(a, b) == 1 else 0.0
    r = _ratio(a, b)                        # процент сходства строк
    if len(a) >= 7 and len(b) >= 7:         # у длинных слов сравниваем ещё и без последних 2 букв (окончание)
        r = max(r, _ratio(a[:-2], b[:-2]))  # «эндомерия» ~ «эндометрий»: окончания разные, основа похожа
    return r / 100 if r >= threshold else 0.0   # ниже порога — считаем, что не совпало


@dataclass
class Token:
    # одно слово текста с позицией и служебной информацией
    raw: str        # слово как в тексте (в нижнем регистре)
    lemma: str      # начальная форма (самая вероятная)
    alts: frozenset # все возможные начальные формы (для омонимов)
    start: int      # позиция начала в тексте (символ)
    end: int        # позиция конца в тексте (символ)
    idx: int        # порядковый номер слова во всём тексте
    sent: int       # номер предложения
    clause: int     # номер части предложения (между запятыми)
    known: bool = False   # обычное слово русского языка (нечётко не сравниваем)
    upper: bool = False   # написано ЗАГЛАВНЫМИ (для аббревиатур: «ОБА» — артерия, «обеих» — нет)


COLON_LINE_RE = re.compile(r":[ \t]*\r?\n[ \t]*$")   # «Ключ:» в конце строки, значение — на следующей строке


def tokenize(text: str, join_colon_lines: bool = False) -> list:
    # разбить текст на слова, проставить номера предложений и частей предложения
    low = norm_text(text)                   # нормализованная копия (той же длины)
    tokens = []                             # результат
    sent, clause = 0, 0                     # счётчики предложений и частей
    prev_end = 0                            # конец предыдущего слова
    for m in WORD_RE.finditer(low):         # идём по всем словам
        gap = low[prev_end : m.start()]     # текст между предыдущим и текущим словом (знаки препинания)
        is_break = bool(SENT_BREAK_RE.search(gap))      # похоже на конец предложения («5.2» не режется: после точки нет пробела)
        if is_break and join_colon_lines and COLON_LINE_RE.search(gap) and gap.count("\n") == 1:
            is_break = False                            # «Ключ:\nзначение» — одно предложение (новая часть)
        if is_break and "\n" not in gap and not re.search(r"[!?;]", gap) and text[m.start()].islower():   # «6 мес. или ...» — сокращение, а не конец
            is_break = False                            # следующее слово с маленькой буквы — предложение продолжается
        if tokens and is_break:                         # конец предложения
            sent += 1                       # новое предложение
            clause += 1                     # и новая часть
        elif tokens and gap == "," and tokens[-1].raw.isdigit() and m.group().isdigit():
            pass                            # «1,2 см» — десятичная запятая, а не граница части предложения
        elif tokens and CLAUSE_BREAK_RE.search(gap):     # в промежутке запятая/двоеточие
            clause += 1                     # новая часть того же предложения
        tokens.append(Token(m.group(), lemma(m.group()), lemma_variants(m.group()), m.start(), m.end(), len(tokens), sent, clause,
                            is_known_word(m.group()), text[m.start():m.end()].isupper() and len(m.group()) > 1))   # сохраняем слово
        prev_end = m.end()                  # запоминаем конец слова
    return tokens                           # список токенов


ABBR = "\x00abbr:"                          # метка аббревиатуры внутри набора форм слова словаря
STOP = {"в", "во", "на", "с", "со", "и", "по", "к", "у", "о", "об", "из", "от", "до"}   # служебные слова


def phrase_lemmas(phrase: str, drop_stop: bool = True) -> list:
    # фразу из словаря превратить в список слов; каждое слово = множество его возможных начальных форм
    raw_words = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", phrase)            # слова фразы с исходным регистром
    words = [norm_text(w) for w in raw_words]
    abbr = {norm_text(w) for w in raw_words if w.isupper() and 1 < len(w) <= 5}   # «ОБА», «ЖП» — аббревиатуры
    kept = [w for w in words if not (drop_stop and w in STOP)] or words   # без предлогов (если что-то осталось)
    if abbr:                                # аббревиатура совпадает только с точным написанием, без начальных форм:
        return [frozenset({ABBR + w}) if w in abbr else (lemma_variants(w) | {lemma(w), w}) for w in kept]
    if _MORPH is None:                      # запасной режим: только основы (сырое слово + обрезка окончаний давали ложные совпадения)
        return [frozenset({lemma(w)}) for w in kept]
    return [lemma_variants(w) | {lemma(w), w} for w in kept]           # варианты форм + само слово
