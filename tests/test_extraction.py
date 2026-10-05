# регрессия по найденным на ревью пропускам (FN) и ложным срабатываниям. Запуск: pytest -q
import pytest
from app.nlp.extractor import extract

CASES = [
    ("Полип эндометрия 12 мм, рекомендована консультация гинеколога.", "endometrial_polyp"),
    ("Полип желчного пузыря 12 мм, рекомендовано динамическое наблюдение.", "gallbladder_polyp"),
    ("Желчный пузырь: стенка 2 мм.\nВ просвете полип 12 мм.", "gallbladder_polyp"),
    ("Желчный пузырь обычных размеров.\nВ просвете конкремент 8 мм с акустической тенью.", "cholelithiasis"),
    ("Фолликулярная киста справа, слева в яичнике образование 35 мм.", "ovarian_mass"),
    ("Состояние после полипэктомии, в полости матки вновь полип 10 мм.", "endometrial_polyp"),
    ("Справа образование 8 мм, BI-RADS 2. Слева в молочной железе образование 15 мм.", "breast_birads"),
    ("Слева в молочной железе образование 15 мм.\nЗаключение: BI-RADS 4.", "breast_birads"),
    ("Полип желчного пузыря 12 мм, васкуляризации нет.", "gallbladder_polyp"),
    ("Палип желчного пузыря 12 мм.", "gallbladder_polyp"),
    ("Полип желчного пузыря 1,2 см.", "gallbladder_polyp"),
    ("Острый холецистит не исключен.", "acute_cholecystitis"),
]
NEGATIVE = [
    ("Рекомендовано: исключить полип эндометрия.", "endometrial_polyp"),
    ("Фолликулярное образование в правом яичнике 20 мм.", "ovarian_mass"),
    ("Полип эндометрия, удален в 2022 году.", "endometrial_polyp"),
    ("Конкрементов, полипов в желчном пузыре не выявлено.", "gallbladder_polyp"),
    ("Без признаков полипа, гиперплазии эндометрия.", "endometrial_hyperplasia"),
    ("Полип желчного пузыря 0,6 см.", "gallbladder_polyp"),
    ("Данных за ЖКБ нет.", "cholelithiasis"),
    ("Грыжа диска L5-S1.", "hernia"),
]


@pytest.mark.parametrize("text,fid", CASES)
def test_found(text, fid):
    assert fid in {f.id for f in extract(text).findings}


@pytest.mark.parametrize("text,fid", NEGATIVE)
def test_not_found(text, fid):
    assert fid not in {f.id for f in extract(text).findings}


def test_size_decimal_comma():
    f = next(f for f in extract("Полип желчного пузыря 1,2 см.").findings if f.id == "gallbladder_polyp")
    assert f.evidence[0].attributes["size_mm"] == 12.0
