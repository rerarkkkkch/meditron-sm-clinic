# словарь: находки и правила разбора, добавленные по реальным и открытым протоколам
import pytest
from app.nlp.extractor import extract

FOUND = [
    ("Заключение: УЗ-признаки ДГПЖ.", "bph"),
    ("Эхографические признаки аденомы предстательной железы.", "bph"),
    ("Заключение: Внутренний эндометриоз.", "adenomyosis"),
    ("Несостоятельность ствола БПВ слева с варикозной трансформацией притоков.", "varicose_veins"),
    ("Недостаточность сафено-феморального соустья, варикозно расширенные надфасциальные притоки.", "varicose_veins"),
    ("Пролонгированный стеноз 60% левой подколенной артерии.", "pad_atherosclerosis"),
    ("Окклюзия левой ЗББА.", "pad_atherosclerosis"),
    ("Окклюзивный тромбоз подколенной вены справа.", "venous_thrombosis"),
    ("Эхографические признаки простой кисты правого яичника.", "ovarian_mass"),
    ("Эхографические признаки хронического калькулёзного холецистита.", "cholelithiasis"),
    ("УЗ-признаки полипа эндометрия, полипа шейки матки.", "cervical_polyp"),
    ("Опущение матки 4 степени.", "uterine_prolapse"),
    ("Гиперэхогенная структура в полости матки (исключить фиброзный полип эндометрия).", "endometrial_polyp"),
]
NOT_FOUND = [
    ("Полип эндометрия исключен.", "endometrial_polyp"),
    ("Данных за флеботромбоз и тромбофлебит на момент исследования не получено.", "venous_thrombosis"),
    ("Сафено-феморальное соустье не расширено, без признаков тромбоза.", "venous_thrombosis"),
    ("ПБА с обеих сторон проходима, со стенозами до 30%.", "pad_atherosclerosis"),
    ("Данные за нестенозирующий атеросклероз.", "pad_atherosclerosis"),
    ("Варикозное расширение вен малого таза не лоцировано.", "varicose_veins"),
    ("ЗАКЛЮЧЕНИЕ: Внутренний эндометриоз.", "ovarian_mass"),
    ("Эхографические признаки простой кисты правого яичника.", "bph"),
]


@pytest.mark.parametrize("text,fid", FOUND)
def test_found(text, fid):
    assert fid in {f.id for f in extract(text, None).findings}


@pytest.mark.parametrize("text,fid", NOT_FOUND)
def test_not_found(text, fid):
    assert fid not in {f.id for f in extract(text, None).findings}


def test_thrombosis_is_urgent():
    f = next(f for f in extract("Окклюзивный тромбоз суральных вен.", None).findings if f.id == "venous_thrombosis")
    assert f.urgent
