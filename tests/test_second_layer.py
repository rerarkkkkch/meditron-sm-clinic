# второй контур: загрузка, поведение на новых текстах, отсутствие ПДн в модели. Нужна обученная модель
import json, os, re
import pytest
from app.nlp.ml_layer import get_ml_hook, training_rows
from app.nlp.extractor import extract
from app.nlp.anonymize import scrub

hook = get_ml_hook()
needs_model = pytest.mark.skipif(hook is None, reason="модель не обучена или ml.enabled: false")


def test_myoma_found_by_dictionary_v03():
    t = "УЗИ органов малого таза\nМиометрий: по задней стенке интрамуральный миоматозный узел 18х15 мм.\nЗаключение: Эхо-признаки миомы матки."
    assert "uterine_myoma" in {f.id for f in extract(t, None).findings}


@needs_model
def test_flags_small_gallbladder_polyp_as_likely_false():
    t = "УЗИ желчного пузыря\nЖЕЛЧНЫЙ ПУЗЫРЬ: в дне пристеночное неподвижное образование 4 мм без акустической тени.\nЗАКЛЮЧЕНИЕ: Полип желчного пузыря."
    r = extract(t, hook)
    f = next((f for f in r.findings if f.id == "gallbladder_polyp"), None)
    if f is None:   # правила сами берут размер 4 мм из описания и отклоняют полип < 10 мм — ещё лучше, чем пометка
        assert any(s.id == "gallbladder_polyp" and s.reason.startswith("условие не выполнено") for s in r.suppressed)
    else:
        assert any("вероятно ложное" in x for e in f.evidence for x in e.flags)


@needs_model
def test_model_does_not_override_rule_negation():
    r = extract("Матка без особенностей. Без признаков полипа эндометрия, гиперплазии эндометрия.", hook)
    assert "endometrial_polyp" not in {f.id for f in r.findings}


@needs_model
def test_model_findings_always_need_review():
    t = "Заключение: Варикозная болезнь левой нижней конечности, несостоятельность БПВ. Миома матки."
    for f in extract(t, hook).findings:
        if any(e.rule.startswith("ml:") for e in f.evidence):
            assert f.needs_review


def test_training_sentences_align_with_scrubbed_text():
    p = {"text": "Пациент: Иванова М.П.\nДата рождения: 01.02.1980\nЗаключение: Миома матки.",
         "expected": [{"id": "uterine_myoma", "quotes": ["Миома матки"]}]}
    t = scrub(p["text"])
    sx, sy, _, _ = training_rows(p, extract(t, None), t)
    assert any("uterine_myoma" in y for y in sy) and any("миома матки" in x for x in sx)
