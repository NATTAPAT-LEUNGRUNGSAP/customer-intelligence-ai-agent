import pytest
from src.narrative_evidence import evidence_references, render_evidence_text


def test_references_keep_segment_and_metric_identity():
    refs = evidence_references({'segments': [{'name': 'Loyal (C1)', 'customers': 1381, 'monetary': 5024.99}]})
    text = render_evidence_text('กลุ่ม {{fact:f0}} มีข้อมูล {{fact:f1}} และ {{fact:f2}}', refs)
    assert '1381' in text and '5024.99' in text and 'customers' in text and 'monetary' in text


@pytest.mark.parametrize('text', ['Revenue 1381', '{{fact:f999}}', '{{fact:wrong}}', 'ลด 10%', '๑๐ คน'])
def test_unsupported_numbers_and_references_rejected(text):
    with pytest.raises(ValueError):
        render_evidence_text(text, evidence_references({'customers': 1381}))


def test_private_ids_and_nonfinite_values_not_exposed():
    refs = evidence_references({'customer_id': 42, 'customer_ids': [42], 'value': float('nan'), 'count': 3})
    assert list(refs.values()) == [{'path': 'count', 'value': 3}]


def test_qualitative_text_allowed():
    assert render_evidence_text('ควรทดลองแคมเปญก่อน', {}) == 'ควรทดลองแคมเปญก่อน'


def test_analyst_renders_reference_without_fallback(monkeypatch):
    import json
    from src import analyst
    monkeypatch.setattr(analyst, '_call_copy_model', lambda *a: json.dumps({
        'headline': 'สรุปลูกค้า', 'interpretation': 'ข้อมูลที่พบ {{fact:f0}}',
        'next_step': 'ทดลองแคมเปญและวัดผล'}))
    intent = analyst.AnalystIntent(action='segment_overview', objective='Retain valuable customers', language='Thai')
    answer, source, warnings = analyst._analyst_narrative(intent, {'customers': 1381}, 'Ollama', 'test')
    assert 'customers = 1381' in answer
    assert source == 'Ollama validated interpretation'
    assert warnings == []
