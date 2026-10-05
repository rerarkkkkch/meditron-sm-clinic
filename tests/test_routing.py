# регрессия по маршрутам: идемпотентность, откат, машина состояний, планировщик, SLA
import threading
import pytest
from app import routing, store
from app.nlp import anonymize

store.init()
GB = "Полип желчного пузыря 12 мм."


def ev(**k):
    return routing.handle_event(k)


def route_of(pid):
    return store.q("SELECT * FROM routes WHERE protocol_id=?", (pid,))


def test_parallel_duplicate_event_creates_one_route():
    res = []
    def go():
        res.append(ev(event_id="T-E1", type="protocol.signed", protocol_id="T-P1", patient_id="1", text=GB).get("status", "ok"))
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert sorted(res) == ["duplicate"] * 7 + ["ok"]
    assert len(route_of("T-P1")) == 1


def test_bad_event_is_not_marked_processed():
    with pytest.raises(ValueError):
        ev(event_id="T-E2", type="protocol.signed", protocol_id="T-P2", patient_id="2", text=None)
    assert ev(event_id="T-E2", type="protocol.signed", protocol_id="T-P2", patient_id="2", text=GB).get("status") != "duplicate"


def test_empty_patient_rejected():
    with pytest.raises(ValueError):
        anonymize.pseudo_id(None)


def test_pseudo_id_no_collisions_1m():
    assert len({anonymize.pseudo_id(str(i)) for i in range(200_000)}) == 200_000


def test_state_machine_blocks_visit_before_verify():
    ev(event_id="T-E3", type="protocol.signed", protocol_id="T-P3", patient_id="3", text=GB)
    with pytest.raises(ValueError):
        routing.transition(route_of("T-P3")[0]["id"], "visit")


def test_no_return_message_after_visit():
    ev(event_id="T-E4", type="protocol.signed", protocol_id="T-P4", patient_id="4", text=GB)
    rid = route_of("T-P4")[0]["id"]
    for a in ("verify", "book", "no_show", "book", "visit"):
        routing.transition(rid, a, slot="s")
    store.advance(2); routing.tick()
    assert not store.q("SELECT 1 FROM notifications WHERE route_id=? AND status='sent' AND text LIKE 'Сегодня не состоялась%'", (rid,))


def test_downtime_does_not_burst_patient_messages():
    ev(event_id="T-E5", type="protocol.signed", protocol_id="T-P5", patient_id="5", text=GB)
    rid = route_of("T-P5")[0]["id"]
    routing.transition(rid, "verify")
    store.advance(24 * 15); routing.tick()
    batches = store.q("SELECT DISTINCT batch FROM notifications WHERE route_id=? AND kind='patient' AND status='sent'", (rid,))
    assert len(batches) <= 2                      # первое уведомление + одно (последнее созревшее) напоминание


def test_urgent_reescalates_until_ack():
    ev(event_id="T-E6", type="protocol.signed", protocol_id="T-P6", patient_id="6", text="Острый холецистит.")
    rid = route_of("T-P6")[0]["id"]
    store.advance(0.5)
    assert (rid, "urgent_reescalated") in routing.tick()["steps"]
    routing.transition(rid, "ack")
    store.advance(0.5)
    assert (rid, "urgent_reescalated") not in routing.tick()["steps"]


def test_rejected_route_reopens_when_correction_changes_findings():
    ev(event_id="T-E7", type="protocol.signed", protocol_id="T-P7", patient_id="7", text=GB)
    rid = route_of("T-P7")[0]["id"]
    routing.transition(rid, "reject")
    ev(event_id="T-E8", type="protocol.corrected", protocol_id="T-P7", patient_id="7", text="Полип желчного пузыря 14 мм.")
    assert store.route(rid)["status"] == "pending_verification"
