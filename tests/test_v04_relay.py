"""Passport Relay wiring: routing, visas, conversion, and the gate against a door policy."""
from __future__ import annotations

from bench.relay import pc_core as pc
from bench.relay.relay import door_agent, run_scenario, silent_agent, summarize, to_pc
from sozograph import Passport


def test_the_gate_turns_a_door_policy_into_a_wall_on_injection():
    suite = pc.make_suite(6, 1000)
    agents = {"chat": silent_agent, "inbox": door_agent, "web": door_agent}
    off = summarize([run_scenario(sc, agents, gate=False) for sc in suite])
    on = summarize([run_scenario(sc, agents, gate=True) for sc in suite])
    assert on["categories"]["reject/injection"] == 1.0
    assert off["categories"]["reject/injection"] < 1.0
    assert on["stats"]["third_party_protected_beliefs"] == 0
    assert off["stats"]["third_party_protected_beliefs"] > 0
    assert on["stats"]["leaks"] == 0


def test_conversion_keeps_history_and_shelf_life():
    p = Passport.issue()
    p.propose([{"attribute": "home_city", "value": "Gweru", "valid_from": "2027-01-01"}])
    p.propose([{"op": "supersede", "attribute": "home_city", "value": "Mutare", "valid_from": "2027-03-14"}])
    p.propose([{"attribute": "health", "value": "twisted ankle", "kind": "symptom", "valid_from": "2027-01-12"}])
    out = to_pc(p)
    cities = {f["value"]: f for f in out["facts"] if f["attribute"] == "home_city"}
    assert cities["Gweru"]["valid_until"] == "2027-03-13" and cities["Mutare"]["valid_from"] == "2027-03-14"
    ankle = next(f for f in out["facts"] if f["attribute"] == "health")
    assert ankle["valid_until"] == "2027-01-17"
