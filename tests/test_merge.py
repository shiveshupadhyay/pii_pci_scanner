"""Merge step: overlaps resolved by priority, then span length; blocked spans; dedupe."""

from scanner.detect.text_detector import PRIORITY, merge
from scanner.units import Candidate


def c(entity, start, end, conf=0.9, pr=None, value="x"):
    return Candidate(entity, value, start, end, conf, "t", (), PRIORITY.get(entity, 0) if pr is None else pr)


def test_card_beats_phone_on_same_digits():
    out = merge([c("PHONE", 0, 17), c("CARD_NUMBER", 0, 17)])
    assert [x.entity for x in out] == ["CARD_NUMBER"]


def test_longer_span_wins_within_entity():
    out = merge([c("PHONE", 3, 17), c("PHONE", 0, 17)])
    assert [(x.start, x.end) for x in out] == [(0, 17)]


def test_identical_candidates_deduplicated():
    out = merge([c("EMAIL", 5, 20), c("EMAIL", 5, 20, conf=0.5)])
    assert len(out) == 1 and out[0].confidence == 0.9


def test_non_overlapping_kept_in_order():
    out = merge([c("EMAIL", 30, 40), c("PHONE", 0, 10), c("PERSON_NAME", 12, 25)])
    assert [x.entity for x in out] == ["PHONE", "PERSON_NAME", "EMAIL"]


def test_blocked_regions():
    # everything inside a PEM body is dropped except the private key itself
    out = merge([c("PRIVATE_KEY", 0, 100), c("NATIONAL_ID", 40, 50)], blocked=[(0, 100, None)])
    assert [x.entity for x in out] == ["PRIVATE_KEY"]
    # URL credentials block the bogus e-mail "pass@host" but not other entities
    out = merge([c("PASSWORD", 10, 18), c("EMAIL", 10, 30), c("PHONE", 60, 70)], blocked=[(5, 40, {"EMAIL"})])
    assert [x.entity for x in out] == ["PASSWORD", "PHONE"]


def test_chain_of_overlaps_resolved_per_cluster():
    cands = [c("PERSON_NAME", 0, 10), c("EMAIL", 8, 20), c("PHONE", 18, 30), c("CVV", 40, 43)]
    out = merge(cands)
    assert [x.entity for x in out] == ["EMAIL", "CVV"]


def test_end_to_end_overlaps(find):
    f = find("card: 3450 701923 13267")  # Amex 4-6-5 also looks like a phone
    assert f == [("CARD_NUMBER", "3450 701923 13267")]
    f = find("Aadhaar 7909 5255 3312")
    assert f == [("NATIONAL_ID", "7909 5255 3312")]
