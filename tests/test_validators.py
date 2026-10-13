"""Checksum and shape validators. Every rule here comes from ANNOTATION_GUIDE.md or the number format's own
specification; the negative cases include the guide's hard negatives."""

import pytest

from scanner.detect import validators as V


@pytest.mark.parametrize("num,ok", [
    ("4539148803436467", True), ("4539148803436468", False), ("378282246310005", True), ("6011000990139424", True),
])
def test_luhn(num, ok):
    assert V.luhn_ok(num) is ok


@pytest.mark.parametrize("num,brand", [
    ("4539148803436467", "visa"), ("5413958719016866", "mastercard"), ("345070192313267", "amex"),
    ("6011658741475981", "discover"), ("6083152413275054", "rupay"), ("8266911767917248", "rupay"),
    ("4539148803436", None),            # Visa must be 16 digits under the guide
    ("5613958719016866", None),         # 56 is not a Mastercard prefix
    ("3450701923132670", None),         # Amex must be 15 digits
])
def test_card_brand(num, brand):
    assert V.card_brand(num) == brand


@pytest.mark.parametrize("num", sorted(V.TEST_CARDS))
def test_well_known_test_cards_rejected(num):
    assert V.luhn_ok(num) and not V.card_ok(num)


@pytest.mark.parametrize("num,ok", [
    ("799039129312", True), ("265297610893", True), ("518100968886", True),
    ("442382495104", False), ("869226933124", False),  # fail Verhoeff
    ("123456789012", False),                            # first digit must be 2-9
])
def test_aadhaar(num, ok):
    assert V.aadhaar_ok(num) is ok


@pytest.mark.parametrize("ssn,ok", [
    ("568308937", True), ("040432685", True), ("799994146", True),
    ("000137997", False), ("666681257", False), ("951323759", False), ("798003196", False), ("123450000", False),
])
def test_ssn_rules(ssn, ok):
    assert V.ssn_ok(ssn) is ok


@pytest.mark.parametrize("pan,ok", [
    ("ABCPE1234F", True), ("OZJJJ0240C", True), ("HOTAB2164Y", True),
    ("RMUOW3131P", False),  # 4th character O is not a valid holder type
    ("ABCP1234F", False),
])
def test_pan(pan, ok):
    assert V.pan_ok(pan) is ok


def test_aws():
    assert V.aws_ok("AKIAJAJQUH5BKX7TSRNK") and V.aws_ok("ASIAQZZB7KEQ6P4SUXUZ")
    assert not V.aws_ok("AKIAIOSFODNN7EXAMPLE")
    assert not V.aws_ok("AKIAJAJQUH5BKX7TSRN")


@pytest.mark.parametrize("v", ["REDACTED", "[FILTERED]", "<your-api-key>", "${DB_PASSWORD}", "{{ vault_secret }}",
                               "xxxxxxxx", "null", "", "********", "YOUR_API_KEY_HERE", "changeme-in-prod",
                               "os.environ[\"API_KEY\"]", "process.env.API_KEY", "$(cat /run/secrets/db_password)",
                               "**** **** **** 1234"])
def test_placeholders(v):
    assert V.is_placeholder(v)


@pytest.mark.parametrize("v", ["Winter2024!", "qvkJJYx^Rv3XhW", "jyrC2w8gpeQ1ooQm*C", "carlos1992"])
def test_real_values_are_not_placeholders(v):
    assert not V.is_placeholder(v)


def test_hash_shapes():
    assert V.is_hash_like("$2b$12$mJiMOdqS0pPldKXcObN533m7yFnYD8hZ0Sq/hnjGb01UjEingh800")
    assert V.is_hash_like("sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
    assert V.is_hash_like("sha512-abcdef")
    assert V.is_hex_digest("84fc711fd78b6ca3a15482cfa6e38d44")
    assert not V.is_hash_like("Winter2024!")


def test_uuid_requires_dashes():
    assert V.is_uuid("6551b8df-fdb3-47f7-b538-d93fc844fcd6")
    assert not V.is_uuid("2f9f7bf72810ead2bf8fe45c0030560f")  # 32 hex without dashes is a token / hash shape


def test_entropy():
    assert V.shannon_entropy("aaaaaaaa") == 0
    assert V.shannon_entropy("TRDq41FZa6xfW5egWiQoTfjty39IyjxB") > 4
