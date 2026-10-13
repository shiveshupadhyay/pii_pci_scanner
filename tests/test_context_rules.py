"""Context rules from ANNOTATION_GUIDE.md, exercised through the detector."""

import pytest


def ents(found, entity):
    return [v for e, v in found if e == entity]


# ------------------------------------------------------------------ phones
@pytest.mark.parametrize("text,value", [
    ("callback number +91 98765 43210", "+91 98765 43210"),
    ("call me back on +1 (415) 555-0132?", "+1 (415) 555-0132"),
    ("otp sent to 98765-43210", "98765-43210"),
    ("my mobile is 9876543210", "9876543210"),
    ("phone: 9876543210", "9876543210"),
])
def test_phone_positive(find, text, value):
    assert ents(find(text), "PHONE") == [value]


@pytest.mark.parametrize("text", ["bytes_sent=8043983594", "ts=1727584919793", "consumer_id=601390479811",
                                  "build 2.14.0.1", "ip 182.53.99.220", "invoice 9876543210",
                                  "ref 000-933-2653"])
def test_phone_negative(find, text):
    assert not ents(find(text), "PHONE")


def test_bare_phone_uses_column_header(find):
    assert ents(find("9876543210", key="mobile", cell=True), "PHONE") == ["9876543210"]
    assert not ents(find("9876543210", key="bytes_sent", cell=True), "PHONE")


# ------------------------------------------------------------------ cards
def test_card_context(find):
    assert ents(find("cc_num=5413 9587 1901 6866"), "CARD_NUMBER") == ["5413 9587 1901 6866"]
    assert ents(find("card: 3450 701923 13267"), "CARD_NUMBER") == ["3450 701923 13267"]
    assert not ents(find("order_id: 4828894923646838"), "CARD_NUMBER")       # Luhn-valid order id
    assert not ents(find("txn_ref: 4351218279938950"), "CARD_NUMBER")
    assert not ents(find("Reference 4346571748097123"), "CARD_NUMBER")
    assert not ents(find("use the sandbox card 4111111111111111"), "CARD_NUMBER")
    assert not ents(find("card: 5397809134142118"), "CARD_NUMBER")            # fails Luhn
    assert not ents(find("paid with card **** **** **** 1234"), "CARD_NUMBER")


def test_uninformative_column_lets_value_speak(find):
    assert ents(find("6087318736623264", key="ref_a", cell=True), "CARD_NUMBER") == ["6087318736623264"]
    assert not ents(find("4126237433819793", key="order_id", cell=True), "CARD_NUMBER")


# ------------------------------------------------------------------ expiry / cvv
def test_expiry_needs_card(find):
    f = find("payment debug cc_num=5413 9587 1901 6866 valid_thru: 10/2029 cvc=947")
    assert ents(f, "CARD_EXPIRY") == ["10/2029"] and ents(f, "CVV") == ["947"]
    assert not ents(find("FYI release 4/29. Note: sprint ends 12/29."), "CARD_EXPIRY")
    assert not ents(find("sprint ends 10/2026. use card 5160640277024269"), "CARD_EXPIRY")
    assert ents(find("card 4869-4407-6758-2752, valid thru 9/29."), "CARD_EXPIRY") == ["9/29"]


def test_expiry_column_needs_card_column(find):
    assert ents(find("10/26", key="valid_thru", cell=True, table={"has_card": True}), "CARD_EXPIRY") == ["10/26"]
    assert not ents(find("10/26", key="valid_thru", cell=True, table={"has_card": False}), "CARD_EXPIRY")


@pytest.mark.parametrize("text,value", [("the CVV is 647", "647"), ("security code 970", "970"), ("cvv2= 460", "460")])
def test_cvv(find, text, value):
    assert ents(find(text), "CVV") == [value]


def test_cvv_negatives(find):
    assert not ents(find("status=200 error code 503 retry_count: 777"), "CVV")
    assert not ents(find("cvv2= 7731"), "CVV")  # 4 digits only with an Amex card
    assert ents(find('"pan": "3791-799999-02781 exp: 7/25 cvv2= 7731'), "CVV") == ["7731"]


# ------------------------------------------------------------------ national ids
def test_national_ids(find):
    assert ents(find("my PAN is WTQCY0698S for the KYC form"), "NATIONAL_ID") == ["WTQCY0698S"]
    assert not ents(find("retrying job batch RMUOW3131P"), "NATIONAL_ID")
    assert ents(find("social security number 568-30-8937"), "NATIONAL_ID") == ["568-30-8937"]
    assert not ents(find("SSN 666-68-1257 / SSN 951-32-3759 / ssn: XXX-XX-4217"), "NATIONAL_ID")
    assert ents(find("Aadhaar 7909 5255 3312"), "NATIONAL_ID") == ["7909 5255 3312"]
    assert not ents(find("ref: 442382495104"), "NATIONAL_ID")


# ------------------------------------------------------------------ secrets
@pytest.mark.parametrize("text,value", [
    ("PASSWORD=Adm1n405!", "Adm1n405!"),
    ('password: "Winter2024!"', "Winter2024!"),
    ("db_password = 9iyAtdC9HOgw", "9iyAtdC9HOgw"),
    ("Your temporary password is priya2002 - please change it", "priya2002"),
    ("VPN password: Monsoon2023! - please change", "Monsoon2023!"),
    ("connecting to postgres://app:S3cret!@db/app", "S3cret!"),
    ("postgres://svc:Cobalt8117#@db-3.internal:5432/app", "Cobalt8117#"),
])
def test_password_positive(find, text, value):
    assert ents(find(text), "PASSWORD") == [value]


@pytest.mark.parametrize("text", ["password_min_length = 12", "require_password: true", '"password": null',
                                  "password_hash: $2b$12$mJiMOdqS0pPldKXcObN533m7yFnYD8hZ0Sq/hnjGb01UjEingh800",
                                  "DB_PASSWORD=${DB_PASSWORD}", "pwd = <your-password>", "reset_password_url: /reset",
                                  "pwd = $(cat /run/secrets/db_password)", "password: {{ vault_db_password }}"])
def test_password_negative(find, text):
    assert not ents(find(text), "PASSWORD")


@pytest.mark.parametrize("text,value", [
    ("api_key=sk_live_79XNSaNl1zkiINe3ayBBzzwW", "sk_live_79XNSaNl1zkiINe3ayBBzzwW"),
    ("token=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789", "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"),
    ("x-api-key: 2f9f7bf72810ead2bf8fe45c0030560f", "2f9f7bf72810ead2bf8fe45c0030560f"),
    ("Authorization: Bearer lbceikWaGNuAFOhaDJRKrAYUNSiNGk", "lbceikWaGNuAFOhaDJRKrAYUNSiNGk"),
    ('API_KEY = "cb1f4934dcc4274ca022b083ffd822cb69edf321bf70eedd5ca53d34a7a4a15a"',
     "cb1f4934dcc4274ca022b083ffd822cb69edf321bf70eedd5ca53d34a7a4a15a"),
    ("Cookie: session=Zq3r9LmW2xYp7TnB4vCk8sHd; theme=dark", "Zq3r9LmW2xYp7TnB4vCk8sHd"),
])
def test_token_positive(find, text, value):
    assert ents(find(text), "API_TOKEN") == [value]


@pytest.mark.parametrize("text", ["sha=c9e3cf2d1d6a4f0ce1cc3af6d1d0e2b4", "commit 526c411f71431c9fd6bbda945b646e395458fb81",
                                  "request_id=02a52100-4ad1-48e4-8e34-ec812377e68e", "api_key: xxxxxxxxxxxxxxxxxxxx",
                                  "auth_token = changeme-in-prod", "API_KEY = os.environ[\"API_KEY\"]",
                                  "const apiKey = process.env.API_KEY;", "sk_live_XXXXXXXXXXXXXXXXXXXXXXXX",
                                  "token_type: Bearer", "checksum: 19a9bfd28a04676e05f77c2ca22a94fe"])
def test_token_negative(find, text):
    assert not ents(find(text), "API_TOKEN")


def test_aws_and_keys(find):
    assert ents(find("aws_access_key_id: AKIAJAJQUH5BKX7TSRNK"), "AWS_ACCESS_KEY") == ["AKIAJAJQUH5BKX7TSRNK"]
    assert not ents(find("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"), "AWS_ACCESS_KEY")
    pem = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBb\n-----END EC PRIVATE KEY-----"
    assert ents(find("ssh_key: |\n" + pem + "\n"), "PRIVATE_KEY") == [pem]
    assert not ents(find("-----BEGIN PUBLIC KEY-----\nMIIBIjAN\n-----END PUBLIC KEY-----"), "PRIVATE_KEY")
    escaped = 'KEY = "-----BEGIN RSA PRIVATE KEY-----\\nMIIEow\\n-----END RSA PRIVATE KEY-----"'
    (v,) = ents(find(escaped), "PRIVATE_KEY")
    assert "\\n" not in v


# ------------------------------------------------------------------ emails
def test_emails(find):
    assert ents(find("user login ok email=david_kelly@bluepeak.com ip=1.2.3.4"), "EMAIL") == ["david_kelly@bluepeak.com"]
    assert not ents(find("contact: support@example.com e.g. john.doe@test.com user@domain.com"), "EMAIL")
    assert not ents(find("remote: git@github.com:team/billing-api.git"), "EMAIL")
    assert not ents(find("ssh deploy@prod-web-01"), "EMAIL")
    assert not ents(find("Message-ID: <3b9f147e07f21c1e@mail.northwind.io>"), "EMAIL")
    assert not ents(find("mysql://app:omar2006@db-2.internal:5432/app"), "EMAIL")


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("text,value", [
    ("From: Hannah Brown <hannah20@ardentlabs.ai>", "Hannah Brown"),
    ("// Author: Olivia Brown <olivia_brown@kestrelcap.com>", "Olivia Brown"),
    ("2025-12-04T20:29:26Z ERROR [api] profile updated for Ananya Khan (id=71714)", "Ananya Khan"),
    ("I've looped in James Fernandes from finance.", "James Fernandes"),
    ("Full name         Priya Wang", "Priya Wang"),
    ("From: Will Smith <will@corp.io>", "Will Smith"),
])
def test_names_positive(find, text, value):
    assert value in ents(find(text), "PERSON_NAME")


@pytest.mark.parametrize("text", ["Hi Priya,", "Thanks,\nAnanya", "Account Manager | Kestrel Capital",
                                  "Will Call | Kestrel Capital", "Google Chrome | Meridian Health",
                                  "request completed in Apache Kafka", "Grace Period ends today",
                                  "Northwind Traders", "cache miss for key Best Regards"])
def test_names_negative(find, text):
    assert not ents(find(text), "PERSON_NAME")
