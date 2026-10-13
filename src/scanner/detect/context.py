"""Context rules: what a key / column header / form label / phrase says about the value next to it.

Every key is normalised to lowercase words ("cc_num" -> ["cc", "num"], "dbPassword" -> ["db", "password"]).
Vocabularies follow ANNOTATION_GUIDE.md and general knowledge; none of them is derived from dev values.
"""

import re

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def key_words(key):
    """Normalise any key style to lowercase words: snake_case, camelCase, kebab-case, dotted JSON paths and
    "Title Case" labels all compare the same way, so every vocabulary below is written once."""
    if not key:
        return []
    k = _CAMEL.sub(r"\1 \2", str(key)).lower()
    return [w for w in _NONALNUM.split(k) if w]


# --------------------------------------------------------------- vocabularies
PASSWORD_WORDS = {"password", "passwd", "pwd", "pass", "passphrase", "passwort", "pword"}
# Words that, placed before the password word, turn the key into metadata about passwords: hashed_password,
# require_password, reset_password, min_password... Their values are hashes, flags, URLs or numbers. (Keys such
# as password_hash or password_min_length are already ruled out because "password" is not their last word.)
PASSWORD_MODIFIERS = {"hash", "hashed", "digest", "salt", "min", "max", "minimum", "maximum", "length", "len",
                      "expiry", "expires", "expiration", "expire", "days", "age", "reset", "url", "uri", "path",
                      "link", "policy", "require", "required", "requires", "rules", "rule", "hint", "strength",
                      "changed", "updated", "enabled", "enable", "field", "column", "label", "prompt",
                      "placeholder", "file", "env", "var", "history", "attempts", "retries", "regex", "pattern",
                      "complexity", "rotation", "encrypted", "enc", "type", "mode", "format", "count", "last",
                      "forgot", "recovery", "change", "set", "new_path", "validation", "valid", "visible", "show"}
TOKEN_KEYS = {"api key", "apikey", "api token", "api secret", "secret key", "secret", "client secret", "auth token",
              "access token", "token", "x api key", "private token", "app secret", "app key", "consumer secret",
              "refresh token", "session", "session id", "sessionid", "session token", "auth", "authorization",
              "bearer", "bearer token", "webhook secret", "signing secret", "signing key", "master key", "secret token",
              "api access token", "personal access token", "access key secret", "secret access key",
              "aws secret access key", "client token", "auth key", "service token", "id token", "security token"}
# A key also counts when it *ends* with one of these (stripe_api_key, github_token, db_client_secret).
TOKEN_SUFFIXES = (("api", "key"), ("secret", "key"), ("access", "token"), ("auth", "token"), ("api", "token"),
                  ("client", "secret"), ("secret",), ("token",), ("apikey",), ("x", "api", "key"))
# ...but not when the last word says the value is something *about* the token: token_id, token_type,
# api_key_name, secret_ttl, access_token_expires_at.
TOKEN_EXCLUDE_LAST = {"id", "ids", "name", "type", "url", "uri", "path", "file", "expiry", "expires", "expiration",
                      "ttl", "length", "len", "count", "enabled", "header", "prefix", "format", "version", "hash",
                      "sha", "md5", "digest", "checksum", "endpoint", "limit", "scope", "scopes", "issuer",
                      "audience", "algorithm", "alg", "kind", "status", "created", "updated", "at", "ms", "seconds"}
PHONE_WORDS = {"phone", "mobile", "mob", "tel", "telephone", "cell", "cellphone", "msisdn", "whatsapp", "sms",
               "callback", "fax", "landline", "contact", "otp", "call", "dial", "ph", "mobileno", "phoneno"}
# Prose cues for a bare number in running text ("call me back on 9876543210"); anchored at the end because they
# are matched against the text immediately left of the number.
PHONE_PHRASES = re.compile(r"(?:call\s+me(?:\s+back)?\s+(?:on|at)|reach\s+me\s+(?:on|at)|my\s+(?:number|no\.?)\s+is|"
                           r"sms\s+(?:queued\s+)?(?:for|to)|otp\s+sent\s+to|text\s+me\s+(?:on|at)|callback\s+number|"
                           r"contact\s+(?:number|no)|whatsapp\s+(?:on|at)?)\W*$", re.I)
# "pan" is ambiguous: card PAN (primary account number) or Indian PAN. The value's shape settles it, since an
# Indian PAN has letters and a card does not.
CARD_WORDS = {"card", "cc", "ccnum", "ccn", "credit", "debit", "pan", "visa", "mastercard", "amex", "rupay",
              "discover", "cardnumber", "cardno"}
# Fields whose numbers are identifiers of something else. Order numbers, AWB/tracking numbers, IMEIs and
# account numbers often pass Luhn by construction or by chance, so under these keys a valid-looking card or
# Aadhaar is not reported.
IDENTIFIER_WORDS = {"order", "invoice", "inv", "txn", "transaction", "trans", "tracking", "awb", "shipment",
                    "loyalty", "imei", "account", "acct", "consumer", "customer", "user", "member", "policy", "serial",
                    "sku", "barcode", "upc", "ean", "isbn", "ticket", "booking", "pnr", "receipt", "voucher", "coupon",
                    "batch", "product", "seq", "sequence", "id", "uuid", "guid", "request", "trace", "correlation",
                    "span", "tenant", "instance", "job", "build", "version", "ts", "timestamp", "epoch", "created",
                    "updated", "bytes", "size", "amount", "count", "port", "pid", "zip", "pin", "postal", "status",
                    "error", "http", "shipping", "parcel", "consignment", "waybill", "msisdn", "iccid", "vin",
                    "subscriber", "contract", "loan", "claim", "case", "employee", "emp", "student", "roll", "registration"}
# Like PHONE_PHRASES, these match the text just left of a value: the keyword may be followed by up to 4
# separator characters (": ", "=\"") and must end exactly where the value starts.
EXPIRY_LEFT = re.compile(r"(?:\bexp(?:iry|iration|ires|ire|_?date|date)?|\bvalid[\s_-]*(?:thru|through|till|until|upto|to)"
                         r"|\bgood[\s_-]*(?:thru|through)|\bcard[\s_-]*exp\w*)\W{0,4}$", re.I)
EXPIRY_KEYS = {"exp", "expiry", "expiration", "expires", "exp date", "expiry date", "expiration date", "valid thru",
               "valid through", "valid till", "valid until", "card expiry", "card exp", "cc exp", "exp month year",
               "good thru", "expdate", "exp_date", "validthru", "valid upto"}
CVV_LEFT = re.compile(r"(?:\bcvv2?|\bcvc2?|\bcvn|\bsecurity[\s_-]*code|\bcard[\s_-]*verification(?:[\s_-]*(?:value|code))?"
                      r"|\bcard[\s_-]*security[\s_-]*code)(?:\s+(?:is|was|=|:))?\W{0,3}$", re.I)
CVV_KEYS = {"cvv", "cvv2", "cvc", "cvc2", "cvn", "security code", "card security code", "card verification value",
            "card verification code", "card cvv", "cc cvv"}
SSN_KEYS = {"ssn", "social security", "social security number", "social security no", "ss number", "ss no", "ssn no",
            "social"}
NAME_KEYS = {"name", "full name", "fullname", "customer name", "client name", "employee", "employee name",
             "contact name", "contact person", "holder name", "card holder", "cardholder", "cardholder name",
             "account holder", "account holder name", "applicant", "applicant name", "author", "reporter",
             "assignee", "requester", "requestor", "owner name", "beneficiary", "beneficiary name", "nominee",
             "patient", "patient name", "signed by", "signatory", "recipient", "recipient name", "sender",
             "sender name", "from", "to", "cc", "bcc", "reply to", "attendee", "person", "person name", "member name",
             "guest name", "passenger name", "student name", "customer", "client", "agent", "agent name",
             "approver", "approved by", "prepared by", "reviewed by", "created by", "submitted by", "contact"}
NAME_KEY_QUALIFIERS = {"full", "customer", "client", "employee", "contact", "holder", "person", "patient",
                       "applicant", "owner", "member", "guest", "passenger", "beneficiary", "nominee", "recipient",
                       "sender", "father", "mother", "spouse", "student", "agent", "legal", "account", "cardholder"}
# Keys whose hex/base64 values are digests or version-control ids. Those look exactly like API keys (32-64 hex
# characters), so the key is the only thing that tells them apart.
HASH_WORDS = {"hash", "hashed", "sha", "sha1", "sha256", "sha512", "sha384", "md5", "checksum", "digest", "commit",
              "revision", "rev", "etag", "integrity", "fingerprint", "merged", "build", "salt", "signature", "crc",
              "blake2", "blake3", "git", "head", "parent"}
# e-mail headers whose <...@...> values are message identifiers, not mailboxes
EMAIL_HEADER_SKIP = {"message id", "in reply to", "references", "msg id", "message"}


def is_password_key(words):
    # The password word must be the *last* word: "db_password" holds a password, "password_policy" does not.
    # Any modifier before it disqualifies the key too ("hashed_password", "min_password").
    if not words:
        return False
    if words[-1] in PASSWORD_WORDS:
        return not any(w in PASSWORD_MODIFIERS for w in words[:-1])
    return False


def is_token_key(words):
    if not words or words[-1] in TOKEN_EXCLUDE_LAST:
        return False
    k = " ".join(words)
    if k in TOKEN_KEYS:
        return True
    for suf in TOKEN_SUFFIXES:
        if len(words) >= len(suf) and tuple(words[-len(suf):]) == suf:
            # CSRF tokens and pagination cursors are random strings by design, but they protect nothing
            return not any(w in PASSWORD_MODIFIERS or w in ("csrf", "xsrf", "page", "next", "pagination",
                                                            "cursor", "continuation", "sync") for w in words)
    return False


def is_phone_context(words, left_text=""):
    if left_text and PHONE_PHRASES.search(left_text):
        return True
    # Only the end of the key matters: "phone_id", "contact_ref" or "sms_count" are about a phone but hold an id
    # or a counter.
    tail = words[-4:]
    if any(w in ("id", "ids", "uuid", "ref", "code", "count") for w in tail[-1:]):
        return False
    return any(w in PHONE_WORDS for w in tail)


def is_card_key(words):
    return any(w in CARD_WORDS for w in words[-3:])


def is_identifier_key(words):
    """Field names another identifier (order_id, txn_ref, tracking_number, imei...) and not a card field.
    'Reference' / 'ref' / 'ref no' name an identifier; 'ref_a', 'ref_2' are uninformative column names
    (the guide's example), so a one-character suffix cancels them."""
    tail = words[-3:]
    if not tail:
        return False
    if any(w in CARD_WORDS for w in tail):  # "card_number", "cc_ref": a card word always wins
        return False
    if any(w in IDENTIFIER_WORDS for w in tail):
        return True
    if tail[-1] in ("ref", "reference", "refno") or (len(tail) >= 2 and tail[-2] in ("ref", "reference")
                                                     and tail[-1] in ("no", "num", "number", "id", "code")):
        return True
    return False


def is_expiry_key(words):
    k = " ".join(words)
    return k in EXPIRY_KEYS or (bool(words) and (words[-1] in ("exp", "expiry", "expiration") or
                                                  k.endswith("valid thru") or k.endswith("exp date")))


def is_cvv_key(words):
    k = " ".join(words)
    return k in CVV_KEYS or bool(words and words[-1] in ("cvv", "cvv2", "cvc", "cvc2"))


def is_ssn_key(words):
    k = " ".join(words)
    return k in SSN_KEYS or "ssn" in words or ("social" in words and "security" in words)


# "<thing>.name" / "<thing>_name" names a thing, not a person (service_name, host.name, bank_name).
NON_PERSON_NAME_PARENTS = {"service", "host", "file", "company", "org", "organization", "organisation", "product",
                           "app", "application", "bank", "branch", "city", "country", "state", "street", "project",
                           "team", "queue", "bucket", "table", "column", "field", "db", "database", "cluster",
                           "region", "zone", "device", "model", "plan", "tier", "sku", "item", "category", "tag",
                           "label", "event", "job", "task", "pipeline", "user", "login", "screen", "display",
                           "domain", "server", "machine", "node", "pod", "container", "image", "repo", "package",
                           "module", "class", "function", "method", "key", "secret", "env", "environment",
                           "variable", "brand", "store", "shop", "merchant", "vendor", "business", "legal_entity",
                           "entity", "document", "doc", "folder", "dir", "path", "site", "page", "course", "metric"}


def is_name_key(words):
    """Person-name keys: 'name', 'full_name', 'customer.name', 'profile.name' - but not 'service_name',
    'host.name', 'username' (a dotted JSON path is judged by its last two components)."""
    k = " ".join(words)
    if k in NAME_KEYS:
        return True
    if words and words[-1] in ("name", "fullname"):
        if len(words) == 1:
            return True
        prev = words[-2]
        if prev in NAME_KEY_QUALIFIERS:
            return True
        return prev not in NON_PERSON_NAME_PARENTS and not prev.isdigit()
    return False


def is_hash_key(words):
    return any(w in HASH_WORDS for w in words[-3:])


def is_email_header_skip(words):
    return " ".join(words[-3:]) in EMAIL_HEADER_SKIP or (len(words) >= 2 and words[-2:] == ["message", "id"])


# --------------------------------------------------------------- left context on the same line
# The key is the run of key-like characters that ends right before the value, once separators are stripped:
# 'cc_num=', '"pan": "', 'Card number<TAB>', 'phone -> '. The context never crosses a line break, so a key on
# the previous line cannot leak into the next record.
_KEY_TAIL = re.compile(r"([A-Za-z][A-Za-z0-9_\-. ]{0,48})$")
_STRIP = " \t\"'`:=>-([{<"


def left_text(text, start, limit=80):
    """Text between the start of the line (at most `limit` chars back) and `start`."""
    ls = text.rfind("\n", max(0, start - limit), start)
    return text[(ls + 1) if ls >= 0 else max(0, start - limit):start]


def left_key(text, start, limit=80):
    """Key-ish phrase immediately before a value: 'cc_num=' -> ['cc','num']; '"pan": "' -> ['pan']."""
    lt = left_text(text, start, limit).rstrip(_STRIP)
    m = _KEY_TAIL.search(lt)
    return key_words(m.group(1)) if m else []
