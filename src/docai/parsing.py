"""Locale-aware parsing of amounts, dates, currencies and tax ids as they appear on French, English and
Arabic invoices. Pure functions, used by the rule-based extractor, the LLM output coercion and the
evaluation."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_SEPARATORS = str.maketrans({"٫": ",", "٬": " ", " ": " ", " ": " ", "'": " "})

CURRENCY_DECIMALS = {"TND": 3, "KWD": 3, "BHD": 3, "OMR": 3, "JOD": 3}
CURRENCY_ALIASES = {
    "€": "EUR",
    "EUR": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "$": "USD",
    "USD": "USD",
    "TND": "TND",
    "DT": "TND",
    "د.ت": "TND",
    "MAD": "MAD",
    "DH": "MAD",
    "DHS": "MAD",
    "د.م.": "MAD",
    "د.م": "MAD",
    "AED": "AED",
    "SAR": "SAR",
    "CHF": "CHF",
    "DZD": "DZD",
}
KNOWN_CURRENCIES = frozenset(CURRENCY_ALIASES.values())

FR_MONTHS = {
    "janvier": 1, "janv": 1, "février": 2, "fevrier": 2, "févr": 2, "fevr": 2, "mars": 3, "avril": 4,
    "avr": 4, "mai": 5, "juin": 6, "juillet": 7, "juil": 7, "août": 8, "aout": 8, "septembre": 9,
    "sept": 9, "octobre": 10, "oct": 10, "novembre": 11, "nov": 11, "décembre": 12, "decembre": 12,
    "déc": 12, "dec": 12,
}  # fmt: skip
EN_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
MONTHS = {**EN_MONTHS, **FR_MONTHS}

AMOUNT_RE = re.compile(r"-?(?:\d{1,3}(?:[ .,]\d{3})+|\d+)(?:[.,]\d{1,3})?")


def normalize_digits(text: str) -> str:
    """Map Arabic-Indic / Persian digits and Arabic separators to ASCII."""
    return text.translate(_DIGITS).translate(_SEPARATORS)


def decimals_for(currency: str | None) -> int:
    return CURRENCY_DECIMALS.get(currency or "", 2)


def parse_amount(text: str | int | float | Decimal | None, decimals: int = 2) -> Decimal | None:
    """Parse '1 234,56', '1,234.56', '1.234,56', '288,760' (TND, 3 decimals) or a number.

    A single separator followed by exactly three digits is ambiguous ('1,234' is 1234 in English
    and 1.234 in French). `decimals` decides it: three-decimal currencies read it as a decimal part.
    """
    if text is None:
        return None
    if isinstance(text, Decimal):
        return text
    if isinstance(text, int | float):
        return Decimal(str(text))
    s = normalize_digits(text)
    match = AMOUNT_RE.search(s)
    if not match:
        return None
    raw = match.group(0).strip()
    last_dot, last_comma = raw.rfind("."), raw.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        dec_sep = "." if last_dot > last_comma else ","
        int_part, frac = raw.rsplit(dec_sep, 1)
        int_part = re.sub(r"[ .,]", "", int_part)
    elif last_dot >= 0 or last_comma >= 0:
        sep = "." if last_dot >= 0 else ","
        int_part, frac = raw.rsplit(sep, 1)
        if raw.count(sep) > 1 or (len(frac) == 3 and decimals != 3):
            int_part, frac = re.sub(r"[ .,]", "", raw), ""
        else:
            int_part = int_part.replace(" ", "").replace(sep, "")
    else:
        int_part, frac = raw.replace(" ", ""), ""
    try:
        return Decimal(f"{int_part}.{frac}" if frac else int_part)
    except InvalidOperation:
        return None


def find_amounts(text: str, decimals: int = 2) -> list[Decimal]:
    out = []
    for m in AMOUNT_RE.finditer(normalize_digits(text)):
        value = parse_amount(m.group(0), decimals)
        if value is not None:
            out.append(value)
    return out


_NUMERIC_DATE = re.compile(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})")
_WORD_DATE_DMY = re.compile(r"(\d{1,2})(?:er)?\s+([A-Za-zÀ-ÿ]+)\.?\s+(\d{4})")
_WORD_DATE_MDY = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str | date | None) -> date | None:
    """Parse the first date in text. Numeric dates are day-first unless the year comes first."""
    if text is None:
        return None
    if isinstance(text, date):
        return text
    s = normalize_digits(text)
    m = _NUMERIC_DATE.search(s)
    if m:
        a, b, c = (int(g) for g in m.groups())
        if len(m.group(1)) == 4:
            return _safe_date(a, b, c)
        if len(m.group(3)) == 4:
            return _safe_date(c, b, a) or _safe_date(c, a, b)
    m = _WORD_DATE_DMY.search(s)
    if m and m.group(2).lower().rstrip(".") in MONTHS:
        return _safe_date(int(m.group(3)), MONTHS[m.group(2).lower().rstrip(".")], int(m.group(1)))
    m = _WORD_DATE_MDY.search(s)
    if m and m.group(1).lower() in MONTHS:
        return _safe_date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    return None


def find_dates(text: str) -> list[date]:
    s = normalize_digits(text)
    out: list[date] = []
    for regex in (_NUMERIC_DATE, _WORD_DATE_DMY, _WORD_DATE_MDY):
        for m in regex.finditer(s):
            d = parse_date(m.group(0))
            if d and d not in out:
                out.append(d)
    return out


def normalize_currency(text: str | None) -> str | None:
    if not text:
        return None
    s = text.strip().upper()
    if s in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[s]
    if text.strip() in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[text.strip()]
    for token in re.findall(r"[A-Z]{3}|[€£$]", s):
        if token in CURRENCY_ALIASES:
            return CURRENCY_ALIASES[token]
    return None


def normalize_id(text: str | None) -> str | None:
    """Uppercase, drop spaces and dots: 'fr 40 303 265 045' -> 'FR40303265045'."""
    if text is None:
        return None
    s = re.sub(r"[\s.\-]", "", normalize_digits(text)).upper()
    return s or None


def normalize_text(text: str | None) -> str:
    """Comparison key for names: NFKC, casefold, punctuation and whitespace collapsed."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text).casefold()
    s = re.sub(r"[ً-ْـ]", "", s)  # Arabic diacritics and tatweel
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


def has_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)
