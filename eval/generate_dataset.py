"""Generate the synthetic labelled dataset of scanned invoices (French, English, Arabic).

Everything is made up: company and person names come from Faker, VAT numbers are generated with valid
check digits but random SIRENs/serials, amounts are random. Each invoice is rendered to an image,
degraded to look like a scan (skew, blur, noise, JPEG artefacts, grey paper) and saved as an image-only
PDF next to a JSON label holding the values printed on it.

    uv run --group datagen python eval/generate_dataset.py --out data/test --per-language 12 --seed 2
    uv run --group datagen python eval/generate_dataset.py --out data/dev --per-language 6 --seed 1

The dev split is what the rules and prompts were written against; the test split is only used by the
evaluation.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import arabic_reshaper
import matplotlib
import numpy as np
from bidi.algorithm import get_display
from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi

FR_MONTHS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]  # fmt: skip
EN_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

ITEMS = {
    "fr": [
        "Maintenance serveur mensuelle", "Licence logiciel annuelle", "Développement module API",
        "Audit sécurité", "Formation équipe (1 jour)", "Hébergement cloud", "Support technique",
        "Conception graphique", "Rédaction documentation", "Migration base de données",
    ],
    "en": [
        "Monthly server maintenance", "Annual software licence", "API module development",
        "Security audit", "Team training (1 day)", "Cloud hosting", "Technical support",
        "Graphic design", "Documentation writing", "Database migration",
    ],
    "ar": [
        "صيانة الخادم الشهرية", "ترخيص برنامج سنوي", "تطوير وحدة برمجية", "تدقيق أمني",
        "تكوين الفريق", "استضافة سحابية", "دعم فني", "تصميم جرافيكي", "تحرير الوثائق",
        "نقل قاعدة البيانات",
    ],
}  # fmt: skip


@dataclass
class Spec:
    lang: str
    country: str
    currency: str
    decimals: int
    number_style: str
    vat_rates: tuple[int, ...]


SPECS = {
    "fr": [Spec("fr", "FR", "EUR", 2, "fr", (20, 10))],
    "en": [Spec("en", "GB", "GBP", 2, "en", (20,)), Spec("en", "IE", "EUR", 2, "en", (23,))],
    "ar": [Spec("ar", "TN", "TND", 3, "fr", (19, 7)), Spec("ar", "MA", "MAD", 2, "fr", (20, 10))],
}


def q(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def fmt_amount(value: Decimal, decimals: int, style: str) -> str:
    whole, _, frac = f"{value:.{decimals}f}".partition(".")
    groups: list[str] = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    if style == "en":
        return ",".join(groups) + "." + frac
    return " ".join(groups).replace(" ", " ") + "," + frac


def fr_vat(rng: random.Random) -> str:
    siren = rng.randrange(100_000_000, 999_999_999)
    key = (12 + 3 * (siren % 97)) % 97
    return f"FR{key:02d}{siren}"


def gb_vat(rng: random.Random) -> str:
    while True:
        digits = [rng.randrange(10) for _ in range(7)]
        total = sum(d * w for d, w in zip(digits, range(8, 1, -1), strict=True))
        check = (-total) % 97
        if check < 100:
            return "GB" + "".join(map(str, digits)) + f"{check:02d}"


def ie_vat(rng: random.Random) -> str:
    digits = [rng.randrange(10) for _ in range(7)]
    total = sum(d * w for d, w in zip(digits, range(8, 1, -1), strict=True))
    letter = "WABCDEFGHIJKLMNOPQRSTUV"[total % 23]
    return "IE" + "".join(map(str, digits)) + letter


def tn_mf(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTVWXYZ"
    return f"{rng.randrange(1_000_000, 9_999_999)}{rng.choice(letters)}/{rng.choice('ABDNP')}/{rng.choice('MCP')}/000"


def ma_ice(rng: random.Random) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(15))


def vat_for(country: str, rng: random.Random) -> str:
    return {"FR": fr_vat, "GB": gb_vat, "IE": ie_vat, "TN": tn_mf, "MA": ma_ice}[country](rng)


def fmt_date(d: date, lang: str, variant: int, arabic_digits: bool) -> str:
    if lang == "fr":
        return f"{d:%d/%m/%Y}" if variant == 0 else f"{d.day} {FR_MONTHS[d.month - 1]} {d.year}"
    if lang == "en":
        return [f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}", f"{d:%d %b %Y}", d.isoformat()][variant % 3]
    text = f"{d:%Y/%m/%d}" if variant == 0 else f"{d:%d/%m/%Y}"
    return text.translate(ARABIC_INDIC) if arabic_digits else text


def ar(text: str) -> str:
    """Shape and reorder Arabic so PIL (without libraqm) draws it right to left."""
    return str(get_display(arabic_reshaper.reshape(text)))


LABELS = {
    "fr": {
        "title": "FACTURE", "number": "Facture N°", "date": "Date", "due": "Échéance", "vat_id": "N° TVA",
        "client": "Client", "desc": "Désignation", "qty": "Qté", "unit": "Prix unitaire", "amount": "Montant",
        "subtotal": "Total HT", "vat": "TVA", "total": "Total TTC", "currency": "Devise",
    },
    "en": {
        "title": "INVOICE", "number": "Invoice No.", "date": "Invoice date", "due": "Due date",
        "vat_id": "VAT No.", "client": "Bill to", "desc": "Description", "qty": "Qty",
        "unit": "Unit price", "amount": "Amount", "subtotal": "Subtotal", "vat": "VAT", "total": "Total due",
        "currency": "Currency",
    },
    "ar": {
        "title": "فاتورة", "number": "رقم الفاتورة", "date": "التاريخ", "due": "تاريخ الاستحقاق",
        "vat_id": "المعرف الجبائي", "client": "العميل", "desc": "البيان", "qty": "الكمية",
        "unit": "سعر الوحدة", "amount": "المبلغ", "subtotal": "المجموع دون الأداءات",
        "vat": "الأداء على القيمة المضافة", "total": "المجموع الجملي", "currency": "العملة",
    },
}  # fmt: skip


def make_invoice(idx: int, lang: str, rng: random.Random) -> tuple[dict[str, object], dict[str, object]]:
    spec = rng.choice(SPECS[lang])
    faker = Faker({"fr": "fr_FR", "en": "en_GB", "ar": "ar_AA"}[lang])
    faker.seed_instance(rng.randrange(1 << 30))
    if lang == "ar":
        seller = f"شركة {faker.last_name()} للخدمات"
        buyer = f"مؤسسة {faker.last_name()}"
        address = f"{rng.randrange(1, 200)} نهج {faker.last_name()}"
        city = rng.choice(["تونس", "صفاقس", "سوسة"] if spec.country == "TN" else ["الرباط", "الدار البيضاء", "فاس"])
    else:
        seller = faker.company()
        buyer = faker.company()
        address = faker.street_address().replace("\n", ", ")
        city = faker.city()
    issue = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 270))
    due = issue + timedelta(days=rng.choice([15, 30, 45, 60]))
    prefix = {"fr": "FA", "en": "INV", "ar": "F"}[lang]
    number = rng.choice([f"{prefix}-{issue.year}-{rng.randrange(1, 9999):04d}", f"{prefix}{rng.randrange(10000, 99999)}"])
    rate = rng.choice(spec.vat_rates)
    lines: list[dict[str, Decimal | str]] = []
    amounts: list[Decimal] = []
    for item in rng.sample(ITEMS[lang], rng.randrange(1, 6)):
        qty = Decimal(rng.choice([1, 1, 2, 3, 4, 5, 10, 12]))
        unit = q(Decimal(rng.randrange(1500, 250000)) / 100, spec.decimals)
        amounts.append(q(qty * unit, spec.decimals))
        lines.append({"description": item, "quantity": qty, "unit_price": unit, "amount": amounts[-1]})
    subtotal = sum(amounts, Decimal(0))
    vat_amount = q(subtotal * rate / 100, spec.decimals)
    total = subtotal + vat_amount
    label: dict[str, object] = {
        "invoice_number": number,
        "issue_date": issue.isoformat(),
        "due_date": due.isoformat(),
        "seller_name": seller,
        "seller_vat_id": vat_for(spec.country, rng),
        "buyer_name": buyer,
        "currency": spec.currency,
        "subtotal": str(subtotal),
        "vat_rate": str(rate),
        "vat_amount": str(vat_amount),
        "total": str(total),
        "line_items": [{k: str(v) for k, v in li.items()} for li in lines],
    }
    meta: dict[str, object] = {
        "lang": lang,
        "country": spec.country,
        "decimals": spec.decimals,
        "number_style": spec.number_style,
        "seller_address": f"{address}, {city}",
        "date_variant": rng.randrange(3),
        "arabic_digits": lang == "ar" and rng.random() < 0.35,
        "injected_error": None,
    }
    # A few invoices print a total that does not add up, to check that validation flags them.
    if idx % 11 == 5:
        printed = total + q(Decimal(rng.choice([10, 25, 100])), spec.decimals)
        label["total"] = str(printed)
        meta["injected_error"] = "totals_mismatch"
    return label, meta


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSerif" if serif else "DejaVuSans"
    return ImageFont.truetype(str(FONT_DIR / f"{name}{'-Bold' if bold else ''}.ttf"), size)


def render(label: dict[str, object], meta: dict[str, object], serif: bool) -> Image.Image:
    lang = str(meta["lang"])
    rtl = lang == "ar"
    serif = serif and not rtl  # DejaVu Serif has no Arabic glyphs
    lab = LABELS[lang]
    dec = int(str(meta["decimals"]))
    style = str(meta["number_style"])
    img = Image.new("L", (PAGE_W, PAGE_H), 255)
    d = ImageDraw.Draw(img)
    f_title, f_bold, f_reg, f_small = font(46, True, serif), font(26, True, serif), font(24, False, serif), font(21)
    margin = 90

    def text(x: float, y: float, s: str, f: ImageFont.FreeTypeFont, anchor: str = "la") -> None:
        if rtl and any("؀" <= c <= "ۿ" for c in s):
            s = ar(s)
        d.text((x, y), s, font=f, fill=0, anchor=anchor)

    def money(v: object) -> str:
        return fmt_amount(Decimal(str(v)), dec, style)

    def datestr(key: str) -> str:
        return fmt_date(date.fromisoformat(str(label[key])), lang, int(str(meta["date_variant"])), bool(meta["arabic_digits"]))

    left, right = margin, PAGE_W - margin
    # Seller block (top left for LTR, top right for RTL)
    sx, sa = (right, "ra") if rtl else (left, "la")
    text(sx, 80, str(label["seller_name"]), f_bold, sa)
    text(sx, 118, str(meta["seller_address"]), f_small, sa)
    text(sx, 150, f"{lab['vat_id']}: " if not rtl else lab["vat_id"], f_small, sa)
    vat_x = (
        (sx - d.textlength(ar(lab["vat_id"]), font=f_small) - 15)
        if rtl
        else sx + d.textlength(f"{lab['vat_id']}: ", font=f_small)
    )
    text(vat_x, 150, str(label["seller_vat_id"]), f_small, "ra" if rtl else "la")
    # Title on the opposite side
    text(left if rtl else right, 80, lab["title"], f_title, "la" if rtl else "ra")

    # Header fields
    y = 260
    rows = [
        (lab["number"], str(label["invoice_number"])),
        (lab["date"], datestr("issue_date")),
        (lab["due"], datestr("due_date")),
    ]
    for key, value in rows:
        if rtl:
            text(right, y, key, f_reg, "ra")
            text(right - 330, y, value, f_reg, "ra")
        else:
            text(left, y, f"{key} :" if lang == "fr" else f"{key}:", f_reg)
            text(left + 260, y, value, f_reg)
        y += 40
    # Buyer block on the other side
    bx, ba = (left, "la") if rtl else (right, "ra")
    text(bx, 260, lab["client"], f_bold, ba)
    text(bx, 300, str(label["buyer_name"]), f_reg, ba)

    # Line item table
    y = 470
    cols = [left, left + 640, left + 850, right] if not rtl else [right, right - 520, right - 640, right - 860]
    heads = [lab["desc"], lab["qty"], lab["unit"], lab["amount"]]
    d.line((left, y - 12, right, y - 12), fill=0, width=2)
    for i, h in enumerate(heads):
        if rtl:
            text(cols[i], y, h, f_bold, "ra")
        else:
            text(cols[i], y, h, f_bold, "la" if i == 0 else "ra")
    y += 48
    d.line((left, y - 8, right, y - 8), fill=0, width=1)
    items = label["line_items"]
    assert isinstance(items, list)
    for li in items:
        vals = [li["description"], str(Decimal(li["quantity"])), money(li["unit_price"]), money(li["amount"])]
        for i, v in enumerate(vals):
            if rtl:
                text(cols[i], y, v, f_reg, "ra")
            else:
                text(cols[i], y, v, f_reg, "la" if i == 0 else "ra")
        y += 44
    d.line((left, y, right, y), fill=0, width=1)

    # Totals
    y += 40
    cur = str(label["currency"])
    totals = [
        (lab["subtotal"], money(label["subtotal"])),
        (f"{lab['vat']} {label['vat_rate']}%", money(label["vat_amount"])),
        (lab["total"], money(label["total"])),
    ]
    for i, (key, value) in enumerate(totals):
        f = f_bold if i == 2 else f_reg
        if rtl:
            text(right, y, key, f, "ra")
            text(left + 260, y, f"{value} {cur}", f, "ra")
        else:
            text(right - 520, y, key, f)
            text(right, y, f"{value} {cur}", f, "ra")
        y += 44
    y += 30
    text(right if rtl else left, y, lab["currency"], f_small, "ra" if rtl else "la")
    text(right - 200 if rtl else left + 200, y, cur, f_small, "ra" if rtl else "la")
    return img


def degrade(img: Image.Image, rng: random.Random, severity: str) -> tuple[Image.Image, float]:
    level = {"light": 0.3, "medium": 0.65, "heavy": 1.0}[severity]
    angle = rng.uniform(-1, 1) * 4.0 * level
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    if level > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1) * level))
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * rng.uniform(0.85, 0.97) + rng.uniform(5, 25) * level  # grey paper, lower contrast
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    arr += nrng.normal(0, 4 + 18 * level, arr.shape)
    speckle = nrng.random(arr.shape)
    arr[speckle < 0.002 * level] = 0
    arr[speckle > 1 - 0.002 * level] = 255
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(80 - 45 * level))
    return Image.open(buf), round(angle, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-language", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = []
    severities = ["light", "medium", "heavy"]
    for lang in ("fr", "en", "ar"):
        for i in range(args.per_language):
            doc_id = f"{lang}-{i + 1:03d}"
            label, meta = make_invoice(i, lang, rng)
            severity = severities[i % 3]
            clean = render(label, meta, serif=rng.random() < 0.4)
            scan, angle = degrade(clean, rng, severity)
            scan.save(args.out / f"{doc_id}.pdf", "PDF", resolution=150)
            meta["severity"] = severity
            meta["skew_degrees"] = angle
            (args.out / f"{doc_id}.json").write_text(
                json.dumps({"id": doc_id, "label": label, "meta": meta}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest.append({"id": doc_id, "lang": lang, "severity": severity, "injected_error": meta["injected_error"]})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} documents to {args.out}")


if __name__ == "__main__":
    main()
