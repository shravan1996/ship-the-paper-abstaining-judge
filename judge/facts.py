"""Four question families whose reference answers come from tables, not from a model.

The paper labels candidate answers with an LLM evaluator (Qwen2.5-7B-Instruct, Appendix B.4).
A false-discovery-rate guarantee is only as good as the labels it is measured against, so every
reference answer here is read from a table that ships with Python or macOS, or from the periodic
table written out below. The admission function is an exact match after normalisation.

Families:
  elements   atomic numbers, names and symbols (the table below)
  ports      TCP port numbers from the IANA registry copy in /etc/services
  unicode    code points from Python's unicodedata (Unicode 16.0 on Python 3.14)
  timezones  UTC offsets on two fixed dates from the system tzdata (2026c on macOS)
"""

from __future__ import annotations

import random
import re
import unicodedata
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

# Atomic number, symbol, name. IUPAC names; American spellings are accepted as variants below.
ELEMENTS = [
    (1, "H", "Hydrogen"), (2, "He", "Helium"), (3, "Li", "Lithium"), (4, "Be", "Beryllium"),
    (5, "B", "Boron"), (6, "C", "Carbon"), (7, "N", "Nitrogen"), (8, "O", "Oxygen"),
    (9, "F", "Fluorine"), (10, "Ne", "Neon"), (11, "Na", "Sodium"), (12, "Mg", "Magnesium"),
    (13, "Al", "Aluminium"), (14, "Si", "Silicon"), (15, "P", "Phosphorus"), (16, "S", "Sulfur"),
    (17, "Cl", "Chlorine"), (18, "Ar", "Argon"), (19, "K", "Potassium"), (20, "Ca", "Calcium"),
    (21, "Sc", "Scandium"), (22, "Ti", "Titanium"), (23, "V", "Vanadium"), (24, "Cr", "Chromium"),
    (25, "Mn", "Manganese"), (26, "Fe", "Iron"), (27, "Co", "Cobalt"), (28, "Ni", "Nickel"),
    (29, "Cu", "Copper"), (30, "Zn", "Zinc"), (31, "Ga", "Gallium"), (32, "Ge", "Germanium"),
    (33, "As", "Arsenic"), (34, "Se", "Selenium"), (35, "Br", "Bromine"), (36, "Kr", "Krypton"),
    (37, "Rb", "Rubidium"), (38, "Sr", "Strontium"), (39, "Y", "Yttrium"), (40, "Zr", "Zirconium"),
    (41, "Nb", "Niobium"), (42, "Mo", "Molybdenum"), (43, "Tc", "Technetium"),
    (44, "Ru", "Ruthenium"), (45, "Rh", "Rhodium"), (46, "Pd", "Palladium"), (47, "Ag", "Silver"),
    (48, "Cd", "Cadmium"), (49, "In", "Indium"), (50, "Sn", "Tin"), (51, "Sb", "Antimony"),
    (52, "Te", "Tellurium"), (53, "I", "Iodine"), (54, "Xe", "Xenon"), (55, "Cs", "Caesium"),
    (56, "Ba", "Barium"), (57, "La", "Lanthanum"), (58, "Ce", "Cerium"),
    (59, "Pr", "Praseodymium"), (60, "Nd", "Neodymium"), (61, "Pm", "Promethium"),
    (62, "Sm", "Samarium"), (63, "Eu", "Europium"), (64, "Gd", "Gadolinium"),
    (65, "Tb", "Terbium"), (66, "Dy", "Dysprosium"), (67, "Ho", "Holmium"), (68, "Er", "Erbium"),
    (69, "Tm", "Thulium"), (70, "Yb", "Ytterbium"), (71, "Lu", "Lutetium"), (72, "Hf", "Hafnium"),
    (73, "Ta", "Tantalum"), (74, "W", "Tungsten"), (75, "Re", "Rhenium"), (76, "Os", "Osmium"),
    (77, "Ir", "Iridium"), (78, "Pt", "Platinum"), (79, "Au", "Gold"), (80, "Hg", "Mercury"),
    (81, "Tl", "Thallium"), (82, "Pb", "Lead"), (83, "Bi", "Bismuth"), (84, "Po", "Polonium"),
    (85, "At", "Astatine"), (86, "Rn", "Radon"), (87, "Fr", "Francium"), (88, "Ra", "Radium"),
    (89, "Ac", "Actinium"), (90, "Th", "Thorium"), (91, "Pa", "Protactinium"),
    (92, "U", "Uranium"), (93, "Np", "Neptunium"), (94, "Pu", "Plutonium"),
    (95, "Am", "Americium"), (96, "Cm", "Curium"), (97, "Bk", "Berkelium"),
    (98, "Cf", "Californium"), (99, "Es", "Einsteinium"), (100, "Fm", "Fermium"),
    (101, "Md", "Mendelevium"), (102, "No", "Nobelium"), (103, "Lr", "Lawrencium"),
    (104, "Rf", "Rutherfordium"), (105, "Db", "Dubnium"), (106, "Sg", "Seaborgium"),
    (107, "Bh", "Bohrium"), (108, "Hs", "Hassium"), (109, "Mt", "Meitnerium"),
    (110, "Ds", "Darmstadtium"), (111, "Rg", "Roentgenium"), (112, "Cn", "Copernicium"),
    (113, "Nh", "Nihonium"), (114, "Fl", "Flerovium"), (115, "Mc", "Moscovium"),
    (116, "Lv", "Livermorium"), (117, "Ts", "Tennessine"), (118, "Og", "Oganesson"),
]
NAME_VARIANTS = {"aluminum": "aluminium", "cesium": "caesium", "sulphur": "sulfur"}

# Unicode blocks to sample from. "Common" blocks hold characters a general model has seen often;
# "rare" blocks hold scripts and symbol sets it has mostly not.
COMMON_BLOCKS = [
    (0x00A0, 0x00FF), (0x0100, 0x017F), (0x0370, 0x03FF), (0x0400, 0x04FF), (0x2000, 0x206F),
    (0x20A0, 0x20CF), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2500, 0x257F), (0x25A0, 0x25FF),
    (0x2600, 0x26FF), (0x2700, 0x27BF),
]
RARE_BLOCKS = [
    (0x0530, 0x058F), (0x0590, 0x05FF), (0x0900, 0x097F), (0x0E00, 0x0E7F), (0x10A0, 0x10FF),
    (0x13A0, 0x13FF), (0x1680, 0x169F), (0x16A0, 0x16FF), (0x1800, 0x18AF), (0x2D30, 0x2D7F),
    (0x3100, 0x312F), (0xA490, 0xA4CF), (0xA500, 0xA63F), (0xA6A0, 0xA6FF),
]
# Blocks indexed for retrieval but never asked about. They make the corpus look less like an
# answer key and give the retriever plausible near-misses.
EXTRA_CORPUS_BLOCKS = [
    (0x0180, 0x024F), (0x0250, 0x02AF), (0x0600, 0x06FF), (0x0980, 0x09FF), (0x1E00, 0x1EFF),
    (0x2100, 0x214F), (0x2300, 0x23FF), (0x2460, 0x24FF), (0x2900, 0x297F), (0x2A00, 0x2AFF),
    (0x2B00, 0x2BFF), (0xFB00, 0xFB4F),
]

TZ_REGIONS = ("Africa", "America", "Antarctica", "Asia", "Atlantic", "Australia", "Europe",
              "Indian", "Pacific")
TZ_DATES = [("15 January 2025", datetime(2025, 1, 15, 12, tzinfo=timezone.utc)),
            ("15 July 2025", datetime(2025, 7, 15, 12, tzinfo=timezone.utc))]

SERVICES_PATH = "/etc/services"


# ---------------------------------------------------------------------------------------------
# Normalisers. Each returns a canonical value or None when nothing parseable is present.
# ---------------------------------------------------------------------------------------------

def first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def norm_element(text: str) -> str | None:
    """An element answer as its lower-case name; a bare symbol is accepted too."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return None
    by_name = {name.lower(): name.lower() for _, _, name in ELEMENTS}
    by_symbol = {sym.lower(): name.lower() for _, sym, name in ELEMENTS}
    for w in words:
        low = w.lower()
        low = NAME_VARIANTS.get(low, low)
        if low in by_name:
            return by_name[low]
    if len(words) == 1 and words[0].lower() in by_symbol:
        return by_symbol[words[0].lower()]
    return None


def norm_symbol(text: str) -> str | None:
    m = re.search(r"\b([A-Z][a-z]?)\b", text.strip())
    if m:
        return m.group(1).lower()
    m = re.search(r"[A-Za-z]{1,2}", text)
    return m.group().lower() if m else None


def norm_codepoint(text: str) -> int | None:
    m = re.search(r"(?:U\+|0x|\\u)([0-9A-Fa-f]{2,6})", text)
    if not m:
        m = re.search(r"\b([0-9A-Fa-f]*\d[0-9A-Fa-f]*)\b", text)
        if m and not 2 <= len(m.group(1)) <= 6:
            m = None
    return int(m.group(1), 16) if m else None


def norm_offset(text: str) -> int | None:
    """A UTC offset in minutes. Accepts +05:45, UTC+5:45, GMT-3, UTC, -0330, and a unicode minus."""
    t = text.replace("−", "-").replace("–", "-").strip()
    m = re.search(r"([+-])\s*(\d{1,2})(?::?(\d{2}))?", t)
    if m:
        minutes = int(m.group(2)) * 60 + int(m.group(3) or 0)
        return -minutes if m.group(1) == "-" else minutes
    if re.fullmatch(r"(?i)\s*(utc|gmt|z)\s*\.?\s*", t) or re.search(r"(?i)(\b(utc|gmt)\s*)?±\s*0+(:00)?\b", t):
        return 0
    return None


def format_offset(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    h, m = divmod(abs(minutes), 60)
    return f"{sign}{h:02d}:{m:02d}"


NORMALISERS = {
    "int": first_int,
    "element": norm_element,
    "symbol": norm_symbol,
    "codepoint": norm_codepoint,
    "offset": norm_offset,
}


def admit(item: dict, candidate: str) -> int:
    """The admission function e*: 1 when the candidate answer matches the reference exactly."""
    got = NORMALISERS[item["kind"]](candidate or "")
    return int(got is not None and got == item["ref"])


# ---------------------------------------------------------------------------------------------
# Families. Each builder returns (questions, corpus_documents).
# ---------------------------------------------------------------------------------------------

def build_elements() -> tuple[list[dict], list[dict]]:
    items, docs = [], []
    for z, sym, name in ELEMENTS:
        docs.append({
            "title": f"Periodic table: {name}",
            "text": f"{name}, chemical symbol {sym}, atomic number {z}.",
            "source": "IUPAC periodic table",
        })
        items += [
            {"q": f"What is the atomic number of {name.lower()}?", "kind": "int", "ref": z,
             "display": str(z), "slice": "atomic_number", "rare": z >= 57},
            {"q": f"Which chemical element has atomic number {z}?", "kind": "element",
             "ref": name.lower(), "display": name, "slice": "element_by_number", "rare": z >= 57},
            {"q": f"What is the chemical symbol for {name.lower()}?", "kind": "symbol",
             "ref": sym.lower(), "display": sym, "slice": "symbol", "rare": z >= 57},
            {"q": f"Which chemical element has the symbol {sym}?", "kind": "element",
             "ref": name.lower(), "display": name, "slice": "element_by_symbol", "rare": z >= 57},
        ]
    for it in items:
        it["family"] = "elements"
    return items, docs


def parse_services(path: str = SERVICES_PATH) -> list[tuple[str, int, str, list[str], str]]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            body, _, comment = line.partition("#")
            parts = body.split()
            if len(parts) < 2 or "/" not in parts[1]:
                continue
            port, _, proto = parts[1].partition("/")
            if not port.isdigit():
                continue
            rows.append((parts[0], int(port), proto, parts[2:], comment.strip()))
    return rows


def build_ports(rng: random.Random, n_low: int = 150, n_high: int = 350):
    rows = parse_services()
    tcp_ports: dict[str, set[int]] = {}
    for name, port, proto, aliases, _ in rows:
        if proto != "tcp":
            continue
        for n in [name, *aliases]:
            tcp_ports.setdefault(n.lower(), set()).add(port)
    unique = {n: next(iter(p)) for n, p in tcp_ports.items() if len(p) == 1}
    primary = {name.lower() for name, _, proto, _, _ in rows if proto == "tcp"}
    names = sorted(n for n in unique if n in primary and unique[n] <= 49151)
    low = [n for n in names if unique[n] < 1024]
    high = [n for n in names if unique[n] >= 1024]
    chosen = [(n, False) for n in rng.sample(low, n_low)] + [(n, True) for n in rng.sample(high, n_high)]
    items = [{
        "q": f'In the IANA service name and port registry, which TCP port number is assigned to the service named "{n}"?',
        "kind": "int", "ref": unique[n], "display": str(unique[n]), "family": "ports",
        "slice": "system_port" if not rare else "registered_port", "rare": rare,
    } for n, rare in chosen]
    docs = [{
        "title": f"IANA service registry: {name}",
        "text": f"{name} {port}/{proto}" + (f" (aliases: {' '.join(aliases)})" if aliases else "")
                + (f". {comment}" if comment else ""),
        "source": "/etc/services",
    } for name, port, proto, aliases, comment in rows]
    return items, docs


def _named(lo: int, hi: int) -> list[int]:
    return [cp for cp in range(lo, hi + 1) if unicodedata.name(chr(cp), None)]


def build_unicode(rng: random.Random, n_common: int = 250, n_rare: int = 250):
    common = [cp for lo, hi in COMMON_BLOCKS for cp in _named(lo, hi)]
    rare = [cp for lo, hi in RARE_BLOCKS for cp in _named(lo, hi)]
    chosen = [(cp, False) for cp in rng.sample(common, n_common)] + [(cp, True) for cp in rng.sample(rare, n_rare)]
    items = [{
        "q": f'What is the Unicode code point of the character named "{unicodedata.name(chr(cp))}"? Answer in the form U+XXXX.',
        "kind": "codepoint", "ref": cp, "display": f"U+{cp:04X}", "family": "unicode",
        "slice": "rare_script" if is_rare else "common_block", "rare": is_rare,
    } for cp, is_rare in chosen]
    corpus_cps = sorted({cp for lo, hi in COMMON_BLOCKS + RARE_BLOCKS + EXTRA_CORPUS_BLOCKS for cp in _named(lo, hi)})
    docs = [{
        "title": f"Unicode character: {unicodedata.name(chr(cp))}",
        "text": f"U+{cp:04X} {unicodedata.name(chr(cp))}, general category {unicodedata.category(chr(cp))}.",
        "source": f"Unicode Character Database {unicodedata.unidata_version}",
    } for cp in corpus_cps]
    return items, docs


def build_timezones(rng: random.Random, n: int = 500):
    zones = sorted(z for z in available_timezones() if z.split("/")[0] in TZ_REGIONS)
    pool, docs = [], []
    for z in zones:
        offs = []
        for label, when in TZ_DATES:
            minutes = int(when.astimezone(ZoneInfo(z)).utcoffset().total_seconds() // 60)
            offs.append(minutes)
            pool.append({
                "q": f'What was the UTC offset of the IANA time zone "{z}" on {label}? Answer in the form +HH:MM or -HH:MM.',
                "kind": "offset", "ref": minutes, "display": format_offset(minutes), "family": "timezones",
                "slice": label.split()[1].lower(), "rare": False,
            })
        docs.append({
            "title": f"Time zone database: {z}",
            "text": f"{z}: UTC offset {format_offset(offs[0])} on {TZ_DATES[0][0]} and {format_offset(offs[1])} on {TZ_DATES[1][0]}.",
            "source": "IANA tz database (system tzdata)",
        })
    return rng.sample(pool, n), docs


def build_all(seed: int = 20260818) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    items, docs = [], []
    for build in (lambda: build_elements(), lambda: build_ports(rng), lambda: build_unicode(rng),
                  lambda: build_timezones(rng)):
        its, ds = build()
        items += its
        docs += ds
    for i, it in enumerate(items):
        it["id"] = f"{it['family'][:2]}{i:04d}"
    for j, d in enumerate(docs):
        d["id"] = j
    return items, docs


def gold_title(item: dict) -> str:
    """Title of the corpus document that answers this item: the row a perfect retriever returns."""
    q = item["q"]
    if item["family"] == "elements":
        by_num = {str(z): n for z, _, n in ELEMENTS}
        by_sym = {s: n for _, s, n in ELEMENTS}
        m = re.search(r"atomic number (\d+)\?", q)
        if m:
            return f"Periodic table: {by_num[m.group(1)]}"
        m = re.search(r"symbol ([A-Z][a-z]?)\?", q)
        if m:
            return f"Periodic table: {by_sym[m.group(1)]}"
        m = re.search(r"(?:of|for) ([a-z]+)\?", q)
        return f"Periodic table: {m.group(1).capitalize()}"
    name = re.search(r'"([^"]+)"', q).group(1)
    return {"ports": f"IANA service registry: {name}", "unicode": f"Unicode character: {name}",
            "timezones": f"Time zone database: {name}"}[item["family"]]
