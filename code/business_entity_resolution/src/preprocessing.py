"""Loading and normalisation of the raw business-entity TSV files.

Scope
-----
Turns an on-disk ``*_source{1,2,3}.tsv`` into normalised records. Blocking,
features and the model consume only this module's output, never raw strings.

Dataset facts to respect
-------------------------
* All files are **tab-separated**. Reading without ``sep="\\t"`` silently yields a
  single column holding the whole line. Use ``encoding="utf-8"`` — the data
  contains Devanagari text and the validator rejects non-UTF-8 output.
* Columns: ``entity_id``, ``business_name``, ``business_address``, ``country``.
  A record's source is given by its ``entity_id`` prefix (``S1-``/``S2-``/``S3-``)
  and the file it lives in; there is no separate source column.
* ``business_name`` and ``business_address`` are frequently empty — allow for
  missing values rather than assuming a string is present. ``"null"`` also occurs
  literally as a missing-value sentinel (see :data:`MISSING_TOKENS`).
* ``country`` is an **open set of string labels**. Training covers ``US`` and
  ``India``; the test set additionally contains ``France``, which never appears
  in training. Do not hard-code, filter, or one-hot the pipeline to a fixed
  country list — a French entity that gets dropped never reaches the submission.
  :func:`country_profile` returns the *generic* profile for any label it has no
  specific rule for, so an unseen country cannot raise or be dropped.
* No identifier is shared across sources, so all linkage must come from text.

Data handling
-------------
The full corpus is ~11.7M records per split (2.2M/1.7M Source-1 rows against
~10.3M/10.0M Source-2+3 rows), so "stream rather than load" is a hard
requirement, not a style preference. Two mechanisms keep peak memory bounded:

* :func:`load_source` yields chunks of :class:`Record` and never materialises a
  whole source file.
* :func:`cache_split` / :func:`load_normalised` round-trip those chunks through a
  Parquet cache, and :class:`NormalisedSet` repacks the cache into fixed-width
  NumPy arrays. Fixed-width ``S`` arrays cost ~1 byte per character instead of
  the ~49 bytes of overhead a Python ``str`` object carries, which is the
  difference between 2.6 GB and 7 GB for the 10M-row match pool.

Token strings are stored space-joined and split on demand: a list of interned
tokens costs ~80 bytes of tuple overhead per record, the joined string costs
~3 bytes per character.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

#: Minimum similarity for a transliterated token to count as a legal form.
#: "praivet" vs "private" scores 86, "limittedd" vs "limited" scores 93; a token
#: that is merely *near* a form, such as "india" vs "inc" (75), is rejected.
_LEGAL_FUZZ_CUTOFF = 85

SOURCE_ID_PREFIXES = {1: "S1-", 2: "S2-", 3: "S3-"}

#: Overrides the location of the training labels; see :func:`override_ground_truth`.
_GROUND_TRUTH_OVERRIDE: Optional[Path] = None

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

#: Where the raw data may live, relative to the repository root. The upstream
#: Kaggle bundle nests it one level deeper than this project's README claims, so
#: resolution is a search rather than a fixed path.
DATA_DIR_CANDIDATES = (
    "dataset/student_resource/dataset",
    "dataset",
)

#: Literal strings the upstream dump uses for "no value". ``keep_default_na=False``
#: means these survive as real text and would otherwise look like content.
MISSING_TOKENS = frozenset({"null", "none", "nan", "n/a", "nil", "-", "--", "?"})

# --------------------------------------------------------------------------- #
# Name normalisation tables
# --------------------------------------------------------------------------- #

#: Entity-type designations that are unambiguous wherever they appear, so they are
#: dropped from anywhere in the name. ``"Holloway Peak Inc Seafood"`` and
#: ``"Holloway Peak Seafood"`` are the same business, but only if the infix
#: ``Inc`` goes too -- a trailing-only rule misses every mid-name designation.
LEGAL_INFIX: Dict[str, str] = {
    "corp": "corp", "corporation": "corp", "inc": "inc", "incorporated": "inc",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc", "gmbh": "gmbh", "sarl": "sarl",
    "srl": "srl", "sas": "sas", "sasu": "sas", "bv": "bv", "nv": "nv",
    "spa": "spa", "kg": "kg", "kk": "kk", "pte": "pte", "pty": "pty",
}

#: Designations only stripped from the end of a name. ``Co`` in particular is
#: also an ordinary word ("Co Op", "Smith Co"), so removing it mid-name would
#: corrupt real names; at the end of a name it is almost always the entity type.
LEGAL_SUFFIX_ONLY: Dict[str, str] = {
    "co": "co", "company": "co", "ag": "ag", "sa": "sa", "as": "as",
    "ab": "ab", "lp": "lp", "sl": "sl",
}

#: Entity-type designations. Keys are the token as it appears after punctuation
#: removal and case folding; values are the canonical form. A trailing run of
#: these is lifted out of the core name into :attr:`Record.name_legal` so the
#: core name can be compared independently of how the entity was incorporated.
LEGAL_FORMS: Dict[str, str] = {**LEGAL_INFIX, **LEGAL_SUFFIX_ONLY}

#: Multi-token legal forms, checked before the single-token pass so that
#: "Private Limited" canonicalises to the same "pvt ltd" as "Pvt. Ltd.".
LEGAL_PHRASES: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("private", "limited"), ("pvt", "ltd")),
    (("pvt", "ltd"), ("pvt", "ltd")),
    (("p", "ltd"), ("pvt", "ltd")),
    (("private", "ltd"), ("pvt", "ltd")),
    (("co", "ltd"), ("co", "ltd")),
    (("and", "co"), ("co",)),
    (("s", "a", "r", "l"), ("sarl",)),
    (("s", "a"), ("sa",)),
)

#: Conservative abbreviation expansion for business names. Only pairs that are
#: unambiguous in ordinary English business usage; anything ambiguous is left for
#: the similarity features rather than being forced to a single surface form.
NAME_ABBREVIATIONS: Dict[str, str] = {
    "intl": "international", "int": "international", "natl": "national",
    "assoc": "associates", "bros": "brothers", "mfg": "manufacturing",
    "mfr": "manufacturer", "ctr": "center", "svcs": "services",
    "svc": "service", "tech": "technology", "ind": "industries",
    "indl": "industrial", "mgmt": "management", "mktg": "marketing",
    "univ": "university", "inst": "institute", "hosp": "hospital",
    "dept": "department", "dist": "distribution", "whse": "warehouse",
    "bldg": "building", "apt": "apartment", "ste": "suite", "blvd": "boulevard",
    "rd": "road", "st": "street", "ave": "avenue", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ter": "terrace", "plz": "plaza",
    "sq": "square", "mt": "mount", "ft": "fort", "no": "number",
    "opp": "opposite", "nr": "near", "nrt": "near", "sec": "section",
    "grp": "group", "cos": "company", "pics": "pictures", "photo": "photography",
}

#: Tokens that carry no identifying signal in a business name. Used for token
#: Jaccard and for deciding which tokens are worth a blocking key.
NAME_STOPWORDS = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "for", "to", "on", "by", "with",
    "or", "de", "la", "le", "les", "el", "du", "des", "di", "da", "et",
    "this", "that", "our", "your", "my", "its",
})

#: Generic commercial words. Not stopwords — they are part of the name — but they
#: are poor blocking keys because they occur in too many records.
NAME_GENERIC = frozenset({
    "service", "services", "general", "company", "group", "international",
    "india", "global", "national", "private", "public", "new", "old", "modern",
    "enterprise", "enterprises", "solution", "solutions", "system", "systems",
    "center", "centre", "store", "shop", "office", "business", "trade", "works",
    "worldwide", "world", "enterprises", "industries", "holdings", "partners",
    "associate", "associates", "management", "consultants", "consulting",
})

# --------------------------------------------------------------------------- #
# Address normalisation tables
# --------------------------------------------------------------------------- #

#: Street-type vocabulary. Long form is canonical; :data:`STREET_SHORT` is the
#: reverse map used to build the "compact" variant used for blocking keys, so
#: "County Road 705" and "Cr 705" collide on the same key.
STREET_TYPES: Dict[str, str] = {
    "street": "street", "st": "street", "str": "street",
    "road": "road", "rd": "road",
    "avenue": "avenue", "ave": "avenue", "av": "avenue", "aven": "avenue",
    "boulevard": "boulevard", "blvd": "boulevard", "boul": "boulevard",
    "lane": "lane", "ln": "lane",
    "drive": "drive", "dr": "drive", "drv": "drive",
    "court": "court", "ct": "court", "crt": "court",
    "place": "place", "pl": "place",
    "square": "square", "sq": "square", "sqr": "square",
    "highway": "highway", "hwy": "highway", "highwy": "highway",
    "parkway": "parkway", "pkwy": "parkway", "pky": "parkway", "parkwy": "parkway",
    "terrace": "terrace", "ter": "terrace", "terr": "terrace",
    "circle": "circle", "cir": "circle", "circl": "circle",
    "way": "way", "wy": "way",
    "loop": "loop", "ln": "lane",
    "trail": "trail", "trl": "trail",
    "crescent": "crescent", "cres": "crescent",
    "gardens": "gardens", "gdns": "gardens", "garden": "gardens",
    "villa": "villa", "villas": "villa",
    "apartments": "apartments", "apartment": "apartments", "apts": "apartments",
    "nagar": "nagar", "colony": "colony", "society": "society",
    "plaza": "plaza", "plz": "plaza", "mall": "mall",
    "complex": "complex", "cx": "complex", "centre": "center", "center": "center",
    "building": "building", "bldg": "building", "bldgs": "building",
    "tower": "tower", "towers": "tower",
    "rua": "rua", "rue": "rue", "avenida": "avenida", "calle": "calle",
    "via": "via", "viale": "viale", "place": "place",
}

STREET_SHORT: Dict[str, str] = {
    "street": "st", "road": "rd", "avenue": "av", "boulevard": "bl",
    "lane": "ln", "drive": "dr", "court": "ct", "place": "pl",
    "square": "sq", "highway": "hwy", "parkway": "pkwy", "terrace": "ter",
    "circle": "cir", "way": "wy", "loop": "lp", "trail": "trl",
    "crescent": "cres", "gardens": "gdns", "villa": "villa",
    "apartments": "apts", "center": "ctr", "building": "bldg",
    "tower": "twr", "nagar": "ngr", "colony": "col", "society": "soc",
    "plaza": "plz", "complex": "cpx", "mall": "mall",
}

#: Compass points. Expanded for the long form, shortened in the compact variant.
DIRECTIONS: Dict[str, str] = {
    "north": "north", "n": "north", "south": "south", "s": "south",
    "east": "east", "e": "east", "west": "west", "w": "west",
    "northeast": "northeast", "ne": "northeast", "northwest": "northwest",
    "nw": "northwest", "southeast": "southeast", "se": "southeast",
    "southwest": "southwest", "sw": "southwest",
}
DIRECTION_SHORT: Dict[str, str] = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}

#: Secondary address vocabulary that is not a street type but behaves like one.
ADDRESS_SYNONYMS: Dict[str, str] = {
    "apartment": "apartment", "apt": "apartment", "flat": "apartment",
    "suite": "suite", "ste": "suite", "unit": "unit", "floor": "floor",
    "fl": "floor", "ground": "floor", "first": "floor", "second": "floor",
    "third": "floor", "fourth": "floor", "fifth": "floor",
    "number": "number", "no": "number", "num": "number",
    "plot": "plot", "house": "plot", "shop": "plot", "block": "plot",
    "door": "plot", "premises": "plot", "property": "plot",
    "near": "near", "nr": "near", "nrt": "near", "opposite": "opposite",
    "opp": "opposite", "beside": "beside", "adjacent": "beside",
    "next": "beside", "behind": "behind", "across": "across",
    "sector": "sector", "sec": "sector", "phase": "phase", "layout": "layout",
    "district": "district", "dist": "district", "city": "city",
    "town": "town", "village": "town", "county": "county", "state": "state",
    "province": "province", "region": "region", "postal": "postal",
    "zip": "postal", "pin": "postal", "pincode": "postal", "zipcode": "postal",
    "post": "postal", "code": "postal",
}

#: Address words carrying no geographic information. Dropped from the "core"
#: token list so that "1087 County Road 705, West Columbia" and "1087 CR 705 W
#: Columbia St" share the same evidence.
#:
#: Note what is *not* here: compass points. "W Columbia" and "West Columbia"
#: are the same place, and the direction is what tells "West Columbia TX" apart
#: from "Columbia SC", so directions stay in the core token set. The state is
#: removed separately, in :func:`parse_address`, once it has been canonicalised —
#: that way "TX" and "Texas" both collapse out and leave the same residue.
ADDRESS_STOPWORDS = frozenset({
    "the", "of", "and", "a", "an", "in", "at", "to", "for", "on", "by", "with",
    "or", "de", "la", "le", "les", "du", "des", "di", "da", "et", "von", "van",
    "del", "della", "dos", "das",
}) | set(STREET_TYPES) | set(ADDRESS_SYNONYMS) | {
    "inc", "corp", "corporation", "co", "company", "ltd", "limited", "pvt",
    "private", "llc", "llp", "lp", "plc", "gmbh", "ag", "sa", "sarl", "bv",
    "nv", "spa", "srl", "mr", "mrs", "ms", "dr", "shri", "smt",
}

#: Tokens that mark the start of a landmark phrase. Everything after one of
#: these, up to the next comma-separated clause, is a landmark reference.
LANDMARK_MARKERS = frozenset({
    "near", "nearby", "opposite", "opp", "beside", "adjacent", "next", "behind",
    "across", "infront", "towards", "toward", "by",
})

#: Vocabulary of things that appear in *every* neighbourhood. Shared between
#: unrelated businesses, so overlap here is deliberately kept out of the core
#: token set and exposed as its own (weak) feature instead.
LANDMARK_WORDS = frozenset({
    "atm", "bank", "banking", "hotel", "hospital", "clinic", "dispensary",
    "school", "college", "university", "institute", "temple", "mosque", "church",
    "gurudwara", "bus", "stop", "railway", "station", "metro", "market",
    "mandi", "bazaar", "post", "office", "circle", "junction", "crossing",
    "bypass", "flyover", "signal", "gate", "campus", "park", "zoo", "theatre",
    "theater", "cinema", "mall", "stadium", "port", "airport", "bridge",
    "residency", "apartments", "centre", "center", "chowk", "ganj", " crossroads",
    "busstop", "police", "station", "court", "temple", "gas", "petrol",
})

#: Words that introduce a person or organisation name rather than a place.
_LANDMARK_TAIL = frozenset({"atm", "bank", "office", "hospital", "hotel", "temple"})
_US_STATE_CODES: Tuple[str, ...] = tuple(
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
    "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA "
    "WA WV WI WY DC".split()
)

_US_STATE_NAME_TO_CODE: Tuple[Tuple[str, str], ...] = (
    ("alabama", "al"), ("alaska", "ak"), ("arizona", "az"), ("arkansas", "ar"),
    ("california", "ca"), ("colorado", "co"), ("connecticut", "ct"),
    ("delaware", "de"), ("florida", "fl"), ("georgia", "ga"), ("hawaii", "hi"),
    ("idaho", "id"), ("illinois", "il"), ("indiana", "in"), ("iowa", "ia"),
    ("kansas", "ks"), ("kentucky", "ky"), ("louisiana", "la"), ("maine", "me"),
    ("maryland", "md"), ("massachusetts", "ma"), ("michigan", "mi"),
    ("minnesota", "mn"), ("mississippi", "ms"), ("missouri", "mo"),
    ("montana", "mt"), ("nebraska", "ne"), ("nevada", "nv"),
    ("new hampshire", "nh"), ("new jersey", "nj"), ("new mexico", "nm"),
    ("new york", "ny"), ("north carolina", "nc"), ("north dakota", "nd"),
    ("ohio", "oh"), ("oklahoma", "ok"), ("oregon", "or"),
    ("pennsylvania", "pa"), ("rhode island", "ri"), ("south carolina", "sc"),
    ("south dakota", "sd"), ("tennessee", "tn"), ("texas", "tx"),
    ("utah", "ut"), ("vermont", "vt"), ("virginia", "va"),
    ("washington", "wa"), ("west virginia", "wv"), ("wisconsin", "wi"),
    ("wyoming", "wy"), ("district of columbia", "dc"),
)

_US_STATE_WORDS: Tuple[str, ...] = tuple(name for name, _ in _US_STATE_NAME_TO_CODE)

#: US state names/abbreviations mapped to their postal code. A feature-only
#: table: an unrecognised token simply yields no state, which is why this does
#: not violate the open-country constraint.
_US_STATE_BY_WORD: Dict[str, str] = {}



#: Country-specific handling. Absent countries fall back to :data:`GENERIC_PROFILE`
#: so an unseen label (e.g. ``France``) is processed, never dropped.
GENERIC_PROFILE: Dict[str, object] = {
    "postal_lengths": (5, 6),
    "house_first": True,
    "state_words": (),
    "state_codes": (),
    "address_order": "house_first",
}

COUNTRY_PROFILES: Dict[str, Dict[str, object]] = {
    "us": {
        "postal_lengths": (5,),
        "house_first": True,
        "state_words": _US_STATE_WORDS,
        "state_codes": _US_STATE_CODES,
        "address_order": "house_first",
    },
    "india": {
        "postal_lengths": (6,),
        "house_first": True,
        # First-level divisions only. City names are deliberately absent: putting
        # "bengaluru" here made it look like a state, so it was lifted out of the
        # core token list and the one city token that carries the location
        # disappeared from every Indian address.
        "state_words": (
            "andhra pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
            "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
            "madhya pradesh", "maharashtra", "manipur", "meghalaya", "odisha",
            "punjab", "rajasthan", "tamil nadu", "telangana", "uttar pradesh",
            "uttarakhand", "west bengal", "delhi", "pondicherry", "chandigarh",
            "andaman nicobar", "dadra nagar", "daman diu", "lakshadweep", "sikkim",
            "arunachal", "nagaland", "tripura", "mizoram", "assam",
        ),
        "state_codes": (),
        "address_order": "house_first",
    },
    "france": {
        "postal_lengths": (5,),
        "house_first": False,
        # Regions, not cities. The five-digit postal code is the strong French
        # geographic signal and is extracted separately; the city stays in the
        # core token list where the similarity features can use it.
        "state_words": (
            "ile de france", "auvergne", "bretagne", "normandie",
            "pays de la loire", "grand est", "hauts de france", "occitanie",
            "provence alpes cote d azur", "corse", "normandy", "brittany",
        ),
        "state_codes": (),
        "address_order": "postal_first",
    },
}

#: US state postal codes, and the full spelling of each. Both the abbreviation
#: and the multi-word spelling are needed: "TX" and "Texas" are the same state
#: and must leave the same residue in the normalised address, and "North
#: Carolina" only matches as an adjacent token pair.

#: Transliteration artefacts. ``unidecode`` renders Devanagari with the inherent
#: schwa as a doubled vowel and doubles the consonant that carries it, so
#: "प्राइवेट लिमिटेड" comes out as "praaivett limittedd". Collapsing both restores
#: something close to the English spelling ("praivet limited"), which is what
#: makes cross-script similarity and legal-form detection work at all. Only
#: applied to text that was *not* already ASCII, so English names are untouched.
_VOWEL_DOUBLING_RE = re.compile(r"(aa|ee|ii|oo|uu)")
_CONSONANT_DOUBLING_RE = re.compile(r"([bcdfghjklmnpqrstvwxz])\1+")


def _collapse_translit(text: str) -> str:
    """Undo the schwa doubling ``unidecode`` introduces for non-Latin scripts."""

    text = _VOWEL_DOUBLING_RE.sub(lambda m: m.group(1)[0], text)
    return _CONSONANT_DOUBLING_RE.sub(r"\1", text)



def _build_us_state_map() -> None:
    assert len(_US_STATE_CODES) == 51, (
        f"expected 50 states + DC, got {len(_US_STATE_CODES)}"
    )
    assert len(_US_STATE_NAME_TO_CODE) == 51, (
        f"expected 51 US state names, got {len(_US_STATE_NAME_TO_CODE)}"
    )
    for code in _US_STATE_CODES:
        _US_STATE_BY_WORD[code.lower()] = code.lower()
    for name, code in _US_STATE_NAME_TO_CODE:
        _US_STATE_BY_WORD[name] = code


_build_us_state_map()

# --------------------------------------------------------------------------- #
# Pre-compiled patterns
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9a-z]+")
#: Clause boundaries inside an address. Commas are the main separator; the
#: spaced hyphen and semicolon catch "Bldg A-1 - 3rd Floor" and "Sector 15; Plot 4".
_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\s-\s")
_DOMAIN_RE = re.compile(r"^(?:www\.)?([a-z0-9][a-z0-9-]*)((?:\.[a-z0-9-]+)+)$")
_TLD = frozenset({
    "com", "net", "org", "co", "in", "us", "biz", "info", "io", "fr", "uk",
    "de", "shop", "store", "online", "site", "website", "co.in", "co.uk",
})
_LEADING_DIGITS_RE = re.compile(r"^\d+(?=[a-z])")
_ALNUM_SPLIT_RE = re.compile(r"[^0-9a-z]+")
_NUMERIC_RE = re.compile(r"^\d+$")
_ALNUM_POSTAL_RE = re.compile(r"^(\d)[a-z]?(\d)[a-z]?(\d)[a-z]?$")


def _strip_accents_and_translit(text: str) -> str:
    """NFKC-fold, strip combining marks, then transliterate non-Latin scripts."""

    text = unicodedata.normalize("NFKC", text)
    if not text.isascii():
        text = _collapse_translit(unidecode(text).lower())
    return text


def _basic_clean(text: str) -> List[str]:
    """Lowercase, drop intra-token dots, split on everything non-alphanumeric.

    Intra-token dots are removed *before* tokenising so abbreviation periods
    collapse: ``"S.A.R.L."`` -> ``"sarl"``, ``"No.-570/13"`` -> ``"no 570"``.
    A bare ``"wilfordhancock.com"`` is special-cased first, because removing the
    dot there would glue the TLD onto the name.
    """

    text = _strip_accents_and_translit(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("+", " and ")
    text = re.sub(r"\.com\b", " ", text) if _looks_like_domain(text) else text
    text = text.replace(".", "")
    return [t for t in _PUNCT_RE.split(text) if t]


def _looks_like_domain(text: str) -> bool:
    stripped = text.strip()
    if stripped.count(".") < 1 or " " in stripped:
        return False
    head, _, tail = stripped.rpartition(".")
    return bool(head) and tail in _TLD


@dataclass(slots=True)
class Record:
    """A single business record from one source.

    Attributes
    ----------
    entity_id:
        Source-prefixed identifier, e.g. ``S2-00047``.
    name_raw, address_raw:
        The original strings as they appeared in the TSV, kept for debugging and
        raw-vs-normalised comparisons. Blocking and features use the normalised
        forms instead.
    country:
        Raw country label. An open-set string, never a fixed enum.
    name_norm, address_norm:
        Output of :func:`normalise_name` / :func:`normalise_address`.
    name_tokens, address_tokens:
        Tokens of the normalised strings, used by :mod:`features`.

    Extended attributes (present on the documented core plus the fields the
    later stages need, all derived from the same normalisation pass):

    source:
        1/2/3, so downstream code never re-parses the ID prefix.
    name_legal:
        Canonical legal form lifted off the end of the name, e.g. ``"pvt ltd"``.
    address_core:
        Space-joined address tokens with street types, directions, stopwords and
        landmark words removed. This is the string the similarity features use:
        it is the part of the address that actually carries geographic signal.
    addr_house, addr_postal, addr_state, addr_city:
        Structured address components, ``""`` when not present.
    landmark_tokens:
        Space-joined landmark reference tokens, kept separate because
        ``"Near SBI ATM"`` style text is shared by unrelated businesses.
    country_norm:
        Case-folded, transliterated country label used for soft comparison.
    """

    entity_id: str
    name_raw: str
    address_raw: str
    country: str
    name_norm: str
    address_norm: str
    name_tokens: List[str]
    address_tokens: List[str]
    source: int = 0
    name_legal: str = ""
    address_core: str = ""
    addr_house: str = ""
    addr_postal: str = ""
    addr_state: str = ""
    addr_city: str = ""
    addr_tail2: str = ""
    landmark_tokens: str = ""
    country_norm: str = ""
    name_present: bool = True
    address_present: bool = True


# --------------------------------------------------------------------------- #
# Country handling
# --------------------------------------------------------------------------- #


def country_profile(country: str) -> Dict[str, object]:
    """Return the rule bundle for ``country``, falling back to generic rules.

    An unknown label is not an error: :data:`GENERIC_PROFILE` describes the
    common "number somewhere, number somewhere else, free text" shape that the
    US, Indian and French conventions share, so an unseen country is normalised
    by the same code path rather than special-cased or dropped.
    """

    key = _strip_accents_and_translit(country or "").lower().strip()
    return COUNTRY_PROFILES.get(key, GENERIC_PROFILE)


def normalise_country(country: str) -> str:
    """Case-folded, transliterated country label for soft string comparison."""

    return _WS_RE.sub(" ", _strip_accents_and_translit(country or "").lower()).strip()


# --------------------------------------------------------------------------- #
# Name / address normalisation
# --------------------------------------------------------------------------- #


def _strip_legal_form(tokens: List[str], fuzzy: bool = False) -> Tuple[List[str], List[str]]:
    """Peel a trailing legal-form run off ``tokens``.

    Returns ``(core_tokens, legal_tokens)``. Handles both the single-token forms
    ("Inc", "Pvt") and the multi-token ones ("Private Limited" -> "pvt ltd") by
    trying phrase matches before the single-token pass.

    ``fuzzy`` enables an approximate match against the legal-form vocabulary. It
    is switched on only for text that had to be transliterated, where no exact
    form can ever hit: the schwa-collapsed Devanagari for "प्राइवेट" is "praivet",
    which is two edits from "private" and would otherwise leave the suffix in the
    core name forever. English names keep exact matching, because an approximate
    match there would risk eating real words that happen to resemble a form.
    """

    if not tokens:
        return [], []

    for phrase, canonical in LEGAL_PHRASES:
        n = len(phrase)
        if len(tokens) > n and tuple(tokens[-n:]) == phrase:
            return tokens[:-n], list(canonical)

    legal: List[str] = []
    idx = len(tokens)
    while idx > 0 and len(legal) < 3:
        head = tokens[idx - 1]
        if head in NAME_STOPWORDS:
            idx -= 1
            continue
        canon = LEGAL_FORMS.get(head)
        if canon is None and fuzzy and len(head) >= 4:
            canon = _fuzzy_legal_form(head)
        if canon is None:
            break
        legal.insert(0, canon)
        idx -= 1
    if not legal:
        return tokens, []
    return tokens[:idx], legal


def _fuzzy_legal_form(token: str) -> Optional[str]:
    """Return the canonical legal form ``token`` is a near-miss for, else ``None``."""

    match = process.extractOne(
        token, list(LEGAL_FORMS), scorer=fuzz.ratio, score_cutoff=_LEGAL_FUZZ_CUTOFF
    )
    return LEGAL_FORMS[match[0]] if match else None


def normalise_name(name: str) -> str:
    """Return a comparison-ready form of a business name.

    Handle the documented noise patterns:

    * Unicode normalisation (NFKC) and case folding, so ``"Café Ltd"`` and
      ``"CAFE LTD"`` collapse together.
    * Transliteration of non-Latin scripts — the data contains Devanagari names
      such as ``"राम मार्केटिंग प्राइवेट लिमिटेड"``, and a transliterated form lets
      similarity work across scripts. Keep the original script too in case exact
      script comparison turns out to matter.
    * Legal-suffix normalisation — ``Corp``/``Corporation``, ``Pvt``/``Private``,
      ``Ltd``/``Limited``, ``Co``/``Company``, ``Inc``, ``LLP``, ``Sarl``, etc.
    * Punctuation and separator noise: ``&`` vs ``and``, hyphens, dots, slashes.
    * DBA/trade names and word-order transpositions. Typos are better handled by
      similarity features than by normalisation.

    Returns ``""`` for missing input rather than raising.
    """

    if not name or not name.strip():
        return ""
    was_non_latin = not name.isascii()
    tokens = _basic_clean(name)
    if not tokens:
        return ""
    if _looks_like_domain(name.lower().strip()):
        # "wilfordhancock.com" -> "wilfordhancock": a domain standing in for a
        # trade name, and the TLD is pure noise shared by every such record.
        head = name.lower().strip().removeprefix("www.")
        head = head.rsplit(".", 1)[0].replace("-", " ")
        tokens = [t for t in _ALNUM_SPLIT_RE.split(head) if t]
    tokens = [NAME_ABBREVIATIONS.get(t, t) for t in tokens]
    tokens = [_LEADING_DIGITS_RE.sub("", t) for t in tokens]
    tokens = [t for t in tokens if t and t not in MISSING_TOKENS]
    if not tokens:
        return ""
    core, _legal = _strip_legal_form(tokens, fuzzy=was_non_latin)
    if not core:
        core = tokens
    core = [t for t in core if t not in LEGAL_INFIX]
    return " ".join(core) if core else ""


def parse_address(address: str, country: str = "") -> Dict[str, str]:
    """Return the normalised address and its extracted structured components.

    Split out from :func:`normalise_address` because the structured parts are
    needed as separate fields downstream. Keys: ``norm`` (full normalised token
    string), ``core`` (geographic tokens only, state removed, house and postal
    included), ``house``, ``postal``, ``state``, ``city``, ``tail2`` (the last
    two core tokens, which is where a city name usually sits once the state is
    gone) and ``landmark``.

    The address is split on commas first. That matters: ``"Near SBI ATM, MG Road,
    Bengaluru"`` is one clause of landmark plus two of geography, and treating it
    as a single token run would discard the part that carries the location.
    """

    empty = {"norm": "", "core": "", "house": "", "postal": "", "state": "",
             "city": "", "tail2": "", "landmark": ""}
    if not address or not address.strip():
        return empty

    profile = country_profile(country)
    postal_lengths = set(profile["postal_lengths"])  # type: ignore[arg-type]

    expanded, landmark = _split_clauses(address)
    if not expanded:
        # Every clause was a landmark reference; retry without clause splitting so
        # a pure-landmark address still contributes whatever else it mentions.
        expanded, _ = _split_clauses(address, split_clauses=False)
    if not expanded:
        return empty

    geo: List[str] = []
    for tok in expanded:
        if tok in MISSING_TOKENS:
            continue
        tok = DIRECTIONS.get(tok, tok)
        tok = STREET_TYPES.get(tok, tok)
        tok = ADDRESS_SYNONYMS.get(tok, tok)
        if tok in LANDMARK_WORDS:
            landmark.append(tok)
            continue
        geo.append(tok)
    landmark = [t for t in landmark if t and t not in MISSING_TOKENS]
    if not geo:
        return empty

    # Structured numeric components. A first-position number is a house/plot
    # number far more often than a postal code, so the postal scan deliberately
    # skips position 0 unless it is the only number present (French "75001 Paris").
    numeric_positions = [i for i, t in enumerate(geo) if _NUMERIC_RE.match(t)]
    alpha_positions = [i for i, t in enumerate(geo) if t and not _NUMERIC_RE.match(t)]
    alpha = [geo[i] for i in alpha_positions]
    postal = ""
    postal_idx = -1
    for i in numeric_positions:
        if i > 0 and len(geo[i]) in postal_lengths:
            postal, postal_idx = geo[i], i
            break
    if not postal and len(numeric_positions) == 1:
        i = numeric_positions[0]
        if len(geo[i]) in postal_lengths:
            postal, postal_idx = geo[i], i
    if not postal and len(numeric_positions) >= 2:
        # US ZIP+4 arrives as two numbers: "12345 6789 Springfield".
        first, second = numeric_positions[0], numeric_positions[1]
        if (first == 0 and len(geo[first]) == 5 and len(geo[second]) == 4
                and second == 1):
            postal, postal_idx = geo[first], first

    house = ""
    for i in numeric_positions:
        if i == postal_idx:
            continue
        house = geo[i].lstrip("0") or "0"
        break

    state, state_idx = _extract_state(alpha, profile)
    state_words = set(profile["state_words"])  # type: ignore[arg-type]
    state_codes = {c.lower() for c in profile["state_codes"]}  # type: ignore[arg-type]
    dropped = {alpha_positions[i] for i in state_idx}

    # Core token list: geography only, with the state lifted out (it is already
    # available as its own feature) and the house/postal put back in (they are
    # strong evidence and would otherwise be lost with the numeric tokens).
    core = [
        t for i, t in enumerate(geo)
        if i not in dropped
        and not _NUMERIC_RE.match(t)
        and t not in ADDRESS_STOPWORDS
        and t not in LANDMARK_WORDS
        and t not in state_words
        and t not in state_codes
    ]
    core = [t for t in core if len(t) >= 2]
    if house and house not in core:
        core.insert(0, house)
    if postal and postal not in core:
        core.append(postal)

    city = ""
    tail2 = ""
    for tok in reversed(core):
        if len(tok) >= 4 and _NUMERIC_RE.match(tok) is None:
            city = tok
            break
    tail_tokens = [t for t in core if _NUMERIC_RE.match(t) is None][-2:]
    tail2 = " ".join(tail_tokens)

    return {
        "norm": " ".join(geo),
        "core": " ".join(core),
        "house": house,
        "postal": postal,
        "state": state,
        "city": city,
        "tail2": tail2,
        "landmark": " ".join(landmark),
    }


def _split_clauses(address: str, split_clauses: bool = True) -> Tuple[List[str], List[str]]:
    """Split an address into geographic and landmark tokens, clause by clause.

    A clause whose first landmark marker is at position ``p`` contributes
    ``tokens[:p]`` to the geography and ``tokens[p + 1:]`` to the landmark list.
    Splitting on commas first is what stops ``"Near SBI ATM, MG Road, Bengaluru"``
    from losing its location to the landmark marker.
    """

    raw_clauses = (
        [c for c in _CLAUSE_SPLIT_RE.split(address) if c.strip()]
        if split_clauses else [address]
    )
    geographic: List[str] = []
    landmark: List[str] = []
    for clause in raw_clauses:
        tokens = [t for t in _basic_clean(clause) if t and t not in MISSING_TOKENS]
        if not tokens:
            continue
        marker_at = next(
            (i for i, t in enumerate(tokens) if t in LANDMARK_MARKERS), None
        )
        if marker_at is None:
            geographic.extend(tokens)
        else:
            geographic.extend(tokens[:marker_at])
            landmark.extend(tokens[marker_at + 1:])
    return geographic, landmark


def _extract_state(
    alpha: Sequence[str], profile: Dict[str, object]
) -> Tuple[str, set]:
    """Find the state/region token, scanning from the end where it usually sits.

    Returns ``(canonical_state, indices_to_drop)``. Two-token names such as
    ``"North Carolina"`` have to be matched as an adjacent pair, and the indices
    are returned so the caller removes exactly the tokens that formed the state
    -- removing a bare ``"north"`` as a compass point instead would be wrong.

    A closed vocabulary is fine here even though ``country`` is an open set: an
    unrecognised token yields ``""`` and the record is still processed, which is
    exactly the behaviour the open-country constraint requires.
    """

    state_words = set(profile["state_words"])  # type: ignore[arg-type]
    state_codes = {c.lower() for c in profile["state_codes"]}  # type: ignore[arg-type]
    for i in range(len(alpha) - 1, -1, -1):
        if alpha[i] in state_codes:
            return alpha[i], {i}
        if alpha[i] in state_words:
            return _US_STATE_BY_WORD.get(alpha[i], alpha[i]), {i}
        if i > 0 and f"{alpha[i - 1]} {alpha[i]}" in state_words:
            pair = f"{alpha[i - 1]} {alpha[i]}"
            return _US_STATE_BY_WORD.get(pair, pair), {i - 1, i}
    return "", set()


def normalise_address(address: str, country: str = "") -> str:
    """Return a comparison-ready form of a business address.

    See :func:`parse_address` for the structured components; this returns the
    full normalised token string (``"1087 county road 705 west columbia tx"``).
    Abbreviations are expanded, landmark clauses are removed, and components are
    reduced to an order-independent token set, which is what makes
    ``"IA, Iowa City, 1064 Newton Rd"`` and ``"1064 Newton Rd, Iowa City, IA"``
    comparable. Returns ``""`` for missing input.
    """

    if not address or not address.strip():
        return ""
    return parse_address(address, country)["norm"]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def resolve_data_dir(data_dir: Optional[str] = None) -> Path:
    """Locate the directory that holds ``train/`` and ``test/``.

    Tries ``data_dir`` first, then :data:`DATA_DIR_CANDIDATES` relative to the
    repository root, then relative to the current directory. The upstream Kaggle
    bundle nests the files one level deeper than this project's README states,
    so the search is what keeps every entry point working unchanged.
    """

    tried: List[Path] = []
    if data_dir:
        tried.append(Path(data_dir))
    here = Path(__file__).resolve()
    # .../<repo>/code/business_entity_resolution/src/preprocessing.py -> <repo>
    repo_root = here.parents[3]
    for cand in DATA_DIR_CANDIDATES:
        tried.append(repo_root / cand)
        tried.append(Path.cwd() / cand)
        tried.append(repo_root / ".." / cand)
    for path in tried:
        if (path / "train").is_dir() and (path / "test").is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "could not locate a data directory containing train/ and test/; tried: "
        + ", ".join(str(p) for p in tried)
    )


def source_path(split: str, source: int, data_dir: Optional[str] = None) -> Path:
    """Return the on-disk path of one source file."""

    base = resolve_data_dir(data_dir)
    return base / split / f"{split}_source{source}.tsv"


def ground_truth_path(data_dir: Optional[str] = None) -> Path:
    """Return the on-disk path of the training ground truth."""

    if _GROUND_TRUTH_OVERRIDE is not None:
        return _GROUND_TRUTH_OVERRIDE
    return resolve_data_dir(data_dir) / "train" / "train_ground_truth.tsv"


def override_ground_truth(path: Optional[str | Path]) -> None:
    """Point :func:`ground_truth_path` at a different labels file, or reset it.

    Used by the pipeline smoke test, which runs against a sliced cache and
    therefore needs labels restricted to the same slice. Passing ``None`` restores
    the default location. Module-level rather than a parameter because the labels
    path is a property of the run, not of any one function's arguments.
    """

    global _GROUND_TRUTH_OVERRIDE
    _GROUND_TRUTH_OVERRIDE = Path(path) if path is not None else None


def make_record(entity_id: str, name: str, address: str, country: str, source: int) -> Record:
    """Normalise one raw row into a :class:`Record`."""

    name = "" if name is None else str(name)
    address = "" if address is None else str(address)
    country = "" if country is None else str(country)
    if name.strip().lower() in MISSING_TOKENS:
        name = ""
    if address.strip().lower() in MISSING_TOKENS:
        address = ""

    name_norm = normalise_name(name)
    parts = parse_address(address, country)
    addr_tokens = parts["norm"].split() if parts["norm"] else []
    name_tokens = name_norm.split() if name_norm else []

    return Record(
        entity_id=entity_id,
        name_raw=name,
        address_raw=address,
        country=country,
        name_norm=name_norm,
        address_norm=parts["norm"],
        name_tokens=name_tokens,
        address_tokens=addr_tokens,
        source=source,
        name_legal=_canonical_legal(name),
        address_core=parts["core"],
        addr_house=parts["house"],
        addr_postal=parts["postal"],
        addr_state=parts["state"],
        addr_city=parts["city"],
        addr_tail2=parts["tail2"],
        landmark_tokens=parts["landmark"],
        country_norm=normalise_country(country),
        name_present=bool(name_norm),
        address_present=bool(parts["norm"]),
    )


_LEGAL_CACHE: Dict[str, str] = {}


def _canonical_legal(name: str) -> str:
    """Canonical legal form for a raw name, memoised per distinct name string."""

    if not name:
        return ""
    cached = _LEGAL_CACHE.get(name)
    if cached is not None:
        return cached
    was_non_latin = not name.isascii()
    tokens = [t for t in _basic_clean(name) if t and t not in MISSING_TOKENS]
    _core, legal = _strip_legal_form(tokens, fuzzy=was_non_latin)
    if not legal:
        # Try again with abbreviation expansion, so "Pvt." resolves like "Pvt".
        expanded = [NAME_ABBREVIATIONS.get(t, t) for t in tokens]
        _core, legal = _strip_legal_form(expanded, fuzzy=was_non_latin)
    value = " ".join(legal)
    if len(_LEGAL_CACHE) < 400_000:
        _LEGAL_CACHE[name] = value
    return value


def _normalise_chunk(args) -> "list[Record]":
    """Worker entry point: normalise a chunk of raw rows."""

    source, entity_ids, names, addresses, countries = args
    out = []
    for entity_id, name, address, country in zip(entity_ids, names, addresses, countries):
        out.append(make_record(entity_id, name, address, country, source))
    return out


def iter_source_chunks(
    split: str,
    source: int,
    data_dir: Optional[str] = None,
    chunksize: int = 200_000,
    limit: Optional[int] = None,
) -> Iterator[List[Record]]:
    """Stream one source file as chunks of :class:`Record`.

    Parallelises the normalisation within each chunk, so the yield granularity is
    ``chunksize`` rows but the CPU work is spread across cores.
    """

    path = source_path(split, source, data_dir)
    reader_kwargs = dict(
        sep="\t", dtype=str, keep_default_na=False, encoding="utf-8", na_filter=False
    )
    seen = 0
    for frame in pd.read_csv(path, chunksize=chunksize, **reader_kwargs):
        if limit is not None and seen >= limit:
            break
        if limit is not None and seen + len(frame) > limit:
            frame = frame.iloc[: limit - seen]
        seen += len(frame)
        _warn_bad_prefixes(frame["entity_id"].to_numpy(), source)
        yield _frame_to_records(frame, source)


def _has_prefix(ids: np.ndarray, prefix: str) -> np.ndarray:
    """Boolean mask of rows whose ID starts with ``prefix``.

    The needle is encoded to match the haystack: under NumPy 2, ``np.char.startswith``
    has no loop mixing a fixed-width byte array with a ``str`` argument.
    """

    if ids.dtype.kind == "S":
        return np.char.startswith(ids, prefix.encode("ascii"))
    return np.char.startswith(ids.astype(str), prefix)


def _warn_bad_prefixes(ids: np.ndarray, source: int) -> None:
    """Report up to five rows whose ``entity_id`` contradicts the file they are in."""

    prefix = SOURCE_ID_PREFIXES[source]
    bad = np.flatnonzero(~_has_prefix(ids, prefix))
    for i in bad[:5]:
        print(
            f"[preprocessing] warning: source{source} row {i} has id {ids[i]!r} "
            f"without the expected {prefix} prefix",
            flush=True,
        )


def load_source(
    split: str,
    source: int,
    path: Optional[str] = None,
    chunksize: int = 200_000,
    data_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[List[Record]]:
    """Stream the records of one source file as chunks of :class:`Record`.

    Parameters
    ----------
    split:
        ``"train"`` or ``"test"``.
    source:
        Which source to read, 1/2/3. Sets the expected ``entity_id`` prefix;
        records with a mismatched prefix are reported, not silently accepted.
    path:
        Override for the file location. Defaults to
        ``dataset/{split}/{split}_source{source}.tsv``.
    chunksize:
        Rows per yielded chunk.

    Yields
    ------
    List[Record]
        Chunks in file order, so the caller can stream onward without holding
        the whole file.
    """

    if path is not None:
        reader_kwargs = dict(
            sep="\t", dtype=str, keep_default_na=False, encoding="utf-8",
            na_filter=False,
        )
        seen = 0
        for frame in pd.read_csv(path, chunksize=chunksize, **reader_kwargs):
            if limit is not None and seen >= limit:
                break
            if limit is not None and seen + len(frame) > limit:
                frame = frame.iloc[: limit - seen]
            seen += len(frame)
            yield _frame_to_records(frame, source)
        return
    yield from iter_source_chunks(split, source, data_dir, chunksize, limit)


def _frame_to_records(frame: "pd.DataFrame", source: int) -> List[Record]:
    """Normalise a raw frame in parallel, preserving row order."""

    ids = frame["entity_id"].to_numpy()
    names = frame["business_name"].fillna("").to_numpy()
    addresses = frame["business_address"].fillna("").to_numpy()
    countries = frame["country"].fillna("").to_numpy()
    step = 50_000
    args = [
        (source, ids[i:i + step], names[i:i + step], addresses[i:i + step],
         countries[i:i + step])
        for i in range(0, len(ids), step)
    ]
    if len(args) == 1:
        return _normalise_chunk(args[0])
    with ProcessPoolExecutor() as pool:
        out: List[Record] = []
        for sub in pool.map(_normalise_chunk, args):
            out.extend(sub)
        return out


def load_all_sources(
    split: str,
    sources: Iterable[int] = (1, 2, 3),
    data_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[List[Record]]:
    """Stream records from several sources of one split, chunk by chunk.

    The usual access pattern is two partitions: Source 1 (the deduplicated
    reference, the entities we predict for) and Sources 2+3 (the pool we match
    into). Pass ``sources=(1,)`` and ``sources=(2, 3)`` to iterate them
    independently rather than interleaved.
    """

    for source in sources:
        for chunk in load_source(split, source, data_dir=data_dir, limit=limit):
            yield chunk


# --------------------------------------------------------------------------- #
# Parquet cache
# --------------------------------------------------------------------------- #

CACHE_COLUMNS = [
    "entity_id", "source", "country_norm", "name_norm", "name_legal",
    "address_core", "address_norm", "addr_house", "addr_postal", "addr_state",
    "addr_city", "addr_tail2", "landmark_tokens", "name_present", "address_present",
]


def _records_to_frame(records: Sequence[Record]) -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "entity_id": [r.entity_id for r in records],
            "source": np.fromiter((r.source for r in records), dtype=np.int8, count=len(records)),
            "country_norm": [r.country_norm for r in records],
            "name_norm": [r.name_norm for r in records],
            "name_legal": [r.name_legal for r in records],
            "address_core": [r.address_core for r in records],
            "address_norm": [r.address_norm for r in records],
            "addr_house": [r.addr_house for r in records],
            "addr_postal": [r.addr_postal for r in records],
            "addr_state": [r.addr_state for r in records],
            "addr_city": [r.addr_city for r in records],
            "addr_tail2": [r.addr_tail2 for r in records],
            "landmark_tokens": [r.landmark_tokens for r in records],
            "name_present": np.fromiter(
                (r.name_present for r in records), dtype=bool, count=len(records)
            ),
            "address_present": np.fromiter(
                (r.address_present for r in records), dtype=bool, count=len(records)
            ),
        },
        columns=CACHE_COLUMNS,
    )


def cache_split(
    split: str,
    cache_dir: str | Path,
    sources: Sequence[int] = (1, 2, 3),
    data_dir: Optional[str] = None,
    chunksize: int = 400_000,
    limit: Optional[int] = None,
    n_jobs: int = 0,
    force: bool = False,
) -> List[Path]:
    """Normalise a whole split and write one Parquet file per source.

    Returns the written paths. Idempotent: an existing file is kept unless
    ``force`` is passed, which is what makes the stages restartable. An existing
    file is only trusted when its footer row count matches the source, because a
    truncated cache that looks complete is far worse than a missing one.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sources:
        out = cache_dir / f"{split}_source{source}.parquet"
        written.append(out)
        expected = _source_row_count(split, source, data_dir, limit)
        if out.exists() and not force:
            have = parquet_row_count(out)
            if have == expected:
                print(
                    f"[preprocessing] {out.name} cached ({have:,} rows)", flush=True
                )
                continue
            print(
                f"[preprocessing] {out.name} has {have:,} rows, expected "
                f"{expected:,} - rebuilding",
                flush=True,
            )
        rows = _write_parquet_split(
            out,
            (
                _records_to_frame(chunk)
                for chunk in load_source(
                    split, source, data_dir=data_dir,
                    chunksize=chunksize, limit=limit,
                )
            ),
        )
        if rows != expected:
            raise RuntimeError(
                f"{out.name}: normalised {rows:,} rows, source has {expected:,}"
            )
        print(f"[preprocessing] cached {out.name}: {rows:,} rows", flush=True)
    return written


def _source_row_count(
    split: str, source: int, data_dir: Optional[str], limit: Optional[int]
) -> int:
    """Row count of a raw source file, used to validate the cache."""

    path = source_path(split, source, data_dir)
    if limit is not None:
        return min(limit, _raw_row_count(path))
    return _raw_row_count(path)


def _raw_row_count(path: Path) -> int:
    """Row count of a raw source TSV, excluding the header.

    Counted in binary chunks rather than by decoding lines, since these files
    are 200-500 MB of UTF-8 with quoted fields and the count is only needed to
    validate the cache.
    """

    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            total += block.count(b"\n")
    # A record that is not newline-terminated is still a record, so the newline
    # count alone undercounts by one; then drop the header line.
    with open(path, "rb") as handle:
        handle.seek(max(0, path.stat().st_size - 1))
        ends_with_newline = handle.read(1) == b"\n"
    return max(total - 1 + (0 if ends_with_newline else 1), 0)


def _append_parquet(path: Path, frame: "pd.DataFrame") -> None:
    """Deprecated shim kept so callers of the old per-chunk helper fail loudly.

    Opening a :class:`pyarrow.parquet.ParquetWriter` on an existing path
    *overwrites* it, so this could only ever leave the final chunk behind. Use
    :func:`_write_parquet_split` instead.
    """

    raise RuntimeError(
        "_append_parquet cannot append; call _write_parquet_split with a single writer"
    )


def _write_parquet_split(path: Path, chunks: Iterable["pd.DataFrame"]) -> int:
    """Write an iterable of frames to one Parquet file and return the row count.

    A single :class:`pyarrow.parquet.ParquetWriter` stays open for the whole
    split. The file is built under a ``.partial`` name and renamed only after the
    writer has been closed and the row count read back from the footer, so a
    crashed or truncated run can never leave a file that looks complete -- the
    earlier per-chunk-writer version silently kept only the last chunk.
    """

    import pyarrow as pa
    import pyarrow.parquet as pq

    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    rows = 0
    writer = None
    try:
        for frame in chunks:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(partial, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
        if writer is None:
            raise ValueError(f"no chunks produced for {path.name}")
    finally:
        if writer is not None:
            writer.close()

    written = pq.ParquetFile(partial).metadata.num_rows
    if written != rows:
        raise RuntimeError(
            f"{path.name}: wrote {written} rows to disk but fed {rows} in"
        )
    partial.replace(path)
    return rows


def parquet_row_count(path: str | Path) -> int:
    """Row count from a Parquet footer, without materialising any columns."""

    import pyarrow.parquet as pq

    return pq.ParquetFile(Path(path)).metadata.num_rows


# --------------------------------------------------------------------------- #
# Fixed-width NumPy view
# --------------------------------------------------------------------------- #

#: Byte widths for the fixed-width arrays. Chosen from the measured length
#: distribution (name p99.9 = 57, max 104; address p99.9 = 148, max 223) with
#: headroom. Truncation only ever removes trailing tokens from a record that is
#: already far into the tail of the distribution.
NAME_WIDTH = 112
ADDRESS_WIDTH = 176
CORE_WIDTH = 128
SHORT_WIDTH = 32

_STR_COLUMNS = (
    "entity_id", "country_norm", "name_norm", "name_legal", "address_core",
    "address_norm", "addr_house", "addr_postal", "addr_state", "addr_city",
    "addr_tail2", "landmark_tokens",
)

_WIDTHS = {
    "entity_id": 16,
    "country_norm": SHORT_WIDTH,
    "name_norm": NAME_WIDTH,
    "name_legal": SHORT_WIDTH,
    "address_core": CORE_WIDTH,
    "address_norm": ADDRESS_WIDTH,
    "addr_house": SHORT_WIDTH,
    "addr_postal": SHORT_WIDTH,
    "addr_state": SHORT_WIDTH,
    "addr_city": SHORT_WIDTH,
    "addr_tail2": 48,
    "landmark_tokens": 96,
}


class RowView:
    """Attribute view over one row of a :class:`NormalisedSet`.

    Exposes exactly the attribute names :class:`Record` exposes for the fields
    the feature functions read, so a ``RowView`` and a ``Record`` are
    interchangeable as feature arguments. Constructing one costs a fraction of a
    microsecond, which matters at ~10^8 candidate pairs.
    """

    __slots__ = (
        "entity_id", "source", "country_norm", "name_norm", "name_legal",
        "address_core", "address_norm", "addr_house", "addr_postal",
        "addr_state", "addr_city", "addr_tail2", "landmark_tokens",
        "name_present", "address_present", "name_tokens", "address_tokens",
    )

    def __init__(self, store: "NormalisedSet", i: int) -> None:
        self.entity_id = store.entity_id[i]
        self.source = store.source[i]
        self.country_norm = store.country_norm[i]
        self.name_norm = store.name_norm[i]
        self.name_legal = store.name_legal[i]
        self.address_core = store.address_core[i]
        self.address_norm = store.address_norm[i]
        self.addr_house = store.addr_house[i]
        self.addr_postal = store.addr_postal[i]
        self.addr_state = store.addr_state[i]
        self.addr_city = store.addr_city[i]
        self.addr_tail2 = store.addr_tail2[i]
        self.landmark_tokens = store.landmark_tokens[i]
        self.name_present = bool(store.name_present[i])
        self.address_present = bool(store.address_present[i])
        self.name_tokens = self.name_norm.split()
        self.address_tokens = self.address_norm.split()


class NormalisedSet:
    """Columnar, fixed-width view of normalised records.

    The full Source-2+3 match pool is ~10M rows; holding it as Python objects
    costs several GB of per-string and per-tuple overhead. Holding it as
    fixed-width ``S`` arrays costs about one byte per character, which keeps the
    pool inside a few GB and leaves room for the blocking index.
    """

    __slots__ = _STR_COLUMNS + ("name_present", "address_present", "_n")

    def __init__(self, arrays: Dict[str, np.ndarray]) -> None:
        for col in _STR_COLUMNS:
            if col in arrays:
                setattr(self, col, arrays[col])
        for col in ("name_present", "address_present"):
            if col in arrays:
                setattr(self, col, arrays[col])
        self._n = len(self.entity_id)

    def __len__(self) -> int:
        return self._n

    def __len__(self) -> int:
        return self._n

    def get(self, i: int) -> RowView:
        return RowView(self, i)

    def nbytes(self) -> int:
        # Absent columns are skipped rather than assumed. A projected load (which
        # is what the blocking and feature stages both use) only carries a subset
        # of _STR_COLUMNS and omits the two bool flags entirely, so iterating the
        # full tuple raised AttributeError on any projected set.
        total = 0
        for col in _STR_COLUMNS + ("name_present", "address_present"):
            array = getattr(self, col, None)
            if array is not None:
                total += array.nbytes
        return total


def _read_parquet_columns(
    path: Path,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    batch_rows: int = 250_000,
    into: Optional[Dict[str, np.ndarray]] = None,
    offset: int = 0,
) -> Dict[str, np.ndarray]:
    """Read the projected columns into fixed-width byte arrays, batch by batch.

    Deliberately *not* ``pq.read_table`` followed by ``to_pylist``. For the
    training pool that would hold, at the same time, the whole Arrow table
    (~2.5 GB for 10.3M rows), a list of 10.3M Python ``str`` objects per column
    (~0.8 GB, transiently), and the finished fixed-width arrays (6.11 GB). Only
    the last of those has to outlive the call, so the reader streams record
    batches and writes each batch straight into its preallocated destination.
    That leaves the peak at the 6.11 GB of destination arrays plus one batch.

    The truncation is silent past ``limit`` rather than an error, matching the
    previous ``table.slice`` behaviour.

    ``into``/``offset`` write into caller-supplied destinations starting at
    ``offset`` instead of allocating, which is how :func:`load_normalised` streams
    the per-source frames straight into one set of on-disk columns. The conversion
    stays here so the memmap and in-RAM paths cannot drift apart.
    """

    import pyarrow.parquet as pq

    str_columns = list(_STR_COLUMNS) if columns is None else list(columns)
    wanted_str = [c for c in _STR_COLUMNS if c in set(str_columns)]
    bool_columns = [] if columns is not None else ["name_present", "address_present"]
    wanted_bool = [c for c in bool_columns if c in ("name_present", "address_present")]

    handle = pq.ParquetFile(path)
    total = handle.metadata.num_rows
    if limit is not None:
        total = min(total, limit)

    if into is None:
        out: Dict[str, np.ndarray] = {
            col: np.zeros(total, dtype=f"S{_WIDTHS[col]}") for col in wanted_str
        }
        for col in wanted_bool:
            out[col] = np.zeros(total, dtype=bool)
    else:
        out = into
        capacity = len(next(iter(out.values()))) if out else 0
        if offset + total > capacity:
            raise ValueError(
                f"{path.name} needs {offset + total:,} rows but the destination "
                f"holds {capacity:,}"
            )

    filled = 0
    for batch in handle.iter_batches(
        batch_size=batch_rows, columns=wanted_str + wanted_bool
    ):
        if filled >= total:
            break
        take = min(batch.num_rows, total - filled)
        stop = filled + take
        for col in wanted_str:
            # Per-batch conversion, so the Python objects are freed with the
            # batch rather than accumulating for every row.
            values = batch.column(batch.schema.get_field_index(col)).to_pylist()
            if take < len(values):
                values = values[:take]
            out[col][offset + filled : offset + stop] = np.array(
                values, dtype=f"S{_WIDTHS[col]}"
            )
        for col in wanted_bool:
            values = batch.column(batch.schema.get_field_index(col)).to_pylist()
            if take < len(values):
                values = values[:take]
            out[col][offset + filled : offset + stop] = np.asarray(values, dtype=bool)
        filled = stop
    if filled < total:
        raise ValueError(
            f"{path} yielded {filled:,} rows but {total:,} were expected; "
            "the cache is truncated or corrupt"
        )
    return out


def _norm_backing_paths(
    cache_dir: Path,
    split: str,
    sources: Sequence[int],
    columns: Sequence[str],
) -> Tuple[str, Dict[str, Path], Path]:
    """Name the on-disk backing for one projection of one split.

    The tag keys the file names on the source list *and* the column set, because
    the same split is projected two ways: blocking wants ``BLOCK_COLUMNS`` and the
    featuriser wants ``FEATURE_COLUMNS``. Sharing one backing between them would
    silently hand a caller the wrong width for a column.
    """

    key = json.dumps(
        {"sources": [int(s) for s in sources], "columns": sorted(columns)},
        sort_keys=True,
    )
    tag = hashlib.blake2b(key.encode("utf-8"), digest_size=4).hexdigest()
    paths = {col: cache_dir / f"{split}_norm_{tag}_{col}.npy" for col in columns}
    meta = cache_dir / f"{split}_norm_{tag}.meta.json"
    return tag, paths, meta


def _norm_backing_is_complete(
    meta_path: Path,
    rows: int,
    columns: Sequence[str],
    sources: Sequence[int],
    paths: Dict[str, Path],
) -> bool:
    """True when every backing file exists and the sidecar certifies the layout.

    The sidecar is written only after the last column is fully populated, so its
    presence is what distinguishes a finished backing from one interrupted
    halfway -- a partially written ``.npy`` has a valid header and would otherwise
    read back as zeros, which is indistinguishable from a legitimately empty
    field and would quietly corrupt every downstream feature.
    """

    if not meta_path.exists():
        return False
    try:
        state = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not state.get("complete"):
        return False
    if int(state.get("rows", -1)) != rows:
        return False
    if sorted(state.get("columns", ())) != sorted(columns):
        return False
    if [int(s) for s in state.get("sources", ())] != [int(s) for s in sources]:
        return False
    return all(paths[col].exists() for col in columns)


def _staging_path(final: Path) -> Path:
    """Scratch name a backing column is built under before being moved into place.

    Building straight onto the final name means an interrupted build leaves a
    ``.npy`` with a valid header and a partially filled body, which reads back as
    a pool of mostly-empty fields. Staging first makes the visible file appear
    only once it is whole.
    """

    return final.with_name(final.name + ".building")


def _build_norm_backing(
    cache_dir: Path,
    split: str,
    sources: Sequence[int],
    wanted_str: Sequence[str],
    wanted_bool: Sequence[str],
    paths: Dict[str, Path],
    meta_path: Path,
) -> int:
    """Stream every requested source into one set of on-disk columns.

    The destination columns are allocated at their final size up front and each
    source is written into its own row band, so the per-source frames never exist
    at the same time as the merged result. That is the whole point: the previous
    shape held the frames *and* their concatenation simultaneously, so the peak
    was twice the payload.
    """

    import pyarrow.parquet as pq

    source_paths = []
    row_counts = []
    for source in sources:
        path = cache_dir / f"{split}_source{source}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run the preprocessing stage first"
            )
        source_paths.append(path)
        row_counts.append(int(pq.ParquetFile(path).metadata.num_rows))
    total = sum(row_counts)

    destination: Dict[str, np.ndarray] = {}
    staged: Dict[str, Path] = {}
    for col in wanted_str:
        staged[col] = _staging_path(paths[col])
        destination[col] = np.lib.format.open_memmap(
            staged[col], mode="w+", dtype=f"S{_WIDTHS[col]}", shape=(total,),
        )
    for col in wanted_bool:
        staged[col] = _staging_path(paths[col])
        destination[col] = np.lib.format.open_memmap(
            staged[col], mode="w+", dtype=bool, shape=(total,),
        )

    offset = 0
    for path, count in zip(source_paths, row_counts):
        _read_parquet_columns(
            path,
            columns=list(wanted_str) + list(wanted_bool),
            into=destination,
            offset=offset,
        )
        offset += count

    # Durability before the sidecar: a sidecar that survived without its data
    # would be a licence to read zeros. The mappings have to be gone before the
    # rename, and the sidecar is written only after every column is in place.
    for col in list(wanted_str) + list(wanted_bool):
        destination[col].flush()
    del destination

    for col, tmp in staged.items():
        try:
            os.replace(tmp, paths[col])
        except OSError as exc:
            raise OSError(
                f"could not install the {col!r} backing at {paths[col]}: {exc}. "
                "On Windows a file with a live memmap cannot be replaced, so a "
                "rebuild is impossible while another NormalisedSet still holds "
                "this projection open. Drop that reference and retry."
            ) from exc

    meta_path.write_text(
        json.dumps(
            {
                "complete": True,
                "rows": total,
                "columns": list(wanted_str) + list(wanted_bool),
                "sources": [int(s) for s in sources],
            }
        ),
        encoding="utf-8",
    )
    return total


def load_normalised(
    cache_dir: str | Path,
    split: str,
    sources: Sequence[int] = (1, 2, 3),
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    memmap: bool = False,
) -> NormalisedSet:
    """Load cached normalised records for a split into a :class:`NormalisedSet`.

    ``columns`` restricts the load to the named fields. Fixed-width storage costs
    roughly 640 bytes per record, so a full three-source train load is about 8 GB;
    stages that only need part of the record (blocking needs the name, the
    address core, the house/postal/tail2 components and the country, and nothing
    else) should project rather than pay for the whole row. ``entity_id`` and
    ``_n`` are always included because the layout depends on them.

    ``memmap=True`` backs the columns with on-disk ``.npy`` files and returns read
    -only ``np.memmap`` views instead of in-RAM arrays. Two reasons it is opt-in
    rather than the default. The backing is built once and reused, so the second
    and later runs skip the Parquet read entirely -- but that also means a stale
    backing is only as fresh as its sidecar, and the caller has to accept that
    the Parquet cache is no longer the source of truth for that projection. And
    the columns come back read-only, so a caller that writes through the
    :class:`NormalisedSet` needs the default path.

    Read-only is also the point of the exercise on Windows. A writeable file
    mapping keeps its dirty pages charged to the system commit budget even after
    ``flush()``, because ``FlushViewOfFile`` writes them out without decommitting
    them; only destroying the mapping releases the charge. A read-only mapping of
    clean pages is reclaimable by the working-set trimmer under pressure. The
    pool is read-only for the whole featurisation, so this is the case that
    actually returns memory.
    """

    cache_dir = Path(cache_dir)
    wanted = None
    if columns is not None:
        wanted = set(columns) | {"entity_id"}
        unknown = wanted - set(_STR_COLUMNS)
        if unknown:
            raise ValueError(f"unknown columns requested: {sorted(unknown)}")

    wanted_str = [c for c in _STR_COLUMNS if wanted is None or c in wanted]
    wanted_bool = [] if wanted is not None else ["name_present", "address_present"]

    if memmap:
        import pyarrow.parquet as pq

        backing_columns = list(wanted_str) + list(wanted_bool)
        tag, paths, meta_path = _norm_backing_paths(
            cache_dir, split, sources, backing_columns
        )
        rows = 0
        for source in sources:
            path = cache_dir / f"{split}_source{source}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing — run the preprocessing stage first"
                )
            rows += int(pq.ParquetFile(path).metadata.num_rows)

        if _norm_backing_is_complete(meta_path, rows, backing_columns, sources, paths):
            build = "reused"
        else:
            # A half-built backing from an interrupted run is worthless; drop the
            # sidecar first so a crash mid-rebuild cannot leave it looking done.
            meta_path.unlink(missing_ok=True)
            rows = _build_norm_backing(
                cache_dir, split, sources, wanted_str, wanted_bool,
                paths, meta_path,
            )
            build = "built"

        arrays: Dict[str, np.ndarray] = {
            col: np.load(paths[col], mmap_mode="r", allow_pickle=False)
            for col in backing_columns
        }
        print(
            f"[load_normalised] {split} sources={tuple(sources)} {build} "
            f"{len(arrays)} columns x {rows:,} rows (memmap, tag={tag})",
            flush=True,
        )
        return NormalisedSet(arrays)

    frames = []
    for source in sources:
        path = cache_dir / f"{split}_source{source}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run the preprocessing stage first"
            )
        frames.append(_read_parquet_columns(path, columns=wanted))
    merged: Dict[str, np.ndarray] = {}
    for col in _STR_COLUMNS:
        if wanted is not None and col not in wanted:
            continue
        merged[col] = np.concatenate([f[col] for f in frames])
    for col in ("name_present", "address_present"):
        if wanted is not None:
            continue
        merged[col] = np.concatenate([f[col] for f in frames])
    return NormalisedSet(merged)


def source_column(norm: NormalisedSet) -> np.ndarray:
    """Return the source number of every row, derived from the ``entity_id`` prefix.

    The prefix is encoded to bytes before the comparison: ``entity_id`` is held in
    a fixed-width byte array, and under NumPy 2's ``BytesDType`` a ``str`` argument
    to :func:`numpy.char.startswith` has no matching loop and raises.
    """

    src = np.zeros(len(norm), dtype=np.int8)
    for source, prefix in SOURCE_ID_PREFIXES.items():
        src[_has_prefix(norm.entity_id, prefix)] = source
    return src


def source_slices(
    norm: NormalisedSet, sources: Sequence[int] = (1, 2, 3)
) -> Dict[int, np.ndarray]:
    """Return ``{source: row indices}`` for the requested sources."""

    src = source_column(norm)
    return {source: np.flatnonzero(src == source) for source in sources}


# --------------------------------------------------------------------------- #
# Ground truth and splitting
# --------------------------------------------------------------------------- #


def load_ground_truth(path: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Return the training labels as ``{source1_entity_id: [matched ids]}``.

    Reads ``dataset/train/train_ground_truth.tsv``, whose columns are
    ``source1_entity_id`` and ``matched_entity_ids`` (comma-separated, **empty**
    when the entity is a singleton). 5.58% of the training entities are
    singletons, so the empty-list case is common, not a corner case.
    """

    if path is None:
        path = str(ground_truth_path(data_dir))
    frame = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8", na_filter=False
    )
    out: Dict[str, List[str]] = {}
    for s1, raw in zip(frame["source1_entity_id"], frame["matched_entity_ids"]):
        raw = (raw or "").strip()
        out[s1] = [t for t in raw.split(",") if t] if raw else []
    return out


def split_train_validation(
    source1_ids: Iterable[str],
    validation_fraction: float = 0.1,
    seed: int = 42,
    strata: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Split Source-1 entity IDs into train and validation partitions.

    Split on **Source-1 entities**, not on individual pairs, and do it before
    computing any feature. A pair-level split leaks: a Source-2 record that truly
    matches a validation Source-1 entity frequently also matches a training one,
    so the model learns to recognise the record rather than the business.

    ``strata`` optionally maps an entity ID to a stratum label (e.g. its
    country); when given, the split is stratified so singletons and the
    US/India mix stay representative in the validation partition.

    Deterministic for a given ``(source1_ids, validation_fraction, seed,
    strata)``: a fixed-seed permutation within each stratum, so reruns
    reproduce the same partition.
    """

    ids = list(source1_ids)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)

    if strata is None:
        order = rng.permutation(len(ids))
        n_val = int(round(len(ids) * validation_fraction))
        val_idx = set(order[:n_val].tolist())
        return (
            [ids[i] for i in range(len(ids)) if i not in val_idx],
            [ids[i] for i in range(len(ids)) if i in val_idx],
        )

    groups: Dict[str, List[int]] = {}
    for i, entity_id in enumerate(ids):
        groups.setdefault(strata.get(entity_id, ""), []).append(i)
    val: set[int] = set()
    for _key, members in groups.items():
        members_arr = np.array(members, dtype=np.int64)
        order = rng.permutation(len(members_arr))
        n_val = int(round(len(members_arr) * validation_fraction))
        val.update(members_arr[order[:n_val]].tolist())
    return (
        [ids[i] for i in range(len(ids)) if i not in val],
        [ids[i] for i in range(len(ids)) if i in val],
    )


__all__ = [
    "Record", "RowView", "NormalisedSet", "SOURCE_ID_PREFIXES", "RECORD_COLUMNS",
    "CACHE_COLUMNS", "LEGAL_FORMS", "NAME_STOPWORDS", "ADDRESS_STOPWORDS",
    "LANDMARK_WORDS", "load_source", "load_all_sources", "normalise_name",
    "normalise_address", "parse_address", "load_ground_truth",
    "split_train_validation", "resolve_data_dir", "source_path",
    "ground_truth_path", "cache_split", "load_normalised", "make_record",
    "country_profile", "normalise_country", "STREET_SHORT", "DIRECTION_SHORT",
]
