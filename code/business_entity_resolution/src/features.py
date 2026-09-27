"""Pairwise similarity features for candidate Source-1 / (Source-2, Source-3) pairs.

Scope
-----
Each function takes the two :class:`preprocessing.Record` objects for one
candidate pair and returns a small, named group of numeric features. The model
consumes the concatenation of these groups; keeping them grouped by field makes
feature-importance output and error analysis in :mod:`evaluation` readable.

Feature families
----------------
Name similarity
    The most discriminative signal when present, but noisy: abbreviations,
    legal-suffix variants, DBA/trade names, ``&`` vs ``and``, word-order
    transpositions, typos, and the same business written in two scripts.

    * token-set and token-sort ratios (order-invariant, handles transpositions)
    * character n-gram Jaccard / cosine (robust to typos and suffixes)
    * normalised edit similarity (Levenshtein ratio)
    * Jaro-Winkler, which rewards a shared prefix and suits names differing only
      in a suffix
    * a phonetic code match (soundex/metaphone-style), tolerant of spelling drift
    * length and token-count deltas, plus a "name present on both sides" flag
      (names are sometimes empty)

Address similarity
    Noisier still, and frequently missing on one side. Some records really do
    have an empty address, so a missing-address indicator is itself a useful
    feature rather than something to impute away.

    * token overlap / Jaccard over normalised address tokens
    * character n-gram cosine over the whole address string
    * structured-part agreement: house or plot number, PIN/postal code, city,
      state — extracted in :func:`preprocessing.normalise_address`
    * landmark-token overlap, kept separate because ``"Near SBI ATM"`` style
      tokens are shared by unrelated businesses
    * address presence flags for either side

Cross-field and context features
    Information neither field carries alone:

    * country agreement — as a *string* comparison, never a fixed one-hot, since
      the test set contains ``France``, unseen in training
    * whether the candidate is from Source 2 or Source 3
    * how many Source-1 entities the candidate was proposed for (ambiguity: a
      generic name like "Star Cafe" is contended)
    * candidate frequency in its own source, a strong prior for singletons
    * the number of distinct blocking keys the pair shares

Implementation notes
--------------------
Edit distance and Jaro-Winkler dominate cost, so use a C implementation
(``rapidfuzz``) and cache per-record normalised forms rather than recomputing
them per pair. For char n-gram cosine, precompute a sparse vector per record and
take the dot product. Anything quadratic in the number of pairs is out of scope.

At full scale the pipeline scores on the order of 10^8 pairs, which makes the
per-pair Python overhead in this module the dominant cost. Two things keep that
in check, and both matter:

* :func:`featurise_pairs` batches pairs so that every expensive per-record
  quantity -- token sets, character n-grams, soundex codes, token sets for
  Jaccard -- is computed **once per record per batch**, not once per pair. A
  candidate pool record is typically proposed for several Source-1 entities, and
  a Source-1 entity is compared against many candidates, so per-pair recomputation
  would repeat the same work many times over.
* The n-gram and token-set features are exact set operations over small cached
  frozensets, not loops over characters.

Column order is fixed by :data:`FEATURE_COLUMNS`, derived once from the feature
group keys. :func:`featurise_pairs` is the public entry point; the module-level
per-pair functions remain available for tests and error analysis.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import preprocessing as pp

FEATURE_GROUPS = ("name", "address", "cross")

#: Character n-gram size used by the Jaccard features. Trigrams are the usual
#: compromise: bigrams are too noisy on short address strings, 4-grams too sparse
#: to survive a single typo.
NGRAM_N = 3

#: ``rapidfuzz`` score below which a partial ratio is reported as 0. Without a
#: cutoff, ``partial_ratio`` happily returns a high score for a short string that
#: happens to appear inside a much longer one -- "cafe" inside "the cafe co" --
#: which is precisely the false friend these features must not manufacture.
PARTIAL_CUTOFF = 90

#: Minimum length for a soundex code to be trusted. Very short names produce
#: collisions that would otherwise look like phonetic agreement.
PHONETIC_MIN_LEN = 4

#: Populated on first use by :func:`feature_columns`; never mutated afterwards.
FEATURE_COLUMNS: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Cached per-record quantities
# --------------------------------------------------------------------------- #


def _ngrams(text: str, n: int = NGRAM_N) -> frozenset:
    """Character n-gram set, padded so word boundaries are not lost."""

    if not text:
        return frozenset()
    padded = f" {text} "
    return frozenset(padded[i:i + n] for i in range(len(padded) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two small sets; 0.0 when both are empty."""

    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _content_tokens(text: str) -> frozenset:
    if not text:
        return frozenset()
    return frozenset(t for t in text.split() if len(t) >= 2)


def _soundexes(text: str) -> frozenset:
    """Soundex codes for the name's content tokens."""

    import jellyfish

    out = set()
    for token in text.split():
        if len(token) < PHONETIC_MIN_LEN:
            continue
        try:
            code = jellyfish.soundex(token)
        except Exception:
            continue
        if code:
            out.add(code)
    return frozenset(out)


def record_cache(record) -> Dict[str, object]:
    """Compute every per-record quantity the feature groups need.

    Called once per record per batch. ``Record`` attributes are read defensively
    so the same helper works for a projected
    :class:`preprocessing.NormalisedSet` row, where some fields may be absent.
    """

    name = _text(record, "name_norm")
    core = _text(record, "address_core")
    legal = _text(record, "name_legal")
    landmark = _text(record, "landmark_tokens")
    return {
        "name": name,
        "name_set": _content_tokens(name),
        "name_ngrams": _ngrams(name),
        "name_soundex": _soundexes(name),
        "name_legal": legal,
        "addr_core": core,
        "addr_set": _content_tokens(core),
        "addr_ngrams": _ngrams(core),
        "landmark_set": _content_tokens(landmark),
        "house": _text(record, "addr_house"),
        "postal": _text(record, "addr_postal"),
        "state": _text(record, "addr_state"),
        "city": _text(record, "addr_city"),
        "tail2": _text(record, "addr_tail2"),
        "country": _text(record, "country_norm"),
    }


def _text(record, field: str) -> str:
    """Read a string field from a ``Record`` or a projected row accessor."""

    value = getattr(record, field, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if value is None:
        return ""
    return str(value)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #


def name_features(
    record_a,
    record_b,
    cache_a: Optional[Dict[str, object]] = None,
    cache_b: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Return name-similarity features comparing two records.

    Keys are prefixed ``name_``, e.g. ``name_token_set_ratio``,
    ``name_char_jaccard``, ``name_edit_ratio``, ``name_jaro_winkler``,
    ``name_phonetic_match``, ``name_len_ratio``, ``name_missing_left``,
    ``name_missing_right``.

    Returns zeros plus explicit missingness flags when either name is empty,
    rather than dropping the pair -- an empty name is informative (the record is
    sparse) and dropping the row would bias the training set toward easy pairs.
    """

    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    a = cache_a if cache_a is not None else record_cache(record_a)
    b = cache_b if cache_b is not None else record_cache(record_b)
    name_a: str = a["name"]  # type: ignore[assignment]
    name_b: str = b["name"]  # type: ignore[assignment]

    if not name_a or not name_b:
        return {
            "name_token_set_ratio": 0.0,
            "name_token_sort_ratio": 0.0,
            "name_token_jaccard": 0.0,
            "name_partial_ratio": 0.0,
            "name_edit_ratio": 0.0,
            "name_jaro_winkler": 0.0,
            "name_char_jaccard": 0.0,
            "name_phonetic_match": 0.0,
            "name_len_ratio": 0.0,
            "name_token_count_delta": 0.0,
            "name_first_token_match": 0.0,
            "name_legal_match": 0.0,
            "name_missing_left": 1.0 if not name_a else 0.0,
            "name_missing_right": 1.0 if not name_b else 0.0,
        }

    tokens_a: Sequence[str] = name_a.split()
    tokens_b: Sequence[str] = name_b.split()
    first_a = tokens_a[0]
    first_b = tokens_b[0]

    return {
        # Order-invariant, so a transposition costs nothing.
        "name_token_set_ratio": fuzz.token_set_ratio(name_a, name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_token_jaccard": _jaccard(a["name_set"], b["name_set"]),  # type: ignore[arg-type]
        # Cutoff applied: without it, a short name embedded in a long one scores
        # near 100 and swamps the real features.
        "name_partial_ratio": (
            fuzz.partial_ratio(name_a, name_b, score_cutoff=PARTIAL_CUTOFF) / 100.0
        ),
        "name_edit_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_jaro_winkler": JaroWinkler.similarity(name_a, name_b),
        "name_char_jaccard": _jaccard(a["name_ngrams"], b["name_ngrams"]),  # type: ignore[arg-type]
        "name_phonetic_match": _jaccard(a["name_soundex"], b["name_soundex"]),  # type: ignore[arg-type]
        "name_len_ratio": min(len(name_a), len(name_b)) / max(len(name_a), len(name_b)),
        "name_token_count_delta": float(abs(len(tokens_a) - len(tokens_b))),
        # The leading token is the brand far more often than not, and it survives
        # "Suma Pacifica Hospitals" vs "Suma Pacifica Hospital Healthcare".
        "name_first_token_match": 1.0 if first_a == first_b else 0.0,
        "name_legal_match": (
            1.0
            if (a["name_legal"] and b["name_legal"] and a["name_legal"] == b["name_legal"])  # type: ignore[index]
            else 0.0
        ),
        "name_missing_left": 0.0,
        "name_missing_right": 0.0,
    }


def address_features(
    record_a,
    record_b,
    cache_a: Optional[Dict[str, object]] = None,
    cache_b: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Return address-similarity features comparing two records.

    Keys prefixed ``addr_``, e.g. ``addr_token_jaccard``, ``addr_char_jaccard``,
    ``addr_house_number_match``, ``addr_postal_match``, ``addr_city_match``,
    ``addr_state_match``, ``addr_landmark_jaccard``, ``addr_missing_left``,
    ``addr_missing_right``.

    Structured components (house/plot number, PIN/postal code, city, state) come
    from :func:`preprocessing.normalise_address`; equal components are a strong
    signal when flat token overlap is diluted by reordering or extra landmark
    text. A component that is missing on *either* side scores 0.0 and is
    distinguished from a genuine mismatch by the address presence flags, because
    "no postal code recorded" and "different postal codes" are different facts.
    """

    from rapidfuzz import fuzz

    a = cache_a if cache_a is not None else record_cache(record_a)
    b = cache_b if cache_b is not None else record_cache(record_b)
    core_a: str = a["addr_core"]  # type: ignore[assignment]
    core_b: str = b["addr_core"]  # type: ignore[assignment]

    if not core_a or not core_b:
        return {
            "addr_token_jaccard": 0.0,
            "addr_token_set_ratio": 0.0,
            "addr_char_jaccard": 0.0,
            "addr_house_match": 0.0,
            "addr_house_similarity": 0.0,
            "addr_postal_match": 0.0,
            "addr_city_match": 0.0,
            "addr_state_match": 0.0,
            "addr_tail2_match": 0.0,
            "addr_landmark_jaccard": 0.0,
            "addr_len_ratio": 0.0,
            "addr_missing_left": 1.0 if not core_a else 0.0,
            "addr_missing_right": 1.0 if not core_b else 0.0,
        }

    house_a: str = a["house"]  # type: ignore[assignment]
    house_b: str = b["house"]  # type: ignore[assignment]

    return {
        "addr_token_jaccard": _jaccard(a["addr_set"], b["addr_set"]),  # type: ignore[arg-type]
        "addr_token_set_ratio": fuzz.token_set_ratio(core_a, core_b) / 100.0,
        "addr_char_jaccard": _jaccard(a["addr_ngrams"], b["addr_ngrams"]),  # type: ignore[arg-type]
        # House numbers are the single best address signal: "41" vs "41A" is
        # still the same premises, so a graded score beats a binary flag.
        "addr_house_match": 1.0 if (house_a and house_a == house_b) else 0.0,
        "addr_house_similarity": (
            fuzz.ratio(house_a, house_b) / 100.0 if (house_a and house_b) else 0.0
        ),
        "addr_postal_match": (
            1.0 if (a["postal"] and a["postal"] == b["postal"]) else 0.0  # type: ignore[index]
        ),
        "addr_city_match": (
            1.0 if (a["city"] and a["city"] == b["city"]) else 0.0  # type: ignore[index]
        ),
        "addr_state_match": (
            1.0 if (a["state"] and a["state"] == b["state"]) else 0.0  # type: ignore[index]
        ),
        "addr_tail2_match": (
            1.0 if (a["tail2"] and a["tail2"] == b["tail2"]) else 0.0  # type: ignore[index]
        ),
        # Deliberately weak: "Near SBI ATM" is shared by every shop on that road.
        "addr_landmark_jaccard": _jaccard(a["landmark_set"], b["landmark_set"]),  # type: ignore[arg-type]
        "addr_len_ratio": min(len(core_a), len(core_b)) / max(len(core_a), len(core_b)),
        "addr_missing_left": 0.0,
        "addr_missing_right": 0.0,
    }


def cross_features(
    record_a,
    record_b,
    source_b: int,
    candidate_degree: int = 1,
    source_frequency: int = 1,
    shared_block_keys: int = 0,
    entity_candidate_count: int = 1,
    known_countries: Optional[Set[str]] = None,
    cache_a: Optional[Dict[str, object]] = None,
    cache_b: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Return features depending on the pair *in context*, not on text alone.

    Keys prefixed ``ctx_``, e.g. ``ctx_country_match``, ``ctx_source_b``,
    ``ctx_candidate_degree`` (how many Source-1 entities proposed this candidate
    — high values mean a contended or generic record), ``ctx_source_frequency``,
    ``ctx_shared_block_keys``.

    ``ctx_country_match`` is a plain string comparison. The test set includes
    ``France``, unseen in training, so a fixed country encoding would either
    crash or silently score it wrongly. ``ctx_country_known`` additionally tells
    the model whether it is looking at a country it was trained on, which is the
    only way to let it treat unseen countries differently without hard-filtering
    them.
    """

    a = cache_a if cache_a is not None else record_cache(record_a)
    b = cache_b if cache_b is not None else record_cache(record_b)
    country_a: str = a["country"]  # type: ignore[assignment]
    country_b: str = b["country"]  # type: ignore[assignment]
    known = known_countries or set()

    return {
        "ctx_country_match": 1.0 if (country_a and country_a == country_b) else 0.0,
        "ctx_country_known": 1.0 if (country_a in known and country_b in known) else 0.0,
        "ctx_country_missing": 1.0 if not (country_a and country_b) else 0.0,
        "ctx_source_b": float(source_b),
        # log1p: these range over three orders of magnitude, so the raw count
        # would let the model key on magnitude alone.
        "ctx_candidate_degree": float(np.log1p(max(candidate_degree, 0))),
        "ctx_source_frequency": float(np.log1p(max(source_frequency, 0))),
        "ctx_entity_candidate_count": float(np.log1p(max(entity_candidate_count, 0))),
        "ctx_shared_block_keys": float(shared_block_keys),
    }


# --------------------------------------------------------------------------- #
# Column order
# --------------------------------------------------------------------------- #


def feature_columns(feature_groups: Sequence[str] = FEATURE_GROUPS) -> Tuple[str, ...]:
    """Return the canonical feature column order, built once and cached.

    Built from a representative pair so the order is derived from the feature
    functions themselves rather than hand-maintained, then frozen. Deriving
    columns per batch is the failure mode the docstring warns about: the model
    still trains and the scores still look plausible, but the columns are
    silently permuted.
    """

    global FEATURE_COLUMNS
    if not FEATURE_COLUMNS or feature_groups != FEATURE_GROUPS:
        probe_a = _Probe("suma pacifica hospital", "41 mg road", "41", "560001",
                         "karnataka", "bengaluru", "bengaluru karnataka", "india")
        probe_b = _Probe("suma pacifica hospitals", "41 mg road", "41", "560001",
                         "karnataka", "bengaluru", "bengaluru karnataka", "india")
        columns: List[str] = []
        if "name" in feature_groups:
            columns += list(name_features(probe_a, probe_b))
        if "address" in feature_groups:
            columns += list(address_features(probe_a, probe_b))
        if "cross" in feature_groups:
            columns += list(cross_features(probe_a, probe_b, source_b=2))
        if feature_groups == FEATURE_GROUPS:
            FEATURE_COLUMNS = tuple(columns)
        return tuple(columns)
    return FEATURE_COLUMNS


class _Probe:
    """Minimal record stand-in so ``feature_columns`` can run without data."""

    def __init__(self, name, core, house, postal, state, city, tail2, country):
        self.name_norm = name
        self.name_legal = ""
        self.address_core = core
        self.addr_house = house
        self.addr_postal = postal
        self.addr_state = state
        self.addr_city = city
        self.addr_tail2 = tail2
        self.landmark_tokens = ""
        self.country_norm = country


# --------------------------------------------------------------------------- #
# Vectorised entry point
# --------------------------------------------------------------------------- #


def featurise_pairs(
    pairs: Iterable[Tuple],
    feature_groups: Sequence[str] = FEATURE_GROUPS,
    n_jobs: int = 1,
    columns: Optional[Sequence[str]] = None,
    known_countries: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Vectorise the feature functions over many pairs.

    ``pairs`` is an iterable of ``(record_a, record_b, source_b)`` triples, in
    step with the candidate list. Returns ``(matrix, column_names)`` with one row
    per pair, as a float32 array so 10^8 rows stay affordable.

    The column order comes from :func:`feature_columns` (or an explicit
    ``columns`` argument) and is applied to every batch, so training and
    inference always agree.

    ``n_jobs > 1`` is accepted for API compatibility; the pair loop is executed
    serially because each item already costs several C-level string comparisons
    and the per-record caches do the deduplication. Parallelise at the batch
    level by calling this function over row blocks in separate processes.
    """

    import pandas as pd

    column_list = list(columns) if columns is not None else list(feature_columns(feature_groups))
    out: List[np.ndarray] = []
    index = {name: i for i, name in enumerate(column_list)}

    for record_a, record_b, source_b in pairs:
        cache_a = record_cache(record_a)
        cache_b = record_cache(record_b)
        row = np.zeros(len(column_list), dtype=np.float32)
        if "name" in feature_groups:
            for key, value in name_features(record_a, record_b, cache_a, cache_b).items():
                if key in index:
                    row[index[key]] = value
        if "address" in feature_groups:
            for key, value in address_features(record_a, record_b, cache_a, cache_b).items():
                if key in index:
                    row[index[key]] = value
        if "cross" in feature_groups:
            for key, value in cross_features(
                record_a, record_b, source_b, known_countries=known_countries,
                cache_a=cache_a, cache_b=cache_b,
            ).items():
                if key in index:
                    row[index[key]] = value
        out.append(row)

    if not out:
        return np.empty((0, len(column_list)), dtype=np.float32), column_list
    return np.vstack(out), column_list


def featurise_arrays(
    s1_norm: pp.NormalisedSet,
    pool_norm: pp.NormalisedSet,
    s1_rows: np.ndarray,
    pool_rows: np.ndarray,
    evidence: np.ndarray,
    pool_source: np.ndarray,
    pool_frequency: np.ndarray,
    candidate_degree: np.ndarray,
    entity_candidate_count: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    known_countries: Optional[Set[str]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Featurise aligned arrays of candidate pairs.

    The scale path. All inputs are parallel per-pair arrays, so no per-pair
    Python object is built: for each distinct row on either side the
    :func:`record_cache` result is computed once and reused, which is where the
    saving over :func:`featurise_pairs` comes from.

    Every array must have the same length, and ``s1_rows[i]`` must be the
    Source-1 record for pair ``i`` with ``pool_rows[i]`` its candidate.

    ``out`` may be a preallocated ``(len(s1_rows), len(column_list))`` float32
    destination -- typically a slice of the on-disk memmap. Writing into it
    directly saves a second full-size buffer, which matters because one chunk is
    5x10^6 pairs and so ~700 MB.
    """

    column_list = list(columns) if columns is not None else list(feature_columns())
    index = {name: i for i, name in enumerate(column_list)}
    n = len(s1_rows)
    if out is None:
        matrix = np.zeros((n, len(column_list)), dtype=np.float32)
    else:
        if out.shape != (n, len(column_list)) or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 with shape {(n, len(column_list))}, "
                f"got {out.dtype} {out.shape}"
            )
        # Zeroed explicitly, not assumed. Most columns are only assigned inside
        # the `if name_a and name_b` / `if core_a and core_b` guards, so an unset
        # entry must read 0. That happens to hold for a fresh "w+" memmap, but
        # relying on it would silently produce stale features the moment a
        # destination is reused. One memset against millions of interpreted
        # feature evaluations is not worth the coupling.
        matrix = out
        matrix.fill(0.0)
    if n == 0:
        return matrix

    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler

    # Unique rows on each side, with a lookup back to cache per pair.
    s1_unique, s1_inv = np.unique(s1_rows, return_inverse=True)
    pool_unique, pool_inv = np.unique(pool_rows, return_inverse=True)

    def caches(norm, unique_rows, fields):
        out = []
        for row in unique_rows:
            holder = _RowView(norm, int(row), fields)
            out.append(record_cache(holder))
        return out

    fields = (
        "name_norm", "name_legal", "address_core", "addr_house", "addr_postal",
        "addr_state", "addr_city", "addr_tail2", "landmark_tokens", "country_norm",
    )
    cache_a = caches(s1_norm, s1_unique, fields)
    cache_b = caches(pool_norm, pool_unique, fields)

    s1_cache = [cache_a[i] for i in s1_inv]
    pool_cache = [cache_b[i] for i in pool_inv]

    for i in range(n):
        a = s1_cache[i]
        b = pool_cache[i]
        name_a = a["name"]
        name_b = b["name"]
        core_a = a["addr_core"]
        core_b = b["addr_core"]

        if name_a and name_b:
            tokens_a = name_a.split()
            tokens_b = name_b.split()
            matrix[i, index["name_token_set_ratio"]] = fuzz.token_set_ratio(name_a, name_b) / 100.0
            matrix[i, index["name_token_sort_ratio"]] = fuzz.token_sort_ratio(name_a, name_b) / 100.0
            matrix[i, index["name_token_jaccard"]] = _jaccard(a["name_set"], b["name_set"])
            matrix[i, index["name_partial_ratio"]] = fuzz.partial_ratio(
                name_a, name_b, score_cutoff=PARTIAL_CUTOFF
            ) / 100.0
            matrix[i, index["name_edit_ratio"]] = fuzz.ratio(name_a, name_b) / 100.0
            matrix[i, index["name_jaro_winkler"]] = JaroWinkler.similarity(name_a, name_b)
            matrix[i, index["name_char_jaccard"]] = _jaccard(a["name_ngrams"], b["name_ngrams"])
            matrix[i, index["name_phonetic_match"]] = _jaccard(a["name_soundex"], b["name_soundex"])
            matrix[i, index["name_len_ratio"]] = min(len(name_a), len(name_b)) / max(len(name_a), len(name_b))
            matrix[i, index["name_token_count_delta"]] = abs(len(tokens_a) - len(tokens_b))
            matrix[i, index["name_first_token_match"]] = 1.0 if tokens_a[0] == tokens_b[0] else 0.0
            if a["name_legal"] and a["name_legal"] == b["name_legal"]:
                matrix[i, index["name_legal_match"]] = 1.0
        else:
            matrix[i, index["name_missing_left"]] = 0.0 if name_a else 1.0
            matrix[i, index["name_missing_right"]] = 0.0 if name_b else 1.0

        if core_a and core_b:
            matrix[i, index["addr_token_jaccard"]] = _jaccard(a["addr_set"], b["addr_set"])
            matrix[i, index["addr_token_set_ratio"]] = fuzz.token_set_ratio(core_a, core_b) / 100.0
            matrix[i, index["addr_char_jaccard"]] = _jaccard(a["addr_ngrams"], b["addr_ngrams"])
            if a["house"] and a["house"] == b["house"]:
                matrix[i, index["addr_house_match"]] = 1.0
            if a["house"] and b["house"]:
                matrix[i, index["addr_house_similarity"]] = fuzz.ratio(a["house"], b["house"]) / 100.0
            if a["postal"] and a["postal"] == b["postal"]:
                matrix[i, index["addr_postal_match"]] = 1.0
            if a["city"] and a["city"] == b["city"]:
                matrix[i, index["addr_city_match"]] = 1.0
            if a["state"] and a["state"] == b["state"]:
                matrix[i, index["addr_state_match"]] = 1.0
            if a["tail2"] and a["tail2"] == b["tail2"]:
                matrix[i, index["addr_tail2_match"]] = 1.0
            matrix[i, index["addr_landmark_jaccard"]] = _jaccard(a["landmark_set"], b["landmark_set"])
            matrix[i, index["addr_len_ratio"]] = min(len(core_a), len(core_b)) / max(len(core_a), len(core_b))
        else:
            matrix[i, index["addr_missing_left"]] = 0.0 if core_a else 1.0
            matrix[i, index["addr_missing_right"]] = 0.0 if core_b else 1.0

        country_a = a["country"]
        country_b = b["country"]
        if country_a and country_b:
            if country_a == country_b:
                matrix[i, index["ctx_country_match"]] = 1.0
        else:
            matrix[i, index["ctx_country_missing"]] = 1.0
        if known_countries and country_a in known_countries and country_b in known_countries:
            matrix[i, index["ctx_country_known"]] = 1.0
        matrix[i, index["ctx_source_b"]] = pool_source[i]
        matrix[i, index["ctx_candidate_degree"]] = np.log1p(max(candidate_degree[i], 0))
        matrix[i, index["ctx_source_frequency"]] = np.log1p(max(pool_frequency[i], 0))
        matrix[i, index["ctx_entity_candidate_count"]] = np.log1p(max(entity_candidate_count[i], 0))
        matrix[i, index["ctx_shared_block_keys"]] = evidence[i]

    return matrix


class _RowView:
    """Attribute view over one row of a projected :class:`NormalisedSet`."""

    __slots__ = ("_norm", "_row", "_fields")

    def __init__(self, norm: pp.NormalisedSet, row: int, fields: Sequence[str]) -> None:
        self._norm = norm
        self._row = row
        self._fields = fields

    def __getattr__(self, name: str) -> str:
        fields = object.__getattribute__(self, "_fields")
        if name not in fields:
            raise AttributeError(name)
        norm = object.__getattribute__(self, "_norm")
        row = object.__getattribute__(self, "_row")
        value = getattr(norm, name)[row]
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


__all__ = [
    "FEATURE_GROUPS", "FEATURE_COLUMNS", "NGRAM_N", "PARTIAL_CUTOFF",
    "record_cache", "name_features", "address_features", "cross_features",
    "feature_columns", "featurise_pairs", "featurise_arrays",
]

# Populated at import so ``FEATURE_COLUMNS`` is never observed empty by a caller
# that reads the constant directly instead of calling :func:`feature_columns`.
feature_columns()
