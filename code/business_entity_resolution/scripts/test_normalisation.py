"""Unit-ish checks for the normalisation rules against observed data patterns.

Run:  python scripts/test_normalisation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import preprocessing as pp  # noqa: E402


NAME_CASES = [
    ("General Printing Worldwide", "general printing worldwide", ""),
    ("GENERAL-PRINTING-WORLDWIDE", "general printing worldwide", ""),
    ("6eneral Printing Wörldwide (Co)", "eneral printing worldwide", "co"),
    ("Pacific Suma LLC", "pacific suma", "llc"),
    ("PACIFIC SUMA", "pacific suma", ""),
    ("Suma Pacific LLC", "suma pacific", "llc"),
    ("Apex Constructions Private Limited", "apex constructions", "pvt ltd"),
    ("APEX CONSTRUCTION PVT. LTD.", "apex construction", "pvt ltd"),
    ("wilfordhancock.com", "wilfordhancock", ""),
    ("www.example.co.in", "example", ""),
    ("-- Holloway Peak Inc Seafood", "holloway peak seafood", ""),
    ("International South Consultants Private Ltd", "international south consultants", "pvt ltd"),
    ("Café Ltd", "cafe", "ltd"),
    ("Smith & Sons", "smith and sons", ""),
    ("Star Cafe", "star cafe", ""),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", None, "ltd"),
    ("", "", ""),
    ("null", "", ""),
    ("S.A.R.L.", None, "sarl"),
    ("Bhatia Autos Pvt Ltd", "bhatia autos", "pvt ltd"),
]

ADDRESS_CASES = [
    ("1795 Westchester Drive, High Point, NC", "US"),
    ("1795 Westchester Dr, High Point, North Carolina", "US"),
    ("1087 Cr 705, West Columbia, TX", "US"),
    ("087 Cr 705, West Columbia, Texas", "US"),
    ("IA, Iowa City, 1064 Newton Rd, Unit 11", "US"),
    ("1064 Newton Road Unit 11, Iowa City, IA", "US"),
    ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
    ("B-14, Sector 15, Rohini, Delhi 110070", "India"),
    ("Near SBI ATM,MG Road, Bengaluru,Karnataka 560001", "India"),
    ("12 rue de la Paix, 75001 Paris", "France"),
    ("75001 Paris, 12 rue de la Paix", "France"),
    ("N SHORE RD, BELFAIR, WA", "US"),
    ("6500 N Shore Road, Belfair, Washington", "US"),
    ("null", "US"),
    ("", "US"),
]


def main() -> int:
    failures = 0
    print("########## NAME NORMALISATION ##########")
    for raw, expect_core, expect_legal in NAME_CASES:
        norm = pp.normalise_name(raw)
        legal = pp._canonical_legal(raw)
        ok_core = expect_core is None or norm == expect_core
        ok_legal = expect_legal is None or legal == expect_legal
        status = "ok  " if (ok_core and ok_legal) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"  [{status}] {raw!r}\n           -> core={norm!r} legal={legal!r}")

    print("\n########## ADDRESS NORMALISATION ##########")
    for raw, country in ADDRESS_CASES:
        p = pp.parse_address(raw, country)
        print(f"  {raw!r} ({country})")
        print(f"      norm={p['norm']!r}")
        print(f"      core={p['core']!r} house={p['house']!r} postal={p['postal']!r} "
              f"state={p['state']!r} city={p['city']!r} landmark={p['landmark']!r}")

    print("\n########## OPEN-SET COUNTRY SAFETY ##########")
    for country in ("France", "FR", "Germany", "Brazil", "", "Atlantis"):
        prof = pp.country_profile(country)
        p = pp.parse_address("1 Some Street, Some City, XX", country)
        print(f"  {country!r}: profile={'generic' if prof is pp.GENERIC_PROFILE else 'specific'} "
              f"-> core={p['core']!r} (no crash)")

    print("\n########## GROUND-TRUTH SPLIT ##########")
    ids = [f"S1-{i}" for i in range(1000)]
    a, b = pp.split_train_validation(ids, 0.1, seed=42)
    a2, b2 = pp.split_train_validation(ids, 0.1, seed=42)
    print(f"  train={len(a)} val={len(b)} deterministic={a == a2 and b == b2} "
          f"disjoint={not (set(a) & set(b))} covers_all={len(a) + len(b) == len(ids)}")
    strata = {i: ("US" if int(i[3:]) % 2 else "India") for i in ids}
    a3, b3 = pp.split_train_validation(ids, 0.1, seed=42, strata=strata)
    from collections import Counter

    print(f"  stratified val mix: "
          f"{Counter(strata[i] for i in b3)} (expected ~50/50)")

    print(f"\n{'ALL NAME CASES PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
