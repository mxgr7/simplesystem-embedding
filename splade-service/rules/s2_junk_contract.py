#!/usr/bin/env python3
"""Loader for contracts/s2_junk.json -- THE s2class dumping-ground lexicon.

MXG-50. Seven codes, six of them found by `build_category_tree.py --step junk`
and human-confirmed, one hand-found seed. They used to live only in the
gitignored artifact `pipeline/out/category_tree.json`, with nine sites carrying
an inline copy of just the seed. The contract is now the source; the artifact
is derived from it.

FAIL-CLOSED on purpose. `category_rules.load_lexicon` used to fail OPEN -- a
missing artifact gave an empty junk set and every filter quietly became a
no-op. A junk filter that silently does nothing is worse than one that is too
narrow, so every accessor here raises rather than returning an empty set.

Two pins (`pins.encoded_in_splade_v1`, `pins.deployed_renderers_now`) exist for
the sites whose job is to reproduce deployed behaviour rather than correct
behaviour. Read the pin by name; never inline the subset. See the contract's
`pins` and `granularity` blocks.

stdlib only, and imports nothing from pipeline/ -- the SQL builders must be
able to get seven strings without pulling in the 4,381-node category tree.
"""
import copy
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_PATH = os.path.join(HERE, "contracts", "s2_junk.json")

CODE_RE = re.compile(r"^\d{8}$")

_cache = None


def load():
    """The raw artifact. Lazy: importing this module must not touch the disk."""
    global _cache
    if _cache is None:
        try:
            with open(CONTRACT_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"s2 junk contract not found at {CONTRACT_PATH}. It is tracked "
                "in git -- run from a full checkout of /workspace.")
    return _cache


def _codes(value, where):
    """Validate a code list. Empty is an error, not an empty filter."""
    codes = tuple(value or ())
    if not codes:
        raise ValueError(
            f"{where} is empty in {CONTRACT_PATH}. This module fails closed: "
            "an empty junk list would turn every filter into a no-op.")
    bad = [c for c in codes if not (isinstance(c, str) and CODE_RE.match(c))]
    if bad:
        raise ValueError(f"{where} holds non-8-digit codes: {bad}")
    return codes


# --------------------------------------------------------------------------
# accessors -- deep copies, so a caller that mutates what it gets back cannot
# poison the next caller through the module cache.
# --------------------------------------------------------------------------
def junk_codes():
    """The full confirmed lexicon, as a frozenset. What a correct filter uses."""
    return frozenset(_codes(load().get("junk"), "junk"))


def junk_sorted():
    """The same list, sorted -- for rendering into SQL, artifacts and logs
    where a stable order matters."""
    return sorted(junk_codes())


def seed():
    """The code build_category_tree seeds a fresh artifact with."""
    return frozenset(_codes(load().get("seed"), "seed"))


def pin(name):
    """A named subset a site applies on purpose. Raises on an unknown name --
    a typo must not silently degrade into 'filter nothing'."""
    pins = load().get("pins") or {}
    if name not in pins:
        known = sorted(k for k in pins if not k.startswith("_"))
        raise KeyError(f"unknown pin {name!r} in {CONTRACT_PATH}; known: {known}")
    return frozenset(_codes(pins[name], f"pins.{name}"))


def code_info(code):
    """Provenance for one code: name, source (seed|scan), rank, z, n, ..."""
    info = (load().get("codes") or {}).get(str(code))
    if info is None:
        raise KeyError(f"{code} is not in {CONTRACT_PATH}")
    return copy.deepcopy(info)


def rule():
    """The confirmation rule, as prose. There is deliberately no z threshold --
    read `rule.rejected_in_band` before you try to invent one."""
    return copy.deepcopy(load()["rule"])


def provenance():
    return copy.deepcopy(load()["provenance"])


# --------------------------------------------------------------------------
# SQL rendering -- these strings are interpolated into ClickHouse DDL, so every
# element is re-validated here even though the accessors already checked it.
# --------------------------------------------------------------------------
def ch_array(codes):
    """['15069090','15090903',...] -- a ClickHouse array literal."""
    codes = _codes(sorted(codes), "ch_array argument")
    return "[" + ",".join(f"'{c}'" for c in codes) + "]"


def ch_record_drop(col, codes=None):
    """`if(hasAny(col, [...]), [], col)` -- the RECORD-level drop.

    ph.article_catalog.s2class holds the full ancestor closure, so filtering
    the junk code out of the array leaves its junk parents behind. §17 rule 1
    drops the whole record; this renders that. See the contract's
    `granularity` block for the three variants and which sites use them.
    """
    return f"if(hasAny({col}, {ch_array(codes or junk_codes())}), [], {col})"


def record_drop(closure, codes=None):
    """The Python twin of `ch_record_drop`: [] if the closure carries a junk
    code, else the closure unchanged.

    For readers that hold a catalog row in memory rather than in SQL
    (build_esci_features). The two must agree, or a candidate article gets
    scored against a behavioural profile built under a different rule --
    MXG-75: reading the closure raw here let a key that MXG-50 removed from
    ph.beh_*_category read back through a `0.0` lookup default as a measured
    zero-affinity rather than a missing value.
    """
    junk = frozenset(_codes(sorted(codes or junk_codes()), "record_drop codes"))
    closure = list(closure or [])
    return [] if junk.intersection(closure) else closure


def main():
    c = load()
    print(f"contracts/s2_junk.json v{c['contract_version']}  "
          f"scan {c['provenance']['scanned_on']}")
    for code in junk_sorted():
        i = code_info(code)
        rank = f"#{i['rank']:<3d}" if i.get("rank") else "  -  "
        z = f"z={i['z']:.2f}" if i.get("z") else "z=  - "
        print(f"  {code}  {rank} {z}  n={i.get('n', 0):>7,}  "
              f"{i['source']:<4}  {i['name'][:52]}")
    print(f"  rejected in the accepted z band: "
          f"{len(rule()['rejected_in_band']) - 1} codes -- there is no threshold")
    for name in sorted(k for k in c["pins"] if not k.startswith("_")):
        print(f"  pin {name:<24} {sorted(pin(name))}")


if __name__ == "__main__":
    main()
