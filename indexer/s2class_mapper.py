"""S2ClassMapper — maps eclass codes to S2CLASS using the binary mapping tables.

Replicates the legacy Java ``S2ClassOfferMapper`` + ``EclassMapper``:
the article-search-indexer derives ``s2classGroups`` at index time from
the highest available non-S2CLASS eclass version through a deterministic
lookup table.

Binary file format (big-endian):
  - 4 bytes: magic ``0x4D415050`` ("MAPP")
  - 4 bytes: format version (1)
  - 4 bytes: entry count N
  - 4 bytes: reserved
  - N × 8 bytes: (from_code, to_code) pairs as int32
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

_MAGIC = 0x4D415050
_FORMAT_VERSION = 1
_DIR = Path(__file__).parent / "classification_mapping"
DEFAULT_S2CLASS_CODE = 90909090

# Legacy `EClassVersion` priority: highest non-S2CLASS version wins. The
# enum names encode dotted versions (`ECLASS_5_1` == eClass 5.1) while the
# mapping resources collapse them to whole-number filenames (`5-s2.bin.gz`).
S2CLASS_VERSION_TO_FILE_VERSION = {
    "ECLASS_5_1": 5,
    "ECLASS_6": 6,
    "ECLASS_7_1": 7,
    "ECLASS_8": 8,
    "ECLASS_9": 9,
    "ECLASS_10": 10,
    "ECLASS_11": 11,
    "ECLASS_12": 12,
    "ECLASS_13": 13,
    "ECLASS_14": 14,
    "ECLASS_15": 15,
    "ECLASS_16": 16,
}
S2CLASS_SOURCE_KEYS_DESC = [
    key for key, _ in sorted(S2CLASS_VERSION_TO_FILE_VERSION.items(), key=lambda kv: kv[1], reverse=True)
]


def _load_mapping(path: Path) -> dict[int, int]:
    data = gzip.decompress(path.read_bytes())
    magic, version, count, _ = struct.unpack_from(">iiii", data)
    if magic != _MAGIC:
        raise ValueError(f"bad magic: {hex(magic)}")
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported format version: {version}")
    mapping: dict[int, int] = {}
    for i in range(count):
        off = 16 + i * 8
        from_code, to_code = struct.unpack_from(">ii", data, off)
        mapping[from_code] = to_code
    return mapping


_S2_MAP_CACHE: dict[str, dict[int, int]] = {}


def mapping_for_version_key(version_key: str) -> dict[int, int]:
    file_version = S2CLASS_VERSION_TO_FILE_VERSION[version_key]
    mapping = _S2_MAP_CACHE.get(version_key)
    if mapping is None:
        mapping = _load_mapping(_DIR / f"{file_version}-s2.bin.gz")
        _S2_MAP_CACHE[version_key] = mapping
    return mapping


def derive_s2class_codes(eclass_groups: dict[str, list] | None) -> set[int]:
    """Derive S2CLASS codes from the highest available eclass version.

    Mirrors the Java ``S2ClassOfferMapper.map()`` logic exactly:
      - ignore source-provided S2CLASS
      - pick the highest non-S2CLASS version whose list is non-empty
      - map every leaf through that version's `{version}-s2.bin.gz`
      - if that chosen version yields no mappings, fall back to the
        default S2CLASS code instead of trying lower versions
    """
    if not eclass_groups:
        return {DEFAULT_S2CLASS_CODE}

    for key in S2CLASS_SOURCE_KEYS_DESC:
        codes = eclass_groups.get(key)
        if not codes:
            continue
        mapping = mapping_for_version_key(key)
        mapped = {s2 for c in codes if (s2 := mapping.get(int(c))) is not None}
        return mapped if mapped else {DEFAULT_S2CLASS_CODE}

    return {DEFAULT_S2CLASS_CODE}
