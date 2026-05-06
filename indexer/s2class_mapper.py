"""S2ClassMapper — maps eclass codes to S2CLASS using the binary mapping tables.

Replicates the legacy Java ``S2ClassOfferMapper`` + ``EclassMapper``:
the article-search-indexer derives ``s2classGroups`` at index time from
the highest available eclass version (ECLASS_8 > ECLASS_5_1) through
a deterministic lookup table.

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


_ECLASS5_TO_S2: dict[int, int] | None = None
_ECLASS8_TO_S2: dict[int, int] | None = None


def _eclass5_map() -> dict[int, int]:
    global _ECLASS5_TO_S2
    if _ECLASS5_TO_S2 is None:
        _ECLASS5_TO_S2 = _load_mapping(_DIR / "5-s2.bin.gz")
    return _ECLASS5_TO_S2


def _eclass8_map() -> dict[int, int]:
    global _ECLASS8_TO_S2
    if _ECLASS8_TO_S2 is None:
        _ECLASS8_TO_S2 = _load_mapping(_DIR / "8-s2.bin.gz")
    return _ECLASS8_TO_S2


def from_eclass5(code: int) -> int | None:
    return _eclass5_map().get(code)


def from_eclass8(code: int) -> int | None:
    return _eclass8_map().get(code)


def derive_s2class_codes(eclass_groups: dict[str, list] | None) -> set[int]:
    """Derive S2CLASS codes from the highest available eclass version.

    Mirrors the Java ``S2ClassOfferMapper.map()`` logic: pick the best
    source (highest version number, non-empty), map each leaf code
    through the mapping table, fall back to DEFAULT_S2CLASS_CODE.
    """
    if not eclass_groups:
        return {DEFAULT_S2CLASS_CODE}

    # Priority: ECLASS_8 > ECLASS_5_1 (higher version wins)
    for key, mapper_fn in [("ECLASS_8", from_eclass8), ("ECLASS_5_1", from_eclass5)]:
        codes = eclass_groups.get(key)
        if not codes:
            continue
        mapped = set()
        for c in codes:
            s2 = mapper_fn(int(c))
            if s2 is not None:
                mapped.add(s2)
        return mapped if mapped else {DEFAULT_S2CLASS_CODE}

    return {DEFAULT_S2CLASS_CODE}
