"""Unit tests for `indexer/friendly_id.py`.

The Java reference is `com.devskiller.friendly_id` 1.1.0. Where we
have known test vectors from the published library docs we assert on
them; everything else is round-trip.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer.friendly_id import to_friendly_id, to_uuid  # noqa: E402


# ---------- known vectors -------------------------------------------------
#
# Test vectors from the friendly-id project README (devskiller/friendly-id).
# https://github.com/devskiller/friendly-id

@pytest.mark.parametrize("uid_str,expected", [
    # README example from devskiller/friendly-id (canonical reference vector).
    ("c3587ec5-0976-497f-8374-61e0c2ea3da5", "5wbwf6yUxVBcr48AMbz9cb"),
    ("00000000-0000-0000-0000-000000000000", "0000000000000000000000"),
])
def test_known_vectors(uid_str: str, expected: str) -> None:
    assert to_friendly_id(uuid.UUID(uid_str)) == expected
    assert to_uuid(expected) == uuid.UUID(uid_str)


def test_decode_handles_short_strings() -> None:
    """The library accepts inputs shorter than 22 chars (no padding required
    on decode). All-zero high digits are implicit."""
    assert to_uuid("0") == uuid.UUID(int=0)
    assert to_uuid("z") == uuid.UUID(int=61)


# ---------- shape ---------------------------------------------------------

def test_output_is_always_22_chars() -> None:
    for _ in range(50):
        assert len(to_friendly_id(uuid.uuid4())) == 22


def test_output_chars_only_in_base62_alphabet() -> None:
    alphabet = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    for _ in range(50):
        assert set(to_friendly_id(uuid.uuid4())) <= alphabet


# ---------- round-trip ----------------------------------------------------

def test_round_trip_random() -> None:
    for _ in range(50):
        u = uuid.uuid4()
        assert to_uuid(to_friendly_id(u)) == u


def test_round_trip_max_uuid() -> None:
    u = uuid.UUID(int=(1 << 128) - 1)
    f = to_friendly_id(u)
    assert to_uuid(f) == u


# ---------- error paths ---------------------------------------------------

def test_decode_rejects_invalid_character() -> None:
    with pytest.raises(ValueError, match="invalid character"):
        to_uuid("not-base62-string!@#$%")


def test_decode_rejects_overflow() -> None:
    # 23 'z's overflows 128 bits.
    with pytest.raises(ValueError, match="too long|> 2\\*\\*128"):
        to_uuid("z" * 23)
