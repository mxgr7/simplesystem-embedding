"""Property-based and fuzz-style tests for the FriendlyId base62 codec.

Tests the two inverse functions:
  - _uuid_to_friendly (response.py): UUID -> base62 FriendlyId string
  - _friendly_to_uuid (request.py):  base62 FriendlyId string -> UUID

Covers: round-trip integrity, boundary values, character coverage,
length invariants, ordering, collision resistance, and compatibility
with the Devskiller Java FriendlyId library's zero-padding convention.
"""

from __future__ import annotations

import random
import string
import uuid

import pytest

from acl.mapping.request import _friendly_to_uuid
from acl.mapping.response import _uuid_to_friendly

_BASE62_CHARS = set(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# ── helpers ──────────────────────────────────────────────────────────

def _random_uuid() -> uuid.UUID:
    return uuid.UUID(int=random.getrandbits(128))


def _pad22(friendly: str) -> str:
    """Zero-pad a FriendlyId to the Devskiller Java convention (22 chars)."""
    return friendly.zfill(22)


# ── 1. Round-trip: UUID -> friendly -> UUID ──────────────────────────

class TestRoundTripUuidToFriendly:
    """For any UUID, decode(encode(uuid)) == uuid."""

    @pytest.mark.parametrize("_run", range(500))
    def test_random_uuid_round_trips(self, _run: int) -> None:
        original = _random_uuid()
        assert _friendly_to_uuid(_uuid_to_friendly(original)) == original

    def test_uuid4_round_trips(self) -> None:
        for _ in range(200):
            original = uuid.uuid4()
            assert _friendly_to_uuid(_uuid_to_friendly(original)) == original


# ── 2. Round-trip inverse: friendly -> UUID -> friendly ──────────────

class TestRoundTripFriendlyToUuid:
    """For any valid (unpadded, canonical) FriendlyId string,
    encode(decode(s)) == s."""

    @pytest.mark.parametrize("_run", range(500))
    def test_canonical_friendly_round_trips(self, _run: int) -> None:
        # Generate a canonical FriendlyId via encode, then verify the
        # inverse round-trip.
        original_uuid = _random_uuid()
        canonical = _uuid_to_friendly(original_uuid)
        assert _uuid_to_friendly(_friendly_to_uuid(canonical)) == canonical


# ── 3. Known test vectors ───────────────────────────────────────────

class TestKnownVectors:
    """Verify against hand-computed / reference Devskiller values."""

    def test_zero_uuid(self) -> None:
        assert _uuid_to_friendly(uuid.UUID(int=0)) == "0" * 22

    def test_max_uuid(self) -> None:
        u = uuid.UUID(int=2**128 - 1)
        friendly = _uuid_to_friendly(u)
        assert friendly == "7n42DGM5Tflk9n8mt7Fhc7"
        assert len(friendly) == 22

    def test_well_known_uuid(self) -> None:
        # RFC 4122 example UUID
        u = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        friendly = _uuid_to_friendly(u)
        assert _friendly_to_uuid(friendly) == u
        assert len(friendly) == 22  # large enough UUID -> 22 chars

    def test_namespace_dns_uuid(self) -> None:
        u = uuid.NAMESPACE_DNS  # 6ba7b810-9dad-11d1-80b4-00c04fd430c8
        friendly = _uuid_to_friendly(u)
        assert _friendly_to_uuid(friendly) == u


# ── 4. Boundary UUIDs ────────────────────────────────────────────────

class TestBoundaryUuids:

    @pytest.mark.parametrize(
        "label, int_val",
        [
            ("zero", 0),
            ("one", 1),
            ("max-1", 2**128 - 2),
            ("max", 2**128 - 1),
            ("half", 2**127),
            ("low-byte-max", 0xFF),
            ("high-bit", 1 << 127),
        ],
    )
    def test_boundary_round_trips(self, label: str, int_val: int) -> None:
        u = uuid.UUID(int=int_val)
        friendly = _uuid_to_friendly(u)
        assert _friendly_to_uuid(friendly) == u, f"failed for {label}"


# ── 5. Character coverage ───────────────────────────────────────────

class TestCharacterCoverage:
    """Output must only contain base62 characters [0-9A-Za-z]."""

    def test_all_chars_are_base62(self) -> None:
        seen: set[str] = set()
        for _ in range(5000):
            friendly = _uuid_to_friendly(_random_uuid())
            seen.update(friendly)
        # Every character produced must be in base62
        assert seen <= _BASE62_CHARS, f"non-base62 chars: {seen - _BASE62_CHARS}"

    def test_boundary_chars_are_base62(self) -> None:
        for int_val in [0, 1, 61, 62, 63, 2**128 - 1]:
            friendly = _uuid_to_friendly(uuid.UUID(int=int_val))
            assert set(friendly) <= _BASE62_CHARS


# ── 6. Length properties ─────────────────────────────────────────────

class TestLengthProperties:
    """ceil(128 * log2 / log62) == 22, so FriendlyId is at most 22 chars.
    The Python encoder doesn't zero-pad, so small UUIDs produce shorter
    strings."""

    def test_max_length_is_22(self) -> None:
        for _ in range(5000):
            friendly = _uuid_to_friendly(_random_uuid())
            assert len(friendly) <= 22

    def test_max_uuid_is_exactly_22(self) -> None:
        assert len(_uuid_to_friendly(uuid.UUID(int=2**128 - 1))) == 22

    def test_zero_uuid_length(self) -> None:
        assert len(_uuid_to_friendly(uuid.UUID(int=0))) == 22

    def test_all_uuids_produce_22_chars(self) -> None:
        lengths = {len(_uuid_to_friendly(uuid.uuid4())) for _ in range(1000)}
        assert lengths == {22}

    def test_min_length_is_22(self) -> None:
        assert len(_uuid_to_friendly(uuid.UUID(int=0))) == 22


# ── 7. Leading zeros / short strings ────────────────────────────────

class TestLeadingZeros:
    """The Python encoder pads to 22 chars (Java Devskiller compat).
    Decoding padded strings must still recover the original UUID."""

    def test_uuid_one_produces_padded_string(self) -> None:
        friendly = _uuid_to_friendly(uuid.UUID(int=1))
        assert friendly == "0000000000000000000001"
        assert len(friendly) == 22

    def test_small_uuids_are_padded(self) -> None:
        for val in range(100):
            friendly = _uuid_to_friendly(uuid.UUID(int=val))
            assert len(friendly) == 22

    def test_short_strings_decode_correctly(self) -> None:
        for val in range(200):
            u = uuid.UUID(int=val)
            assert _friendly_to_uuid(_uuid_to_friendly(u)) == u


# ── 8. Lexicographic ordering ───────────────────────────────────────

class TestLexicographicOrdering:
    """Base62 string ordering matches UUID integer ordering ONLY when
    FriendlyIds are zero-padded to a uniform width.  The Python encoder
    does NOT pad, so raw string comparison is NOT order-preserving.
    Padded to 22 chars it IS order-preserving."""

    def test_padded_order_matches_uuid_order(self) -> None:
        uuids = sorted(_random_uuid() for _ in range(500))
        padded = [_pad22(_uuid_to_friendly(u)) for u in uuids]
        assert padded == sorted(padded)

    def test_padded_output_preserves_ordering(self) -> None:
        """With 22-char zero-padding, lexicographic and numeric order agree."""
        small = uuid.UUID(int=9)
        large = uuid.UUID(int=62)
        f_small = _uuid_to_friendly(small)
        f_large = _uuid_to_friendly(large)
        assert f_small < f_large
        assert small.int < large.int


# ── 9. Collision resistance ──────────────────────────────────────────

class TestCollisionResistance:
    """10 000 random UUIDs must produce 10 000 distinct FriendlyIds."""

    def test_no_collisions_in_10k(self) -> None:
        friendlies = {_uuid_to_friendly(_random_uuid()) for _ in range(10_000)}
        assert len(friendlies) == 10_000


# ── 10. Padding / Java compatibility ────────────────────────────────

class TestJavaPaddingCompatibility:
    """The Devskiller Java FriendlyId library zero-pads all FriendlyIds
    to exactly 22 characters — matching the Java library's output."""

    def test_python_pads_to_22(self) -> None:
        """Python now matches Java: UUID(1) → '0000000000000000000001'."""
        friendly = _uuid_to_friendly(uuid.UUID(int=1))
        assert friendly == "0000000000000000000001"
        assert len(friendly) == 22

    def test_decoder_accepts_padded_input(self) -> None:
        for val in [0, 1, 42, 61, 62, 2**64, 2**127, 2**128 - 1]:
            u = uuid.UUID(int=val)
            encoded = _uuid_to_friendly(u)
            assert len(encoded) == 22
            assert _friendly_to_uuid(encoded) == u

    def test_round_trip_through_padding(self) -> None:
        for _ in range(500):
            u = _random_uuid()
            friendly = _uuid_to_friendly(u)
            assert len(friendly) == 22
            assert _friendly_to_uuid(friendly) == u

    def test_encode_reproduces_java_padding(self) -> None:
        """Encoding after decoding a Java-padded string preserves padding."""
        java_padded = "0000000000000000000001"
        decoded = _friendly_to_uuid(java_padded)
        re_encoded = _uuid_to_friendly(decoded)
        assert re_encoded == java_padded

    def test_large_uuids_produce_20_to_22_chars(self) -> None:
        """UUID4 (122 random bits) typically produces 20-22 char strings.
        The padding gap (vs Java's fixed 22) only matters for very small
        UUIDs, but even UUID4 can be shorter than 22."""
        for _ in range(1000):
            friendly = _uuid_to_friendly(uuid.uuid4())
            assert 20 <= len(friendly) <= 22


# ── Decoder robustness ──────────────────────────────────────────────

class TestDecoderRobustness:
    """Fuzz the decoder with invalid inputs -- it should raise, not
    silently corrupt."""

    def test_empty_string_silently_decodes_to_zero(self) -> None:
        """BUG / QUIRK: the decoder accepts "" and returns UUID(0).
        The for-loop body never executes, so n stays 0. This is
        arguably incorrect -- an empty string is not a valid FriendlyId
        -- but documenting the actual behavior here."""
        result = _friendly_to_uuid("")
        assert result == uuid.UUID(int=0)

    @pytest.mark.parametrize(
        "bad_input",
        [
            "!!!",       # non-base62
            "abc-def",   # hyphen
            "abc def",   # space
            "ZZZZZZZZZZZZZZZZZZZZZZZz",  # 24 chars -- too long, overflows 128 bits
        ],
    )
    def test_invalid_input_raises(self, bad_input: str) -> None:
        with pytest.raises((KeyError, ValueError)):
            _friendly_to_uuid(bad_input)

    def test_overflow_raises_value_error(self) -> None:
        """A base62 string whose numeric value exceeds 2^128-1 must
        be rejected by uuid.UUID(int=...)."""
        # "7n42DGM5Tflk9n8mt7Fhc8" is max+1 in base62
        overflow = "7n42DGM5Tflk9n8mt7Fhc8"
        with pytest.raises(ValueError):
            _friendly_to_uuid(overflow)
