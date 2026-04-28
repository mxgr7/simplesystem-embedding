"""Port of `com.devskiller.friendly_id` (1.1.0) — UUID ↔ 22-char base62.

The legacy `articleId` PK is `{friendlyId}:{base64Url(articleNumber)}`
where `friendlyId = FriendlyId.toFriendlyId(vendorId)` (see
`commons/.../domain/ArticleId.java:23`). We have to reproduce the Java
encoding bit-for-bit so PKs round-trip across the indexer (Python) and
any service that decodes them via the Java library.

Algorithm (mirrors `FriendlyId.java` in the upstream library):
  encode: UUID → 128-bit unsigned int → base62 string, left-padded with
          `'0'` to a fixed length of 22 chars (since 62**22 > 2**128).
  decode: base62 → unsigned int → UUID. Leading `'0'`s contribute 0 to
          the int so padding is transparent.

Alphabet ordering matches the upstream `Base62.GMP` style: digits, then
uppercase A-Z, then lowercase a-z (NOT the URL-safe variant).
"""

from __future__ import annotations

import uuid as _uuid

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_ALPHABET_INDEX = {c: i for i, c in enumerate(_ALPHABET)}
_BASE = 62
_LENGTH = 22


def to_friendly_id(value: _uuid.UUID) -> str:
    n = value.int
    if n == 0:
        return _ALPHABET[0] * _LENGTH
    chars: list[str] = []
    while n > 0:
        n, rem = divmod(n, _BASE)
        chars.append(_ALPHABET[rem])
    chars.reverse()
    out = "".join(chars)
    if len(out) < _LENGTH:
        out = _ALPHABET[0] * (_LENGTH - len(out)) + out
    return out


def to_uuid(friendly_id: str) -> _uuid.UUID:
    if len(friendly_id) > _LENGTH:
        raise ValueError(f"friendly_id too long: {friendly_id!r}")
    n = 0
    for c in friendly_id:
        try:
            n = n * _BASE + _ALPHABET_INDEX[c]
        except KeyError as exc:
            raise ValueError(f"invalid character {c!r} in friendly_id {friendly_id!r}") from exc
    if n >= 1 << 128:
        raise ValueError(f"friendly_id decodes to value > 2**128: {friendly_id!r}")
    return _uuid.UUID(int=n)
