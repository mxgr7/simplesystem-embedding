#!/usr/bin/env python3
"""German-style text fold — the ä→ae / ö→oe / ü→ue / ß→ss transliteration variant.

Alternative to fold_text.fold (which STRIPS diacritics: ä→a). This EXPANDS the
German umlauts to their two-letter forms (the standard German search normalization
for keyboards without umlaut keys), then strips any remaining non-German diacritics
(é→e). casefold already maps ß→ss and lowercases. Idempotent.

  fold_de: NFC -> casefold -> NFC -> {ä→ae, ö→oe, ü→ue, ß→ss} -> drop remaining
  combining marks -> NFC.
"""
import unicodedata

_MAP = (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"))


def fold_de(text):
    if not text:
        return "" if text is None else text
    t = unicodedata.normalize("NFC", str(text)).casefold()
    t = unicodedata.normalize("NFC", t)          # recompose umlauts after casefold
    for a, b in _MAP:
        t = t.replace(a, b)
    t = "".join(c for c in unicodedata.normalize("NFKD", t)
                if not unicodedata.combining(c))  # strip non-German diacritics (é→e)
    return unicodedata.normalize("NFC", t)


if __name__ == "__main__":
    checks = {
        "Kühlschrank": "kuehlschrank",
        "Schloßschraube GROß": "schlossschraube gross",
        "für": "fuer",
        "Größe": "groesse",
        "Lüfter": "luefter",
        "Mülltonne": "muelltonne",
        "V4034PX": "v4034px",
        "Café": "cafe",
        "": "",
    }
    for src, want in checks.items():
        got = fold_de(src)
        assert got == want, (src, got, want)
        assert fold_de(got) == got, ("not idempotent", src, got, fold_de(got))
    print("fold_de self-checks OK")
