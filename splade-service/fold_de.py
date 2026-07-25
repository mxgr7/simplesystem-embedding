import unicodedata


_MAP = (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss"))


def fold_de(text):
    if not text:
        return "" if text is None else text
    value = unicodedata.normalize("NFC", str(text)).casefold()
    value = unicodedata.normalize("NFC", value)
    for source, target in _MAP:
        value = value.replace(source, target)
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return unicodedata.normalize("NFC", value)
