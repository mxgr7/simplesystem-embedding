"""Shared helpers for the article->category classifier (catclass).

Tree = /data/s2class-categories.json: 4,381 8-digit eclass-style codes with
complete parent chains (36 L1 / 325 L2 / 1,107 L3 / 2,913 L4). Depth is
encoded in trailing '00' digit pairs. Observed vendor codes go deeper /
off-tree (~16k distinct) and are resolved to their deepest named ancestor.
"""
import json
import os

TREE_PATH = "/data/s2class-categories.json"
ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "out", "category_tree.json")
# The junk lexicon lives in contracts/s2_junk.json (MXG-50) — read it through
# s2_junk_contract.py. This module stays free of file IO at import: it is the
# bootstrap layer that build_category_tree seeds the artifact from.


def code_depth(code):
    if len(code) != 8 or not code.isdigit():
        return 0
    if code[2:] == "000000":
        return 1
    if code[4:] == "0000":
        return 2
    if code[6:] == "00":
        return 3
    return 4


def code_ancestors(code):
    """Ancestor chain including the code itself, L1 first."""
    out = []
    for pfx, pad in ((2, "000000"), (4, "0000"), (6, "00"), (8, "")):
        a = code[:pfx] + pad
        if a not in out:
            out.append(a)
        if a == code:
            break
    return out


def load_tree(path=TREE_PATH):
    with open(path) as f:
        return json.load(f)


def load_artifact(path=ARTIFACT_PATH):
    with open(path) as f:
        return json.load(f)


def resolve_raw(code, tree):
    """Observed code -> deepest named tree ancestor (or None)."""
    code = str(code)
    if len(code) != 8 or not code.isdigit():
        return None
    for a in reversed(code_ancestors(code)):
        if a in tree:
            return a
    return None


def majority_walk(sigs, tree, children, tau=0.6, junk=frozenset(),
                  interim=frozenset()):
    """Top-down majority walk over weighted signatures.

    sigs: list of (codeset, weight); codeset = raw code strings of one offer
    signature. Signatures containing a junk code are dropped wholesale
    (their parent codes are the same junk); interim codes are ignored
    code-wise. Each surviving signature votes with its resolved ancestor
    closure. Descend while the best child's weight >= tau * classified
    weight. Returns (label_code or None, support_share, n_signatures).
    """
    votes = []                       # (ancestor-closure set, weight)
    for codeset, w in sigs:
        raw = {str(c) for c in codeset}
        if any(resolve_raw(c, tree) in junk or c in junk for c in raw):
            continue
        closure = set()
        for c in raw:
            r = resolve_raw(c, tree)
            if r and r not in interim:
                closure.update(code_ancestors(r))
        if closure:
            votes.append((closure, w))
    if not votes:
        return None, 0.0, 0
    total = sum(w for _, w in votes)
    node, share = None, 0.0
    while True:
        cands = children.get(node) if node else children.get("")
        if not cands:
            break
        best, bw = None, 0.0
        for ch in cands:
            w = sum(w for s, w in votes if ch in s)
            if w > bw:
                best, bw = ch, w
        if best is None or bw < tau * total:
            break
        node, share = best, bw / total
    return node, share, len(votes)


def build_children(tree):
    """code -> sorted child codes; '' -> L1 roots."""
    children = {}
    for c in tree:
        d = code_depth(c)
        parent = "" if d == 1 else code_ancestors(c)[d - 2]
        children.setdefault(parent, []).append(c)
    for v in children.values():
        v.sort()
    return children


def maximal_chains(codes):
    """Deepest codes not ancestors of another code in the set."""
    cs = set(codes)
    return sorted(c for c in cs
                  if not any(o != c and c in code_ancestors(o) for o in cs))
