"""I1 — Indexer pipeline (MongoDB → Milvus projection + bulk loader).

Phase A (this commit) ships the canonical projection module + a
deterministic-vector test loader. The production bulk pipeline (real
TEI, MongoDB scan, resume) is Phase B.
"""
