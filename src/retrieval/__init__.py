"""Retrieval stage: BM25, dense, hybrid, and the records they return.

``base.py`` holds the contract they share. Read it before adding a method: a
retriever satisfies ``Retriever`` and drops into the ablation loop, rather than
the loop growing a branch for it.
"""
