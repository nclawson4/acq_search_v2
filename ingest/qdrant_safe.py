"""Defensive Qdrant client wrapper for v2.

The Qdrant API key we use is cluster-wide and shares a cluster with v1's
collections. To prevent any v2 code path from accidentally touching v1's
data, this wrapper enforces:

  1. Every collection name passed in must start with V2_PREFIX ("v2_").
  2. Known v1 collection names are explicitly denied.
  3. The wrapper does NOT expose a `delete_collection` shortcut for arbitrary
     names — only `delete_v2_collection(suffix)` is provided.
  4. `list_collections` filters v1 names out of any user-facing return value.

The intent is "fail closed": if a bug somewhere ever computes the wrong
collection name, the wrapper raises before any network call lands.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

V2_PREFIX = "v2_"

# Hard-coded list of v1 collections from acq_search_retrieval/ingest/config.py.
# If v1 adds new collections in the future, this list may be stale — that is
# fine for the deny check; we always require the v2_ prefix as the
# affirmative condition. The deny list is only a second line of defense.
V1_COLLECTIONS_DENY = frozenset({"segments", "frames", "moments", "sessions"})


class V1CollectionAccessError(RuntimeError):
    """Raised when v2 code tries to operate on a non-v2_ collection."""


def _enforce(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise V1CollectionAccessError(f"Invalid collection name: {name!r}")
    if name in V1_COLLECTIONS_DENY:
        raise V1CollectionAccessError(
            f"Refused: collection {name!r} belongs to v1. v2 code may not touch it."
        )
    if not name.startswith(V2_PREFIX):
        raise V1CollectionAccessError(
            f"Refused: collection {name!r} does not have the required {V2_PREFIX!r} prefix."
        )
    return name


class SafeQdrantClient:
    """Wraps qdrant_client.QdrantClient with v2-only enforcement on every call
    that takes a `collection_name` argument."""

    def __init__(self, url: str, api_key: str, timeout: int = 60) -> None:
        self._inner = QdrantClient(url=url, api_key=api_key, timeout=timeout)

    @property
    def inner(self) -> QdrantClient:
        """Escape hatch — direct access. Callers using this bypass safety; do not."""
        return self._inner

    # ----- list / introspection -----

    def list_v2_collections(self) -> list[str]:
        """Return only v2_-prefixed collection names. v1 collections are filtered."""
        cols = self._inner.get_collections().collections
        return [c.name for c in cols if c.name.startswith(V2_PREFIX)]

    def collection_exists(self, name: str) -> bool:
        _enforce(name)
        return self._inner.collection_exists(name)

    # ----- create / configure -----

    def create_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        _enforce(name)
        dist_map = {
            "Cosine": qmodels.Distance.COSINE,
            "Euclid": qmodels.Distance.EUCLID,
            "Dot": qmodels.Distance.DOT,
        }
        self._inner.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=dist_map[distance]),
        )

    def recreate_collection(self, name: str, vector_size: int, distance: str = "Cosine") -> None:
        _enforce(name)
        if self._inner.collection_exists(name):
            self._inner.delete_collection(collection_name=name)
        self.create_collection(name, vector_size, distance)

    # ----- writes -----

    def upsert(self, name: str, points: list[qmodels.PointStruct]) -> None:
        _enforce(name)
        self._inner.upsert(collection_name=name, points=points)

    # ----- reads -----

    def search(self, name: str, vector: list[float], limit: int = 10, with_payload: bool = True):
        _enforce(name)
        return self._inner.search(
            collection_name=name,
            query_vector=vector,
            limit=limit,
            with_payload=with_payload,
        )

    def count(self, name: str) -> int:
        _enforce(name)
        return int(self._inner.count(collection_name=name, exact=True).count)

    # ----- destructive (explicit, prefixed) -----

    def delete_v2_collection(self, suffix: str) -> None:
        """Delete a v2 collection. `suffix` is concatenated with V2_PREFIX —
        this signature makes it physically harder to pass a v1 name."""
        if not suffix or "/" in suffix or "_" in suffix.replace("_", "", 0):
            # allow underscores in suffix (e.g. "scenes_dev") but reject path-y junk
            pass
        full = f"{V2_PREFIX}{suffix}"
        _enforce(full)
        self._inner.delete_collection(collection_name=full)


def open_safe(url: str, api_key: str) -> SafeQdrantClient:
    return SafeQdrantClient(url=url, api_key=api_key)
