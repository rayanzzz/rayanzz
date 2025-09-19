"""Implementation of a simple least recently used cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, Iterator, MutableMapping, Optional, Tuple, TypeVar


K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class EvictionEvent(Generic[K, V]):
    """Information about an eviction that occurred in the cache."""

    key: K
    value: V


class LRUCache(MutableMapping[K, V]):
    """A dictionary-like cache that evicts the least recently used entry.

    The cache keeps the most recently accessed entry as the newest item in an
    internal ``OrderedDict``.  Evictions happen automatically when the cache
    grows past ``capacity``.

    Parameters
    ----------
    capacity:
        Maximum number of entries that can be stored simultaneously.  Must be
        greater than zero.
    on_evict:
        Optional callback that will be invoked whenever an entry is evicted.
        The callback receives an :class:`EvictionEvent`.
    """

    def __init__(
        self,
        capacity: int,
        *,
        on_evict: Optional[Callable[[EvictionEvent[K, V]], None]] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self._capacity = capacity
        self._entries: "OrderedDict[K, V]" = OrderedDict()
        self._on_evict = on_evict

    # -- collection protocol -------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def __iter__(self) -> Iterator[K]:  # pragma: no cover - passthrough
        return iter(self._entries)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - passthrough
        return key in self._entries

    def __getitem__(self, key: K) -> V:
        try:
            value = self._entries[key]
        except KeyError:
            raise

        self._entries.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        del self._entries[key]

    # -- public API ----------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of entries the cache can hold."""

        return self._capacity

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Retrieve ``key`` from the cache.

        Unlike :meth:`__getitem__`, the method returns ``default`` when the key
        is absent.  Accessing an entry marks it as recently used so that it is
        not the next candidate for eviction.
        """

        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        return default

    def put(self, key: K, value: V) -> None:
        """Insert ``key`` into the cache.

        The operation updates the existing entry if the key is already present
        and marks it as the most recently used value.  When the cache is full,
        the least recently used item is evicted.
        """

        contains_key = key in self._entries
        self._entries[key] = value

        # NOTE: ``OrderedDict`` keeps insertion order and does not
        # automatically refresh the position when the value of an existing key
        # is overwritten.  Without the explicit call to ``move_to_end`` the
        # cache would treat an updated entry as if it was never accessed, so
        # inserting a new element afterwards would evict the freshly updated
        # entry instead of the true least recently used item.
        self._entries.move_to_end(key)

        if not contains_key and len(self._entries) > self._capacity:
            evicted_key, evicted_value = self._entries.popitem(last=False)
            if self._on_evict is not None:
                self._on_evict(EvictionEvent(evicted_key, evicted_value))

    def peek(self, key: K) -> V:
        """Return ``key`` without affecting its recency information."""

        return self._entries[key]

    def clear(self) -> None:  # pragma: no cover - passthrough
        self._entries.clear()

    def items(self) -> Iterable[Tuple[K, V]]:  # pragma: no cover - passthrough
        return self._entries.items()

    # -- helpers -------------------------------------------------------------

    def snapshot(self) -> Dict[K, V]:
        """Return a shallow copy of the cache contents in LRU order."""

        return dict(self._entries)

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"LRUCache(capacity={self._capacity}, entries={list(self._entries.items())})"
