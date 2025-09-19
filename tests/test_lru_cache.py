import pytest

from rayanzz import LRUCache


def test_put_refreshes_recency_on_update():
    cache = LRUCache(capacity=2)
    cache.put("alpha", 1)
    cache.put("beta", 2)

    # Updating an existing key should mark it as the most recently used entry.
    cache.put("alpha", 3)
    cache.put("gamma", 4)

    assert "alpha" in cache  # ``alpha`` must survive as the most recent key.
    assert "gamma" in cache
    assert "beta" not in cache  # ``beta`` is the true LRU entry and should go.


def test_get_promotes_entry():
    cache = LRUCache(capacity=2)
    cache.put("left", 1)
    cache.put("right", 2)

    assert cache.get("left") == 1
    cache.put("extra", 3)

    assert "left" in cache  # Accessing ``left`` should protect it from eviction.
    assert "right" not in cache


def test_peek_does_not_change_order():
    cache = LRUCache(capacity=2)
    cache.put("first", 1)
    cache.put("second", 2)

    assert cache.peek("first") == 1
    cache.put("third", 3)

    assert "first" not in cache
    assert "second" in cache
    assert "third" in cache


def test_on_evict_receives_event():
    events = []

    def collector(event):
        events.append((event.key, event.value))

    cache = LRUCache(capacity=1, on_evict=collector)
    cache.put("one", 1)
    cache.put("two", 2)

    assert events == [("one", 1)]


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
