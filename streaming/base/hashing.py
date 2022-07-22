import hashlib
from typing import Any, Callable, Dict
import xxhash


def _get() -> Dict[str, Callable[[bytes], Any]]:
    algo2hash = {}

    for algo in hashlib.algorithms_available:
        if hasattr(hashlib, algo):
            algo2hash[algo] = getattr(hashlib, algo)

    for algo in xxhash.algorithms_available:  # pyright: ignore
        algo2hash[algo] = getattr(xxhash, algo)

    return algo2hash


_algo2hash = _get()


def get_hash_algorithms():
    return sorted(_algo2hash)


def is_valid_hash_algorithm(algo):
    return algo in _algo2hash


def get_hash(algo, data):
    func = _algo2hash[algo]
    return func(data).hexdigest()
