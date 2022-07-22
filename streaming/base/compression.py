import brotli
import gzip
import snappy
import zstd


class Compression(object):
    def compress(self, data: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError


class Brotli(Compression):
    qualities = list(range(12))

    def __init__(self, quality: int = 11) -> None:
        assert quality in self.qualities
        self.quality = quality

    def compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=self.quality)

    def decompress(self, data: bytes) -> bytes:
        return brotli.decompress(data)


class Gzip(Compression):
    levels = list(range(10))

    def __init__(self, level: int = 9) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class Snappy(Compression):
    def compress(self, data: bytes) -> bytes:
        return snappy.compress(data)

    def decompress(self, data: bytes) -> bytes:
        return snappy.decompress(data)


class Zstd(Compression):
    levels = list(range(1, 23))

    def __init__(self, level: int = 3) -> None:
        assert level in self.levels
        self.level = level

    def compress(self, data) -> bytes:
        return zstd.compress(data, self.level)

    def decompress(self, data: bytes) -> bytes:
        return zstd.decompress(data)


_name2cls = {
    'brotli': Brotli,
    'gzip': Gzip,
    'snappy': Snappy,
    'zstd': Zstd,
}


def _get(algo):
    idx = algo.find(':')
    if idx == -1:
        name = algo
        args = ()
    else:
        name, level = algo.split(':')
        args = int(level),
    cls = _name2cls[name]
    return cls(*args)


def is_valid_compression(algo):
    if algo is None:
        return True
    try:
        _get(algo)
    except:
        return False
    return True


def compress(algo, data):
    if algo is None:
        return data
    lib = _get(algo)
    return lib.compress(data)


def decompress(algo, data):
    if algo is None:
        return data

    lib = _get(algo)
    return lib.decompress(data)
