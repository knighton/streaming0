from io import BytesIO
import json
import numpy as np
from PIL import Image
import pickle
from typing import Any, Optional


class Encoding(object):
    """Encodes and decodes between objects of a certain type and serialized bytes."""
    
    def encode(self, obj: Any) -> bytes:
        """Encode the given data from the original object to bytes.

        Args:
            obj (Any): Decoded object.

        Returns:
            bytes: Encoded data.
        """
        raise NotImplementedError
    
    def decode(self, data: bytes) -> Any:
        """Decode the given data from bytes to the original object.

        Args:
            data (bytes): Encoded data.

        Returns:
            Any: Decoded object.
        """
        raise NotImplementedError

    def get_size(self) -> Optional[int]:
        """Get the fixed size of all encodings of this type, if there is one, else None.

        Returns:
            Optional[int]: Fixed size of encoded data.
        """
        return None


class Bytes(Encoding):
    """No-op encoding."""

    def encode(self, obj: Any) -> bytes:
        return obj

    def decode(self, data: bytes) -> Any:
        return data


class Str(Encoding):
    """Store as UTF-8."""

    def encode(self, obj: str) -> bytes:
        return obj.encode('utf-8')

    def decode(self, data: bytes) -> str:
        return data.decode('utf-8')


class Int(Encoding):
    """Store as int64."""

    def encode(self, obj: int) -> bytes:
        return np.int64(obj).tobytes()

    def decode(self, data: bytes) -> int:
        return int(np.frombuffer(data, np.int64)[0])

    def get_size(self) -> Optional[int]:
        return 8


class PIL(Encoding):
    """Store PIL image raw."""

    def encode(self, obj: Image.Image) -> bytes:
        mode = obj.mode.encode('utf-8')
        width, height = obj.size
        raw = obj.tobytes()
        ints = np.array([width, height, len(mode)], np.uint32) 
        return ints.tobytes() + mode + raw

    def decode(self, data: bytes) -> Image.Image:
        idx = 3 * 4
        width, height, mode_size = np.frombuffer(data[:idx], np.uint32)
        idx2 = idx + mode_size
        mode = data[idx:idx2].decode('utf-8')
        size = width, height
        raw = data[idx2:]
        return Image.frombytes(mode, size, raw)  # pyright: ignore


class JPEG(Encoding):
    """Store PIL image as JPEG."""

    def encode(self, obj: Image.Image) -> bytes:
        out = BytesIO()
        obj.save(out, format='JPEG')
        return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)
        

class PNG(Encoding):
    """Store PIL image as PNG."""

    def encode(self, obj: Image.Image) -> bytes:
        out = BytesIO()
        obj.save(out, format='PNG')
        return out.getvalue()

    def decode(self, data: bytes) -> Image.Image:
        inp = BytesIO(data)
        return Image.open(inp)


class Pickle(Encoding):
    """Store arbitrary data as pickle."""

    def encode(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def decode(self, data: bytes) -> Any:
        return pickle.loads(data)


class JSON(Encoding):
    """Store arbitrary data as JSON."""

    def encode(self, obj: Any) -> bytes:
        return json.dumps(obj).encode('utf-8')

    def decode(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))


_encodings = {
    'bytes': Bytes,
    'str': Str,
    'int': Int,
    'pil': PIL,
    'jpeg': JPEG,
    'png': PNG,
    'pkl': Pickle,
    'json': JSON,
}


def decode(encoding: str, data: bytes) -> Any:
    """Decode the given data from bytes to the original object.

    Args:
        encoding (str): Encoding.
        data (bytes): Encoded data.

    Returns:
        Any: Decoded object.
    """
    coder = _encodings[encoding]()
    return coder.decode(data)


def encode(encoding: str, obj: Any) -> bytes:
    """Encode the given data from the original object to bytes.

    Args:
        encoding (str): Encoding.
        obj (Any): Decoded object.

    Returns:
        bytes: Encoded data.
    """
    if isinstance(obj, bytes):
        return obj
    coder = _encodings[encoding]()
    return coder.encode(obj)


def get_fixed_encoded_size(encoding: str) -> Optional[int]:
    """Get the fixed size of all encodings of this type, if there is one, else None.

    Args:
        encoding (str): Encoding.

    Returns:
        Optional[int]: Fixed size of encoded data.
    """
    coder = _encodings[encoding]()
    return coder.get_size()


def is_valid_encoding(encoding: str) -> bool:
    """Get whether this is a valid encoding.

    Args:
        encoding (str): Encoding.

    Returns:
        bool: Whether the encoding is valid.
    """
    return encoding in _encodings
