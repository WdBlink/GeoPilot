"""Strict JSON boundary shared by the Node launcher and numerical adapter."""
import json
import math
import sys
from pathlib import Path


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON number')
    if type(value) is int and abs(value) > 2**53 - 1:
        raise ValueError('Unsafe JSON integer')
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    return value


def read_json(path):
    text = sys.stdin.read() if str(path) == '-' else Path(path).read_text()
    return finite(json.loads(text, object_pairs_hook=unique))


if __name__ == '__main__':
    print(json.dumps(read_json(sys.argv[1]), allow_nan=False))
