"""Fixed UTF-8 byte vocabulary over a safe, lossless canonical tree format."""

import json
from pathlib import Path

import sympy as sp

from mathai import RULES, Action, State, apply_action

SERIALIZATION_VERSION = 1
VOCABULARY_VERSION = 1
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[BOS]", "[CURRENT]", "[TARGET]", "[ACTION]", "[NEXT]", "[EOS]")
FIELDS = ("current", "target", "action", "next")


def expression_to_data(expression):
    expression = State(expression, 0).expression

    def convert(node):
        if isinstance(node, sp.Symbol):
            return ["S", node.name]
        if isinstance(node, sp.Rational):
            return ["Q", str(node.p), str(node.q)]
        return [{sp.Add: "A", sp.Mul: "M", sp.Pow: "P"}[node.func],
                [convert(child) for child in node.args]]

    return convert(expression)


def expression_from_data(data):
    """Decode only allowlisted tree nodes; never evaluate serialized source."""
    if not isinstance(data, list) or not data or not isinstance(data[0], str):
        raise ValueError("Invalid expression tree")
    tag = data[0]
    if tag == "S" and len(data) == 2 and isinstance(data[1], str) and data[1]:
        result = sp.Symbol(data[1])
    elif tag == "Q" and len(data) == 3 and all(isinstance(part, str) for part in data[1:]):
        try:
            numerator, denominator = int(data[1]), int(data[2])
        except ValueError:
            raise ValueError("Invalid rational integer encoding") from None
        if denominator <= 0:
            raise ValueError("Rational denominator must be positive")
        result = sp.Rational(numerator, denominator)
    elif tag in ("A", "M", "P") and len(data) == 2 and isinstance(data[1], list):
        if (tag == "P" and len(data[1]) != 2) or (tag != "P" and len(data[1]) < 2):
            raise ValueError("Invalid expression arity")
        args = [expression_from_data(child) for child in data[1]]
        if tag == "P" and (not isinstance(args[1], sp.Integer) or args[1] < 0):
            raise ValueError("Invalid polynomial exponent")
        result = {"A": sp.Add, "M": sp.Mul, "P": sp.Pow}[tag](*args)
    else:
        raise ValueError("Unknown or malformed expression node")
    # Reject noncanonical encodings instead of silently changing tree structure.
    if expression_to_data(result) != data:
        raise ValueError("Expression encoding is not canonical")
    return result


def action_to_data(action):
    if action.rule not in RULES:
        raise ValueError("Unknown rule identifier")
    for values in (action.path, action.selection):
        if not isinstance(values, tuple) or any(type(i) is not int or i < 0 for i in values):
            raise ValueError("Action indices must be nonnegative integer tuples")
    selective = action.rule in ("distribute_term", "factor_pair")
    if selective != (len(action.selection) == 2 and action.expected is not None):
        raise ValueError("Invalid selective action metadata")
    if not selective and (action.selection or action.expected is not None):
        raise ValueError("Unexpected action metadata")
    return {"rule": action.rule, "path": list(action.path), "selection": list(action.selection),
            "expected": expression_to_data(action.expected) if action.expected is not None else None}


def action_from_data(data):
    if not isinstance(data, dict) or set(data) != {"rule", "path", "selection", "expected"}:
        raise ValueError("Invalid action object")
    if not isinstance(data["path"], list) or not isinstance(data["selection"], list):
        raise ValueError("Action indices must be lists")
    action = Action(data["rule"], tuple(data["path"]), tuple(data["selection"]),
                    expression_from_data(data["expected"]) if data["expected"] is not None else None)
    if action_to_data(action) != data:
        raise ValueError("Invalid action encoding")
    return action


def validate_record(record):
    if (not isinstance(record, dict) or set(record) != {"version", *FIELDS}
            or type(record["version"]) is not int or record["version"] != SERIALIZATION_VERSION):
        raise ValueError("Unsupported candidate record format")
    for field in ("current", "target", "next"):
        expression_from_data(record[field])
    action_from_data(record["action"])


def serialize_candidate(state, action, successor):
    verified, _ = apply_action(state, action)
    if verified != successor:
        raise ValueError("Supplied successor does not match verified action")
    return {"version": SERIALIZATION_VERSION, "current": expression_to_data(state.expression),
            "target": expression_to_data(state.target), "action": action_to_data(action),
            "next": expression_to_data(successor.expression)}


class PolynomialTokenizer:
    """Eight special IDs followed by 256 byte IDs; unknown ID is reserved.

    Every UTF-8 string is representable, including arbitrary symbol names. Field
    separators are distinct IDs, never recognized inside JSON payload bytes.
    Inputs exceeding max_length raise ValueError; nothing is truncated.
    """

    pad_id = 0
    unknown_id = 1
    vocab_size = len(SPECIAL_TOKENS) + 256

    def __init__(self, max_length=2048):
        if type(max_length) is not int or max_length < 10:
            raise ValueError("max_length must be an integer >= 10")
        self.max_length = max_length

    def encode(self, record):
        validate_record(record)
        ids = [2]
        for separator, field in enumerate(FIELDS, 3):
            payload = json.dumps(record[field], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            ids.append(separator)
            ids.extend(byte + len(SPECIAL_TOKENS) for byte in payload.encode("utf-8"))
        ids.append(7)
        if len(ids) > self.max_length:
            raise ValueError(f"Input has {len(ids)} tokens, exceeding max_length={self.max_length}; truncation is disabled")
        return ids

    def decode(self, ids):
        ids = list(ids)
        while ids and ids[-1] == self.pad_id:
            ids.pop()
        if not ids or ids[0] != 2 or ids[-1] != 7:
            raise ValueError("Invalid sequence boundaries")
        cursor, record = 1, {"version": SERIALIZATION_VERSION}
        for separator, field in enumerate(FIELDS, 3):
            if cursor >= len(ids) or ids[cursor] != separator:
                raise ValueError("Invalid field separator")
            cursor += 1
            payload = []
            while cursor < len(ids) and type(ids[cursor]) is int and 8 <= ids[cursor] < self.vocab_size:
                payload.append(ids[cursor]-8)
                cursor += 1
            try:
                record[field] = json.loads(bytes(payload).decode("utf-8"))
            except (ValueError, UnicodeError):
                raise ValueError("Invalid field payload") from None
        if cursor != len(ids)-1:
            raise ValueError("Unexpected sequence tokens")
        validate_record(record)
        return record

    def batch_encode(self, records):
        encoded = [self.encode(record) for record in records]
        if not encoded:
            raise ValueError("Cannot encode an empty batch")
        length = max(map(len, encoded))
        return {"input_ids": [ids + [0]*(length-len(ids)) for ids in encoded],
                "attention_mask": [[True]*len(ids) + [False]*(length-len(ids)) for ids in encoded]}

    def save(self, path):
        data = {"vocabulary_version": VOCABULARY_VERSION, "serialization_version": SERIALIZATION_VERSION,
                "max_length": self.max_length, "special_tokens": list(SPECIAL_TOKENS),
                "byte_values": list(range(256)), "overlength_policy": "error"}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"vocabulary_version": 1, "serialization_version": 1,
                    "special_tokens": list(SPECIAL_TOKENS), "byte_values": list(range(256)),
                    "overlength_policy": "error"}
        if set(data) != {*expected, "max_length"} or any(data[key] != value for key, value in expected.items()):
            raise ValueError("Unsupported tokenizer configuration or vocabulary")
        return cls(data["max_length"])
