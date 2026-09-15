"""Bounded, deadline-aware framing for the formal Planning socket boundary."""

from __future__ import annotations

import json
import math
import select
import socket
import struct
import time
from typing import Any, Mapping


FRAME_MAGIC = b"LRC2"
FRAME_KIND_JSON = 1
FRAME_KIND_OBSERVATION = 2
MAX_JSON_FRAME_BYTES = 2 * 1024 * 1024
MAX_OBSERVATION_FRAME_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 18
MAX_JSON_NODES = 24_000
MAX_JSON_STRING_CHARS = 262_144
MAX_JSON_CONTAINER_ITEMS = 2_048
_HEADER = struct.Struct("!4sBQ")
_LIMITS = {
    FRAME_KIND_JSON: MAX_JSON_FRAME_BYTES,
    FRAME_KIND_OBSERVATION: MAX_OBSERVATION_FRAME_BYTES,
}


class FormalProcessTransportError(RuntimeError):
    """Raised on framing, strict-codec, closure, or absolute-deadline failure."""


def _strict_timeout(value: object) -> float:
    if (
        type(value) not in (int, float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("transport timeout must be a finite positive number")
    return float(value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_json_tree(value: object) -> None:
    """Bound decoded/encoded JSON structure as well as byte length."""

    remaining = MAX_JSON_NODES
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        remaining -= 1
        if remaining < 0:
            raise FormalProcessTransportError("JSON message has too many values")
        if depth > MAX_JSON_DEPTH:
            raise FormalProcessTransportError("JSON message nesting is too deep")
        if current is None or type(current) is bool:
            continue
        if type(current) is int:
            if not -(2**63) <= current < 2**63:
                raise FormalProcessTransportError("JSON integer is outside int64")
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise FormalProcessTransportError("JSON number is not finite")
            continue
        if type(current) is str:
            if len(current) > MAX_JSON_STRING_CHARS:
                raise FormalProcessTransportError("JSON string is too long")
            continue
        if type(current) is list:
            if len(current) > MAX_JSON_CONTAINER_ITEMS:
                raise FormalProcessTransportError("JSON list has too many items")
            stack.extend((item, depth + 1) for item in current)
            continue
        if type(current) is dict:
            if len(current) > MAX_JSON_CONTAINER_ITEMS:
                raise FormalProcessTransportError("JSON object has too many fields")
            for key, item in current.items():
                if type(key) is not str or len(key) > 256:
                    raise FormalProcessTransportError(
                        "JSON object keys must be short native strings"
                    )
                stack.append((item, depth + 1))
            continue
        raise FormalProcessTransportError(
            f"JSON contains forbidden type {type(current).__name__}"
        )


def strict_json_bytes(payload: Mapping[str, Any]) -> bytes:
    if type(payload) is not dict:
        raise FormalProcessTransportError("process JSON must be an exact object")
    _validate_json_tree(payload)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalProcessTransportError(f"message is not strict JSON: {exc}") from exc
    if not encoded or len(encoded) > MAX_JSON_FRAME_BYTES:
        raise FormalProcessTransportError("JSON frame exceeds its exact byte limit")
    return encoded


def strict_json_loads(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise FormalProcessTransportError("JSON frame body must be non-empty bytes")
    if len(payload) > MAX_JSON_FRAME_BYTES:
        raise FormalProcessTransportError("JSON frame exceeds its exact byte limit")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FormalProcessTransportError(f"invalid strict process JSON: {exc}") from exc
    if type(value) is not dict:
        raise FormalProcessTransportError("process JSON must be an exact object")
    _validate_json_tree(value)
    return value


class BoundedFrameSocket:
    """One non-blocking socket with whole-frame absolute I/O deadlines."""

    __slots__ = ("_socket", "_default_timeout_s", "_closed")

    def __init__(self, value: socket.socket, *, default_timeout_s: float) -> None:
        if type(value) is not socket.socket:
            raise TypeError("formal transport requires an exact socket.socket")
        self._socket = value
        self._socket.setblocking(False)
        self._default_timeout_s = _strict_timeout(default_timeout_s)
        self._closed = False

    @property
    def fileno(self) -> int:
        return self._socket.fileno()

    def _deadline(self, timeout_s: float | None) -> float:
        timeout = (
            self._default_timeout_s
            if timeout_s is None
            else _strict_timeout(timeout_s)
        )
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise FormalProcessTransportError("formal socket absolute deadline expired")
        return remaining

    def _read_exact(self, size: int, deadline: float) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            try:
                readable, _, _ = select.select(
                    [self._socket], [], [], self._remaining(deadline)
                )
            except (OSError, ValueError) as exc:
                raise FormalProcessTransportError(
                    f"formal socket read wait failed: {exc}"
                ) from exc
            if not readable:
                raise FormalProcessTransportError(
                    "formal socket absolute read deadline expired"
                )
            try:
                chunk = self._socket.recv(size - len(chunks))
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise FormalProcessTransportError(
                    f"formal socket read failed: {exc}"
                ) from exc
            if not chunk:
                raise FormalProcessTransportError("formal socket closed mid-frame")
            chunks.extend(chunk)
        return bytes(chunks)

    def _write_all(self, payload: bytes, deadline: float) -> None:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            try:
                _, writable, _ = select.select(
                    [], [self._socket], [], self._remaining(deadline)
                )
            except (OSError, ValueError) as exc:
                raise FormalProcessTransportError(
                    f"formal socket write wait failed: {exc}"
                ) from exc
            if not writable:
                raise FormalProcessTransportError(
                    "formal socket absolute write deadline expired"
                )
            try:
                count = self._socket.send(view[written:])
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise FormalProcessTransportError(
                    f"formal socket write failed: {exc}"
                ) from exc
            if count <= 0:
                raise FormalProcessTransportError("formal socket closed mid-frame")
            written += count

    def send_frame(
        self,
        kind: int,
        payload: bytes,
        *,
        timeout_s: float | None = None,
    ) -> None:
        if self._closed:
            raise FormalProcessTransportError("formal socket is closed")
        if type(kind) is not int or kind not in _LIMITS:
            raise FormalProcessTransportError("unknown formal frame kind")
        if type(payload) is not bytes or not payload:
            raise FormalProcessTransportError("formal frame body must be non-empty bytes")
        if len(payload) > _LIMITS[kind]:
            raise FormalProcessTransportError("formal frame exceeds its kind limit")
        deadline = self._deadline(timeout_s)
        header = _HEADER.pack(FRAME_MAGIC, kind, len(payload))
        self._write_all(header, deadline)
        self._write_all(payload, deadline)

    def recv_frame(
        self,
        expected_kind: int,
        *,
        timeout_s: float | None = None,
    ) -> bytes:
        if self._closed:
            raise FormalProcessTransportError("formal socket is closed")
        if type(expected_kind) is not int or expected_kind not in _LIMITS:
            raise FormalProcessTransportError("unknown expected formal frame kind")
        deadline = self._deadline(timeout_s)
        raw_header = self._read_exact(_HEADER.size, deadline)
        magic, kind, size = _HEADER.unpack(raw_header)
        if magic != FRAME_MAGIC:
            raise FormalProcessTransportError("formal frame magic is invalid")
        if kind != expected_kind:
            raise FormalProcessTransportError("formal frame kind is invalid")
        if size < 1 or size > _LIMITS[kind]:
            raise FormalProcessTransportError("formal frame length is invalid")
        return self._read_exact(size, deadline)

    def send_json(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> None:
        self.send_frame(
            FRAME_KIND_JSON,
            strict_json_bytes(payload),
            timeout_s=timeout_s,
        )

    def recv_json(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return strict_json_loads(
            self.recv_frame(FRAME_KIND_JSON, timeout_s=timeout_s)
        )

    def send_observation_bytes(
        self,
        payload: bytes,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self.send_frame(FRAME_KIND_OBSERVATION, payload, timeout_s=timeout_s)

    def recv_observation_bytes(
        self,
        *,
        timeout_s: float | None = None,
    ) -> bytes:
        return self.recv_frame(FRAME_KIND_OBSERVATION, timeout_s=timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()


__all__ = [
    "BoundedFrameSocket",
    "FRAME_KIND_JSON",
    "FRAME_KIND_OBSERVATION",
    "FRAME_MAGIC",
    "FormalProcessTransportError",
    "MAX_JSON_FRAME_BYTES",
    "MAX_OBSERVATION_FRAME_BYTES",
    "strict_json_bytes",
    "strict_json_loads",
]
