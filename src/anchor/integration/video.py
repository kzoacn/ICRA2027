"""Streaming dual-camera episode video recording."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class FrameWriter(Protocol):
    def append_data(self, image: np.ndarray) -> None: ...

    def close(self) -> None: ...


def _default_writer(path: Path, fps: int) -> FrameWriter:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise RuntimeError("video recording requires imageio and an ffmpeg backend") from exc
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)


def _camera_mapping(observation: object) -> Mapping[str, Any]:
    cameras = getattr(observation, "cameras", None)
    if not isinstance(cameras, Mapping):
        raise TypeError("sensor observation must expose a cameras mapping")
    return cameras


def _rgb(frame: object) -> np.ndarray:
    image = np.asarray(getattr(frame, "rgb", None))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera RGB frame must be HxWx3 uint8")
    return image


def dual_view_frame(
    observation: object,
    camera_names: tuple[str, str] = ("agentview", "wrist"),
) -> np.ndarray:
    """Compose the external and wrist images without accessing raw simulator state."""

    cameras = _camera_mapping(observation)
    try:
        left, right = (_rgb(cameras[name]) for name in camera_names)
    except KeyError as exc:
        raise KeyError(f"dual-view recording requires cameras {camera_names!r}") from exc
    if left.shape[0] != right.shape[0]:
        height = min(left.shape[0], right.shape[0])
        left, right = left[:height], right[:height]
    return np.concatenate((left, right), axis=1)


class DualViewVideoRecorder:
    """Write frames incrementally to keep memory bounded during 300-step runs."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: int = 20,
        stride: int = 1,
        camera_names: tuple[str, str] = ("agentview", "wrist"),
        writer_factory: Callable[[Path, int], FrameWriter] = _default_writer,
    ) -> None:
        if fps < 1 or stride < 1:
            raise ValueError("fps and stride must be positive")
        self.path = Path(path)
        self.fps = fps
        self.stride = stride
        self.camera_names = camera_names
        self.writer_factory = writer_factory
        self._writer: FrameWriter | None = None
        self._seen = 0
        self._written = 0

    @property
    def written_frames(self) -> int:
        return self._written

    def add(self, observation: object) -> None:
        take = self._seen % self.stride == 0
        self._seen += 1
        if not take:
            return
        frame = dual_view_frame(observation, self.camera_names)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self.writer_factory(self.path, self.fps)
        self._writer.append_data(frame)
        self._written += 1

    def close(self) -> Mapping[str, str]:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return {"dual": str(self.path)} if self._written else {}

    def __enter__(self) -> "DualViewVideoRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
