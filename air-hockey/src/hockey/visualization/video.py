"""H.264 MP4 writing through an FFmpeg pipe (no extra Python dependency)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


class VideoWriter:
    def __init__(
        self, path: str | Path, width: int, height: int, fps: int = 30, crf: int = 20
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found (it is installed in the Dev Container)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.frames = 0
        cmd = [
            *("ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"),
            *("-s", f"{width}x{height}", "-r", str(fps), "-i", "-"),
            *("-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"),
            *("-movflags", "+faststart", str(self.path)),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError(f"frame shape {frame.shape} != {self.size[::-1]}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {rc} for {self.path}")

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def probe(path: str | Path) -> dict:
    """ffprobe summary: duration, codec, resolution, frame rate."""
    entries = "format=duration:stream=codec_name,width,height,r_frame_rate,pix_fmt"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    return {
        "duration": float(data["format"]["duration"]),
        "codec": stream["codec_name"],
        "width": stream["width"],
        "height": stream["height"],
        "fps": stream["r_frame_rate"],
        "pix_fmt": stream.get("pix_fmt"),
    }
