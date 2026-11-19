"""Render a vertical (9:16) rough-cut video from a script.

* Every scene is drawn with Pillow (no stock footage needed).
* A simple beat (kick + hi-hat + pad) is synthesised with NumPy, and every cut and
  text animation is timed to that beat.
* MoviePy joins the frames and the audio into an MP4 (and optionally a GIF preview).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 720, 1280
FPS = 24
SR = 44100
BG, FG, MUTED, ACCENT = (8, 8, 9), (245, 245, 245), (140, 140, 146), (60, 60, 66)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# ---------------- audio ----------------
def synth_beat(duration: float, bpm: int = 120, seed: int = 0) -> np.ndarray:
    """A short royalty-free beat, returned as a (samples, 2) float array in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    out = np.zeros(n)
    beat = 60 / bpm

    t = np.arange(int(0.35 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 9)
    th = np.arange(int(0.05 * SR)) / SR
    hat = rng.standard_normal(len(th)) * np.exp(-th * 90) * 0.25

    def place(sample: np.ndarray, at: float, gain: float = 1.0) -> None:
        i = int(at * SR)
        j = min(n, i + len(sample))
        if i < n:
            out[i:j] += sample[: j - i] * gain

    k = 0
    while k * beat < duration:
        place(kick, k * beat, 0.9)
        place(hat, k * beat + beat / 2, 1.0)
        k += 1

    tt = np.arange(n) / SR
    chord = sum(np.sin(2 * np.pi * f * tt) for f in (110.0, 164.81, 220.0, 261.63)) / 4
    env = np.clip(tt / 1.0, 0, 1) * np.clip((duration - tt) / 1.0, 0, 1)
    out += 0.18 * chord * env

    out /= max(1e-9, np.abs(out).max())
    out *= 0.85
    return np.stack([out, out], axis=1)


# ---------------- scenes ----------------
def _scene_image(text: str, index: int, total: int, label: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # soft spotlight
    for r in range(520, 0, -40):
        shade = int(8 + 22 * (1 - r / 520))
        d.ellipse([W / 2 - r, H * 0.42 - r, W / 2 + r, H * 0.42 + r], fill=(shade, shade, shade + 1))
    small = _font(26)
    d.text((48, 64), label.upper(), font=small, fill=MUTED)
    d.text((W - 48, 64), f"{index + 1:02d}/{total:02d}", font=small, fill=MUTED, anchor="ra")
    size = 74 if len(text) < 28 else 60
    big = _font(size)
    lines = textwrap.wrap(text, width=14 if size == 74 else 18)
    lh = int(size * 1.15)
    y0 = H * 0.42 - lh * len(lines) / 2
    for i, line in enumerate(lines):
        d.text((W / 2, y0 + i * lh), line, font=big, fill=FG, anchor="ma")
    d.text((W / 2, H - 110), "viral-trend-predictor", font=small, fill=ACCENT, anchor="ma")
    return img


def render_video(scenes: list[str], out_path: str | Path, label: str = "trend alert",
                 bpm: int = 120, beats_per_scene: int = 4, gif_path: str | Path | None = None) -> Path:
    """Write an MP4 where each scene lasts ``beats_per_scene`` beats."""
    from moviepy import AudioArrayClip, VideoClip  # imported here so the rest works without ffmpeg

    beat = 60 / bpm
    scene_len = beat * beats_per_scene
    duration = scene_len * len(scenes)
    frames = [np.asarray(_scene_image(s, i, len(scenes), label)) for i, s in enumerate(scenes)]
    bar_y = H - 160

    def frame_at(t: float) -> np.ndarray:
        i = min(int(t // scene_len), len(scenes) - 1)
        local = t - i * scene_len
        # text slides up and fades in over the first beat of each scene
        k = min(1.0, local / (beat * 0.9))
        ease = 1 - (1 - k) ** 3
        shift = int((1 - ease) * 60)
        base = frames[i]
        f = np.empty_like(base)
        f[:] = BG
        f[: H - shift] = base[shift:]
        f = (BG + (f.astype(np.float32) - BG) * (0.25 + 0.75 * ease)).astype(np.uint8)
        # flash on every beat + progress bar
        phase = (t % beat) / beat
        pulse = int(10 * max(0.0, 1 - phase * 4))
        if pulse:
            f = np.clip(f.astype(np.int16) + pulse, 0, 255).astype(np.uint8)
        f[bar_y:bar_y + 4, 48:W - 48] = ACCENT
        f[bar_y:bar_y + 4, 48:48 + int((W - 96) * t / duration)] = FG
        return f

    clip = VideoClip(frame_function=frame_at, duration=duration)
    clip = clip.with_audio(AudioArrayClip(synth_beat(duration, bpm), fps=SR))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(out_path), fps=FPS, codec="libx264", audio_codec="aac", logger=None)
    if gif_path:
        clip.without_audio().resized(width=270).write_gif(str(gif_path), fps=10, logger=None)
    return out_path
