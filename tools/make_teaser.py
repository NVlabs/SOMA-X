# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared README teaser compositor.

Assembles the SOMA Body and SOMA Hand demo renders into one GIF with a single
title/label specification (regular-weight 18 px titles, 12 px labels), so
typography and alignment stay consistent across both sections (see issue
#95). Rows are stacked vertically on a shared-width canvas; every panel is
scaled and center-cropped, so figures in both rows are center-aligned.

Example (after rendering with demo_soma_vis.py / demo_soma_hand_vis.py):

    python tools/make_teaser.py \\
        --body-videos out/teaser/body/soma_fixed_shape_skel.mp4,... \\
        --body-labels "SOMA native,MHR,SMPL-X,Anny,GM" \\
        --hand-videos out/teaser/hand/hand_right_soma_..._skel.mp4,... \\
        --hand-labels "SOMA native,MHR,MANO" \\
        --output out/teaser/soma-in-action.gif

Requires pillow (bundled with the imageio dependency stack) and ffmpeg for the
palette-optimized GIF encode.
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

TITLE_BAR_HEIGHT = 40
TITLE_FONT_SIZE = 18
TITLE_Y = 10
LABEL_FONT_SIZE = 12
LABEL_STRIP_HEIGHT = 24
REGULAR_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _load_frames(path: str, stride: int) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if i % stride == 0:
            frames.append(np.asarray(frame))
    reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def _panel(
    frame: np.ndarray,
    panel_w: int,
    panel_h: int,
    crop_frac: float = 1.0,
    recenter: bool = False,
) -> Image.Image:
    """Crop a render to `crop_frac` of its height, scale to the row, center it.

    With `recenter`, the crop window follows the per-frame bounding box of the
    rendered content (non-background pixels) so a subject that pivots in place
    (e.g. a hand with wrist rotation) stays centered in its panel.
    """
    img = Image.fromarray(frame)
    keep_h = round(img.height * crop_frac)
    scale = panel_h / keep_h
    keep_w = min(img.width, round(panel_w / scale))
    cy, cx = img.height / 2, img.width / 2
    if recenter:
        content = np.asarray(frame).min(axis=-1) < 245
        ys, xs = np.nonzero(content)
        if len(ys):
            cy = (ys.min() + ys.max()) / 2
            cx = (xs.min() + xs.max()) / 2
    top = int(min(max(cy - keep_h / 2, 0), img.height - keep_h))
    left = int(min(max(cx - keep_w / 2, 0), img.width - keep_w))
    img = img.crop((left, top, left + keep_w, top + keep_h))
    img = img.resize((round(keep_w * scale), panel_h), Image.LANCZOS)
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    panel.paste(img, ((panel_w - img.width) // 2, 0))
    return panel


def _row(
    sources: list[list[np.ndarray]],
    labels: list[str],
    t: int,
    canvas_w: int,
    row_h: int,
    label_font: ImageFont.FreeTypeFont,
    crop_frac: float = 1.0,
    recenter: bool = False,
) -> Image.Image:
    """Panels scaled to `row_h`, with a centered label strip underneath."""
    panel_w = canvas_w // len(sources)
    row = Image.new("RGB", (canvas_w, row_h + LABEL_STRIP_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(row)
    x0 = (canvas_w - panel_w * len(sources)) // 2
    for i, frames in enumerate(sources):
        panel = _panel(frames[t % len(frames)], panel_w, row_h, crop_frac, recenter)
        row.paste(panel, (x0 + i * panel_w, 0))
    for i, label in enumerate(labels):
        cx = x0 + i * panel_w + panel_w / 2
        w = draw.textlength(label, font=label_font)
        # Keep long labels inside the canvas instead of clipping at the edges.
        x = min(max(cx - w / 2, 2), canvas_w - w - 2)
        draw.text((x, row_h + 4), label, fill=(30, 30, 30), font=label_font)
    return row


def _title_bar(title: str, canvas_w: int, font: ImageFont.FreeTypeFont) -> Image.Image:
    bar = Image.new("RGB", (canvas_w, TITLE_BAR_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    w = draw.textlength(title, font=font)
    draw.text(((canvas_w - w) / 2, TITLE_Y), title, fill=(0, 0, 0), font=font)
    return bar


def main():
    parser = argparse.ArgumentParser(description="Compose the combined SOMA README teaser GIF")
    parser.add_argument("--body-videos", required=True, help="Comma-separated body demo videos")
    parser.add_argument("--body-labels", required=True, help="Comma-separated body panel labels")
    parser.add_argument("--hand-videos", required=True, help="Comma-separated hand demo videos")
    parser.add_argument("--hand-labels", required=True, help="Comma-separated hand panel labels")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--canvas-width", type=int, default=720)
    parser.add_argument("--row-height", type=int, default=200)
    parser.add_argument("--body-title", default="SOMA Body")
    parser.add_argument("--hand-title", default="SOMA Hand")
    parser.add_argument(
        "--hand-crop-frac",
        type=float,
        default=0.7,
        help="Fraction of the hand render height to keep (center crop) so the hands sit "
        "close to their title instead of floating in empty canvas.",
    )
    parser.add_argument(
        "--section-gap",
        type=int,
        default=28,
        help="Blank pixels between the body labels and the hand title.",
    )
    parser.add_argument("--fps", type=float, default=12.5, help="Output GIF frame rate")
    parser.add_argument(
        "--body-stride",
        type=int,
        default=2,
        help="Keep every Nth body frame (body demos render at 30 fps)",
    )
    parser.add_argument(
        "--hand-stride", type=int, default=2, help="Keep every Nth hand frame; shorter rows loop"
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args)

    body = [_load_frames(p, args.body_stride) for p in args.body_videos.split(",")]
    hand = [_load_frames(p, args.hand_stride) for p in args.hand_videos.split(",")]
    body_labels = [s.strip() for s in args.body_labels.split(",")]
    hand_labels = [s.strip() for s in args.hand_labels.split(",")]
    if len(body_labels) != len(body) or len(hand_labels) != len(hand):
        raise ValueError("Label count must match video count for each row.")

    # Regular weight, modest size: reviewers found bold 22 px titles too heavy.
    title_font = ImageFont.truetype(REGULAR_FONT, TITLE_FONT_SIZE)
    label_font = ImageFont.truetype(REGULAR_FONT, LABEL_FONT_SIZE)

    num_frames = max(len(f) for f in body)
    logger.info(f"Compositing {num_frames} frames at {args.fps} fps...")

    with tempfile.TemporaryDirectory(prefix="soma_teaser_") as tmp:
        for t in range(num_frames):
            sections = [
                _title_bar(args.body_title, args.canvas_width, title_font),
                _row(body, body_labels, t, args.canvas_width, args.row_height, label_font),
                Image.new("RGB", (args.canvas_width, args.section_gap), (255, 255, 255)),
                _title_bar(args.hand_title, args.canvas_width, title_font),
                _row(
                    hand,
                    hand_labels,
                    t,
                    args.canvas_width,
                    args.row_height,
                    label_font,
                    args.hand_crop_frac,
                    recenter=True,
                ),
            ]
            canvas = Image.new(
                "RGB",
                (args.canvas_width, sum(s.height for s in sections)),
                (255, 255, 255),
            )
            y = 0
            for section in sections:
                canvas.paste(section, (0, y))
                y += section.height
            canvas.save(f"{tmp}/{t:05d}.png")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Palette-optimized encode with the settings agreed in issue #95.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                f"{tmp}/%05d.png",
                "-filter_complex",
                "split[v0][v1];[v0]palettegen=stats_mode=diff[p];"
                "[v1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                "-loop",
                "0",
                str(out_path),
            ],
            check=True,
        )
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
