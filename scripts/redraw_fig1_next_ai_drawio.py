#!/usr/bin/env python3
"""Recolor the original Fig. 1 workflow and package it for draw.io.

This script intentionally preserves the original figure geometry, text, arrows,
and step numbering. It only remaps the colored elements to a cleaner
publication palette and writes:

* figs/Framework.png   -- paper preview
* figs/Framework.pdf   -- LaTeX figure used by main.tex
* figs/Framework.svg   -- SVG wrapper around the recolored image
* figs/Framework.drawio -- next-ai-drawio/draw.io source file
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


PALETTE = {
    "red": np.array([214, 39, 40], dtype=np.float32),       # step marks and difficulty text
    "coral": np.array([231, 111, 81], dtype=np.float32),    # complex/pink nodes
    "amber": np.array([242, 184, 75], dtype=np.float32),    # orange nodes/features
    "teal": np.array([42, 157, 143], dtype=np.float32),     # green/simple nodes
    "blue": np.array([76, 120, 168], dtype=np.float32),     # student trajectory nodes
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render_pdf_with_ghostscript(input_pdf: Path, output_png: Path, dpi: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gs",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pngalpha",
        f"-r{dpi}",
        "-dFirstPage=1",
        "-dLastPage=1",
        f"-sOutputFile={output_png}",
        str(input_pdf),
    ]
    subprocess.run(cmd, check=True)


def apply_luminance(base_rgb: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply target hue while retaining antialiasing and lightness variation."""
    luminance = (
        0.2126 * base_rgb[..., 0:1]
        + 0.7152 * base_rgb[..., 1:2]
        + 0.0722 * base_rgb[..., 2:3]
    )
    scale = np.clip(luminance / 180.0, 0.45, 1.25)
    return np.clip(target.reshape(1, 1, 3) * scale, 0, 255)


def recolor_image(input_png: Path, output_png: Path) -> Image.Image:
    im = Image.open(input_png).convert("RGBA")
    arr = np.asarray(im).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    chroma = maxc - minc

    visible = alpha > 20
    saturated = visible & (chroma > 18) & (maxc < 252)

    red_mask = visible & (r > 150) & (g < 105) & (b < 105)
    coral_mask = saturated & (r > g + 20) & (r > b + 10) & (g >= 80) & (b >= 80)
    amber_mask = saturated & (r > 160) & (g > 115) & (b < 150) & (r >= g + 8)
    teal_mask = saturated & (g >= r - 5) & (g > b + 8) & (r > 80) & (b > 70)
    blue_mask = saturated & (b > r + 12) & (b >= g) & (r < 145)

    out = rgb.copy()
    for mask, target in [
        (coral_mask, PALETTE["coral"]),
        (amber_mask, PALETTE["amber"]),
        (teal_mask, PALETTE["teal"]),
        (blue_mask, PALETTE["blue"]),
        (red_mask, PALETTE["red"]),
    ]:
        if np.any(mask):
            recolored = apply_luminance(rgb, target)
            out[mask] = 0.18 * rgb[mask] + 0.82 * recolored[mask]

    final = np.dstack([np.clip(out, 0, 255), alpha]).astype(np.uint8)
    result = Image.fromarray(final, mode="RGBA")
    result = crop_whitespace(result)
    result.save(output_png)
    return result


def crop_whitespace(image: Image.Image, padding: int = 30) -> Image.Image:
    arr = np.asarray(image.convert("RGBA"))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    non_white = (alpha > 20) & np.any(rgb < 246, axis=2)
    ys, xs = np.where(non_white)
    if len(xs) == 0 or len(ys) == 0:
        return image
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding, image.width - 1)
    bottom = min(int(ys.max()) + padding, image.height - 1)
    return image.crop((left, top, right + 1, bottom + 1))


def write_pdf(image: Image.Image, output_pdf: Path, dpi: int) -> None:
    rgb = Image.new("RGB", image.size, "white")
    rgb.paste(image.convert("RGB"), mask=image.getchannel("A"))
    rgb.save(output_pdf, "PDF", resolution=dpi)


def write_svg(image_png: Path, output_svg: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(image_png.read_bytes()).decode("ascii")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>
</svg>
'''
    output_svg.write_text(svg, encoding="utf-8")


def write_drawio(image_png: Path, output_drawio: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(image_png.read_bytes()).decode("ascii")
    # draw.io style fields are semicolon-delimited, so encode the semicolon in
    # the data URI header.
    data_uri = f"data:image/png%3Bbase64,{encoded}"
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "modified": "2026-06-01T00:00:00.000Z",
            "agent": "next-ai-draw-io-compatible",
            "version": "24.7.17",
            "type": "device",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "hgpra-framework-recolored", "name": "Framework"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "1",
            "dy": "1",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(width),
            "pageHeight": str(height),
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": "framework_recolored_image",
            "value": "",
            "style": f"shape=image;html=1;imageAspect=0;aspect=fixed;image={data_uri};",
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": "0", "y": "0", "width": str(width), "height": str(height), "as": "geometry"},
    )
    output_drawio.write_text(ET.tostring(mxfile, encoding="unicode"), encoding="utf-8")
    ET.parse(output_drawio)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pdf",
        default="/Users/qiluo/Downloads/Framework (1).pdf",
        help="Original Framework PDF to preserve and recolor.",
    )
    parser.add_argument("--out-dir", default=str(repo_root() / "figs"))
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    input_pdf = Path(args.input_pdf)
    out_dir = Path(args.out_dir)
    tmp_dir = repo_root() / "tmp" / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_png = tmp_dir / "framework_source_for_recolor.png"
    output_png = out_dir / "Framework.png"
    output_pdf = out_dir / "Framework.pdf"
    output_svg = out_dir / "Framework.svg"
    output_drawio = out_dir / "Framework.drawio"

    render_pdf_with_ghostscript(input_pdf, source_png, args.dpi)
    recolored = recolor_image(source_png, output_png)
    write_pdf(recolored, output_pdf, args.dpi)
    write_svg(output_png, output_svg, recolored.width, recolored.height)
    write_drawio(output_png, output_drawio, recolored.width, recolored.height)

    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_svg}")
    print(f"Wrote {output_drawio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
