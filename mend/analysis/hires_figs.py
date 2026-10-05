# SPDX-License-Identifier: Apache-2.0
"""Paper-ready high-resolution qualitative figures from gen_hires.py (or gen_compare.py) images, composed with PIL
at native pixels (no matplotlib resampling), written as PNG (lossless) and JPEG (quality --quality, 4:4:4) with a
sidecar JSON recording exactly which images were used.

Subcommands:

``grid``     rows = methods, columns = prompts (DiffusionOPSD Fig. 11 / Self-OPD Fig. 1 layout). Each method is
             ``LABEL=ROOT/METHOD`` (or ``ROOT/METHOD``; label = method dir name), so rows may mix roots, e.g.
             512 vs 1024 of the same LoRA. Each column is ``PID[:SEED][@x0/y0/x1/y1]``; the optional box (fractions
             of the image) makes that column a full-resolution crop instead of the whole image. ``--tile`` sets the
             cell size in pixels (images are resized with Lanczos only when their size differs from the tile).
             ``--transpose`` puts prompts on rows and methods on columns.
``zoom``     one image per method on a row with a full-resolution crop inset below each (detail comparison).
``gallery``  justified-rows collage of images with mixed aspect ratios (DiffusionOPSD Fig. 2 style): ``--items``
             ROOT/METHOD/PID_SEED entries, ``--width`` and ``--row_height`` in pixels.

Printed-size guide: paper text width 5.5 in; a 4400 px wide grid prints at 800 dpi, which keeps 1024 px tiles
sharp when zoomed in the PDF. JPEG q92 of a 6 x 5 grid of 1024 tiles is about 3-5 MB; use --max_width to cap.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from mend.paths import OUTPUT_ROOT  # noqa: E402

OUT_DIR = str(OUTPUT_ROOT / "hires/figs")
FONT_DIRS = ["/usr/share/fonts/urw-base35"]
FONTS = {"regular": "NimbusRoman-Regular.otf", "bold": "NimbusRoman-Bold.otf", "italic": "NimbusRoman-Italic.otf"}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        p = os.path.join(d, FONTS[kind])
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    import matplotlib

    name = {"regular": "DejaVuSerif.ttf", "bold": "DejaVuSerif-Bold.ttf", "italic": "DejaVuSerif-Italic.ttf"}[kind]
    return ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf", name), size)


def parse_method(spec: str) -> Tuple[str, str]:
    label, _, path = spec.partition("=")
    if not path:
        path, label = label, os.path.basename(label.rstrip("/"))
    return label.replace("\\n", "\n"), path


def parse_col(spec: str, default_seed: int) -> Dict[str, Any]:
    main, _, box = spec.partition("@")
    pid, _, seed = main.partition(":")
    return {"pid": pid, "seed": int(seed) if seed else default_seed,
            "box": tuple(float(x) for x in box.split("/")) if box else None}


def load_prompt_text(mdir: str, pid: str) -> str:
    man = os.path.join(mdir, "manifest.jsonl")
    if os.path.exists(man):
        with open(man) as f:
            for line in f:
                r = json.loads(line)
                if r["prompt_id"] == pid:
                    return r["prompt"]
    return pid


def cell(path: str, box, tw: int, th: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if box:
        w, h = im.size
        im = im.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
    if im.size != (tw, th):
        # cover-fit: keep aspect, center-crop to the tile
        s = max(tw / im.width, th / im.height)
        im = im.resize((max(tw, round(im.width * s)), max(th, round(im.height * s))), Image.LANCZOS)
        l, t = (im.width - tw) // 2, (im.height - th) // 2
        im = im.crop((l, t, l + tw, t + th))
    return im


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int, max_lines: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while draw.textlength(lines[-1] + " ...", font=fnt) > max_w and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += " ..."
    return lines


def save(canvas: Image.Image, stem: str, args, record: Dict[str, Any]) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    if args.max_width and canvas.width > args.max_width:
        s = args.max_width / canvas.width
        canvas = canvas.resize((args.max_width, round(canvas.height * s)), Image.LANCZOS)
    base = os.path.join(args.out_dir, stem)
    canvas.save(base + ".jpg", quality=args.quality, subsampling=0, optimize=True, dpi=(600, 600))
    if not args.no_png:
        canvas.save(base + ".png", optimize=False, compress_level=6, dpi=(600, 600))
    record.update({"size_px": list(canvas.size), "jpg_bytes": os.path.getsize(base + ".jpg"),
                   "png_bytes": os.path.getsize(base + ".png") if not args.no_png else None})
    with open(base + ".json", "w") as f:
        json.dump(record, f, indent=1)
    print(f"[hires_figs] {base}.jpg {canvas.size[0]}x{canvas.size[1]} "
          f"{record['jpg_bytes'] / 1e6:.2f} MB" + ("" if args.no_png else f", png {record['png_bytes'] / 1e6:.1f} MB"))


# ---------------------------------------------------------------------------------------------------------------
def cmd_grid(args) -> None:
    methods = [parse_method(s) for s in args.methods.split(",") if s.strip()]
    cols = [parse_col(s, args.seed) for s in args.cols.split(",") if s.strip()]
    tw, th = args.tile, args.tile
    gap, fs = args.gap, max(14, round(args.tile * args.font_frac))
    f_lab, f_txt = font("bold", fs), font("italic", round(fs * 0.85))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rows_spec, col_spec = (cols, methods) if args.transpose else (methods, cols)
    # header texts (top of columns) and row labels (left)
    ref_dir = methods[0][1]
    ptext = {c["pid"]: load_prompt_text(ref_dir, c["pid"]) for c in cols}
    if args.transpose:
        heads = [m[0] for m in methods]
        rlabels = [ptext[c["pid"]] for c in cols]
    else:
        heads = [ptext[c["pid"]] for c in cols]
        rlabels = [m[0] for m in methods]
    head_lines = [wrap(probe, h, f_lab if args.transpose else f_txt, tw - 8, args.prompt_lines) for h in heads]
    head_h = (max(len(x) for x in head_lines) * round(fs * 1.15) + 2 * gap) if not args.no_header else 0
    if args.transpose:
        lab_w = round(tw * 0.55)
        lab_lines = [wrap(probe, t, f_txt, lab_w - 12, 12) for t in rlabels]
    else:
        lab_w = max(max(probe.textlength(l, font=f_lab) for l in lbl.split("\n")) for lbl in rlabels) + 2 * gap
        lab_w = int(lab_w)
    W = lab_w + len(col_spec) * tw + (len(col_spec) - 1) * gap
    H = head_h + len(rows_spec) * th + (len(rows_spec) - 1) * gap
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    used = []
    for j, lines in enumerate(head_lines):
        if args.no_header:
            break
        x0 = lab_w + j * (tw + gap)
        for k, ln in enumerate(lines):
            fnt = f_lab if args.transpose else f_txt
            d.text((x0 + (tw - d.textlength(ln, font=fnt)) / 2, gap // 2 + k * round(fs * 1.15)), ln, font=fnt, fill="black")
    for i in range(len(rows_spec)):
        y0 = head_h + i * (th + gap)
        if args.transpose:
            lines = lab_lines[i]
            ty = y0 + (th - len(lines) * round(fs * 1.0)) / 2
            for k, ln in enumerate(lines):
                d.text((6, ty + k * round(fs * 1.0)), ln, font=f_txt, fill="black")
        else:
            lines = rlabels[i].split("\n")
            ty = y0 + (th - len(lines) * round(fs * 1.15)) / 2
            for k, ln in enumerate(lines):
                d.text((lab_w - gap - d.textlength(ln, font=f_lab), ty + k * round(fs * 1.15)), ln, font=f_lab,
                       fill="black")
        for j in range(len(col_spec)):
            (label, mdir), c = (col_spec[j], rows_spec[i]) if args.transpose else (rows_spec[i], col_spec[j])
            path = os.path.join(mdir, f"{c['pid']}_{c['seed']}.png")
            canvas.paste(cell(path, c["box"], tw, th), (lab_w + j * (tw + gap), y0))
            used.append({"label": label, "file": path, "box": c["box"]})
    save(canvas, args.name, args, {"cmd": "grid", "methods": methods, "cols": cols, "tile": args.tile,
                                   "prompts": ptext, "images": used})


def cmd_zoom(args) -> None:
    methods = [parse_method(s) for s in args.methods.split(",") if s.strip()]
    c = parse_col(args.col, args.seed)
    box = c["box"] or (0.375, 0.375, 0.625, 0.625)
    tw, gap, fs = args.tile, args.gap, max(14, round(args.tile * args.font_frac))
    f_lab = font("bold", fs)
    im0 = Image.open(os.path.join(methods[0][1], f"{c['pid']}_{c['seed']}.png"))
    th = round(tw * im0.height / im0.width)
    head = round(fs * 1.4)
    W = len(methods) * tw + (len(methods) - 1) * gap
    H = head + th + gap + tw
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    used = []
    for j, (label, mdir) in enumerate(methods):
        path = os.path.join(mdir, f"{c['pid']}_{c['seed']}.png")
        x0 = j * (tw + gap)
        d.text((x0 + (tw - d.textlength(label, font=f_lab)) / 2, 0), label, font=f_lab, fill="black")
        full = cell(path, None, tw, th)
        # outline the crop box on the overview
        dd = ImageDraw.Draw(full)
        lw = max(2, tw // 200)
        dd.rectangle((box[0] * tw, box[1] * th, box[2] * tw, box[3] * th), outline=(230, 40, 40), width=lw)
        canvas.paste(full, (x0, head))
        canvas.paste(cell(path, box, tw, tw), (x0, head + th + gap))
        used.append({"label": label, "file": path})
    save(canvas, args.name, args, {"cmd": "zoom", "col": c, "box": box, "images": used,
                                   "prompt": load_prompt_text(methods[0][1], c["pid"])})


def cmd_gallery(args) -> None:
    items = [s.strip() for s in args.items.split(",") if s.strip()]
    ims = [(p if p.endswith(".png") else p + ".png") for p in items]
    sizes = [Image.open(p).size for p in ims]
    W, rh, gap = args.width, args.row_height, args.gap
    rows, cur = [], []
    for p, (w, h) in zip(ims, sizes):
        cur.append((p, w / h))
        if sum(a for _, a in cur) * rh + gap * (len(cur) - 1) >= W:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    placed, y = [], 0
    for r in rows:
        avail = W - gap * (len(r) - 1)
        h = round(avail / sum(a for _, a in r)) if r is not rows[-1] or sum(a for _, a in r) * rh > avail else rh
        x = 0
        for k, (p, a) in enumerate(r):
            w = (W - x) if (k == len(r) - 1 and r is not rows[-1]) else round(a * h)
            placed.append((p, x, y, w, h))
            x += w + gap
        y += h + gap
    canvas = Image.new("RGB", (W, y - gap), "white")
    for p, x, yy, w, h in placed:
        canvas.paste(cell(p, None, w, h), (x, yy))
    save(canvas, args.name, args, {"cmd": "gallery", "images": [p for p, *_ in placed]})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["grid", "zoom", "gallery"]:
        s = sub.add_parser(name)
        s.add_argument("--name", required=True, help="output stem under --out_dir")
        s.add_argument("--out_dir", default=OUT_DIR)
        s.add_argument("--gap", type=int, default=12)
        s.add_argument("--quality", type=int, default=92)
        s.add_argument("--max_width", type=int, default=0)
        s.add_argument("--no_png", action="store_true")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--tile", type=int, default=768)
        s.add_argument("--font_frac", type=float, default=0.055, help="label font size / tile size")
        if name == "grid":
            s.add_argument("--methods", required=True)
            s.add_argument("--cols", required=True)
            s.add_argument("--transpose", action="store_true")
            s.add_argument("--prompt_lines", type=int, default=4)
            s.add_argument("--no_header", action="store_true")
        elif name == "zoom":
            s.add_argument("--methods", required=True)
            s.add_argument("--col", required=True, help="PID[:SEED][@x0/y0/x1/y1]")
        else:
            s.add_argument("--items", required=True)
            s.add_argument("--width", type=int, default=4400)
            s.add_argument("--row_height", type=int, default=900)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = parse_args(argv)
    {"grid": cmd_grid, "zoom": cmd_zoom, "gallery": cmd_gallery}[a.cmd](a)


if __name__ == "__main__":
    main()
