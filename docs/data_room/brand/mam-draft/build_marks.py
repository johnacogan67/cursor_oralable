#!/usr/bin/env python3
"""MAM monogram as filled outlines.

Each M has a valley that ends on the baseline. The A is a sharp peak.
The red crossbar is the same weight as the strokes and runs from the
outer edge of one leg to the outer edge of the other.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

BLACK = "#111111"
RED = "#C8102E"
GREEN = "#157A45"
BLUE = "#1A4E89"
WHITE = "#FFFFFF"


def unit(x, y):
    length = math.hypot(x, y)
    if length == 0:
        return 0.0, 0.0
    return x / length, y / length


def line_hit(origin, direction, other_origin, other_direction):
    cross = direction[0] * other_direction[1] - direction[1] * other_direction[0]
    if abs(cross) < 1e-9:
        return None
    delta = (other_origin[0] - origin[0], other_origin[1] - origin[1])
    t = (delta[0] * other_direction[1] - delta[1] * other_direction[0]) / cross
    return (origin[0] + t * direction[0], origin[1] + t * direction[1])


def stroke_polygon(points, width, miter_limit=6.0, pinch_below=None):
    """Open polyline to a filled outline. Butt caps, miter joins.

    pinch_below: if a miter tip falls below this y (y grows downward),
    pull that tip back to this y so a valley sits on the baseline.
    """
    half = width / 2.0
    segs = []
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = unit(x2 - x1, y2 - y1)
        segs.append({
            "a": (x1, y1),
            "b": (x2, y2),
            "d": (dx, dy),
            "n": (-dy, dx),
        })

    def side_point(seg, end, sign):
        px, py = seg["b"] if end == "b" else seg["a"]
        nx, ny = seg["n"]
        return (px + sign * nx * half, py + sign * ny * half)

    def join(prev, nxt, sign, joint):
        p0 = side_point(prev, "b", sign)
        p1 = side_point(nxt, "a", sign)
        hit = line_hit(p0, prev["d"], p1, nxt["d"])
        if hit is None:
            return [p0, p1]
        reach = math.hypot(hit[0] - joint[0], hit[1] - joint[1])
        if reach / half > miter_limit:
            return [p0, p1]
        if pinch_below is not None and hit[1] > pinch_below and joint[1] >= pinch_below - 0.5:
            hit = (joint[0], pinch_below)
        return [hit]

    left = [side_point(segs[0], "a", 1)]
    right = [side_point(segs[0], "a", -1)]
    for i, seg in enumerate(segs):
        if i:
            joint = seg["a"]
            left.extend(join(segs[i - 1], seg, 1, joint))
            right.extend(join(segs[i - 1], seg, -1, joint))
    left.append(side_point(segs[-1], "b", 1))
    right.append(side_point(segs[-1], "b", -1))
    return left + list(reversed(right))


def poly_path(polygon):
    parts = [f"M {polygon[0][0]:.2f} {polygon[0][1]:.2f}"]
    for x, y in polygon[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    parts.append("Z")
    return " ".join(parts)


def geometry():
    # Cap height 400. M is wide enough that the valley reads.
    yb = 500.0
    yc = 100.0
    ya = 72.0  # A overshoots the cap line
    stroke = 46.0
    m_width = 430.0
    a_span = 390.0
    gap = 78.0  # white between stroke edges is gap - stroke
    x0 = 80.0

    lm_l = x0
    lm_v = lm_l + m_width / 2
    lm_r = lm_l + m_width
    a_l = lm_r + gap
    a_p = a_l + a_span / 2
    a_r = a_l + a_span
    rm_l = a_r + gap
    rm_v = rm_l + m_width / 2
    rm_r = rm_l + m_width

    left_m = [(lm_l, yb), (lm_l, yc), (lm_v, yb), (lm_r, yc), (lm_r, yb)]
    a_pts = [(a_l, yb), (a_p, ya), (a_r, yb)]
    right_m = [(rm_l, yb), (rm_l, yc), (rm_v, yb), (rm_r, yc), (rm_r, yb)]

    # 58% of the way down the A. The bar is a trapezoid whose ends
    # follow the inner edges of the legs and overlap them slightly.
    bar_y = ya + 0.58 * (yb - ya)

    return {
        "left_m": left_m,
        "a": a_pts,
        "right_m": right_m,
        "bar_y": bar_y,
        "apex_x": a_p,
        "stroke": stroke,
        "baseline": yb,
        "bounds": (lm_l, ya, rm_r, yb),
    }


def bar_polygon(geom):
    """Trapezoid from inner leg to inner leg, overlapping each leg by a few units."""
    stroke = geom["stroke"]
    half = stroke / 2.0
    overlap = 4.0
    (a_l, yb), (a_p, ya), (a_r, _) = geom["a"]
    bar_y = geom["bar_y"]
    ux, uy = unit(a_p - a_l, ya - yb)
    inward_x = -uy  # horizontal part of the inward normal; positive
    y0 = bar_y - half
    y1 = bar_y + half

    def center(foot_x, y):
        t = (yb - y) / (yb - ya)
        return foot_x + t * (a_p - foot_x)

    def left_x(y):
        return center(a_l, y) + inward_x * half - overlap

    def right_x(y):
        return center(a_r, y) - inward_x * half + overlap

    return [(left_x(y0), y0), (right_x(y0), y0), (right_x(y1), y1), (left_x(y1), y1)]


def outlines(geom):
    stroke = geom["stroke"]
    base = geom["baseline"]
    left = stroke_polygon(geom["left_m"], stroke, pinch_below=base)
    right = stroke_polygon(geom["right_m"], stroke, pinch_below=base)
    # A peak should stay sharp. Feet are butt caps on the baseline.
    a_poly = stroke_polygon(geom["a"], stroke, miter_limit=12)
    bar = bar_polygon(geom)
    return left, a_poly, right, bar


def bounds_of(polygons, pad):
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def svg_mark(geom, mode, width_px):
    left, a_poly, right, bar = outlines(geom)
    min_x, min_y, max_x, max_y = bounds_of([left, a_poly, right, bar], pad=8)
    vw, vh = max_x - min_x, max_y - min_y
    height_px = max(1, round(width_px * vh / vw))
    apex = geom["apex_x"]

    def filled(polygon, color, clip=None):
        clip_attr = f' clip-path="url(#{clip})"' if clip else ""
        return f'  <path d="{poly_path(polygon)}" fill="{color}"{clip_attr}/>'

    if mode == "black":
        body = "\n".join([
            filled(left, BLACK),
            filled(a_poly, BLACK),
            filled(right, BLACK),
            filled(bar, RED),
        ])
        defs = ""
    else:
        defs = f"""  <defs>
    <clipPath id="clipL">
      <rect x="{min_x - 10:.2f}" y="{min_y - 10:.2f}" width="{apex - (min_x - 10):.2f}" height="{vh + 20:.2f}"/>
    </clipPath>
    <clipPath id="clipR">
      <rect x="{apex:.2f}" y="{min_y - 10:.2f}" width="{max_x - apex + 10:.2f}" height="{vh + 20:.2f}"/>
    </clipPath>
  </defs>
"""
        body = "\n".join([
            filled(left, GREEN),
            filled(a_poly, GREEN, "clipL"),
            filled(a_poly, BLUE, "clipR"),
            filled(right, BLUE),
            filled(bar, RED),
        ])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" viewBox="{min_x:.2f} {min_y:.2f} {vw:.2f} {vh:.2f}">
  <rect x="{min_x:.2f}" y="{min_y:.2f}" width="{vw:.2f}" height="{vh:.2f}" fill="{WHITE}"/>
{defs}{body}
</svg>
"""
    return svg, width_px, height_px, (min_x, min_y, vw, vh)


def page(body, width, height):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0; padding: 0; background: {WHITE};
    width: {width}px; height: {height}px; overflow: hidden;
  }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    geom = geometry()
    black, bw, bh, _ = svg_mark(geom, "black", 1500)
    color, cw, ch, _ = svg_mark(geom, "channels", 1500)
    small_w = 420
    small, sw, sh, _ = svg_mark(geom, "black", small_w)

    (OUT / "mam-mark-black.svg").write_text(black)
    (OUT / "mam-mark-channels.svg").write_text(color)

    pad = 96
    jobs = []

    def add(name, html, w, h):
        (OUT / name).write_text(html)
        png = name.replace("render-", "").replace(".html", ".png")
        jobs.append((name, png, w, h))

    add("render-mark-black.html", page(f'<div style="padding:{pad}px;line-height:0;">{black}</div>', bw + pad * 2, bh + pad * 2), bw + pad * 2, bh + pad * 2)
    add("render-mark-channels.html", page(f'<div style="padding:{pad}px;line-height:0;">{color}</div>', cw + pad * 2, ch + pad * 2), cw + pad * 2, ch + pad * 2)
    add("render-mark-small.html", page(f'<div style="padding:36px;line-height:0;">{small}</div>', sw + 72, sh + 72), sw + 72, sh + 72)

    # Company lockup: the mark, then DIAGNOSTICS set to the same width.
    lock_svg, lw, lh, box = svg_mark(geom, "black", 1100)
    text_size = 42
    text_gap = 36
    lock_w = lw + pad * 2
    lock_h = pad + lh + text_gap + text_size + pad
    lock_body = f"""
<div style="box-sizing:border-box;width:{lock_w}px;height:{lock_h}px;padding:{pad}px {pad}px 0 {pad}px;background:{WHITE};">
  <div style="line-height:0;">{lock_svg}</div>
  <div style="margin-top:{text_gap}px;width:{lw}px;text-align:center;font-family:'Avenir Next',Avenir,'Helvetica Neue',sans-serif;font-weight:500;font-size:{text_size}px;letter-spacing:0.46em;color:{BLACK};line-height:1;text-indent:0.46em;">DIAGNOSTICS</div>
</div>
"""
    add("render-diagnostics.html", page(lock_body, lock_w, lock_h), lock_w, lock_h)

    # Horizontal lockup: mark at the height of the two type lines, then the name.
    side_svg, side_w, side_h, _ = svg_mark(geom, "black", 520)
    side_canvas_w = 96 + side_w + 48 + 560 + 96
    side_canvas_h = 320
    side_body = f"""
<div style="box-sizing:border-box;width:{side_canvas_w}px;height:{side_canvas_h}px;padding:64px 96px;display:flex;align-items:center;gap:48px;background:{WHITE};">
  <div style="line-height:0;">{side_svg}</div>
  <div>
    <div style="font-family:'Avenir Next',Avenir,'Helvetica Neue',sans-serif;font-weight:600;font-size:92px;letter-spacing:0.06em;color:{BLACK};line-height:0.8;">MAM</div>
    <div style="margin-top:16px;font-family:'Avenir Next',Avenir,'Helvetica Neue',sans-serif;font-weight:500;font-size:18px;letter-spacing:0.34em;color:{BLACK};line-height:1;text-indent:0.34em;">DIAGNOSTICS</div>
  </div>
</div>
"""
    add("render-lockup.html", page(side_body, side_canvas_w, side_canvas_h), side_canvas_w, side_canvas_h)

    (OUT / "render-jobs.txt").write_text("\n".join(f"{a} {b} {c} {d}" for a, b, c, d in jobs) + "\n")
    print("jobs", len(jobs))
    for job in jobs:
        print(*job)


if __name__ == "__main__":
    main()
