"""Render the site's animated GIFs from the exported CIFAR-10 outputs (site/data/cifar10.json).

    python scripts/make_gifs.py        # writes site/gifs/{sets,worst,resplit}.gif + a .png poster for each

Uses the same numbers as the live demos: logits stored as int16/100, temperature from D_design, the naive (LAC)
score 1 - p_y, and q = k-th smallest calibration score with k = ceil((n+1)(1-alpha)).
Frames are drawn in the Catppuccin Mocha palette, like a framed screen recording.
"""
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "gifs"
C = dict(base="#1e1e2e", mantle="#181825", surface0="#313244", surface1="#45475a", overlay0="#6c7086", text="#cdd6f4",
         sub="#a6adc8", green="#a6e3a1", red="#f38ba8", peach="#fab387", blue="#89b4fa", mauve="#cba6f7", crust="#11111b")
W, H = 760, 340
SHORT = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def font(size, mono=False):
    for p in (["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"] if mono else
              ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


F_T, F_S, F_M, F_MS = font(20), font(14), font(14, True), font(12, True)


def decode(block, K):
    a = np.frombuffer(base64.b64decode(block["logits"]), dtype="<i2").astype(np.float32) / 100
    return a.reshape(block["n"], K), np.array(block["labels"])


def softmax(z, T):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def qhat(scores, alpha):
    n = len(scores)
    k = math.ceil(round((n + 1) * (1 - alpha), 9))
    return np.inf if k > n else np.sort(scores)[k - 1]


def aps(p, seed):
    """Randomised APS (as src/scores.py): mass of strictly likelier labels + u * p_y, one u ~ U(0,1) per image."""
    order = np.argsort(-p, 1)
    ps = np.take_along_axis(p, order, 1)
    u = np.random.default_rng(seed).random((len(p), 1))
    cum = np.cumsum(ps, 1) - ps + u * ps
    out = np.empty_like(p)
    np.put_along_axis(out, order, cum, 1)
    return out


def canvas(title, sub):
    im = Image.new("RGB", (W, H), C["base"])
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill=C["mantle"])
    for i, col in enumerate((C["red"], C["peach"], C["green"])):  # window dots
        d.ellipse([14 + i * 18, 12, 24 + i * 18, 22], fill=col)
    d.text((80, 9), title, font=F_S, fill=C["sub"])
    d.text((W - 16, 9), sub, font=F_MS, fill=C["overlay0"], anchor="ra")
    return im, d


def save(frames, name, ms):
    OUT.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(OUT / f"{name}.gif", save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    frames[len(frames) // 3].save(OUT / f"{name}.png", optimize=True)
    print(f"{name}.gif  {len(frames)} frames  {(OUT / f'{name}.gif').stat().st_size // 1024} KB")


def gif_sets(d, p_cal, y_cal, p_test, y_test):
    """Eight hard test pictures; their prediction sets grow and shrink as alpha sweeps 0.30 -> 0.02 -> 0.30."""
    s_cal = aps(p_cal, 1)[np.arange(len(y_cal)), y_cal]
    s_test = aps(p_test, 2)
    hard = [g for g in d["gallery"] if p_test[g["i"]].max() < 0.85][:8]
    imgs = [Image.open(io.BytesIO(base64.b64decode(g["img"].split(",", 1)[1]))).convert("RGB").resize((76, 76), Image.NEAREST) for g in hard]
    alphas = list(np.round(np.linspace(0.30, 0.02, 22), 3))
    alphas = alphas + alphas[::-1][1:-1]
    frames = []
    for a in alphas:
        q = qhat(s_cal, a)
        sets = s_test <= q
        cov, size = sets[np.arange(len(y_test)), y_test].mean(), sets.sum(1).mean()
        im, dr = canvas("prediction sets · CIFAR-10 · live from the exported model outputs", "APS score, T fitted on D_design")
        dr.text((24, 50), f"α = {a:.2f}", font=F_T, fill=C["text"])
        dr.text((130, 55), f"target {round((1 - a) * 100)}%   coverage {cov * 100:.1f}%   mean size {size:.2f}", font=F_M, fill=C["sub"])
        bar = int((1 - a) / 1 * (W - 48))  # coverage target bar
        dr.rounded_rectangle([24, 84, W - 24, 88], 2, fill=C["surface0"])
        dr.rounded_rectangle([24, 84, 24 + bar, 88], 2, fill=C["mauve"])
        for j, (g, img) in enumerate(zip(hard, imgs)):
            x, y = 24 + j * 91, 104
            i, yt = g["i"], y_test[g["i"]]
            ok = bool(sets[i, yt])
            dr.rounded_rectangle([x - 3, y - 3, x + 79, y + 79], 6, outline=C["green"] if ok else C["red"], width=2)
            im.paste(img, (x, y))
            labels = sorted(np.flatnonzero(sets[i]), key=lambda k: -p_test[i, k])
            if not labels:
                dr.rounded_rectangle([x, y + 88, x + 50, y + 106], 4, fill=C["red"])
                dr.text((x + 6, y + 91), "empty", font=F_MS, fill=C["crust"])
            for r, k in enumerate(labels[:4]):
                yy = y + 88 + r * 22
                fill = C["text"] if k == yt else C["surface1"]
                dr.rounded_rectangle([x, yy, x + 62, yy + 18], 4, fill=fill)
                dr.text((x + 5, yy + 2), SHORT[k], font=F_MS, fill=C["crust"] if k == yt else C["text"])
            if len(labels) > 4:
                dr.text((x + 66, y + 88 + 3 * 22 + 2), f"+{len(labels) - 4}", font=F_MS, fill=C["sub"])
        dr.text((24, H - 26), "white chip = true label   green frame = covered   red frame = missed", font=F_MS, fill=C["overlay0"])
        frames.append(im)
    save(frames, "sets", 260)


def gif_worst(p_cal, y_cal, p_test, y_test):
    """Redraw the calibration set again and again; per-class coverage wobbles and some class always falls behind."""
    P, Y = np.concatenate([p_cal, p_test]), np.concatenate([y_cal, y_test])
    rng, n_cal, alpha, frames = np.random.default_rng(7), 500, 0.1, []
    worst_hist = []
    for t in range(36):
        idx = rng.permutation(len(Y))
        c, e = idx[:n_cal], idx[n_cal:]
        q = qhat(1 - P[c, Y[c]], alpha)
        hit = (1 - P[e, Y[e]]) <= q
        cc = np.array([hit[Y[e] == k].mean() for k in range(10)])
        w = int(cc.argmin())
        worst_hist.append(cc[w])
        im, dr = canvas("who gets left out? · per-class coverage over random calibration draws", f"n_cal = {n_cal}, α = 0.1")
        dr.text((24, 48), f"draw #{t + 1:02d}", font=F_T, fill=C["text"])
        dr.text((140, 53), f"average {hit.mean() * 100:.1f}%   worst class: {SHORT[w]} {cc[w] * 100:.1f}%", font=F_M, fill=C["sub"])
        x0, y0, y1, bw = 60, 90, 290, 40
        lo, hi = 0.7, 1.0
        Yp = lambda v: y1 - (min(max(v, lo), hi) - lo) / (hi - lo) * (y1 - y0)
        for v in (0.7, 0.8, 0.9, 1.0):
            dr.line([x0 - 6, Yp(v), x0 + 10 * 54, Yp(v)], fill=C["surface0"])
            dr.text((x0 - 10, Yp(v) - 7), f"{int(v * 100)}%", font=F_MS, fill=C["overlay0"], anchor="ra")
        for k in range(10):
            x = x0 + k * 54
            col = C["red"] if k == w else (C["blue"] if cc[k] >= 0.9 else C["peach"])
            dr.rounded_rectangle([x, Yp(cc[k]), x + bw, y1], 3, fill=col)
            dr.text((x + bw / 2, y1 + 8), SHORT[k], font=F_MS, fill=C["text"] if k == w else C["sub"], anchor="ma")
        for xx in range(x0 - 6, x0 + 540, 12):  # dashed target
            dr.line([xx, Yp(0.9), xx + 6, Yp(0.9)], fill=C["green"], width=2)
        sx, sy = 622, 110  # sparkline of the worst class over draws
        dr.text((sx, sy - 18), "worst class", font=F_MS, fill=C["overlay0"])
        dr.rounded_rectangle([sx - 6, sy - 4, W - 14, sy + 104], 6, outline=C["surface0"])
        ty = sy + 100 - (0.9 - 0.7) / 0.3 * 100
        dr.line([sx, ty, W - 20, ty], fill=C["green"], width=1)
        pts = [(sx + i * 3.4, sy + 100 - (min(max(v, 0.7), 1.0) - 0.7) / 0.3 * 100) for i, v in enumerate(worst_hist)]
        if len(pts) > 1:
            dr.line(pts, fill=C["red"], width=2)
        frames.append(im)
    save(frames, "worst", 420)


def gif_resplit(p_cal, y_cal, p_test, y_test):
    """Coverage of 160 independent re-splits piles up into the exact Beta-binomial shape."""
    P, Y = np.concatenate([p_cal, p_test]), np.concatenate([y_cal, y_test])
    S = 1 - P[np.arange(len(Y)), Y]
    rng, n, alpha = np.random.default_rng(3), 200, 0.1
    covs = []
    for _ in range(160):
        idx = rng.permutation(len(Y))
        covs.append(float((S[idx[n:]] <= qhat(S[idx[:n]], alpha)).mean()))
    k = math.ceil((n + 1) * (1 - alpha))
    a_, b_ = k, n + 1 - k
    xs = np.linspace(0.8, 0.999, 200)
    lpdf = (a_ - 1) * np.log(xs) + (b_ - 1) * np.log(1 - xs) + math.lgamma(a_ + b_) - math.lgamma(a_) - math.lgamma(b_)
    pdf = np.exp(lpdf)
    bins = np.linspace(0.8, 1.0, 26)
    frames = []
    for t in list(range(4, 161, 4)) + [160] * 6:
        im, dr = canvas("is the 90% promise real? · coverage over re-splits vs the exact Beta law", f"n_cal = {n}")
        c = np.array(covs[:t])
        dr.text((24, 48), f"{t:3d} re-splits", font=F_T, fill=C["text"])
        dr.text((190, 53), f"mean {c.mean() * 100:.2f}%   (promise: 90% on average over draws)", font=F_M, fill=C["sub"])
        x0, x1, y0, y1 = 50, W - 30, 90, 290
        X = lambda v: x0 + (v - 0.8) / 0.2 * (x1 - x0)
        h, _ = np.histogram(c, bins)
        dens = h / max(t, 1) / (bins[1] - bins[0])
        top = max(pdf.max(), 1) * 1.15
        Yp = lambda v: y1 - v / top * (y1 - y0)
        for i, v in enumerate(dens):
            if v > 0:
                dr.rectangle([X(bins[i]) + 1, Yp(v), X(bins[i + 1]) - 1, y1], fill=C["blue"])
        dr.line([(X(x), Yp(v)) for x, v in zip(xs, pdf)], fill=C["peach"], width=3)
        dr.line([x0, y1, x1, y1], fill=C["overlay0"])
        for v in (0.8, 0.85, 0.9, 0.95, 1.0):
            dr.text((X(v), y1 + 8), f"{int(round(v * 100))}%", font=F_MS, fill=C["overlay0"], anchor="ma")
        dr.line([X(0.9), y0, X(0.9), y1], fill=C["green"], width=1)
        dr.text((x1, y0 - 4), "— exact Beta law   ■ observed", font=F_MS, fill=C["sub"], anchor="ra")
        frames.append(im)
    save(frames, "resplit", 140)


def main():
    d = json.loads((ROOT / "site" / "data" / "cifar10.json").read_text())
    K, T = d["K"], d["tuned"]["T"]
    zc, yc = decode(d["cal"], K)
    zt, yt = decode(d["test"], K)
    pc, pt = softmax(zc, T), softmax(zt, T)
    gif_sets(d, pc, yc, pt, yt)
    gif_worst(pc, yc, pt, yt)
    gif_resplit(pc, yc, pt, yt)


if __name__ == "__main__":
    main()
