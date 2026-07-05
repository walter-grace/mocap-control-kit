#!/usr/bin/env python3
"""Mocap joint data → pose+depth control videos for AI video models.

Takes a joint-position JSON ({fps, parents, pos[F][J][3]} — Z-up world space,
the MuJoCo convention) and renders an OpenPose-style colored stick figure
through a simple pinhole camera → mp4 ready for Wan-VACE, LTX-2.3
IC-LoRA Union-Control, or any pose-conditioned video model. Physics-simulated
stick figures in, photoreal characters out — no skinning, ever.

With --depth-combo the same 3D data also renders a TRUE geometric depth pass
(shaded ground-plane strips + depth-shaded bones) UNDER the colored skeleton:
one control video carrying both pose and depth.

Usage:
  mocap_control.py <motion.json.gz> <out.mp4> [--start S --dur D --w 832 --h 480]
                   [--dist 2.1 --height 0.35 --damp 1.0 --static --inplace]
                   [--depth-combo --fps 24 --frames N]
  mocap_control.py --multi "<m1.json.gz>:dx,dz,ry" "<m2.json.gz>:dx,dz,ry" <out.mp4> [--dur D]
    → N characters in ONE scene (per-character XZ offset + Y rotation), camera
      tracks the midpoint of all hips — fight scenes, group choreography.

All camera dials also accept env vars (SKEL_DIST, SKEL_HEIGHT, SKEL_DAMP,
SKEL_STATIC, SKEL_INPLACE, SKEL_DEPTH_COMBO, SKEL_FPS, SKEL_FRAMES) as
defaults; CLI flags win when given.

Requires: numpy, Pillow, ffmpeg on PATH.
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

PALETTE = [(255, 212, 0), (255, 140, 0), (46, 204, 113), (231, 76, 60), (52, 152, 219),
           (155, 89, 182), (0, 229, 255), (255, 0, 204), (26, 188, 156), (241, 196, 15)]


def _load(src, start, dur, dx=0.0, dz=0.0, ry=0.0):
    d = json.load(gzip.open(src, "rt"))
    fps_in = d["fps"]
    pos = np.array(d["pos"], dtype=np.float32)[:, :, [0, 2, 1]] * np.array([1, 1, -1])
    f0, f1 = int(start * fps_in), min(len(pos), int((start + dur) * fps_in))
    pos = pos[f0:f1]
    pos[:, :, 1] -= pos[:, :, 1].min()
    if ry:
        c, s_ = math.cos(ry), math.sin(ry)
        x, z = pos[:, :, 0].copy(), pos[:, :, 2].copy()
        pos[:, :, 0] = c * x + s_ * z
        pos[:, :, 2] = -s_ * x + c * z
    ctr0 = pos[0, 0]
    pos[:, :, 0] += dx - ctr0[0]
    pos[:, :, 2] += dz - ctr0[2]
    return pos, d["parents"], fps_in


# real-world half-widths (m) per SOMA-23 bone index (child joint of each bone)
BONE_W = {1: 0.16, 2: 0.17, 3: 0.18, 4: 0.07, 5: 0.07, 6: 0.11,
          7: 0.07, 8: 0.06, 9: 0.05, 10: 0.04, 11: 0.07, 12: 0.06, 13: 0.05, 14: 0.04,
          15: 0.09, 16: 0.065, 17: 0.05, 18: 0.045, 19: 0.09, 20: 0.065, 21: 0.05, 22: 0.045}
MANNEQUIN_TINTS = [(202, 196, 186), (168, 178, 194), (196, 176, 176), (176, 194, 176)]


def render_multi(specs: list, out: str, dur: float = 5.0, W: int = 832, H: int = 480,
                 volumetric: bool = False, orbit_deg: float = 0.0):
    from PIL import Image, ImageDraw
    chars = []
    for spec in specs:
        path, _, ofs = spec.partition(":")
        dx, dz, ry = ([float(x) for x in ofs.split(",")] + [0, 0, 0])[:3] if ofs else (0, 0, 0)
        chars.append(_load(path, 0.0, dur, dx, dz, ry))
    fig_h = 1.7
    dist = fig_h * 1.85 + 0.45 * (len(chars) - 1)
    base_az = math.atan2(0.72, 0.72)
    cam_r = dist * 0.72 * math.sqrt(2)
    fl = 1.15 * min(W, H)
    fps_out = 24
    n_out = int(dur * fps_out)
    tmp = tempfile.mkdtemp(prefix="mocapctl_")
    for i in range(n_out):
        t = i / fps_out
        hips = np.mean([c[0][min(len(c[0]) - 1, int(t * c[2]))][0] for c in chars], axis=0)
        look = hips.copy(); look[1] = fig_h * 0.55
        az = base_az + math.radians(orbit_deg) * (t / dur)
        cam = look + np.array([math.sin(az) * cam_r, fig_h * 0.4, math.cos(az) * cam_r])
        fwd = (look - cam); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        img = Image.new("RGB", (W, H), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        if volumetric:
            # painter's algorithm across ALL characters: far bones first
            bones = []
            for ci, (pos, parents, fps_in) in enumerate(chars):
                frame = pos[min(len(pos) - 1, int(t * fps_in))]
                for j, par in enumerate(parents):
                    if par < 0: continue
                    mid = (frame[j] + frame[par]) / 2
                    z = float((mid - cam) @ fwd)
                    bones.append((z, ci, frame[par], frame[j], j))
                # head as its own "bone"
                bones.append((float((frame[6] - cam) @ fwd), ci, frame[6], frame[6], 6))
            bones.sort(key=lambda b: -b[0])
            def prj(p):
                v = p - cam
                x, y, z = v @ right, v @ up, v @ fwd
                if z < 0.1: z = 0.1
                return (W / 2 + fl * x / z, H / 2 - fl * y / z), z
            for z, ci, pa, pb, j in bones:
                tint = MANNEQUIN_TINTS[ci % len(MANNEQUIN_TINTS)]
                # subtle depth shading so limbs separate visually
                shade = max(0.78, min(1.0, 3.2 / max(z, 0.5)))
                col = tuple(int(c * shade) for c in tint)
                (xa, ya), za = prj(pa)
                (xb, yb), zb = prj(pb)
                wpx = max(2, int(BONE_W.get(j, 0.06) * fl / max((za + zb) / 2, 0.5)))
                if j == 6:   # head sphere
                    dr.ellipse([xb - wpx, yb - wpx, xb + wpx, yb + wpx], fill=col)
                else:
                    dr.line([(xa, ya), (xb, yb)], fill=col, width=wpx * 2)
                    for (x, y) in ((xa, ya), (xb, yb)):
                        dr.ellipse([x - wpx, y - wpx, x + wpx, y + wpx], fill=col)
        else:
            for ci, (pos, parents, fps_in) in enumerate(chars):
                frame = pos[min(len(pos) - 1, int(t * fps_in))]
                def prj2(p):
                    v = p - cam
                    x, y, z = v @ right, v @ up, v @ fwd
                    if z < 0.1: z = 0.1
                    return (W / 2 + fl * x / z, H / 2 - fl * y / z)
                pts = [prj2(frame[j]) for j in range(len(frame))]
                shift = ci * 3
                for j, par in enumerate(parents):
                    if par < 0: continue
                    dr.line([pts[par], pts[j]], fill=PALETTE[(j + shift) % len(PALETTE)], width=max(3, int(H / 90)))
                for j in range(len(pts)):
                    r = max(3, int(H / 110))
                    dr.ellipse([pts[j][0] - r, pts[j][1] - r, pts[j][0] + r, pts[j][1] + r],
                               fill=PALETTE[(j + shift) % len(PALETTE)])
        img.save(os.path.join(tmp, f"f_{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps_out),
                    "-i", os.path.join(tmp, "f_%04d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out],
                   check=True, timeout=180)
    print(json.dumps({"ok": True, "out": out, "chars": len(chars), "frames": n_out}))


def render(src: str, out: str, start: float = 0.0, dur: float = 5.0, W: int = 832, H: int = 480,
           dist_mult: float = 2.1, cam_height: float = 0.35, damp: float = 1.0,
           static: bool = False, inplace: bool = False, depth_combo: bool = False,
           fps_out: int = 24, frames: int = 0):
    from PIL import Image, ImageDraw
    d = json.load(gzip.open(src, "rt"))
    fps_in = d["fps"]
    parents = d["parents"]
    pos = np.array(d["pos"], dtype=np.float32)          # [F, J, 3] Z-up
    pos = pos[:, :, [0, 2, 1]] * np.array([1, 1, -1])   # → Y-up (x, z, -y)
    f0 = int(start * fps_in)
    f1 = min(len(pos), int((start + dur) * fps_in))
    pos = pos[f0:f1]
    # ground the motion: min Y over clip = 0
    pos[:, :, 1] -= pos[:, :, 1].min()
    # --inplace: cancel hip XZ travel per-frame (treadmill mode) — for
    # compositing onto real footage whose own camera supplies the travel
    if inplace:
        hipxz = pos[:, 0:1, [0, 2]].copy()
        pos[:, :, 0] -= hipxz[:, :, 0]
        pos[:, :, 2] -= hipxz[:, :, 1]
    # TRACKING camera: follows the hips (joint 0) with a fixed 3/4 offset so the
    # figure stays centered and large while the motion travels through space.
    fig_h = float(pos[0, :, 1].max() - pos[0, :, 1].min()) or 1.7
    dist = fig_h * dist_mult
    offset = np.array([dist * 0.72, fig_h * cam_height, dist * 0.72])
    fl = 1.15 * min(W, H)   # focal length in px

    _smooth = {"look": None}

    def make_cam(hips):
        # lerp-damped follow (--damp, default 1.0 = hard lock; 0.08-0.15 =
        # cinematic lag) — kills hip-jolt camera shake on violent motions
        if static and _smooth["look"] is not None:
            hips = _smooth["look"]
            look = hips.copy(); look[1] = fig_h * 0.55
            cam = look + offset
            fwd = (look - cam); fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            return cam, right, up, fwd
        if _smooth["look"] is None or damp >= 1.0:
            _smooth["look"] = np.array(hips, dtype=float)
        else:
            _smooth["look"] = _smooth["look"] + (np.array(hips, dtype=float) - _smooth["look"]) * damp
        hips = _smooth["look"]
        look = hips.copy(); look[1] = fig_h * 0.55
        cam = look + offset
        fwd = (look - cam); fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return cam, right, up, fwd

    def project(p, camdat):
        cam, right, up, fwd = camdat
        v = p - cam
        x, y, z = v @ right, v @ up, v @ fwd
        if z < 0.1: z = 0.1
        return (W / 2 + fl * x / z, H / 2 - fl * y / z)

    n_out = frames or int(dur * fps_out)
    tmp = tempfile.mkdtemp(prefix="mocapctl_")
    n_src = len(pos)
    for i in range(n_out):
        t = i / fps_out
        fi = int(t * fps_in)
        if fi >= n_src:                       # pingpong loop past motion end
            cyc = fi % (2 * n_src - 2) if n_src > 1 else 0
            fi = cyc if cyc < n_src else (2 * n_src - 2 - cyc)
        frame = pos[fi]
        camdat = make_cam(frame[0])
        img = Image.new("RGB", (W, H), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        pts = [project(frame[j], camdat) for j in range(len(frame))]
        if depth_combo:
            # DEPTH LAYER first: true geometric depth from the same 3D data.
            # Ground plane: quad strips receding along camera forward, shaded 1/z.
            cam, right, up, fwd = camdat
            def zdepth(pw):
                v = np.asarray(pw, dtype=float) - cam
                return max(0.12, float(v @ fwd))
            znear, zfar = 1.0, dist * 4.5
            def shade(z):
                g = int(255 * max(0.0, min(1.0, (1.0/z - 1.0/zfar) / (1.0/znear - 1.0/zfar))))
                return (g, g, g)
            for k in range(28):
                z0w, z1w = 0.6 + k * 0.9, 0.6 + (k + 1) * 0.9
                # project 4 corners of the floor strip at y=0 in world
                c0 = cam + fwd * z0w; c1 = cam + fwd * z1w
                quad = []
                for base, sx in ((c0, -4.0), (c0, 4.0), (c1, 4.0), (c1, -4.0)):
                    wpt = np.array([base[0] + right[0]*sx, 0.0, base[2] + right[2]*sx])
                    quad.append(project(wpt, camdat))
                dr.polygon(quad, fill=shade(zdepth((c0 + c1) / 2)))
            # character: bones drawn as thick strokes shaded by joint depth
            for j, par in enumerate(parents):
                if par < 0: continue
                z = (zdepth(frame[j]) + zdepth(frame[par])) / 2
                g = shade(z)
                bg = (min(255, g[0] + 70),) * 3   # figure pops brighter than floor
                dr.line([pts[par], pts[j]], fill=bg, width=max(9, int(H / 32)))
            for j in range(len(pts)):
                z = zdepth(frame[j]); g = shade(z)
                bg = (min(255, g[0] + 70),) * 3
                r = max(6, int(H / 60))
                dr.ellipse([pts[j][0]-r, pts[j][1]-r, pts[j][0]+r, pts[j][1]+r], fill=bg)
        for j, par in enumerate(parents):
            if par < 0: continue
            c = PALETTE[j % len(PALETTE)]
            dr.line([pts[par], pts[j]], fill=c, width=max(3, int(H / 90)))
        for j in range(len(pts)):
            r = max(3, int(H / 110))
            dr.ellipse([pts[j][0] - r, pts[j][1] - r, pts[j][0] + r, pts[j][1] + r],
                       fill=PALETTE[j % len(PALETTE)])
        img.save(os.path.join(tmp, f"f_{i:04d}.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps_out),
                    "-i", os.path.join(tmp, "f_%04d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out],
                   check=True, timeout=180)
    print(json.dumps({"ok": True, "out": out, "frames": n_out, "src_fps": fps_in, "joints": len(parents)}))


def _envf(name, default):
    return float(os.environ.get(name, default))


def _envb(name):
    return os.environ.get(name, "0") == "1"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render mocap joint data as pose(+depth) control videos for AI video models.")
    ap.add_argument("input", nargs="*",
                    help="motion.json.gz out.mp4 — or with --multi: one or more "
                         "'<motion.json.gz>:dx,dz,ry' specs followed by out.mp4")
    ap.add_argument("--multi", action="store_true",
                    help="multi-character mode: each input is '<path>:dx,dz,ry'")
    ap.add_argument("--start", type=float, default=0.0, help="start time in source motion (s)")
    ap.add_argument("--dur", type=float, default=5.0, help="clip duration (s)")
    ap.add_argument("--w", type=int, default=832, help="output width px")
    ap.add_argument("--h", type=int, default=480, help="output height px")
    ap.add_argument("--dist", type=float, default=_envf("SKEL_DIST", 2.1),
                    help="camera distance as multiple of figure height "
                         "(1.25 = medium-full, 2.1 = wide) [env SKEL_DIST]")
    ap.add_argument("--height", type=float, default=_envf("SKEL_HEIGHT", 0.35),
                    help="camera height as multiple of figure height [env SKEL_HEIGHT]")
    ap.add_argument("--damp", type=float, default=_envf("SKEL_DAMP", 1.0),
                    help="follow-cam lerp factor: 1.0 = hard lock, 0.08-0.15 = "
                         "cinematic lag [env SKEL_DAMP]")
    ap.add_argument("--static", action="store_true", default=_envb("SKEL_STATIC"),
                    help="lock the camera to its first-frame position [env SKEL_STATIC=1]")
    ap.add_argument("--inplace", action="store_true", default=_envb("SKEL_INPLACE"),
                    help="treadmill mode: cancel hip XZ travel per-frame "
                         "(for compositing onto traveling plates) [env SKEL_INPLACE=1]")
    ap.add_argument("--depth-combo", action="store_true", default=_envb("SKEL_DEPTH_COMBO"),
                    help="render true geometric depth (ground strips + depth-shaded "
                         "bones) under the colored skeleton [env SKEL_DEPTH_COMBO=1]")
    ap.add_argument("--fps", type=int, default=int(_envf("SKEL_FPS", 24)),
                    help="output frame rate [env SKEL_FPS]")
    ap.add_argument("--frames", type=int, default=int(_envf("SKEL_FRAMES", 0)),
                    help="exact output frame count (pingpong-loops past motion end; "
                         "0 = dur*fps) [env SKEL_FRAMES]")
    ap.add_argument("--volumetric", action="store_true",
                    help="(--multi only) mannequin-style volumetric bones with painter's sort")
    ap.add_argument("--orbit", type=float, default=0.0,
                    help="(--multi only) camera orbit over the clip, degrees")
    args = ap.parse_args(argv)

    if args.multi:
        if len(args.input) < 2:
            ap.error("--multi needs at least one motion spec and an output mp4")
        render_multi(args.input[:-1], args.input[-1], dur=args.dur, W=args.w, H=args.h,
                     volumetric=args.volumetric, orbit_deg=args.orbit)
    else:
        if len(args.input) != 2:
            ap.error("expected: <motion.json.gz> <out.mp4>")
        render(args.input[0], args.input[1], start=args.start, dur=args.dur,
               W=args.w, H=args.h, dist_mult=args.dist, cam_height=args.height,
               damp=args.damp, static=args.static, inplace=args.inplace,
               depth_combo=args.depth_combo, fps_out=args.fps, frames=args.frames)


if __name__ == "__main__":
    main()
