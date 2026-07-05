#!/usr/bin/env python3
"""Extract pose + depth control videos from your own footage.

The inverse companion to the renderer: instead of synthesizing control videos
from mocap data, this pulls them out of real video. Three outputs:

  depth.mp4  — monocular depth (Depth-Anything-V2-Small), grayscale, near=bright
  pose.mp4   — MediaPipe pose landmarks drawn as an OpenPose-style colored
               skeleton on black (same palette spirit as the renderer)
  combo.mp4  — the colored skeleton overlaid ON the depth video: one control
               video, two channels, matching the renderer's --depth-combo idea

Feed combo.mp4 to union/multi-condition control models (e.g. LTX-2.3 IC-LoRA
Union-Control) exactly like a --depth-combo render; pose.mp4 alone drives any
pose-conditioned model.

Plain wrappers over public models — no API keys, everything runs locally.
First run downloads the depth model (~100 MB) from the Hugging Face hub and
the MediaPipe pose landmarker (~6 MB) from Google's model storage.

Usage:
  python3 extract_controls.py --video in.mp4 [--outdir out/]
                              [--max-frames 0] [--scale 1.0] [--fps 0]
                              [--skip-depth] [--skip-pose]

Requires: pip install -r requirements.txt  (torch, transformers, mediapipe,
opencv-python) and ffmpeg on PATH.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import cv2
import numpy as np

# same palette family as the renderer, one color per limb/joint (BGR for cv2)
PALETTE = [(0, 212, 255), (0, 140, 255), (113, 204, 46), (60, 76, 231), (219, 152, 52),
           (182, 89, 155), (255, 229, 0), (204, 0, 255), (156, 188, 26), (15, 196, 241)]

# BlazePose 33-landmark topology (MediaPipe POSE_CONNECTIONS)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


def read_frames(path, max_frames=0, scale=1.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"error: cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        sys.exit(f"error: no frames decoded from {path}")
    return frames, fps


def encode(frames, fps, out):
    tmp = tempfile.mkdtemp(prefix="extractctl_")
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"f_{i:05d}.png"), f)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:g}",
                    "-i", os.path.join(tmp, "f_%05d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", out], check=True)
    for name in os.listdir(tmp):
        os.unlink(os.path.join(tmp, name))
    os.rmdir(tmp)


def depth_pass(frames):
    """Depth-Anything-V2-Small via transformers pipeline → grayscale BGR frames."""
    from PIL import Image
    from transformers import pipeline
    pipe = pipeline("depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf")
    out = []
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = pipe(Image.fromarray(rgb))["depth"]           # PIL L, near=bright
        d = np.asarray(pred.resize((frame.shape[1], frame.shape[0])))
        out.append(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
        if (i + 1) % 10 == 0:
            print(f"  depth {i + 1}/{len(frames)}", file=sys.stderr)
    return out


def draw_skeleton(canvas, pts):
    """pts: [(x, y, visibility), ...] in pixels; draws colored bones + joints."""
    h = canvas.shape[0]
    lw = max(3, h // 90)
    for k, (a, b) in enumerate(POSE_CONNECTIONS):
        if a >= len(pts) or b >= len(pts):
            continue
        if pts[a][2] < 0.5 or pts[b][2] < 0.5:
            continue
        cv2.line(canvas, pts[a][:2], pts[b][:2], PALETTE[k % len(PALETTE)],
                 lw, cv2.LINE_AA)
    r = max(3, h // 110)
    for j, p in enumerate(pts):
        if p[2] < 0.5:
            continue
        cv2.circle(canvas, p[:2], r, PALETTE[j % len(PALETTE)], -1, cv2.LINE_AA)


def _pose_pass_legacy(mp, frames):
    """mediapipe <= 0.10.2x: classic Solutions API."""
    hits, out = 0, []
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        for frame in frames:
            h, w = frame.shape[:2]
            canvas = np.zeros_like(frame)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                hits += 1
                pts = [(int(p.x * w), int(p.y * h), p.visibility)
                       for p in res.pose_landmarks.landmark]
                draw_skeleton(canvas, pts)
            out.append(canvas)
    return out, hits


def _pose_pass_tasks(mp, frames, fps):
    """mediapipe >= 0.10.3x: Tasks API (Solutions removed). Auto-downloads the
    lite pose-landmarker model (~6 MB) to ~/.cache/mocap-control-kit/."""
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    cache = os.path.join(os.path.expanduser("~"), ".cache", "mocap-control-kit")
    os.makedirs(cache, exist_ok=True)
    model_path = os.path.join(cache, "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        print(f"downloading pose landmarker model → {model_path}", file=sys.stderr)
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    hits, out = 0, []
    with vision.PoseLandmarker.create_from_options(opts) as lmk:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            canvas = np.zeros_like(frame)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(mp_img, int(i * 1000.0 / fps))
            if res.pose_landmarks:
                hits += 1
                for person in res.pose_landmarks:
                    pts = [(int(p.x * w), int(p.y * h),
                            1.0 if p.visibility is None else p.visibility)
                           for p in person]
                    draw_skeleton(canvas, pts)
            out.append(canvas)
    return out, hits


def pose_pass(frames, fps):
    """MediaPipe pose landmarks → OpenPose-style colored skeleton on black.

    Returns (frames, hit_count). MediaPipe is trained on real humans; it will
    not fire on stick figures or abstract footage — hit_count says how many
    frames actually produced a skeleton.
    """
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        return _pose_pass_legacy(mp, frames)
    return _pose_pass_tasks(mp, frames, fps)


def main():
    ap = argparse.ArgumentParser(
        description="Extract depth/pose/combo control videos from real footage.")
    ap.add_argument("--video", required=True, help="input video file")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frame count (0 = all)")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor before inference")
    ap.add_argument("--fps", type=float, default=0.0, help="override output fps (0 = source fps)")
    ap.add_argument("--skip-depth", action="store_true", help="skip the depth pass")
    ap.add_argument("--skip-pose", action="store_true", help="skip the pose pass")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    frames, src_fps = read_frames(args.video, args.max_frames, args.scale)
    fps = args.fps or src_fps
    report = {"ok": True, "frames": len(frames), "fps": fps}

    depth = pose = None
    if not args.skip_depth:
        print(f"depth pass: {len(frames)} frames …", file=sys.stderr)
        depth = depth_pass(frames)
        p = os.path.join(args.outdir, "depth.mp4")
        encode(depth, fps, p)
        report["depth"] = p
    if not args.skip_pose:
        print(f"pose pass: {len(frames)} frames …", file=sys.stderr)
        pose, hits = pose_pass(frames, fps)
        p = os.path.join(args.outdir, "pose.mp4")
        encode(pose, fps, p)
        report["pose"] = p
        report["pose_frames_detected"] = hits
        if hits == 0:
            print("warning: MediaPipe found no person in any frame "
                  "(it needs real human footage, not stick figures)", file=sys.stderr)
    if depth is not None and pose is not None:
        combo = []
        for d, s in zip(depth, pose):
            c = d.copy()
            mask = s.any(axis=2)
            c[mask] = s[mask]
            combo.append(c)
        p = os.path.join(args.outdir, "combo.mp4")
        encode(combo, fps, p)
        report["combo"] = p

    print(json.dumps(report))


if __name__ == "__main__":
    main()
