#!/usr/bin/env python3
"""Synthesize a parametric walk/jog cycle in the motion format mocap_control.py
consumes: {fps, parents, pos[F][J][3]} json.gz, Z-up world space (MuJoCo
convention: x/y ground plane, z is height).

No mocap data needed — a plain sine-based gait on the 23-joint skeleton
(pelvis → spine ×3 → neck → head ×2, two 4-joint arms off the chest, two
4-joint legs off the pelvis). Good enough to smoke-test the renderer and to
drive a pose-conditioned video model with a generic walk.

Usage:
  make_sample_motion.py out.json.gz [--dur 4.0 --fps 30 --style walk|jog]
"""
import argparse
import gzip
import json
import math

# 23-joint skeleton: parent index per joint (-1 = root)
PARENTS = [-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21]
# joints: 0 pelvis, 1-3 spine, 4 neck, 5 head base, 6 head top,
#         7-10 left arm (shoulder/elbow/wrist/hand), 11-14 right arm,
#         15-18 left leg (hip/knee/ankle/toe), 19-22 right leg


def gait_frame(t, style="walk"):
    """One frame of joint positions [23][3] at time t (seconds), Z-up."""
    if style == "jog":
        f_cycle, stride, lift, bounce, arm_amp = 1.5, 0.38, 0.22, 0.055, 0.75
    else:
        f_cycle, stride, lift, bounce, arm_amp = 1.0, 0.30, 0.14, 0.03, 0.5
    theta = 2 * math.pi * f_cycle * t
    speed = 4 * stride * f_cycle          # so stance feet roughly don't slide
    px = speed * t                        # pelvis travels along +x
    pz = 0.95 + bounce * math.sin(2 * theta)
    py = 0.03 * math.sin(theta)           # lateral sway

    J = [[0.0, 0.0, 0.0] for _ in range(23)]
    J[0] = [px, py, pz]
    # spine → head, slight forward lean
    for k, (dz, dx) in enumerate([(0.12, 0.01), (0.25, 0.02), (0.38, 0.03),
                                  (0.51, 0.04), (0.59, 0.045), (0.73, 0.05)], start=1):
        J[k] = [px + dx, py * 0.5, pz + dz]

    def arm(base, side, phase):
        # side: +1 left / -1 right; swing opposite the same-side leg
        sw = arm_amp * math.sin(theta + phase)
        sx, sy, sz = px + 0.03, side * 0.22, pz + 0.44
        base[0][:] = [sx, sy, sz]                                   # shoulder
        base[1][:] = [sx + 0.15 * sw, sy * 1.05, sz - 0.28]         # elbow
        base[2][:] = [sx + 0.38 * sw, sy * 1.05, sz - 0.50]         # wrist
        base[3][:] = [sx + 0.46 * sw, sy * 1.05, sz - 0.57]         # hand

    def leg(base, side, phase):
        th = theta + phase
        hx, hy, hz = px, side * 0.10, pz - 0.03
        base[0][:] = [hx, hy, hz]                                   # hip
        ax = px + stride * math.sin(th)                             # ankle fore-aft
        swing = max(0.0, math.cos(th))                              # 1 at mid-swing
        az = 0.06 + lift * swing
        bend = 0.08 + 0.14 * swing                                  # knee forward
        base[1][:] = [0.55 * hx + 0.45 * ax + bend, hy, 0.5 * (hz + az) + 0.05]  # knee
        base[2][:] = [ax, hy, az]                                   # ankle
        base[3][:] = [ax + 0.16, hy, max(0.02, az - 0.04)]          # toe

    arm([J[7], J[8], J[9], J[10]], +1, math.pi)   # left arm w/ right leg
    arm([J[11], J[12], J[13], J[14]], -1, 0.0)    # right arm w/ left leg
    leg([J[15], J[16], J[17], J[18]], +1, 0.0)    # left leg
    leg([J[19], J[20], J[21], J[22]], -1, math.pi)
    return J


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic walk/jog motion file.")
    ap.add_argument("out", help="output .json.gz path")
    ap.add_argument("--dur", type=float, default=4.0, help="duration (s)")
    ap.add_argument("--fps", type=int, default=30, help="sample rate")
    ap.add_argument("--style", choices=["walk", "jog"], default="walk")
    args = ap.parse_args()

    n = int(args.dur * args.fps)
    pos = [gait_frame(i / args.fps, args.style) for i in range(n)]
    with gzip.open(args.out, "wt") as f:
        json.dump({"fps": args.fps, "parents": PARENTS, "pos": pos}, f)
    print(json.dumps({"ok": True, "out": args.out, "frames": n,
                      "joints": len(PARENTS), "style": args.style}))


if __name__ == "__main__":
    main()
