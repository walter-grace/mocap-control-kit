# Launch thread draft (X/Twitter)

**1/**
Open-sourced the tool I use to turn physics mocap into control videos for AI video models.

One Python file. Joint positions in ({fps, parents, pos} json.gz), OpenPose-style skeleton video out, ready to drive Wan-VACE or LTX-2.3 Union-Control. Deps: numpy, Pillow, ffmpeg.

[LINK]

**2/**
The part I actually care about: --depth-combo.

The source data is 3D, so the script renders true geometric depth (shaded ground bands + depth-shaded bones) under the colored skeleton. One control video, two channels. Union-control models read pose and depth from the same frames.

**3/**
Side effect I didn't plan: the ground shading is quantized into luminance bands, and video models read the steps as geometry. The same control video produces stairs in one style, terraces in another, floor seams in a third.

You can sculpt the set by painting a gradient. Depth as a level editor.

**4/**
Also in the box: framing dials (--dist 1.25 medium-full, 2.1 wide), a lerp-damped follow cam, exact frame counts with pingpong looping, multi-character scenes with one shared camera, and --inplace treadmill mode for keying a generated character onto traveling plates.

Ships with a synthetic walk generator, so you can test it without owning a single mocap file. MIT.

[LINK]
