# mocap-control-kit

Turn mocap/joint data into pose+depth control videos for AI video models (Wan-VACE, LTX-2.3, and friends).

One Python file, no rendering stack. Feed it joint positions from any source (physics sim, retargeted mocap, procedural gait) as `{fps, parents, pos[F][J][3]}` json.gz, and it renders an OpenPose-style colored skeleton through a pinhole camera into an mp4 you hand to a pose-conditioned video model. The model does the skinning, lighting, and wardrobe. You keep the motion.

Requires `numpy`, `Pillow`, and `ffmpeg` on PATH. Nothing else.

## Pose and depth in one control video

Pose skeletons tell a video model where the limbs are but nothing about the world. `--depth-combo` fixes that with zero extra passes: since the source data is 3D, the script renders true geometric depth (a shaded ground plane receding in quantized bands, plus depth-shaded thick bones) *underneath* the colored skeleton. One video, two control channels. Models that accept union/multi-condition control (LTX-2.3 IC-LoRA Union-Control is the one we tested) read both at once: the skeleton drives the body, the grayscale depth drives camera perspective and world layout.

The side effect worth knowing about: because the ground shading is quantized into bands, models interpret the luminance steps as geometry. Depth bands render as stairs, terraces, or floor seams depending on the style prompt. You can sculpt world structure by editing a grayscale gradient. Treat the depth layer as a level editor.

## Quickstart

```bash
git clone https://github.com/walter-grace/mocap-control-kit
cd mocap-control-kit

# 1. synthesize a walk cycle (no mocap files needed)
python3 examples/make_sample_motion.py walk.json.gz --dur 4 --style walk

# 2. pose-only control video
python3 mocap_control.py walk.json.gz walk_pose.mp4 --dur 4 --dist 1.6

# 3. pose + depth in one control video
python3 mocap_control.py walk.json.gz walk_combo.mp4 --dur 4 --dist 1.6 --depth-combo
```

Then send `walk_combo.mp4` to your video model as the control/reference video with a prompt describing the character and world.

## Motion format

A gzipped JSON object:

```jsonc
{
  "fps": 30,                  // sample rate of pos
  "parents": [-1, 0, 1, ...], // parent joint index per joint, -1 = root
  "pos": [                    // [frames][joints][3] world-space positions, meters
    [[x, y, z], ...],         // Z-up (MuJoCo convention): x/y ground, z height
    ...
  ]
}
```

Any skeleton topology works; the renderer just draws parent→child bones. The bundled sample generator and the `BONE_W` table use a 23-joint layout (pelvis, 3 spine, neck, 2 head, 4 per arm, 4 per leg) with parents `[-1,0,1,2,3,4,5,3,7,8,9,3,11,12,13,0,15,16,17,0,19,20,21]`. If your data is Y-up, swap axes before export (`pos_zup = pos_yup[:, :, [0, 2, 1]] * [1, -1, 1]`).

`examples/make_sample_motion.py` is also the reference exporter: ~90 lines that produce a valid file from pure math. Port its output block to dump motions from your own pipeline.

## Camera dials

Framing is the generation's framing doctrine: video models compose the shot the control video composes. Set the shot here, not in the prompt.

| Flag | Env var | Default | What it does |
|---|---|---|---|
| `--dist` | `SKEL_DIST` | 2.1 | Camera distance as a multiple of figure height. 1.25 = medium-full (character work, faces), 2.1 = wide (choreography, travel). |
| `--height` | `SKEL_HEIGHT` | 0.35 | Camera height as a multiple of figure height. Lower = heroic low angle. |
| `--damp` | `SKEL_DAMP` | 1.0 | Follow-cam lerp. 1.0 hard-locks to the hips; 0.08–0.15 gives cinematic lag and kills hip-jolt shake on violent motions. |
| `--static` | `SKEL_STATIC` | off | Lock the camera at its first-frame position; the figure moves through frame. |
| `--inplace` | `SKEL_INPLACE` | off | Treadmill mode: cancel hip XZ travel per frame. See recipe below. |
| `--depth-combo` | `SKEL_DEPTH_COMBO` | off | Render geometric depth (ground bands + depth-shaded bones) under the skeleton. |
| `--fps` | `SKEL_FPS` | 24 | Output frame rate. |
| `--frames` | `SKEL_FRAMES` | 0 | Exact output frame count. Past the end of the motion the renderer pingpong-loops, so you can hit model-required frame counts from any clip length. |

Env vars are fallback defaults (handy in pipelines); CLI flags win.

## Driving LTX-2.3 Union-Control and Wan-VACE

- **LTX-2.3 IC-LoRA Union-Control**: pass the combo video as the control input. Keep output resolution a multiple of 64 per side (the IC-LoRA requires it); the default 832×480 render works as-is. Use `--frames` to match the model's expected frame count.
- **Wan-VACE**: pass the pose-only render as `src_video`. VACE reads the OpenPose color coding directly. For multi-character scenes use `--multi` (below) so all characters share one camera.
- Anything pose-conditioned: the skeleton is plain OpenPose-style colored sticks on black, which most pose ControlNets and pose LoRAs accept without preprocessing.

Prompt for the world; the control video already owns the body and the camera.

## Treadmill + traveling plate recipe

To composite a generated character onto real footage whose camera already travels (walking shots, vehicle plates):

1. Render the motion with `--inplace --static`. Hip travel is cancelled per frame, so the character runs in place under a locked camera, framed and stable.
2. Generate the character from that control video on a plain background (green, if your model holds it).
3. Key and overlay onto the traveling plate. The plate's camera motion supplies the travel; the character's gait matches because the limb cycle survives the hip cancellation.

Without `--inplace`, the follow camera chases the hips and your composite inherits two competing camera moves.

## Multi-character

```bash
python3 mocap_control.py --multi \
  "fighter_a.json.gz:0,0,0" \
  "fighter_b.json.gz:1.4,0,3.14" \
  fight.mp4 --dur 5
```

Each spec is `path:dx,dz,ry` (XZ offset in meters, Y rotation in radians). Characters get shifted palettes so the model can tell them apart, and the camera tracks the midpoint of all hips. `--volumetric` switches to depth-sorted mannequin-style bones; `--orbit 30` sweeps the camera 30° over the clip.

## License

MIT. The sample motion is synthesized from sine functions, so the repo ships zero mocap data and zero dataset license strings attached.
