// mocap-control — mocap joint data → pose+depth control videos for AI video models.
//
// Takes a joint-position JSON ({fps, parents, pos[F][J][3]} — Z-up world space,
// the MuJoCo convention) and renders an OpenPose-style colored stick figure
// through a simple pinhole camera → mp4 ready for Wan-VACE, LTX-2.3
// IC-LoRA Union-Control, or any pose-conditioned video model. Physics-simulated
// stick figures in, photoreal characters out — no skinning, ever.
//
// With -depth-combo the same 3D data also renders a TRUE geometric depth pass
// (shaded ground-plane strips + depth-shaded bones) UNDER the colored skeleton:
// one control video carrying both pose and depth.
//
// Usage:
//
//	mocap-control [flags] <motion.json.gz> <out.mp4>
//	mocap-control -multi "<m1.json.gz>:dx,dz,ry" -multi "<m2.json.gz>:dx,dz,ry" <out.mp4>
//	mocap-control sample -out walk.json.gz [-dur 4 -fps 30 -style walk|jog]
//
// All camera dials also accept env vars (SKEL_DIST, SKEL_HEIGHT, SKEL_DAMP,
// SKEL_STATIC, SKEL_INPLACE, SKEL_DEPTH_COMBO, SKEL_FPS, SKEL_FRAMES) as
// defaults; CLI flags win when given.
//
// Requires: ffmpeg on PATH. Nothing else — single static binary.
package main

import (
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// OpenPose-style palette, one color per joint index (mod len).
var palette = []color.RGBA{
	{255, 212, 0, 255}, {255, 140, 0, 255}, {46, 204, 113, 255}, {231, 76, 60, 255},
	{52, 152, 219, 255}, {155, 89, 182, 255}, {0, 229, 255, 255}, {255, 0, 204, 255},
	{26, 188, 156, 255}, {241, 196, 15, 255},
}

// real-world half-widths (m) per SOMA-23 bone index (child joint of each bone)
var boneW = map[int]float64{
	1: 0.16, 2: 0.17, 3: 0.18, 4: 0.07, 5: 0.07, 6: 0.11,
	7: 0.07, 8: 0.06, 9: 0.05, 10: 0.04, 11: 0.07, 12: 0.06, 13: 0.05, 14: 0.04,
	15: 0.09, 16: 0.065, 17: 0.05, 18: 0.045, 19: 0.09, 20: 0.065, 21: 0.05, 22: 0.045,
}

var mannequinTints = []color.RGBA{
	{202, 196, 186, 255}, {168, 178, 194, 255}, {196, 176, 176, 255}, {176, 194, 176, 255},
}

// ---------------------------------------------------------------- vec3 math

type vec3 [3]float64

func (a vec3) add(b vec3) vec3      { return vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }
func (a vec3) sub(b vec3) vec3      { return vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }
func (a vec3) scale(s float64) vec3 { return vec3{a[0] * s, a[1] * s, a[2] * s} }
func (a vec3) dot(b vec3) float64   { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
func (a vec3) cross(b vec3) vec3 {
	return vec3{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
}
func (a vec3) norm() vec3 {
	l := math.Sqrt(a.dot(a))
	if l == 0 {
		return a
	}
	return a.scale(1 / l)
}

// ---------------------------------------------------------------- motion IO

type motionFile struct {
	FPS     float64       `json:"fps"`
	Parents []int         `json:"parents"`
	Pos     [][][]float64 `json:"pos"`
}

type motion struct {
	fps     float64
	parents []int
	pos     [][]vec3 // [F][J], Y-up (x, z, -y from the Z-up source)
}

// loadMotion reads {fps, parents, pos[F][J][3]} json.gz, converts Z-up → Y-up,
// slices [start, start+dur), grounds min-Y to 0, then applies an optional Y
// rotation (ry, radians) and XZ placement so frame-0 hips land at (dx, dz).
func loadMotion(src string, start, dur, dx, dz, ry float64, place bool) (*motion, error) {
	f, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", src, err)
	}
	defer zr.Close()
	var mf motionFile
	if err := json.NewDecoder(zr).Decode(&mf); err != nil {
		return nil, fmt.Errorf("%s: %w", src, err)
	}
	f0 := int(start * mf.FPS)
	f1 := int((start + dur) * mf.FPS)
	if f1 > len(mf.Pos) {
		f1 = len(mf.Pos)
	}
	if f0 < 0 || f0 >= f1 {
		return nil, fmt.Errorf("%s: empty frame range [%d:%d]", src, f0, f1)
	}
	pos := make([][]vec3, f1-f0)
	minY := math.Inf(1)
	for i := range pos {
		raw := mf.Pos[f0+i]
		row := make([]vec3, len(raw))
		for j, p := range raw {
			// Z-up (x, y, z) → Y-up (x, z, -y)
			row[j] = vec3{p[0], p[2], -p[1]}
			if row[j][1] < minY {
				minY = row[j][1]
			}
		}
		pos[i] = row
	}
	for i := range pos {
		for j := range pos[i] {
			pos[i][j][1] -= minY
		}
	}
	if ry != 0 {
		c, s := math.Cos(ry), math.Sin(ry)
		for i := range pos {
			for j := range pos[i] {
				x, z := pos[i][j][0], pos[i][j][2]
				pos[i][j][0] = c*x + s*z
				pos[i][j][2] = -s*x + c*z
			}
		}
	}
	if place {
		ctr := pos[0][0]
		for i := range pos {
			for j := range pos[i] {
				pos[i][j][0] += dx - ctr[0]
				pos[i][j][2] += dz - ctr[2]
			}
		}
	}
	return &motion{fps: mf.FPS, parents: mf.Parents, pos: pos}, nil
}

// ---------------------------------------------------------------- rasterizer

type canvas struct {
	img  *image.RGBA
	w, h int
}

func newCanvas(w, h int) *canvas {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// black background
	for i := 3; i < len(img.Pix); i += 4 {
		img.Pix[i] = 255
	}
	return &canvas{img: img, w: w, h: h}
}

func (c *canvas) set(x, y int, col color.RGBA) {
	if x < 0 || y < 0 || x >= c.w || y >= c.h {
		return
	}
	i := c.img.PixOffset(x, y)
	c.img.Pix[i] = col.R
	c.img.Pix[i+1] = col.G
	c.img.Pix[i+2] = col.B
	c.img.Pix[i+3] = 255
}

// fillCircle paints a filled disc of radius r centered at (cx, cy).
func (c *canvas) fillCircle(cx, cy, r float64, col color.RGBA) {
	x0, x1 := int(math.Floor(cx-r)), int(math.Ceil(cx+r))
	y0, y1 := int(math.Floor(cy-r)), int(math.Ceil(cy+r))
	r2 := r * r
	for y := y0; y <= y1; y++ {
		dy := float64(y) - cy
		for x := x0; x <= x1; x++ {
			dx := float64(x) - cx
			if dx*dx+dy*dy <= r2 {
				c.set(x, y, col)
			}
		}
	}
}

// thickLine draws a line of the given pixel width by stamping discs along the
// segment (round caps — the joint discs drawn on top hide any difference from
// PIL's butt caps).
func (c *canvas) thickLine(x0, y0, x1, y1 float64, width int, col color.RGBA) {
	r := float64(width) / 2
	dx, dy := x1-x0, y1-y0
	length := math.Hypot(dx, dy)
	step := 0.4
	if r*0.5 > step {
		step = r * 0.5
	}
	n := int(length/step) + 1
	for i := 0; i <= n; i++ {
		t := float64(i) / float64(n)
		c.fillCircle(x0+dx*t, y0+dy*t, r, col)
	}
}

// fillPoly scanline-fills a polygon (even-odd rule, pixel-center sampling).
func (c *canvas) fillPoly(pts [][2]float64, col color.RGBA) {
	yMin, yMax := math.Inf(1), math.Inf(-1)
	for _, p := range pts {
		yMin = math.Min(yMin, p[1])
		yMax = math.Max(yMax, p[1])
	}
	iy0 := int(math.Floor(yMin))
	iy1 := int(math.Ceil(yMax))
	if iy0 < 0 {
		iy0 = 0
	}
	if iy1 >= c.h {
		iy1 = c.h - 1
	}
	var xs []float64
	for y := iy0; y <= iy1; y++ {
		fy := float64(y) + 0.5
		xs = xs[:0]
		for i := range pts {
			a, b := pts[i], pts[(i+1)%len(pts)]
			if (a[1] <= fy) == (b[1] <= fy) {
				continue
			}
			xs = append(xs, a[0]+(fy-a[1])*(b[0]-a[0])/(b[1]-a[1]))
		}
		sort.Float64s(xs)
		for i := 0; i+1 < len(xs); i += 2 {
			x0 := int(math.Ceil(xs[i] - 0.5))
			x1 := int(math.Ceil(xs[i+1]-0.5)) - 1
			if x0 < 0 {
				x0 = 0
			}
			if x1 >= c.w {
				x1 = c.w - 1
			}
			for x := x0; x <= x1; x++ {
				c.set(x, y, col)
			}
		}
	}
}

// ---------------------------------------------------------------- camera

type camera struct{ pos, right, up, fwd vec3 }

func makeCamera(look, offset vec3) camera {
	cam := look.add(offset)
	fwd := look.sub(cam).norm()
	right := fwd.cross(vec3{0, 1, 0}).norm()
	up := right.cross(fwd)
	return camera{cam, right, up, fwd}
}

func (cm camera) project(p vec3, w, h int, fl float64) (float64, float64) {
	v := p.sub(cm.pos)
	x, y, z := v.dot(cm.right), v.dot(cm.up), v.dot(cm.fwd)
	if z < 0.1 {
		z = 0.1
	}
	return float64(w)/2 + fl*x/z, float64(h)/2 - fl*y/z
}

func (cm camera) zdepth(p vec3) float64 {
	z := p.sub(cm.pos).dot(cm.fwd)
	if z < 0.12 {
		z = 0.12
	}
	return z
}

// ---------------------------------------------------------------- encoding

func writeFrames(render func(i int) *image.RGBA, nOut, fpsOut int, out string) error {
	tmp, err := os.MkdirTemp("", "mocapctl_")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	for i := 0; i < nOut; i++ {
		f, err := os.Create(filepath.Join(tmp, fmt.Sprintf("f_%04d.png", i)))
		if err != nil {
			return err
		}
		enc := png.Encoder{CompressionLevel: png.BestSpeed}
		if err := enc.Encode(f, render(i)); err != nil {
			f.Close()
			return err
		}
		f.Close()
	}
	cmd := exec.Command("ffmpeg", "-y", "-v", "error", "-framerate", strconv.Itoa(fpsOut),
		"-i", filepath.Join(tmp, "f_%04d.png"),
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-time.After(180 * time.Second):
		_ = cmd.Process.Kill()
		return fmt.Errorf("ffmpeg timed out after 180s")
	}
}

// ---------------------------------------------------------------- single render

type renderOpts struct {
	start, dur                  float64
	w, h                        int
	distMult, camHeight, damp   float64
	static, inplace, depthCombo bool
	fpsOut, frames              int
}

func renderSingle(src, out string, o renderOpts) error {
	m, err := loadMotion(src, o.start, o.dur, 0, 0, 0, false)
	if err != nil {
		return err
	}
	pos, parents, fpsIn := m.pos, m.parents, m.fps

	// -inplace: cancel hip XZ travel per-frame (treadmill mode) — for
	// compositing onto real footage whose own camera supplies the travel
	if o.inplace {
		for i := range pos {
			hx, hz := pos[i][0][0], pos[i][0][2]
			for j := range pos[i] {
				pos[i][j][0] -= hx
				pos[i][j][2] -= hz
			}
		}
	}
	// TRACKING camera: follows the hips (joint 0) with a fixed 3/4 offset so the
	// figure stays centered and large while the motion travels through space.
	minY, maxY := math.Inf(1), math.Inf(-1)
	for _, p := range pos[0] {
		minY = math.Min(minY, p[1])
		maxY = math.Max(maxY, p[1])
	}
	figH := maxY - minY
	if figH == 0 {
		figH = 1.7
	}
	dist := figH * o.distMult
	offset := vec3{dist * 0.72, figH * o.camHeight, dist * 0.72}
	fl := 1.15 * math.Min(float64(o.w), float64(o.h)) // focal length in px

	var smoothLook *vec3

	makeCam := func(hips vec3) camera {
		// lerp-damped follow (-damp, default 1.0 = hard lock; 0.08-0.15 =
		// cinematic lag) — kills hip-jolt camera shake on violent motions
		if o.static && smoothLook != nil {
			hips = *smoothLook
		} else if smoothLook == nil || o.damp >= 1.0 {
			h := hips
			smoothLook = &h
		} else {
			s := smoothLook.add(hips.sub(*smoothLook).scale(o.damp))
			smoothLook = &s
			hips = s
		}
		if !o.static {
			hips = *smoothLook
		}
		look := hips
		look[1] = figH * 0.55
		return makeCamera(look, offset)
	}

	nOut := o.frames
	if nOut == 0 {
		nOut = int(o.dur * float64(o.fpsOut))
	}
	nSrc := len(pos)

	render := func(i int) *image.RGBA {
		t := float64(i) / float64(o.fpsOut)
		fi := int(t * fpsIn)
		if fi >= nSrc { // pingpong loop past motion end
			cyc := 0
			if nSrc > 1 {
				cyc = fi % (2*nSrc - 2)
			}
			if cyc < nSrc {
				fi = cyc
			} else {
				fi = 2*nSrc - 2 - cyc
			}
		}
		frame := pos[fi]
		cam := makeCam(frame[0])
		cv := newCanvas(o.w, o.h)
		pts := make([][2]float64, len(frame))
		for j := range frame {
			x, y := cam.project(frame[j], o.w, o.h, fl)
			pts[j] = [2]float64{x, y}
		}
		if o.depthCombo {
			// DEPTH LAYER first: true geometric depth from the same 3D data.
			// Ground plane: quad strips receding along camera forward, shaded 1/z.
			znear, zfar := 1.0, dist*4.5
			shade := func(z float64) uint8 {
				v := (1.0/z - 1.0/zfar) / (1.0/znear - 1.0/zfar)
				if v < 0 {
					v = 0
				}
				if v > 1 {
					v = 1
				}
				return uint8(255 * v)
			}
			for k := 0; k < 28; k++ {
				z0w, z1w := 0.6+float64(k)*0.9, 0.6+float64(k+1)*0.9
				c0 := cam.pos.add(cam.fwd.scale(z0w))
				c1 := cam.pos.add(cam.fwd.scale(z1w))
				// project 4 corners of the floor strip at y=0 in world
				quad := make([][2]float64, 0, 4)
				for _, bs := range []struct {
					base vec3
					sx   float64
				}{{c0, -4.0}, {c0, 4.0}, {c1, 4.0}, {c1, -4.0}} {
					wpt := vec3{bs.base[0] + cam.right[0]*bs.sx, 0.0, bs.base[2] + cam.right[2]*bs.sx}
					x, y := cam.project(wpt, o.w, o.h, fl)
					quad = append(quad, [2]float64{x, y})
				}
				g := shade(cam.zdepth(c0.add(c1).scale(0.5)))
				cv.fillPoly(quad, color.RGBA{g, g, g, 255})
			}
			// character: bones drawn as thick strokes shaded by joint depth
			lift := func(g uint8) color.RGBA {
				v := int(g) + 70 // figure pops brighter than floor
				if v > 255 {
					v = 255
				}
				return color.RGBA{uint8(v), uint8(v), uint8(v), 255}
			}
			bw := o.h / 32
			if bw < 9 {
				bw = 9
			}
			for j, par := range parents {
				if par < 0 {
					continue
				}
				z := (cam.zdepth(frame[j]) + cam.zdepth(frame[par])) / 2
				bg := lift(shade(z))
				cv.thickLine(pts[par][0], pts[par][1], pts[j][0], pts[j][1], bw, bg)
			}
			jr := float64(o.h / 60)
			if jr < 6 {
				jr = 6
			}
			for j := range pts {
				bg := lift(shade(cam.zdepth(frame[j])))
				cv.fillCircle(pts[j][0], pts[j][1], jr, bg)
			}
		}
		lw := o.h / 90
		if lw < 3 {
			lw = 3
		}
		for j, par := range parents {
			if par < 0 {
				continue
			}
			cv.thickLine(pts[par][0], pts[par][1], pts[j][0], pts[j][1], lw, palette[j%len(palette)])
		}
		jr := float64(o.h / 110)
		if jr < 3 {
			jr = 3
		}
		for j := range pts {
			cv.fillCircle(pts[j][0], pts[j][1], jr, palette[j%len(palette)])
		}
		return cv.img
	}

	if err := writeFrames(render, nOut, o.fpsOut, out); err != nil {
		return err
	}
	report(map[string]any{"ok": true, "out": out, "frames": nOut, "src_fps": fpsIn, "joints": len(parents)})
	return nil
}

// ---------------------------------------------------------------- multi render

func renderMulti(specs []string, out string, dur float64, w, h int, volumetric bool, orbitDeg float64) error {
	chars := make([]*motion, 0, len(specs))
	for _, spec := range specs {
		path, ofs, has := strings.Cut(spec, ":")
		var dx, dz, ry float64
		if has && ofs != "" {
			parts := strings.Split(ofs, ",")
			vals := [3]float64{}
			for i := 0; i < len(parts) && i < 3; i++ {
				v, err := strconv.ParseFloat(strings.TrimSpace(parts[i]), 64)
				if err != nil {
					return fmt.Errorf("bad offset in spec %q: %w", spec, err)
				}
				vals[i] = v
			}
			dx, dz, ry = vals[0], vals[1], vals[2]
		}
		m, err := loadMotion(path, 0.0, dur, dx, dz, ry, true)
		if err != nil {
			return err
		}
		chars = append(chars, m)
	}
	figH := 1.7
	dist := figH*1.85 + 0.45*float64(len(chars)-1)
	baseAz := math.Atan2(0.72, 0.72)
	camR := dist * 0.72 * math.Sqrt2
	fl := 1.15 * math.Min(float64(w), float64(h))
	fpsOut := 24
	nOut := int(dur * float64(fpsOut))

	frameAt := func(m *motion, t float64) []vec3 {
		fi := int(t * m.fps)
		if fi > len(m.pos)-1 {
			fi = len(m.pos) - 1
		}
		return m.pos[fi]
	}

	render := func(i int) *image.RGBA {
		t := float64(i) / float64(fpsOut)
		var hips vec3
		for _, m := range chars {
			hips = hips.add(frameAt(m, t)[0])
		}
		hips = hips.scale(1 / float64(len(chars)))
		look := hips
		look[1] = figH * 0.55
		az := baseAz + orbitDeg*math.Pi/180*(t/dur)
		camPos := look.add(vec3{math.Sin(az) * camR, figH * 0.4, math.Cos(az) * camR})
		fwd := look.sub(camPos).norm()
		right := fwd.cross(vec3{0, 1, 0}).norm()
		up := right.cross(fwd)
		cam := camera{camPos, right, up, fwd}
		cv := newCanvas(w, h)
		if volumetric {
			// painter's algorithm across ALL characters: far bones first
			type bone struct {
				z      float64
				ci     int
				pa, pb vec3
				j      int
			}
			var bones []bone
			for ci, m := range chars {
				frame := frameAt(m, t)
				for j, par := range m.parents {
					if par < 0 {
						continue
					}
					mid := frame[j].add(frame[par]).scale(0.5)
					bones = append(bones, bone{mid.sub(camPos).dot(fwd), ci, frame[par], frame[j], j})
				}
				// head as its own "bone"
				bones = append(bones, bone{frame[6].sub(camPos).dot(fwd), ci, frame[6], frame[6], 6})
			}
			sort.SliceStable(bones, func(a, b int) bool { return bones[a].z > bones[b].z })
			prj := func(p vec3) (float64, float64, float64) {
				v := p.sub(camPos)
				x, y, z := v.dot(right), v.dot(up), v.dot(fwd)
				if z < 0.1 {
					z = 0.1
				}
				return float64(w)/2 + fl*x/z, float64(h)/2 - fl*y/z, z
			}
			for _, b := range bones {
				tint := mannequinTints[b.ci%len(mannequinTints)]
				// subtle depth shading so limbs separate visually
				zs := b.z
				if zs < 0.5 {
					zs = 0.5
				}
				sh := 3.2 / zs
				if sh > 1.0 {
					sh = 1.0
				}
				if sh < 0.78 {
					sh = 0.78
				}
				col := color.RGBA{uint8(float64(tint.R) * sh), uint8(float64(tint.G) * sh), uint8(float64(tint.B) * sh), 255}
				xa, ya, za := prj(b.pa)
				xb, yb, zb := prj(b.pb)
				zm := (za + zb) / 2
				if zm < 0.5 {
					zm = 0.5
				}
				hw, ok := boneW[b.j]
				if !ok {
					hw = 0.06
				}
				wpx := int(hw * fl / zm)
				if wpx < 2 {
					wpx = 2
				}
				if b.j == 6 { // head sphere
					cv.fillCircle(xb, yb, float64(wpx), col)
				} else {
					cv.thickLine(xa, ya, xb, yb, wpx*2, col)
					cv.fillCircle(xa, ya, float64(wpx), col)
					cv.fillCircle(xb, yb, float64(wpx), col)
				}
			}
		} else {
			lw := h / 90
			if lw < 3 {
				lw = 3
			}
			jr := float64(h / 110)
			if jr < 3 {
				jr = 3
			}
			for ci, m := range chars {
				frame := frameAt(m, t)
				pts := make([][2]float64, len(frame))
				for j := range frame {
					x, y := cam.project(frame[j], w, h, fl)
					pts[j] = [2]float64{x, y}
				}
				shift := ci * 3
				for j, par := range m.parents {
					if par < 0 {
						continue
					}
					cv.thickLine(pts[par][0], pts[par][1], pts[j][0], pts[j][1], lw, palette[(j+shift)%len(palette)])
				}
				for j := range pts {
					cv.fillCircle(pts[j][0], pts[j][1], jr, palette[(j+shift)%len(palette)])
				}
			}
		}
		return cv.img
	}

	if err := writeFrames(render, nOut, fpsOut, out); err != nil {
		return err
	}
	report(map[string]any{"ok": true, "out": out, "chars": len(chars), "frames": nOut})
	return nil
}

// ---------------------------------------------------------------- sample gait

// 23-joint skeleton: parent index per joint (-1 = root)
// joints: 0 pelvis, 1-3 spine, 4 neck, 5 head base, 6 head top,
//
//	7-10 left arm (shoulder/elbow/wrist/hand), 11-14 right arm,
//	15-18 left leg (hip/knee/ankle/toe), 19-22 right leg
var soma23Parents = []int{-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 3, 11, 12, 13, 0, 15, 16, 17, 0, 19, 20, 21}

// gaitFrame returns one frame of joint positions [23][3] at time t (seconds), Z-up.
func gaitFrame(t float64, style string) [][]float64 {
	var fCycle, stride, lift, bounce, armAmp float64
	if style == "jog" {
		fCycle, stride, lift, bounce, armAmp = 1.5, 0.38, 0.22, 0.055, 0.75
	} else {
		fCycle, stride, lift, bounce, armAmp = 1.0, 0.30, 0.14, 0.03, 0.5
	}
	theta := 2 * math.Pi * fCycle * t
	speed := 4 * stride * fCycle // so stance feet roughly don't slide
	px := speed * t              // pelvis travels along +x
	pz := 0.95 + bounce*math.Sin(2*theta)
	py := 0.03 * math.Sin(theta) // lateral sway

	J := make([][]float64, 23)
	for i := range J {
		J[i] = []float64{0, 0, 0}
	}
	J[0] = []float64{px, py, pz}
	// spine → head, slight forward lean
	spine := [][2]float64{{0.12, 0.01}, {0.25, 0.02}, {0.38, 0.03}, {0.51, 0.04}, {0.59, 0.045}, {0.73, 0.05}}
	for k, dzdx := range spine {
		J[k+1] = []float64{px + dzdx[1], py * 0.5, pz + dzdx[0]}
	}

	arm := func(a, b, c, d int, side, phase float64) {
		// side: +1 left / -1 right; swing opposite the same-side leg
		sw := armAmp * math.Sin(theta+phase)
		sx, sy, sz := px+0.03, side*0.22, pz+0.44
		J[a] = []float64{sx, sy, sz}                         // shoulder
		J[b] = []float64{sx + 0.15*sw, sy * 1.05, sz - 0.28} // elbow
		J[c] = []float64{sx + 0.38*sw, sy * 1.05, sz - 0.50} // wrist
		J[d] = []float64{sx + 0.46*sw, sy * 1.05, sz - 0.57} // hand
	}
	leg := func(a, b, c, d int, side, phase float64) {
		th := theta + phase
		hx, hy, hz := px, side*0.10, pz-0.03
		J[a] = []float64{hx, hy, hz}       // hip
		ax := px + stride*math.Sin(th)     // ankle fore-aft
		swing := math.Max(0, math.Cos(th)) // 1 at mid-swing
		az := 0.06 + lift*swing
		bend := 0.08 + 0.14*swing                                          // knee forward
		J[b] = []float64{0.55*hx + 0.45*ax + bend, hy, 0.5*(hz+az) + 0.05} // knee
		J[c] = []float64{ax, hy, az}                                       // ankle
		J[d] = []float64{ax + 0.16, hy, math.Max(0.02, az-0.04)}           // toe
	}

	arm(7, 8, 9, 10, +1, math.Pi) // left arm w/ right leg
	arm(11, 12, 13, 14, -1, 0.0)  // right arm w/ left leg
	leg(15, 16, 17, 18, +1, 0.0)  // left leg
	leg(19, 20, 21, 22, -1, math.Pi)
	return J
}

func sampleMain(args []string) error {
	fs := flag.NewFlagSet("mocap-control sample", flag.ExitOnError)
	out := fs.String("out", "", "output .json.gz path")
	dur := fs.Float64("dur", 4.0, "duration (s)")
	fps := fs.Int("fps", 30, "sample rate")
	style := fs.String("style", "walk", "gait style: walk|jog")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *out == "" && fs.NArg() > 0 {
		*out = fs.Arg(0)
	}
	if *out == "" {
		return fmt.Errorf("sample: -out <path.json.gz> required")
	}
	if *style != "walk" && *style != "jog" {
		return fmt.Errorf("sample: -style must be walk or jog")
	}
	n := int(*dur * float64(*fps))
	pos := make([][][]float64, n)
	for i := 0; i < n; i++ {
		pos[i] = gaitFrame(float64(i)/float64(*fps), *style)
	}
	f, err := os.Create(*out)
	if err != nil {
		return err
	}
	defer f.Close()
	zw := gzip.NewWriter(f)
	if err := json.NewEncoder(zw).Encode(map[string]any{
		"fps": *fps, "parents": soma23Parents, "pos": pos,
	}); err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	report(map[string]any{"ok": true, "out": *out, "frames": n, "joints": len(soma23Parents), "style": *style})
	return nil
}

// ---------------------------------------------------------------- CLI

func report(v map[string]any) {
	b, _ := json.Marshal(v)
	fmt.Println(string(b))
}

func envFloat(name string, def float64) float64 {
	if s, ok := os.LookupEnv(name); ok {
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			return v
		}
	}
	return def
}

func envBool(name string) bool { return os.Getenv(name) == "1" }

type multiSpecs []string

func (m *multiSpecs) String() string     { return strings.Join(*m, " ") }
func (m *multiSpecs) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	if len(os.Args) > 1 && os.Args[1] == "sample" {
		if err := sampleMain(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

	fs := flag.NewFlagSet("mocap-control", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), `mocap-control — render mocap joint data as pose(+depth) control videos for AI video models.

Usage:
  mocap-control [flags] <motion.json.gz> <out.mp4>
  mocap-control -multi "<motion.json.gz>:dx,dz,ry" [-multi ...] <out.mp4>
  mocap-control sample -out walk.json.gz [-dur 4 -fps 30 -style walk|jog]

Flags:
`)
		fs.PrintDefaults()
	}
	var multi multiSpecs
	fs.Var(&multi, "multi", "multi-character spec '<motion.json.gz>:dx,dz,ry' (repeatable)")
	start := fs.Float64("start", 0.0, "start time in source motion (s)")
	dur := fs.Float64("dur", 5.0, "clip duration (s)")
	w := fs.Int("w", 832, "output width px")
	h := fs.Int("h", 480, "output height px")
	dist := fs.Float64("dist", envFloat("SKEL_DIST", 2.1),
		"camera distance as multiple of figure height (1.25 = medium-full, 2.1 = wide) [env SKEL_DIST]")
	height := fs.Float64("height", envFloat("SKEL_HEIGHT", 0.35),
		"camera height as multiple of figure height [env SKEL_HEIGHT]")
	damp := fs.Float64("damp", envFloat("SKEL_DAMP", 1.0),
		"follow-cam lerp factor: 1.0 = hard lock, 0.08-0.15 = cinematic lag [env SKEL_DAMP]")
	static := fs.Bool("static", envBool("SKEL_STATIC"),
		"lock the camera to its first-frame position [env SKEL_STATIC=1]")
	inplace := fs.Bool("inplace", envBool("SKEL_INPLACE"),
		"treadmill mode: cancel hip XZ travel per-frame (for compositing onto traveling plates) [env SKEL_INPLACE=1]")
	depthCombo := fs.Bool("depth-combo", envBool("SKEL_DEPTH_COMBO"),
		"render true geometric depth (ground strips + depth-shaded bones) under the colored skeleton [env SKEL_DEPTH_COMBO=1]")
	fps := fs.Int("fps", int(envFloat("SKEL_FPS", 24)), "output frame rate [env SKEL_FPS]")
	frames := fs.Int("frames", int(envFloat("SKEL_FRAMES", 0)),
		"exact output frame count (pingpong-loops past motion end; 0 = dur*fps) [env SKEL_FRAMES]")
	volumetric := fs.Bool("volumetric", false, "(-multi only) mannequin-style volumetric bones with painter's sort")
	orbit := fs.Float64("orbit", 0.0, "(-multi only) camera orbit over the clip, degrees")
	_ = fs.Parse(os.Args[1:])

	var err error
	switch {
	case len(multi) > 0:
		if fs.NArg() != 1 {
			fmt.Fprintln(os.Stderr, "error: -multi needs one or more -multi specs and exactly one output mp4")
			os.Exit(2)
		}
		err = renderMulti(multi, fs.Arg(0), *dur, *w, *h, *volumetric, *orbit)
	case fs.NArg() == 2:
		err = renderSingle(fs.Arg(0), fs.Arg(1), renderOpts{
			start: *start, dur: *dur, w: *w, h: *h,
			distMult: *dist, camHeight: *height, damp: *damp,
			static: *static, inplace: *inplace, depthCombo: *depthCombo,
			fpsOut: *fps, frames: *frames,
		})
	default:
		fs.Usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
