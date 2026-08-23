# v3 — streaming multiresolution

v2 ships one 90 MB glTF: the browser downloads all of it before drawing anything, and the mesh has
to be decimated from 17 050 230 triangles to 2.5 M to fit in a file GitHub will accept. v3 is the
Sketchfab-style alternative — the model is cut into a tile pyramid and the viewer downloads only the
tiles inside the frustum, only to the depth the screen can resolve. **Leaves are the original,
undecimated mesh**, so this is the full-quality reference.

Nothing here is compressed or quantised. That is deliberate: the point of the exercise was to see the
17 M mesh at top quality first. Draco/meshopt on the geometry and KTX2 on the textures would cut the
payload several-fold and are the obvious next step.

## The pyramid

A quadtree over the wall's UV frame — v2's orthophoto parametrisation makes the spatial index
obvious, exactly like a map tile pyramid.

```
level 0   2 x 1 tiles     coarse                    -> first paint
level 1   4 x 2
level 2   8 x 4
level 3  16 x 8
level 4  32 x 16 tiles    the original triangles, undecimated
```

Every triangle of the full mesh lands in exactly one leaf, sorted by the UV of its centroid. Each
parent is its four children merged and decimated back to one tile's budget, so cost per tile is
roughly constant and each level up is 4× coarser per unit area.

Three details make it work:

- **UV is analytic.** `to_uv` is a plane projection, so texture coordinates are recomputed from the
  vertex positions after every decimation rather than carried through it — just as well, since
  Open3D's quadric decimation drops UVs.
- **Skirts, not stitching.** Neighbouring tiles drawn at different levels leave cracks. Every *cut*
  edge — one lying on the tile's nominal grid line, as opposed to the wall's genuine ragged rim —
  grows a flange extruded backwards along the view axis, deep enough to cover that level's own
  geometric error. Because the view axis is orthogonal to the UV basis, a skirt vertex has exactly
  the same UV as its source, so the texture stays continuous and the tile's UV extent is unchanged.
- **Measured geometric error.** Each node records the 99th-percentile distance from the original
  triangles under it to its own decimated surface. The viewer turns that into a screen-space error,
  so refinement follows measured deviation rather than a guess about level.

## The viewer

`index.html` walks the tree every frame:

```
sse = error * (viewportHeight / (2 tan(fov/2))) / distanceToBox
refine into children when sse > MAX_SSE  (default 2 px, override with ?sse=)
```

A node keeps being drawn until **all** of its visible children have arrived, so refinement never
punches a hole — the "replacement refinement with parent fallback" that 3D Tiles uses. Requests are
queued nearest-first, six in flight, and tiles unused for 25 s have their GPU memory released.

The route, bolt and camera overlays are read from `../v2/` — v3 shares v2's coordinate frame and
does not duplicate the metadata.

## Result

| level | tiles | triangles | MB | error (median) |
|---|---|---|---|---|
| 0 | 2 | 29 056 | 0.9 | 0.0214 (~7 cm) |
| 1 | 8 | 118 572 | 3.7 | 0.0093 |
| 2 | 32 | 468 557 | 14.2 | 0.0047 |
| 3 | 125 | 1 536 863 | 48.2 | 0.0027 |
| 4 | 480 | **17 519 252** | 435.9 | 0.0018 (~6 mm) |
| | **647** | | **502.8** | |

Measured in the viewer at 1400 × 900, from a cold cache:

| view | tiles drawn | triangles on screen | downloaded |
|---|---|---|---|
| first paint | 2 (L0) | — | **0.47 MB**, 1.3 s |
| whole wall | 8 (L1) | 0.12 M | 3.1 MB |
| zoomed in | 26 (L1+L2) | 0.39 M | 10.8 MB |
| closer | 56 (L2+L3) | 0.84 M | 26.7 MB |
| hard zoom | 78 (L3+L4) | 3.14 M | 127 MB |

So even at the deepest zoom a session pulls about a quarter of the 503 MB, and what is on screen is the
original mesh. Build time is 7.5 min on a Kaggle CPU kernel.

Three bugs were worth the two extra runs it took to find them, all of them invisible in the numbers
and obvious on screen:

- `Object.assign(t, { children: [] })` wrote into the same object the child keys were about to be
  read from, so every node came out childless and the tree never refined at all.
- `viewing_direction()` points *from* the camera into the scene, so `-VIEW` extruded the skirts
  towards the viewer and pasted them over the wall.
- `TriangleMesh += TriangleMesh` concatenates without welding. The children's seam vertices stayed
  duplicated, so every seam remained a boundary, propagated up the pyramid, and the decimator could
  not collapse across any of them.

Skirt depth also matters more than it looks: at 3 × the node's error the seams showed as dark wedges
at grazing angles. The gap a skirt has to cover is bounded by that error, so 1.5 × is margin enough.

## Hosting

The pyramid is ~half a gigabyte, which does not fit in the 1 GB GitHub Pages budget alongside
everything else, so **it is not in this repo** (`v3/tiles/` is gitignored). It is served from

**https://filehost.openclimbing.org/korno-tiles/**

which is the viewer's default; `?tiles=https://other-host/path/` overrides it. That box runs nginx
with `root /srv/www`, `Access-Control-Allow-Origin: *` on everything (the viewer fetches geometry
with `fetch()` and textures through WebGL, both cross-origin from Pages), gzip on `.bin` and
`.json`, a seven-day `Cache-Control` on tiles and `no-cache` on `index.json` so a rebuilt pyramid
cannot be read against a stale tree. Cloudflare's tunnel maps the hostname to port 80.

To redeploy after a rebuild:

```bash
kaggle kernels output pavelzbytovsk/korno-v3-tiles -p ./out
rsync -a --delete ./out/tiles/ ubuntu@filehost:/srv/www/korno-tiles/
```

Any static host with CORS works just as well:

The layout is flat and needs no server logic:

```
tiles/index.json          the tree: per node the error, bbox, uv rect, children
tiles/<level>/<i>_<j>.bin geometry
tiles/<level>/<i>_<j>.jpg texture crop
```

`.bin` is little-endian: `"KRN3"`, `u32 version`, `u32 nverts`, `u32 ntris`, `u32 indexBytes`, then
`f32 position[3n]`, `f32 normal[3n]`, `f32 uv[2n]`, then indices as `u16` or `u32`.

## Building it

```bash
python3 v3/kaggle/tiles/build_notebook.py
kaggle kernels push -p v3/kaggle/tiles
```

The kernel takes the cleaned mesh and the camera poses from `korno-v2-mvs` and the photo-baked atlas
straight out of `korno_v2_desmudged.glb` (`korno-v2-retexture`), so it re-derives nothing.
