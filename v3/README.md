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

**The whole mesh is decimated once per level and only then cut into that level's grid.** Every
triangle goes wholly to one tile, so neighbours on the same level share their boundary vertices
exactly and meet with no crack at all. The obvious construction — build each tile by decimating its
own four children — cracks along *every* seam instead, because two neighbours simplified
independently no longer agree on the boundary they share; skirts only hide that, and the grid of
seams stays visible. Levels are cascaded (17 M → 1.9 M → 480 k → 119 k → 29 k), so each level up is
about 4× coarser per unit area and costs one decimation, not one per tile.

Only transitions *between* levels can still gap, and shallow skirts cover those. Sketchfab and Nexus
fix those too, with a batched multi-triangulation: the partition alternates between levels so each
level's seams fall inside the next level's cells and get simplified away. That needs a DAG rather
than a tree — this is the tree-shaped 90 % of it.

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

## Lighting

The texture carries no baked shadows — the wall faces north and was shot in flat light — which is a
nuisance for looking at it and a gift for lighting it: nothing here fights an added light, the way it
would on a model with golden-hour shadows painted in. Four modes, in the HUD:

| | |
|---|---|
| `sky` | hemisphere plus an environment map. What the wall honestly looks like. Default. |
| `relief` | a hard light raked across the face at 15°, direction on a slider. A tool, not a picture: it is how edges and holds become legible. |
| `sun` | warm key at 30°, openly artificial — this wall never gets direct sun. |
| `flat` | unlit, texture only. What the photo bake actually produced, with no shading on top. |

Rendering is ACES filmic with an environment map either way; two bare lights and no tone mapping
made a photogrammetry texture look like wallpaper.

## The viewer

`index.html` walks the tree every frame:

```
sse   = error * (viewportHeight * devicePixelRatio / (2 tan(fov/2))) / distanceToBox
level = the coarsest whose worst visible tile still meets MAX_SSE   (default 0.7 px, ?sse= to override)
```

**One level for the whole frustum.** Refining tile by tile on its own error is what a 3D-Tiles
traversal does, and on a wall it looks wrong: neighbours land on different levels, so a sharp tile
sits beside a soft one with a hard line between them, and the line moves as you turn. Choosing a
single level for everything on screen costs some bandwidth and gives a picture that is uniformly
sharp. The level is the coarsest that leaves nothing on screen under-resolved, and a 140-tile budget
stops a grazing view — where the far end of the wall is in frame with the near end — from demanding
the leaf level across the whole face. The next level down is prefetched wholesale, so stepping to it
is not a wait.

`devicePixelRatio` matters more than it looks. Counting CSS pixels made a phone at DPR 3 pick a level
far too coarse: the initial view came up as **2 tiles at level 0** where a desktop got level 1, and
the first zoom step then jumped two levels at once. Counting device pixels puts the phone on level 1
from the start.

A node keeps being drawn until **all** of its visible children have arrived, so refinement never
punches a hole — the "replacement refinement with parent fallback" that 3D Tiles uses. Requests are
queued nearest-first, six in flight, and tiles unused for 25 s have their GPU memory released.

The route, bolt and camera overlays are read from `../v2/` — v3 shares v2's coordinate frame and
does not duplicate the metadata.

## Result

| level | tiles | triangles | MB | error (median) |
|---|---|---|---|---|
| 0 | 2 | 29 084 | 1.6 | 0.0194 (~6 cm) |
| 1 | 8 | 119 272 | 6.5 | 0.0073 |
| 2 | 32 | 479 942 | 24.7 | 0.0027 |
| 3 | 125 | 1 916 904 | 71.0 | 0.0018 |
| 4 | 480 | **17 519 252** | 437.9 | 0.0018 (~6 mm) |
| | **647** | | **541.7** | |

A tile's error is the larger of its measured geometric deviation and its texel size, and below level 1
it is the **texel** that dominates — so tile texture resolution, not triangle count, is what decides
when the viewer refines. Capping tile textures at 512 px made the coarse levels blurry and refinement
lazy; at 1024 px only levels 0–2 are capped at all (42 tiles), which costs 30 MB and halves the error
at levels 2 and 3.

Measured in the viewer at 1400 × 900, from a cold cache:

| view | tiles drawn | triangles on screen | downloaded |
|---|---|---|---|
| first paint | 2 (L0) | — | **0.47 MB**, 1.3 s |
| whole wall | 8 (L1) | 0.12 M | 3.1 MB |
| zoomed in | 26 (L1+L2) | 0.39 M | 10.8 MB |
| closer | 56 (L2+L3) | 0.84 M | 26.7 MB |
| hard zoom | 78 (L3+L4) | 3.14 M | 127 MB |

So even at the deepest zoom a session pulls about a quarter of the 511 MB, and what is on screen is the
original mesh. Build time is 8.5 min on a Kaggle CPU kernel.

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

One more, and the reason a fourth run was needed: the notebook generator was invoked as
`python3 build_notebook.py >/dev/null 2>&1; <check>`. It was failing — a `"""` docstring inside a
`r"""…"""` cell string closes the cell — but the output was discarded and the `;` let the next
command run anyway, so the *stale* `.ipynb` got pushed and syntax-checked clean. Two runs produced
byte-identical triangle counts before that was noticed. Generate with `&&` and let it speak.

## Hosting

The pyramid is ~half a gigabyte, which does not fit in the 1 GB GitHub Pages budget alongside
everything else, so **it is not in this repo** (`v3/tiles/` is gitignored). It is served from

**https://filehost.openclimbing.org/korno-tiles/**

which is the viewer's default; `?tiles=https://other-host/path/` overrides it. That box runs nginx
with `root /srv/www`, `Access-Control-Allow-Origin: *` on everything (the viewer fetches geometry
with `fetch()` and textures through WebGL, both cross-origin from Pages), gzip on `.bin` and
`.json`, a seven-day `Cache-Control` on tiles and `no-cache` on `index.json`. Cloudflare's tunnel maps the
hostname to port 80 — and caches the tiles at its edge, so `index.json` carries a build `version`
that the viewer appends to every tile URL. Without it a rebuilt pyramid is read against whatever
tiles the edge still holds; that is exactly what happened on the first redeploy, and the two tiles
that had been fetched during verification came back stale while everything else was current.

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
