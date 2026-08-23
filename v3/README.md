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

## Hosting

The pyramid is ~half a gigabyte, which does not fit in the 1 GB GitHub Pages budget alongside
everything else, so **it is not in this repo** (`v3/tiles/` is gitignored). Put the `tiles/`
directory anywhere that serves static files with CORS enabled and point the viewer at it:

```
v3/index.html?tiles=https://your-host/korno-tiles/
```

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
