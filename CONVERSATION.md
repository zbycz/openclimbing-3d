# How this repo was built

A log of the whole session, chapter by chapter. Every prompt is verbatim (one Kaggle token
redacted); the answers are summarised to a few sentences each.

---

## 1. Gaussian splat on a Kaggle P100

> Nahrál jsem set fotek z dronu na https://www.kaggle.com/datasets/pavelzbytovsk/korno-rockface-photogrammetry — chtěl bych abys vyrobil notebook pro kaggle, který udělá gaussian splat pomocí GPU P100, které jsou na kaggle k dispozici. Potom z toho udělej 3D soubor v plné kvalitě a druhý který bude mít cca ~70 MB a bude vhodný pro prohlížení na webu. (Všechny 3 soubory mají být jako výstupní artefakty toho notebooku) Každou verzi notebooku commitni a pushni do `main`. Pak spusť pomocí kaggle CLI a sleduj živě logy: použij "kaggle kernels logs --follow" a na konci &, ať se mrkneš jak to běží. Pokud bude problém s verzemi v kaggle, tak udělej únik do shellu a nainstaluj si tam co potřebuješ abys to spustil. Soubor 3D pro web pushni do `main` a k tomu udělej index.html, kde bude JS prohlížečka. V repu je zapnutý Github Pages, takže by to mělo být vidět, dej mi pak výslednou URL. Můj access_token pro kaggle je `KGAT_…` *(redacted)*

`kaggle/splat/` — COLMAP SfM then inria 3DGS. Two things had to be worked around: Kaggle's stock
PyTorch has no Pascal kernels (`sm_70`+, the P100 is `sm_60`), so the notebook probes
`nvidia-smi --query-gpu=compute_cap` before importing torch and reinstalls cu124 wheels if needed;
and training OOMed at iteration 11 700 on 16 GB, fixed with `expandable_segments`, `--data_device
cpu` and a lower densification budget. Result: 3 718 543 gaussians, `korno_full.ply` 922 MB and
`korno_web.splat` 73 MB (61.7 % of the gaussians carrying 98.4 % of screen coverage). GitHub Pages
turned out not to be enabled — the workflow failed with "Resource not accessible by integration".

## 2. Turning the splat into a textured mesh

> Funguje to pěkně, dík. Ty github pages jsem už zapnul. Teď bych potřeboval, aby se z toho splatu vyrobil normálně 3D model s texturami. Použij asi ten velký https://www.kaggle.com/code/pavelzbytovsk/korno-gaussian-splat/output?select=korno_full.ply vstup, jestli to udělá nějaký rozdíl. Mrkni se jak to vypadá a udělej to hezky. Aby tam nebyla ta mlha před kamerou ale bylo tam to podstatné - ta skála a vegetace na ní. Zase na to použij kaggle P100 nebo kaggle CPU, nevím co je lepší. A pak commitni výsledný model (možná udělej tři verze třeba 10MB, 50 MB a 100MB, a v index.html udělej jen rozcestník na tyhle tři verze a na tu původní (splat).

`kaggle/mesh/` on a CPU kernel: filter the floaters out of the 922 MB cloud, Poisson surface,
planar UV projection, KD-tree colour bake, three LODs at 9.8 / 47.6 / 95.4 MB. `index.html` became
a hub linking all four models.

## 3. Undo the hub

> vrať zpátky ten index, aby ukazovalo přímo ten splat. To nové tři budou jen v repu, ale bez odkazů odnikud.

Reverted; the meshes stayed in the repo unlinked.

## 4. v2 — dense MVS photogrammetry from scratch

> A teď bych rád zkusil úplně jiný přístup "v2" - úplně nový notebook na kaggle, který rovnou udělá photogrametrii přímo ze zdrojového datasetu https://www.kaggle.com/datasets/pavelzbytovsk/korno-rockface-photogrammetry . Nevyužívej žádné výstupy které vznikly doteď. Nepoužívej Gaussian splat. Celé to dělej ve složce v2 - tj v2/index.html bude obsahovat zase prohlížečku toho modelu a ještě bych tam chtěl mít checkbox, který ukáže pozice kamer/fotek. Chtěl bych velmi přesnou skálu včetně všech spár a hran. ve složce v2/kaggle budou ty notebooky atd.

> pokračuj

COLMAP SfM → `patch_match_stereo` → `stereo_fusion` → `poisson_mesher`. `patch_match_stereo` is
CUDA-only with no pip wheel, so COLMAP came from conda-forge via micromamba — a package with three
undeclared-dependency traps (`libfaiss` 1.10 specifically, `openblas` not MKL, `openimageio`), all
of which were found locally with `ldd` instead of burning kernel runs. A throwaway 8-image probe
notebook sized the real run. COLMAP's PoissonRecon segfaults at depth 13 *after* writing a complete
mesh, so the notebook validates the PLY byte-exactly against its own header and tolerates the
non-zero exit. 102/103 photos registered, 1.35 px error, 5.2 M dense points, 17 M triangles,
`korno_v2.glb` 90 MiB, 2 h 21 m on one P100.

## 5. Where heavy work belongs

> když to bude trvat moc dlouho, tak to dělej na kaggle, tento kontejner má malý výkon a běží vedle i produkce

## 6. Bolts

> Jo, zapiš si to do globálního. Stáhni si tohle repo https://github.com/zbycz/openclimbing-bolts-ai, prozkoumej a použij model onnx abys pro každou fotku z datasetu určil všechny její borháky. Udělej to zase přes kaggle CPU, můžeš se inspirovat openclimbing-bolts-ai/training/workspace/11_bolt_infer_v1_kaggle.py Stáhni pak json pro každou fotku a zkus použít projekci na 3D model. Chtěl bych zachovat natrénované 3D modely tak jak jsou a jen přidat tyhle metadata, je to možné?

Yes — the models are never touched, the bolts live in their own JSON the viewer overlays. Two CPU
kernels: YOLOv8-nano ONNX over native-resolution 1024 px tiles (152 detections), then a ray from
each camera through each detection onto the mesh, clustered across views. 113 candidates → 19 bolts
confirmed by ≥2 photos, 94 single-view rejects kept for review. Multi-view agreement is the real
filter: the detector is out of its training domain on drone frames (median score 0.30), so
thresholding on score alone would keep noise and drop real bolts. Proof the models were untouched:
`git rev-parse HEAD:v2/korno_v2.glb 4218f66:v2/korno_v2.glb` → identical hash.

## 7. Why parts of the texture are smudged

> A zkus mi mezitím ještě odpovědět proč je na tom 3D modelu místa kde je krásně vidět textura jak na fotce, a místa kde to je úplně zašmudlané? viz třeba tenhle screenshot. Jo a pak po celé délce té skály vede taková čára jakoby hrana, ale přitom v reálu tam není. A právě nad ní je to špatné rozlišení textury. Jen zkus zamyslet čím to je, zatím to nefixuj.

Measured rather than guessed: dense-point density drops ~16× above h ≈ 1.55 (the top edge of drone
coverage — the last camera row sits at h = 1.32–1.85 and there is nothing between 1.85 and 2.38).
The "edge" is that coverage boundary, sharpened by `stereo_fusion --min_num_pixels 4`. The texel is
0.00184 units, so where points are 0.0029 apart the texture is sharp and where they are 0.011 apart
one photo pixel is smeared over six texels. Secondary cause: the planar UV projection stretches on
steep faces (17 % of triangles > 45°, 4 % > 60°). Full write-up and the fix in
[§ Texture sharpness](#texture-sharpness) below.

## 8. Bolt toggle and mouse inversion

> víš co, udělej mi v té prohlížečce ještě přepínač na zobrazení těch 19ti vs. zobrazení všech 94 a ještě tam dej checkbox na inverzi obou os pro ovládání myší

## 9. Climbing routes from the production database

> Stáhni si produkční databázi přes POST https://openclimbing.org/api/climbing-tiles/export a najdi si tam group s názvem Korno. Ta má na sobě několik fotek. Tak jedna z těch fotek by měla bejt, ta první by měla bejt právě totožná s jednou fotkou tady z dronu. Tak zkus to najít, která to je, nevím, jestli ti to povede. Zkus to asi podle obsahu, abys našel úplně, která je to, která je to přesně fotka. No a potom tam vede, na ní jsou zakreslený vlastně cesty lezecké na nodech. Takže máš už souřadnice v rámci tý fotky pro každou cestu. A teď bych chtěl, abys udělal vlastně další checkbox, kterej vykreslí ty zakreslený cesty, naprojektuje na to 3Dčko a vykreslí je přes to 3D. pro referenci si případně checkoutni repo https://github.com/jvaclavik/openclimbing

`Tomáškův lom - Korno.jpg` = `DJI_20260811211418_0016_D.JPG`, established by SIFT + RANSAC with
4000 inliers (the feature cap — effectively every feature matched) against 1861 for the next best.
The other four Commons photos are ground-level phone shots and correctly matched nothing (9
inliers). Keypoints are normalised to [0,1] before fitting, so the homography maps fractions of one
image to fractions of the other and the route paths map straight through it. 13 routes projected.
The path format needed openclimbing's own `pathUtils.ts`: a trailing letter on `y` is the point
type, a trailing colon means the next segment is dotted — 13 points here end in `A`.

## 10. Viewer polish

> je to super, ještě udělej aby šlo ten menu s nápisem název a ty check boxy tak aby šlo nějak sklopit třeba nahoře může být taková ta šipka směrem dolů která se pak po kliknutí jako zabalí směrem nahoru / oprav velikost stahování – ten progres bar ukazuje celkem 60 mega, ale stahuje se asi 90 / A ještě udělej ty čáry na cesty dvakrát tlustší a udělej checkbox který schová názvy

The progress bar was right and the header was wrong: Pages gzips the GLB, so the browser counts
decompressed bytes against a compressed `content-length`. Hardcoded the real sizes. Line thickness
needed `Line2`/`LineMaterial` — `LineBasicMaterial.linewidth` is ignored on almost every platform.

## 11. Keep the ropes out of the rock, colour them like openclimbing

> Ještě uprav, aby ty cesty, ty vlastně jakoby lana nešly skrz skálu. Někde ti, někde je vidět, že to jako je nad povrchem, ale někde se ti to prostě dostane pod povrch. Tak to uprav, aby to prostě v tom místě přidej třeba body, aby ti to šlo prostě pěkně nad povrchem. A ty barvy těch čar udělej podle toho, jak jsou udělaný v tom repozitáři OpenClimbing. to tam najdeš

The problem was not the points but the *chords* between them: a route has 2–6 points and a straight
line across a bulge runs inside the wall. Now the path is resampled every 0.25 % of the photo before
casting, lifted along the mesh normal and verified by casting back from the camera (anything still
occluded gets its lift doubled), then simplified with a tolerance below the lift. Measured: the old
lines cut up to 94 cm into the rock. Colours come from openclimbing's `gradeData.ts`.

## 12. One solid stroke, labels that lay themselves out

> Ještě oprav dvě věci. Jedna, že ta čára je teď pěkná, ale má správnou barvu, ale je jakoby obarvená jakože po segmentech a pak tam je vždycky takovej černej okraj vidět, takže to je takový rozkouskovaný. A možná by to bylo hezčí, když by to byla selistvá ta barva. No a druhá věc, že jak se udělaly ty popisky, tak zkus tam nechat jenom ty obtížnosti a aby se zobrazovaly vlastně až tehdy, když bude celej label vidět, jako spočítat nějaký kolizní boxy a tak, aby se prostě vykreslily třeba jenom některý a postupně, když člověk zoomuje, tak aby se jako ukázaly všechny. A když nazoomuje ještě dál a bude na to místo, tak se vypíše i ten název, ale ten se normálně vypisovat nebude. Vlastně to by se muselo nazoomovat fakt hodně, aby se to tam vešlo.

Three causes, each isolated with an A/B render: too few points (denser simplification), surface
noise (two binomial smoothing passes + Catmull-Rom), and — the actual "chopping" — the outline
z-fighting with the line it outlines. Labels moved from sprites to a 2D overlay with real collision
boxes.

## 13. Defaults, unbroken lines, pinned clickable labels

> udělej defaultně zapnuté cesty, popisky ne.
>
> když to přiblížím, tak ty cesty jsou rozkouskované na 1000 malých kousíčků, líbilo by se mi kdyby to bylo stále v kuse.
>
> A když zapnu popisky tak chci aby zůstavaly pouze u dolního okraje cesty, nesmí se pohybovat.
>
> A ještě kliknutím na label by šlo otevřít openclimbing.org/node/XXX

`polygonOffset` had only papered over the z-fighting — the depth error grows with zoom, so the
dashes came back. Replaced with scaling the outline group about the camera each frame: every point
moves straight away from the eye, so the screen position is unchanged but there is real distance
between the strokes. Labels now hang off a fixed world point (the foot of the route) and open
`openclimbing.org/node/<osmId>` on click, with a drag guard so an orbit ending on a label does not
navigate.

## 14. This file

> udělej shrnutí celé konverzace do CONVERSATION.md - pěkně po kapitolách, zachovej celé prompty verbatim. Odpovědi shrň do pár vět.
>
> A pak všechny poznatky do CLAUDE.md. Pushni do main (zkontroluj že tam neleakuje token).
>
> A ještě mi sem napiš, proč je ta textura od toho 1.5m taková rozmazaná a jestli by to šlo nějak řešit.

## 15. Fixing the smudge

> ok, vyřeš to nějak nejlíp jak to půjde. Klidně spusť i nový běh na kaggle kdyby to bylo potřeba.
>
> Zachovej původní glb fily v repu, vyrob třeba nové korno_v2_desmudged.glb abych si to mohl v prohlížečce porovnat.

conda-forge has neither `openmvs` nor `mvs-texturing`, and building one to find out is a gamble worth
a kernel run, so the bake was written directly: `v2/kaggle/texture/` reproduces the geometry and the
planar UV frame byte-for-byte and replaces only where the colour comes from. Every texel is ray-cast
to its surface point, projected into all 102 posed photos, filtered against a per-photo depth map,
scored by sampling density and sampled from the original 8000 × 4500 pixels; the best three views are
blended narrowly and per-photo exposure gains are fitted against the old bake. 31 min on a CPU kernel.
Local contrast went up **7.4×** overall — 9–20× in the upper bands, and still 4–7× at the foot,
because blending six nearest points was softening the whole wall rather than only the top. The viewer
gained a picker so both bakes can be compared over the same geometry.

---

## Texture sharpness

The full answer to chapter 7, what fixed it and what is still open, is in
[`CLAUDE.md`](CLAUDE.md#why-the-texture-was-smudged-and-what-fixed-it).
