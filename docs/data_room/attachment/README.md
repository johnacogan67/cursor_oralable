# Attachment (data room)

**As at:** 3 Oct 2026 · Pack **1.1.74**  
**Canonical spec:** [../../ATTACHMENT.md](../../ATTACHMENT.md)  
**Agent topic:** `attachment`

Housing and skin attach for the MAM. Working model: ANR M40 pad-and-snap class. PCB / BOM / charge case stay [hardware/](../hardware/). Charge path: MAM RX coil aligned with case TX coil on this build. A magnetic dock is a possible later case.

**Print now:** Gen1 cavity (lab) and Gen2 cavity (October boards). **Do not** rewrite Beacon kit-ship as Gen2 until charge-to-temple clears.

**Beacon loaner ops (Gen1/Gen2):** charge bare MAM → seat in housing → one night → return to Beacon → remove → charge again. Patient does not take the charge case. Detail: [ATTACHMENT.md](../../ATTACHMENT.md) §0b.

**Gen4:** combined MAM + ANR-class EMG, **USB-C on the pod** — concept, not a BOM generation. USB-C is the intended fix for the remove / reseat / coil-align loop (charge the closed pod at clinic). Do not wait for Gen4 to start Beacon.

## Measured dimensions

Units **mm**. Gen1 numbers are an STL axis-aligned box. Gen2 numbers are from the REV11 STEP: **do not** use the full-board box as the temple cavity.

| Item | Value | Source |
|------|-------|--------|
| M40 length | TBD | Calipers |
| M40 width | TBD | Calipers |
| M40 height | TBD | Calipers |
| Snap-to-snap pitch | TBD | Calipers |
| Snap hole ID | TBD | Calipers |
| Gen1 MAM L×W×H | **26.24 × 7.26 × 5.98 mm** (mesh Y × X × Z) | [attachment_gen1_MAM.stl](./attachment_gen1_MAM.stl) AABB · 40 464 triangles. Mesh is not at origin. |
| MAXM86161 window L×W and offset | TBD Gen1 · REV11 U2 ~ **(+10.8, 0)** mm | Calipers / [PCB00003_GEN2_REV11_HARDWARE.md](../../PCB00003_GEN2_REV11_HARDWARE.md) |
| Coil OD | TBD | Calipers · STEP has `L1` / `F_copper_coil` |
| Gen2 clip island (AABB only) | **~32 × 12 mm** in XY (X −13.7 → +18.6). **Not** the cavity outline. | [PCB00003-TGM-STEP-REV11.step](./PCB00003-TGM-STEP-REV11.step) clip band |
| Gen2 cavity | **Conformal** to clip solids: BAT1, U2, U5, L1/coil, PCB (+0.2–0.4 mm). Not a rectangle. | Boolean cut from same STEP |
| Gen2 full STEP (do not pocket) | **82.2 × 17.5 × 22.1 mm** | Same file: clip + flex + charger half (~47–69 mm) |
| Gen2 v4 bottom shell | **48 × 11 × 6.0 mm** | [FIG-CO-080](../../figures/FIG-CO-080-gen2-v4-bottom-two-views.png) · STL `mam_gen2_version4_latest_for_cursor` (not in git). Thin bar, not the M40 outer |
| Gen2 v4 bottom holes | **Ø4.00 mm**, 3.0 mm from each end, **42.0 mm** apart, on the centre line. Bore open through the underside; wall rises 4.0 mm | Same figure |
| Gen2 v4 top shell | **48 × 11 × 4.5 mm** | [FIG-CO-081](../../figures/FIG-CO-081-gen2-v4-top-two-views.png) |
| Gen2 v4 top square | **6 × 6 mm**, centred on the width, 7 mm in from one end | Same figure. No Ø4 holes on the lid |

Gen1 STL is PCB + cell + coil, so larger than the PCB-only “~20×7×4 mm” class in the Altium brief. Gen2 clip is a bit longer and wider than Gen1 (LP260820 pouch + U2 still near +11 mm while BAT1/U5 sit on negative X). STEP Z ~22 mm is both CAD faces — cut the cavity from the imported clip **solids** (conformal), not from the AABB box.

## CAD

| File | What |
|------|------|
| [attachment_gen1_MAM.stl](./attachment_gen1_MAM.stl) | Gen1 MAM solid for Fusion `MAM_gen1` (binary STL) |
| [PCB00003-TGM-STEP-REV11.step](./PCB00003-TGM-STEP-REV11.step) | Gen2 REV11 PCB assembly (Open CASCADE, mm). Fusion `MAM_gen2`: import, hide charger half, **boolean-cut clip solids** into a conformal cavity (+0.2–0.4 mm). |

Housing `.f3d` / ventral+dorsal STLs stay out of git until the M40 outer is measured. Recipe: [ATTACHMENT.md](../../ATTACHMENT.md) §3.

## Figures

| ID | Path |
|----|------|
| FIG-CO-058 | [../../figures/FIG-CO-058-anr-m40-label-pads-mam.jpg](../../figures/FIG-CO-058-anr-m40-label-pads-mam.jpg) |
| FIG-CO-059 | [../../figures/FIG-CO-059-ppg-window-mark.jpg](../../figures/FIG-CO-059-ppg-window-mark.jpg) |
| FIG-CO-060 | [../../figures/FIG-CO-060-red-dot-pad-gap.jpg](../../figures/FIG-CO-060-red-dot-pad-gap.jpg) |
| FIG-CO-061 | [../../figures/FIG-CO-061-anr-exploded-pads-shell-mam.jpg](../../figures/FIG-CO-061-anr-exploded-pads-shell-mam.jpg) |
| FIG-CO-062 | [../../figures/FIG-CO-062-mam-between-snaps.jpg](../../figures/FIG-CO-062-mam-between-snaps.jpg) |
| FIG-CO-063 | [../../figures/FIG-CO-063-gen2-housing-concept.svg](../../figures/FIG-CO-063-gen2-housing-concept.svg) · [PNG](../../figures/FIG-CO-063-gen2-housing-concept.png) — PPG window **offset to U2 (~+10.8 mm)**, not mid-capsule · data-room symlink `figures/FIG-CO-063-*` |
| FIG-CO-064 | [../../figures/FIG-CO-064-gen2-conformal-cavity-3d.png](../../figures/FIG-CO-064-gen2-conformal-cavity-3d.png) — 3D conformal cavity (BAT1 / coil / U2 pockets) · data-room symlink `figures/FIG-CO-064-*` |
| FIG-CO-079 | [../../figures/FIG-CO-079-gen2-v4-housing-shells.png](../../figures/FIG-CO-079-gen2-v4-housing-shells.png) — v4 STL, both shells · data-room symlink `figures/FIG-CO-079-*` |
| FIG-CO-080 | [../../figures/FIG-CO-080-gen2-v4-bottom-two-views.png](../../figures/FIG-CO-080-gen2-v4-bottom-two-views.png) — bottom, looking down and from underneath · two Ø4 holes |
| FIG-CO-081 | [../../figures/FIG-CO-081-gen2-v4-top-two-views.png](../../figures/FIG-CO-081-gen2-v4-top-two-views.png) — top, looking down and from underneath · 6×6 mm square |
| FIG-CO-065 | [../../figures/FIG-CO-065-gen2-mam-pcb.jpg](../../figures/FIG-CO-065-gen2-mam-pcb.jpg) — bare Gen2 MAM PCB |
| FIG-CO-066 | [../../figures/FIG-CO-066-gen1-vs-gen2-mam.jpg](../../figures/FIG-CO-066-gen1-vs-gen2-mam.jpg) — Gen1 with cell vs bare Gen2 |
| FIG-CO-067 | [../../figures/FIG-CO-067-gen2-housing-vs-anr-m40.jpg](../../figures/FIG-CO-067-gen2-housing-vs-anr-m40.jpg) — first print (thin bar) vs ANR M40 · same Red Dots |
| FIG-CO-068 | [../../figures/FIG-CO-068-gen2-housing-scale-stack.jpg](../../figures/FIG-CO-068-gen2-housing-scale-stack.jpg) — boards + housing + ANR |
| FIG-CO-076 | [../../figures/FIG-CO-076-scale-stack-plus-sunrise.jpg](../../figures/FIG-CO-076-scale-stack-plus-sunrise.jpg) — copy of 068 + Sunrise pod · size peer, not housing |
| FIG-CO-077 | [../../figures/FIG-CO-077-scale-stack-plus-acupebble.jpg](../../figures/FIG-CO-077-scale-stack-plus-acupebble.jpg) — copy of 068 + AcuPebble disc · attach class, not AHI |
| FIG-CO-078 | [../../figures/FIG-CO-078-scale-stack-four-extraoral.jpg](../../figures/FIG-CO-078-scale-stack-four-extraoral.jpg) — Oralable · ANR · AcuPebble · Sunrise |
| FIG-CO-069 | [../../figures/FIG-CO-069-red-dot-snap-housing.jpg](../../figures/FIG-CO-069-red-dot-snap-housing.jpg) — snap face + housing strip |
| FIG-CO-070 | [../../figures/FIG-CO-070-matisse-photo-gen2-housing-wear.png](../../figures/FIG-CO-070-matisse-photo-gen2-housing-wear.png) — hybrid wear (line face + photo housing) |
| FIG-CO-082 | [../../figures/FIG-CO-082-matisse-photo-gen2-housing-9oct.png](../../figures/FIG-CO-082-matisse-photo-gen2-housing-9oct.png) — same line face · 9 Oct closed housing |
| FIG-CO-071 | [../../figures/FIG-CO-071-matisse-photo-gen1-gen2-mam.png](../../figures/FIG-CO-071-matisse-photo-gen1-gen2-mam.png) — hybrid boards |
| FIG-CO-072 | [../../figures/FIG-CO-072-matisse-photo-housing-vs-anr.png](../../figures/FIG-CO-072-matisse-photo-housing-vs-anr.png) — hybrid class match · not the Beacon box |
