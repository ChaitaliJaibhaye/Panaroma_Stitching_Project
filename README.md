# 🖼️ Panorama Image Stitcher — CV + Data Science Edition

A production-quality panorama stitching application built with **Python, OpenCV, Streamlit, and scikit-learn**. Upload two or more overlapping photos of the same scene and get back a seamless, automatically-cropped panorama — plus a full **quality dashboard** and an **ML-based success predictor**, comparable to the panorama mode on a modern smartphone with a data-science layer on top.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Installation](#installation)
5. [Running the App](#running-the-app)
6. [Command-Line Interface](#command-line-interface)
7. [How It Works (Algorithm)](#how-it-works-algorithm)
8. [Quality Dashboard & Confidence Score](#quality-dashboard--confidence-score)
9. [ML Success-Prediction Pipeline](#ml-success-prediction-pipeline)
10. [Configuration Options](#configuration-options)
11. [Error Handling](#error-handling)
12. [Screenshots](#screenshots)
13. [Limitations](#limitations)
14. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a full panorama-stitching pipeline from first principles using classical computer vision (no deep learning required for the stitching itself):

`Cylindrical Projection (optional) → Feature Detection → Automatic Ordering → Feature Matching → Homography Estimation (RANSAC) → Warping → Exposure Compensation → Blending → Auto-Cropping → Quality Scoring → ML Success Prediction`

It ships with:
- A **Streamlit web UI** (`app.py`) with drag-and-drop upload, live progress, debug visualizations, an interactive zoom/pan panorama viewer, a **quality metrics dashboard**, an **ML success-prediction panel**, and one-click download of both the panorama and a JSON stitching report.
- A **command-line interface** (`cli.py`) for scripted / batch use, with an optional `--report` flag to emit the same JSON quality/ML report.
- A clean, modular codebase (`stitcher.py`, `feature_matching.py`, `blending.py`, `cylindrical_projection.py`, `quality_metrics.py`, `ml_predictor.py`, `utils.py`) that is easy to read, extend, and test independently of the UI.
- A lightweight **RandomForest classifier** (`ml_predictor.py`) that predicts stitch success probability from pre-stitch metrics, trained on a synthetic-but-principled dataset and cached to `models/panorama_success_model.pkl`.

---

## Features

**Core requirements**
- ✅ Upload 2+ images (JPG / JPEG / PNG) with thumbnail previews
- ✅ SIFT feature detection (auto-falls back to ORB if SIFT is unavailable)
- ✅ BFMatcher (ORB) / FLANN (SIFT) matching + Lowe's ratio test
- ✅ Homography estimation with `cv2.findHomography` + RANSAC
- ✅ Perspective warping onto a shared canvas (`cv2.warpPerspective`)
- ✅ Seamless blending — **multi-band (Laplacian pyramid, default)** or **feather** blending, user-selectable
- ✅ Automatic detection & cropping of black borders
- ✅ Modern UI: upload button, stitch button, progress bar, status log, preview, download button
- ✅ Robust error handling with user-friendly messages

**Bonus features implemented**
- ✅ Automatic image ordering (no need to upload in left-to-right order) — with a human-readable "Detected image order: Image3 → Image1 → Image2" display
- ✅ Cylindrical projection (optional, own module, automatic focal-length estimation with a safe fallback)
- ✅ Simple exposure/brightness compensation between overlapping images
- ✅ Progress indicator with live status log
- ✅ Drag-and-drop upload (native to Streamlit's file uploader)
- ✅ Interactive zoom & pan viewer for the final panorama (scroll to zoom, drag to pan)
- ✅ Debug mode: view feature-match visualizations and intermediate warped images
- ✅ Save intermediate debug images to disk (`outputs/debug/`)
- ✅ Batch panorama generation via CLI (`--batch-input`)
- ✅ Full command-line interface
- ✅ Graceful CPU fallback when CUDA-enabled OpenCV isn't present

**Data-science / ML layer (new)**
- ✅ **Quality Dashboard**: images uploaded, processing time, avg. keypoints, avg. good matches, total RANSAC inliers, avg. inlier ratio, estimated overlap %, panorama resolution, blend method, projection on/off
- ✅ **Confidence Score** (0–100, weighted: 40% inlier ratio / 30% match count / 20% overlap / 10% sharpness) with a ★-to-☆ star rating (Excellent → Failed)
- ✅ **ML Success Predictor**: a RandomForestClassifier trained on a synthetic-but-heuristic-grounded dataset predicts stitch success probability and expected quality from 7 pre-stitch features, with a low-confidence warning surfaced in the UI
- ✅ **Downloadable JSON stitching report** combining all of the above (`app.py` download button, or `cli.py --report`)

---

## Folder Structure

```
panorama_project/
│
├── app.py                     # Streamlit web application (UI + dashboard + ML panel)
├── stitcher.py                 # Core pipeline: PanoramaStitcher orchestration class
├── feature_matching.py          # Feature detection (SIFT/ORB) + matching + Lowe's ratio test
├── blending.py                  # Feather blending & multi-band (Laplacian pyramid) blending
├── cylindrical_projection.py     # Cylindrical warp + automatic focal-length estimation
├── quality_metrics.py            # Confidence score, star rating, JSON quality report
├── ml_predictor.py                # RandomForest success predictor (synthetic training data)
├── utils.py                       # Validation, cropping, exposure comp, sharpness, I/O helpers
├── cli.py                         # Command-line interface (single + batch modes, --report)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── models/                          # Cached ML model (panorama_success_model.pkl)
├── sample_images/                   # Example overlapping images to try the app with
└── outputs/                         # Default location for saved panoramas, reports & debug images
```

---

## Installation

### 1. Prerequisites
- Python 3.9+
- pip

### 2. Clone / copy the project
Place all files in a folder named `panorama_project/` (matching the structure above).

### 3. Create a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
opencv-contrib-python>=4.8.0
numpy>=1.24.0
streamlit>=1.32.0
Pillow>=10.0.0
scikit-image>=0.22.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

> **Note:** `opencv-contrib-python` (not plain `opencv-python`) is required so that `cv2.SIFT_create()` is available. If SIFT cannot be created for any reason, the app automatically falls back to ORB — no code changes needed. `scikit-learn` + `joblib` power the ML success predictor; if either is missing, the app detects this and simply disables that panel rather than crashing.

---

## Running the App

From inside the `panorama_project/` folder:

```bash
streamlit run app.py
```

This opens the app in your browser (default: `http://localhost:8501`). Then:

1. **Upload** 2 or more overlapping JPG/PNG images.
2. Adjust options in the sidebar if desired (blending method, cylindrical projection, exposure compensation, debug mode, working resolution).
3. Click **🧵 Stitch Panorama**.
4. Watch the progress bar and status log.
5. Review the **Detected image order**, the **Quality Dashboard** (confidence score, star rating, keypoints/matches/inliers/overlap), and the **ML Success Prediction** panel (success probability + expected quality, with a warning if confidence is low).
6. Explore the result: feature-match debug view (if enabled), the warped images before blending, and the final panorama in the interactive zoom/pan viewer.
7. Click **⬇️ Download Panorama (PNG)** and/or **⬇️ Download Stitching Report (JSON)** to save your results.

---

## Command-Line Interface

Stitch a single folder of images:
```bash
python cli.py --input sample_images --output outputs/panorama.png
```

Batch mode — every sub-folder of `--batch-input` becomes its own panorama:
```bash
python cli.py --batch-input path/to/many_scenes/ --output-dir outputs/
```

Useful flags:
```
--blend {multiband,feather}     Blending method (default: multiband)
--cylindrical                   Enable cylindrical projection
--no-exposure-comp              Disable brightness/exposure compensation
--no-auto-order                 Stitch images in the given (not auto-detected) order
--max-dimension INT             Max working resolution in pixels (default: 1600)
--debug                         Save intermediate feature-match debug images
--report                        Also save a JSON quality/ML stitching report next to each output
```

---

## How It Works (Algorithm)

1. **Pre-processing** — Images are resized so their largest dimension does not exceed a configurable limit (keeps runtime reasonable). Optionally, each image is projected onto a **cylindrical surface** (`cylindrical_projection.py`) using an automatically estimated focal length (derived from image width + an assumed field of view, with a safe width-based fallback if estimation is degenerate) to reduce perspective distortion for wide, rotation-based panoramas. All images in a set share one focal length so the projection stays geometrically consistent across the sequence.

2. **Feature Detection** — For every image, keypoints and descriptors are extracted using **SIFT** (scale/rotation invariant, sub-pixel accurate). If the installed OpenCV build lacks SIFT, the pipeline transparently switches to **ORB** (fast, binary descriptors, patent-free).

3. **Automatic Ordering** *(optional)* — When more than 2 images are uploaded, a pairwise "good match count" similarity matrix is built between every pair of images (an *O(n²)* comparison, practical for the supported 2–10 image range). A chain is grown greedily from the strongest pair outward — at each step, the unused image with the strongest connection to either end of the current chain is attached there — producing a sensible stitching order regardless of upload order. The result is surfaced in the UI as, e.g., `Detected image order: Image3 → Image1 → Image2 → Image5`.

4. **Feature Matching** — Descriptors between adjacent images (in stitching order) are matched with **FLANN** (for SIFT's float descriptors) or **BFMatcher with Hamming distance** (for ORB's binary descriptors). **Lowe's ratio test** (default threshold 0.75) discards ambiguous matches, keeping only distinctive, reliable correspondences.

5. **Homography Estimation** — `cv2.findHomography()` with **RANSAC** computes the 3×3 projective transform between each adjacent image pair, automatically rejecting outlier matches. The inlier ratio is reported so you can judge match quality.

6. **Homography Chaining** — Rather than warping everything relative to one edge image (which maximizes distortion), all per-pair homographies are composed into a single **reference frame anchored at the middle image**, minimizing cumulative geometric distortion across the panorama.

7. **Canvas Computation & Warping** — The transformed corners of every image are used to compute the bounding box of the final canvas and a translation that keeps all coordinates positive. Each image is then warped onto this shared canvas with `cv2.warpPerspective()`.

8. **Exposure Compensation** *(optional)* — Before blending each new image in, its brightness is compared to the already-built panorama in their overlapping region, and a clamped gain correction is applied so that stitched photos taken in slightly different lighting don't produce a visible brightness seam.

9. **Blending** — Two strategies are available:
   - **Feather blending**: per-pixel weights derived from a distance transform (how far a pixel is from the edge of its image's content), producing a smooth linear cross-fade in the overlap region.
   - **Multi-band (Laplacian pyramid) blending**: images are decomposed into Gaussian/Laplacian pyramids; low frequencies (broad lighting/color) are blended over a wide region while high frequencies (texture/edges) are blended more narrowly. This avoids both hard seams and blurring/ghosting of fine detail — the technique used in most professional stitching software.

10. **Auto-Cropping** — The final canvas nearly always has ragged black borders where no image data was warped. These are automatically detected via contour analysis on a binarized non-black mask and cropped away so the output panorama is a clean rectangle of useful pixels only.

---

## Quality Dashboard & Confidence Score

After every stitch, `quality_metrics.py` assembles a `QualityReport` from measurements gathered during the run:

| Metric | Meaning |
|---|---|
| Images uploaded | Count of input photos |
| Processing time | Wall-clock seconds for the full pipeline |
| Avg. keypoints | Mean detected keypoints per image |
| Avg. good matches | Mean ratio-tested matches per adjacent pair |
| RANSAC inliers (total) | Sum of inlier correspondences across all pairs |
| Avg. inlier ratio | Mean fraction of matches RANSAC accepted |
| Estimated overlap | Mean % of shared content area between adjacent warped images |
| Panorama resolution | Final cropped output size |
| Blend method / Projection | Configuration used for this run |

These roll up into a single **Confidence Score** (0–100), weighted as:

```
Confidence Score = 40% × inlier_ratio + 30% × match_count(normalized)
                  + 20% × overlap% + 10% × sharpness(normalized)
```

...which maps to a star rating: **★★★★★ Excellent (90+)**, **★★★★ Good (75+)**, **★★★ Average (55+)**, **★★ Poor (35+)**, **★ Failed (<35)**. The dashboard also flags specific warnings (low inlier ratio, low overlap, blurry images, large brightness gaps) and the full report — including all raw metrics — can be downloaded as JSON from the app or written to disk with `cli.py --report`.

---

## ML Success-Prediction Pipeline

`ml_predictor.py` adds a small, self-contained ML layer on top of the classical CV pipeline, intended to demonstrate an end-to-end data-science workflow (feature engineering → training → persistence → inference) rather than to replace the deterministic quality score above:

1. **Feature extraction** — 7 numeric features per stitch job: `num_images`, `avg_keypoints`, `avg_good_matches`, `overlap_percent`, `sharpness_score` (variance of Laplacian), `brightness_diff` (max−min mean brightness across inputs), `avg_inlier_ratio`. These are exactly the same measurements used by the Confidence Score, so no extra computation is needed at inference time.
2. **Synthetic training data** — Since no labeled real-world "success/failure" dataset exists for this task, `_generate_synthetic_dataset()` samples ~1,200 plausible feature vectors and assigns a binary label via a noisy, domain-grounded heuristic (more matches / higher inlier ratio / more overlap / sharper images / smaller brightness gaps ⇒ more likely to succeed; more images in the set ⇒ mildly more likely to fail, since one weak link can break the chain). Gaussian noise is added so the model must learn a real decision boundary instead of memorizing a threshold.
3. **Model** — A `RandomForestClassifier` (200 trees, `class_weight="balanced"`) is trained on an 80/20 split and persisted with `joblib` to `models/panorama_success_model.pkl`, so training only happens once per environment.
4. **Inference** — At runtime, `SuccessPredictor.predict()` loads the cached model, extracts the 7 features from the completed stitch's `PipelineMetrics`, and returns a success probability + expected-quality label (`Excellent`/`Good`/`Average`/`Poor`/`Failed`). If the probability is below 55%, a warning is surfaced in the UI suggesting concrete fixes (more overlap, sharper photos, closer exposures).
5. **Graceful degradation** — If `scikit-learn`/`joblib` aren't installed, `SuccessPredictor` detects this and reports the panel as unavailable instead of crashing the rest of the app.

---

## Configuration Options

| Option | Where | Description |
|---|---|---|
| Blending method | Sidebar / `--blend` | `multiband` (recommended, smoothest, default) or `feather` (faster) |
| Cylindrical projection | Sidebar / `--cylindrical` | Reduces distortion for wide, many-image rotational panoramas; auto-estimates focal length |
| Exposure compensation | Sidebar / `--no-exposure-comp` | Corrects brightness differences between photos |
| Automatic ordering | Sidebar / `--no-auto-order` | Figures out stitching order automatically; shown as "Detected image order: ..." |
| Debug mode | Sidebar / `--debug` | Shows/saves feature-match visualizations |
| Max working resolution | Sidebar / `--max-dimension` | Speed/quality trade-off |
| JSON report | Download button / `--report` | Saves the full quality + ML report alongside the panorama |

---

## Error Handling

The application gracefully detects and reports, with user-friendly messages instead of crashing:

- Fewer than 2 images uploaded
- Corrupted or unreadable image files
- Unsupported file types
- Images that don't overlap enough (too few reliable matches)
- Homography estimation failure (degenerate geometry / too few RANSAC inliers)
- Images of very different sizes (handled naturally via independent resizing before warping)

---

## Screenshots

> _Add screenshots of your running app here, e.g.:_
>
> `docs/screenshot_upload.png` — Upload screen with thumbnails
> `docs/screenshot_debug.png` — Feature-match debug view
> `docs/screenshot_result.png` — Final panorama with zoom/pan viewer

---

## Limitations

- Classical feature-based stitching struggles with **scenes lacking texture** (blank walls, clear sky) since too few reliable keypoints can be found.
- Works best for **static scenes**; moving objects (people, cars) can produce ghosting artifacts in overlap regions.
- Homography-based warping assumes either a **rotating camera** or a **planar scene** — panoramas of scenes with significant depth/parallax (close-up objects at varying distances) may show local misalignment.
- GPU acceleration depends on OpenCV being built with CUDA support; the default `opencv-contrib-python` wheel from PyPI does **not** include CUDA, so the app automatically runs on CPU. Users who build OpenCV with CUDA support can extend `feature_matching.py` to use `cv2.cuda` equivalents.
- Very large panoramas (many high-resolution images) can require significant RAM during pyramid blending; use the "Max working resolution" setting to manage this.

---

## Future Improvements

- Deep-learning-based feature matching (e.g., SuperGlue/LoFTR) for scenes with low texture or large viewpoint changes.
- Bundle adjustment across all images simultaneously (rather than sequential pairwise chaining) for large panoramas, as done in tools like Hugin/AutoStitch.
- Seam-optimal cutting (graph-cut based) prior to blending, in addition to the current feather/multi-band approaches.
- Train the ML success predictor on real, labeled stitching outcomes (rather than synthetic data) once such a dataset is collected — the feature-extraction and persistence pipeline already supports a drop-in replacement.
- True GPU-accelerated feature detection/matching when CUDA-enabled OpenCV is available.
- 360° / spherical panorama support for full-rotation photo sets.
- Maximum-spanning-tree (rather than greedy chain) image ordering for very irregular, non-linear photo sets.

---

Built with ❤️ using Python, OpenCV, NumPy, and Streamlit.
