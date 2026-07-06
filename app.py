
from __future__ import annotations

import base64
import time
from typing import List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from ml_predictor import PredictionResult, SuccessPredictor
from quality_metrics import QualityReport, build_quality_report
from stitcher import PanoramaStitcher, StitchingError
from utils import (
    ImageValidationError,
    bgr_to_rgb,
    bytes_to_image,
    encode_image_to_bytes,
    make_thumbnail,
    validate_image_file,
)


@st.cache_resource(show_spinner=False)
def load_success_predictor() -> SuccessPredictor:
    """Load (training + caching on first call) the ML success predictor.

    Cached as a Streamlit resource so the RandomForest model is trained
    only once per server process, not on every script rerun.

    Returns:
        A ready-to-use `SuccessPredictor`.
    """
    return SuccessPredictor()

st.set_page_config(
    page_title="Panorama Stitcher",
    page_icon="🖼️",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .status-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initialize Streamlit session state keys used across reruns."""
    defaults = {
        "images": [],       # list of (filename, BGR np.ndarray)
        "result": None,     # last StitchResult
        "report": None,     # last QualityReport
        "prediction": None, # last ML PredictionResult
        "error": None,      # last error message, if any
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Render the page title and description."""
    st.markdown('<div class="main-title"> Panorama Image Stitcher</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload overlapping photos of the same scene and '
        "generate a seamless, high-quality panorama — right in your browser.</div>",
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render sidebar controls and return the chosen configuration.

    Returns:
        Dictionary of stitching configuration options.
    """
    st.sidebar.header("⚙️ Stitching Options")

    blend_method = st.sidebar.selectbox(
        "Blending method",
        options=["multiband", "feather"],
        help="Multi-band blending gives the smoothest results for most "
        "scenes. Feather blending is faster and simpler.",
    )

    use_cylindrical = st.sidebar.checkbox(
        "Cylindrical projection",
        value=False,
        help="Useful for wide panoramas captured by rotating the camera. "
        "May reduce quality for simple 2-3 image side-by-side shots.",
    )

    exposure_compensation = st.sidebar.checkbox(
        "Exposure compensation",
        value=True,
        help="Automatically corrects brightness differences between photos.",
    )

    auto_order = st.sidebar.checkbox(
        "Automatic image ordering",
        value=True,
        help="Automatically figures out the left-to-right order of your "
        "photos instead of using upload order.",
    )

    debug_mode = st.sidebar.checkbox(
        "Debug mode (show feature matches)",
        value=False,
    )

    st.sidebar.markdown("---")
    max_dimension = st.sidebar.slider(
        "Max working resolution (px)",
        min_value=800,
        max_value=3000,
        value=1600,
        step=100,
        help="Larger values preserve more detail but are slower.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Algorithm: SIFT/ORB features → BFMatcher/FLANN + Lowe's ratio "
        "test → RANSAC homography → warp → blend → auto-crop."
    )

    return {
        "blend_method": blend_method,
        "use_cylindrical": use_cylindrical,
        "exposure_compensation": exposure_compensation,
        "auto_order": auto_order,
        "debug": debug_mode,
        "max_dimension": max_dimension,
    }


def render_upload_section() -> None:
    """Render the file uploader and thumbnail previews."""
    st.subheader("1. Upload Images")
    uploaded_files = st.file_uploader(
        "Drag and drop or browse for 2+ overlapping images (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    images: List[tuple] = []
    errors: List[str] = []

    if uploaded_files:
        for file in uploaded_files:
            file_bytes = file.getvalue()
            try:
                validate_image_file(file.name, file_bytes)
                image = bytes_to_image(file_bytes)
                images.append((file.name, image))
            except ImageValidationError as exc:
                errors.append(str(exc))

    st.session_state["images"] = images

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")

    if images:
        st.caption(f"{len(images)} valid image(s) loaded.")
        cols = st.columns(min(len(images), 6))
        for i, (name, img) in enumerate(images):
            with cols[i % len(cols)]:
                thumb = make_thumbnail(img)
                st.image(bgr_to_rgb(thumb), caption=name, use_container_width=True)


def render_dashboard() -> None:
    """Render the quality-metrics dashboard and ML success-prediction cards."""
    report: QualityReport = st.session_state.get("report")
    if report is None:
        return

    st.markdown("### 📊 Panorama Quality Dashboard")

    m = report.metrics
    row1 = st.columns(4)
    row1[0].metric("Images Uploaded", m.num_images)
    row1[1].metric("Processing Time", f"{report.processing_time_sec:.2f}s")
    row1[2].metric("Avg. Keypoints", f"{m.avg_keypoints:.0f}")
    row1[3].metric("Avg. Good Matches", f"{m.avg_good_matches:.0f}")

    row2 = st.columns(4)
    row2[0].metric("RANSAC Inliers (total)", m.total_ransac_inliers)
    row2[1].metric("Avg. Inlier Ratio", f"{m.avg_inlier_ratio:.0%}")
    row2[2].metric("Estimated Overlap", f"{m.overlap_percent:.0f}%")
    row2[3].metric("Panorama Resolution", f"{report.panorama_width}×{report.panorama_height}")

    row3 = st.columns(4)
    row3[0].metric("Blend Method", report.blend_method.capitalize())
    row3[1].metric("Projection", "Enabled" if report.projection_enabled else "Disabled")
    row3[2].metric("Confidence Score", f"{report.confidence_score:.0f}/100")
    row3[3].metric("Rating", f"{report.stars_display()}", help=report.rating_label)
    st.caption(f"**{report.rating_label}** — {report.stars_display()}")

    if report.warnings:
        with st.expander("⚠️ Quality Warnings", expanded=True):
            for w in report.warnings:
                st.warning(w)

    st.markdown("###  ML Success Prediction")
    if report.ml_success_probability is not None and not np.isnan(report.ml_success_probability):
        ml_cols = st.columns(2)
        ml_cols[0].metric("Success Probability", f"{report.ml_success_probability:.0%}")
        ml_cols[1].metric("Expected Quality", report.ml_expected_quality)
        st.progress(min(max(report.ml_success_probability, 0.0), 1.0))
    else:
        st.info("ML success predictor unavailable (scikit-learn/joblib not installed).")

    report_json = report.to_json()
    st.download_button(
        label="⬇ Download Stitching Report (JSON)",
        data=report_json,
        file_name="stitching_report.json",
        mime="application/json",
    )
    st.markdown("---")


def run_stitching(config: dict) -> None:
    """Execute the stitching pipeline with a progress bar and status log.

    Args:
        config: Stitching configuration from the sidebar.
    """
    images = [img for _, img in st.session_state["images"]]

    if len(images) < 2:
        st.warning("Please upload at least 2 overlapping images before stitching.")
        return

    progress_bar = st.progress(0, text="Initializing...")
    status_placeholder = st.empty()
    log_lines: List[str] = []

    def log(message: str, progress: int) -> None:
        log_lines.append(message)
        status_placeholder.markdown(
            f'<div class="status-box">{"<br>".join(log_lines)}</div>',
            unsafe_allow_html=True,
        )
        progress_bar.progress(progress, text=message)

    try:
        log("Validating images and initializing pipeline...", 5)
        stitcher = PanoramaStitcher(
            blend_method=config["blend_method"],
            use_cylindrical=config["use_cylindrical"],
            exposure_compensation=config["exposure_compensation"],
            auto_order=config["auto_order"],
            max_dimension=config["max_dimension"],
            debug=config["debug"],
        )

        log("Detecting features and matching image pairs...", 25)
        time.sleep(0.1)  # small pause so the UI visibly updates
        log("Estimating homographies with RANSAC...", 50)
        result = stitcher.stitch(images)
        result.filenames_in_order = [
            st.session_state["images"][i][0] for i in result.order
        ]

        for i, msg in enumerate(result.messages):
            log(msg, min(95, 55 + 5 * i))

        log("Running ML success prediction...", 97)
        predictor = load_success_predictor()
        prediction: PredictionResult = predictor.predict(result.metrics)

        report: QualityReport = build_quality_report(
            metrics=result.metrics,
            processing_time_sec=result.processing_time_sec,
            panorama_shape=result.panorama.shape[:2],
            blend_method=config["blend_method"],
            projection_enabled=config["use_cylindrical"],
            ml_success_probability=prediction.success_probability,
            ml_expected_quality=prediction.expected_quality,
        )

        log("Done! Panorama generated successfully.", 100)
        st.session_state["result"] = result
        st.session_state["report"] = report
        st.session_state["prediction"] = prediction
        st.session_state["error"] = None

        if prediction.low_confidence_warning:
            st.warning(f" {prediction.low_confidence_warning}")

    except StitchingError as exc:
        st.session_state["result"] = None
        st.session_state["report"] = None
        st.session_state["prediction"] = None
        st.session_state["error"] = str(exc)
        progress_bar.progress(100, text="Failed.")
        st.error(f" Stitching failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to the user
        st.session_state["result"] = None
        st.session_state["report"] = None
        st.session_state["prediction"] = None
        st.session_state["error"] = str(exc)
        progress_bar.progress(100, text="Failed.")
        st.error(f" Unexpected error: {exc}")


def render_results(config: dict) -> None:
    """Render the stitching results: matches, warped images, panorama, download.

    Args:
        config: Stitching configuration (used to decide what to show).
    """
    result = st.session_state.get("result")
    if result is None:
        return

    st.subheader("3. Results")

    st.markdown(f"**Detected image order:** {result.order_description()}")
    if result.focal_length_used:
        st.caption(f"Cylindrical projection focal length used: ≈{result.focal_length_used:.0f}px")

    render_dashboard()

    if config["debug"] and result.match_debug_images:
        with st.expander(" Feature matches (debug mode)", expanded=False):
            for i, match_img in enumerate(result.match_debug_images):
                st.image(bgr_to_rgb(match_img), caption=f"Match pair {i + 1}", use_container_width=True)

    with st.expander(" Warped images (before blending)", expanded=False):
        cols = st.columns(min(len(result.warped_images), 4) or 1)
        for i, warped in enumerate(result.warped_images):
            with cols[i % len(cols)]:
                st.image(bgr_to_rgb(warped), caption=f"Warped #{i + 1}", use_container_width=True)

    st.markdown("###  Final Panorama")
    png_bytes = encode_image_to_bytes(result.panorama, ext=".png")
    render_zoom_pan_viewer(png_bytes)

    st.download_button(
        label="⬇ Download Panorama (PNG)",
        data=png_bytes,
        file_name="panorama.png",
        mime="image/png",
    )


def render_zoom_pan_viewer(image_bytes: bytes) -> None:
    """Render an interactive zoom/pan viewer for the final panorama.

    Uses scroll-wheel to zoom (centered on the cursor) and click-drag to
    pan, implemented as a small self-contained HTML/JS component since
    Streamlit's native st.image does not support this out of the box.

    Args:
        image_bytes: Encoded PNG bytes of the panorama to display.
    """
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    html = f"""
    <div id="viewer-wrap" style="
        width: 100%; height: 520px; overflow: hidden; position: relative;
        background: #111827; border-radius: 10px; cursor: grab;
        border: 1px solid #2a2f3a;">
      <img id="pano-img" src="data:image/png;base64,{b64_img}"
           style="position: absolute; top: 0; left: 0; transform-origin: 0 0;
                  transform: translate(0px, 0px) scale(1); user-select: none;
                  -webkit-user-drag: none; will-change: transform;" draggable="false" />
    </div>
    <div style="color:#9ca3af; font-size:0.8rem; margin-top:6px;">
      Scroll to zoom &middot; Click and drag to pan &middot; Double-click to reset
    </div>
    <script>
      (function() {{
        const wrap = document.getElementById('viewer-wrap');
        const img = document.getElementById('pano-img');
        let scale = 1, originX = 0, originY = 0;
        let isDragging = false, startX = 0, startY = 0;

        function applyTransform() {{
          img.style.transform = `translate(${{originX}}px, ${{originY}}px) scale(${{scale}})`;
        }}

        wrap.addEventListener('wheel', function(e) {{
          e.preventDefault();
          const rect = wrap.getBoundingClientRect();
          const mouseX = e.clientX - rect.left;
          const mouseY = e.clientY - rect.top;
          const prevScale = scale;
          const delta = e.deltaY < 0 ? 1.15 : 1 / 1.15;
          scale = Math.min(Math.max(scale * delta, 0.2), 8);

          // Zoom centered on cursor position.
          originX = mouseX - (mouseX - originX) * (scale / prevScale);
          originY = mouseY - (mouseY - originY) * (scale / prevScale);
          applyTransform();
        }}, {{ passive: false }});

        wrap.addEventListener('mousedown', function(e) {{
          isDragging = true;
          wrap.style.cursor = 'grabbing';
          startX = e.clientX - originX;
          startY = e.clientY - originY;
        }});
        window.addEventListener('mouseup', function() {{
          isDragging = false;
          wrap.style.cursor = 'grab';
        }});
        window.addEventListener('mousemove', function(e) {{
          if (!isDragging) return;
          originX = e.clientX - startX;
          originY = e.clientY - startY;
          applyTransform();
        }});
        wrap.addEventListener('dblclick', function() {{
          scale = 1; originX = 0; originY = 0;
          applyTransform();
        }});
      }})();
    </script>
    """
    components.html(html, height=560)


def main() -> None:
    """Application entry point."""
    init_session_state()
    render_header()
    config = render_sidebar()
    render_upload_section()

    st.subheader("2. Stitch")
    if st.button(" Stitch Panorama", type="primary"):
        run_stitching(config)

    render_results(config)


if __name__ == "__main__":
    main()
