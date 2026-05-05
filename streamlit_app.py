import base64
import io

import cv2
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from typing import Optional

from image_analysis import ImageAnalyzer


def format_color_box(
    rgb: tuple[int, int, int], percentage: float, distance: Optional[float] = None
) -> str:
    hex_color = ImageAnalyzer.rgb_to_hex(rgb)
    distance_line = ""
    if distance is not None:
        if distance <= 85:
            color = "green"
        elif distance <= 170:
            color = "orange"
        else:
            color = "red"
        distance_line = (
            f"Distance from average: "
            f"<span style=\"color:{color}; font-weight:bold;\">{distance:.1f}</span>"
        )
    return (
        f"<div style=\"display:flex;align-items:center;margin-bottom:10px;\">"
        f"<div style=\"width:96px;height:64px;border:1px solid #ddd;margin-right:12px;background:{hex_color};\"></div>"
        f"<div>"
        f"<strong>{hex_color}</strong><br>"
        f"RGB: {rgb}<br>"
        f"Share: {percentage:.2%}<br>"
        f"{distance_line}"
        f"</div>"
        f"</div>"
    )


def format_color_summary(title: str, rgb: tuple[int, int, int]) -> str:
    hex_color = ImageAnalyzer.rgb_to_hex(rgb)
    return (
        f"<div style=\"display:flex;align-items:center;gap:12px;margin-bottom:16px;\">"
        f"<div style=\"width:48px;height:48px;border:1px solid #ddd;background:{hex_color};\"></div>"
        f"<div>"
        f"<strong>{title}</strong><br>"
        f"RGB: {rgb}<br>"
        f"HEX: {hex_color}"
        f"</div>"
        f"</div>"
    )


def image_to_base64(image: np.ndarray) -> str:
    # OpenCV expects BGR for encoding, but our image is stored in RGB.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError('Unable to encode image to PNG')
    return base64.b64encode(buffer).decode('utf-8')


def render_analysis_panel(panel, image_uri: str, image: Optional[np.ndarray], result: Optional[dict]) -> None:
    if not image_uri:
        panel.info("Enter an image URI or local path to analyze.")
        return

    if result is None or image is None:
        panel.error("No analysis available for this image.")
        return

    panel.subheader("Image Preview")
    image_base64 = image_to_base64(image)
    image_html = (
        f"<div style=\"height:320px; display:flex; align-items:center; justify-content:center; "
        f"overflow:hidden; border:1px solid #ddd; background:#fafafa; margin-bottom:8px;\">"
        f"<img src=\"data:image/png;base64,{image_base64}\" "
        f"style=\"max-height:100%; max-width:100%; object-fit:contain;\" />"
        f"</div>"
        f"<div style=\"font-size:0.9rem; color:#555; margin-bottom:16px;\">{image_uri}</div>"
    )
    panel.markdown(image_html, unsafe_allow_html=True)

    panel.subheader("Color Analysis Results")
    panel.markdown(format_color_summary("Average color", result["average_color"]), unsafe_allow_html=True)
    panel.markdown(format_color_summary("Median color", result["median_color"]), unsafe_allow_html=True)
    panel.markdown(f"**Warmth score:** {result['warmth_score']}  \nPositive values indicate a warmer red bias; negative values indicate a cooler blue bias.")
    panel.markdown(f"**Brightness score:** {result['brightness_score']:.2f}  \nPerceived luminance of the average color (0-255 scale). Higher values indicate brighter images.")

    panel.markdown("---")
    panel.subheader("Dominant Colors")
    for rgb, pct, dist in zip(
        result["dominant_colors"], result["dominant_percentages"], result["dominant_distances"]
    ):
        panel.markdown(format_color_box(rgb, pct, dist), unsafe_allow_html=True)

    panel.markdown("---")
    panel.subheader("Color Histogram")
    histogram = result["histogram"]
    histogram_data = []
    num_bins = len(histogram["red"])
    bin_width = 256 // num_bins
    range_labels = []
    for idx in range(num_bins):
        start = idx * bin_width
        end = (idx + 1) * bin_width - 1
        if idx == num_bins - 1:
            end = 255
        range_labels.append(f"{start}-{end}")

    for channel in ("red", "green", "blue"):
        for idx, value in enumerate(histogram[channel].tolist()):
            histogram_data.append({
                "channel": channel,
                "range": range_labels[idx],
                "range_order": idx,
                "count": value,
            })

    histogram_df = pd.DataFrame(histogram_data)
    chart = alt.Chart(histogram_df).mark_bar().encode(
        x=alt.X(
            "range:O",
            title="Intensity range",
            sort=alt.SortArray(range_labels),
        ),
        y=alt.Y("count:Q", title="Pixel count"),
        color=alt.Color(
            "channel:N",
            scale=alt.Scale(domain=["red", "green", "blue"], range=["red", "green", "blue"]),
            title="Channel",
        ),
        tooltip=["channel", "range", "count"],
    ).properties(width=320)

    panel.altair_chart(chart, use_container_width=True)

    panel.markdown("---")
    panel.subheader("Style Analysis")
    panel.markdown(f"**Angular score:** {result['angular_score']:.2f}  \nHigher values indicate more angular features (straight lines, geometric shapes).")
    panel.markdown(f"**Flowing score:** {result['flowing_score']:.2f}  \nHigher values indicate more flowing features (curves, organic shapes).")


def main() -> None:
    st.set_page_config(page_title="Painting Analysis", layout="wide")
    st.title("Painting Analysis")
    st.write("Analyze two images side by side using local paths or remote URIs.")

    left, right = st.columns(2)
    image_uri_1 = left.text_input("Image 1 URI or local file path", "", key="image_uri_1")
    image_uri_2 = right.text_input("Image 2 URI or local file path", "", key="image_uri_2")
    clusters = st.slider("Dominant color clusters", min_value=2, max_value=10, value=5)

    if st.button("Analyze images"):
        image_uris = [image_uri_1, image_uri_2]
        results = []

        for image_uri in image_uris:
            if not image_uri:
                results.append((None, None, None))
                continue

            try:
                analyzer = ImageAnalyzer(image_uri)
                image = analyzer.load_image()
                result = analyzer.analyze_colors(num_clusters=clusters)
                style_result = analyzer.analyze_style()
                result.update(style_result)
                results.append((image_uri, image, result))
            except Exception as exc:
                results.append((image_uri, None, {"error": str(exc)}))

        panels = st.columns(2)
        for panel, (image_uri, image, result) in zip(panels, results):
            if image_uri is None:
                panel.info("Enter an image URI or local path to analyze.")
            elif result is None:
                panel.error("No analysis available for this image.")
            elif "error" in result:
                panel.error(f"Unable to analyze image: {result['error']}")
            else:
                render_analysis_panel(panel, image_uri, image, result)


if __name__ == "__main__":
    main()
