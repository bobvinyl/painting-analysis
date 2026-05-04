import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from typing import Optional

from image_analysis import ImageAnalyzer


def format_color_box(rgb: tuple[int, int, int], percentage: float) -> str:
    hex_color = ImageAnalyzer.rgb_to_hex(rgb)
    return (
        f"<div style=\"display:flex;align-items:center;margin-bottom:10px;\">"
        f"<div style=\"width:96px;height:64px;border:1px solid #ddd;margin-right:12px;background:{hex_color};\"></div>"
        f"<div>"
        f"<strong>{hex_color}</strong><br>"
        f"RGB: {rgb}<br>"
        f"Share: {percentage:.2%}"
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


def render_analysis_panel(panel, image_uri: str, image: Optional[np.ndarray], result: Optional[dict]) -> None:
    if not image_uri:
        panel.info("Enter an image URI or local path to analyze.")
        return

    if result is None or image is None:
        panel.error("No analysis available for this image.")
        return

    panel.subheader("Image Preview")
    panel.image(image, caption=image_uri, width=320)

    panel.subheader("Color Analysis Results")
    panel.markdown(format_color_summary("Average color", result["average_color"]), unsafe_allow_html=True)
    panel.markdown(format_color_summary("Median color", result["median_color"]), unsafe_allow_html=True)

    panel.markdown("---")
    panel.subheader("Dominant Colors")
    for rgb, pct in zip(result["dominant_colors"], result["dominant_percentages"]):
        panel.markdown(format_color_box(rgb, pct), unsafe_allow_html=True)

    panel.markdown("---")
    panel.subheader("Color Histogram")
    histogram = result["histogram"]
    histogram_data = []
    for channel in ("red", "green", "blue"):
        for idx, value in enumerate(histogram[channel].tolist()):
            histogram_data.append({
                "channel": channel,
                "bin": idx,
                "count": value,
            })

    histogram_df = pd.DataFrame(histogram_data)
    chart = alt.Chart(histogram_df).mark_bar().encode(
        x=alt.X("bin:O", title="Intensity bin"),
        y=alt.Y("count:Q", title="Pixel count"),
        color=alt.Color(
            "channel:N",
            scale=alt.Scale(domain=["red", "green", "blue"], range=["red", "green", "blue"]),
            title="Channel",
        ),
        tooltip=["channel", "bin", "count"],
    ).properties(width=320)

    panel.altair_chart(chart, use_container_width=True)


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
