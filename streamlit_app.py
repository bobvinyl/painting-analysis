import streamlit as st

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


def main() -> None:
    st.set_page_config(page_title="Painting Analysis", layout="wide")
    st.title("Painting Analysis")
    st.write("Analyze image colors from a local path or a remote URI.")

    image_uri = st.text_input("Image URI or local file path", "")
    clusters = st.slider("Dominant color clusters", min_value=2, max_value=10, value=5)

    if image_uri:
        if st.button("Analyze image"):
            try:
                analyzer = ImageAnalyzer(image_uri)
                image = analyzer.load_image()
                result = analyzer.analyze_colors(num_clusters=clusters)

                st.subheader("Image Preview")
                st.image(image, caption=image_uri, width=700)

                st.subheader("Color Analysis Results")
                cols = st.columns(2)
                cols[0].metric("Average color (RGB)", str(result["average_color"]))
                cols[1].metric("Average color (HEX)", ImageAnalyzer.rgb_to_hex(result["average_color"]))
                cols[0].metric("Median color (RGB)", str(result["median_color"]))
                cols[1].metric("Median color (HEX)", ImageAnalyzer.rgb_to_hex(result["median_color"]))

                st.markdown("---")
                st.subheader("Dominant Colors")
                for rgb, pct in zip(result["dominant_colors"], result["dominant_percentages"]):
                    st.markdown(format_color_box(rgb, pct), unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Color Histogram")
                histogram = result["histogram"]
                st.bar_chart(
                    {
                        "red": histogram["red"].tolist(),
                        "green": histogram["green"].tolist(),
                        "blue": histogram["blue"].tolist(),
                    }
                )
            except Exception as exc:
                st.error(f"Unable to analyze image: {exc}")


if __name__ == "__main__":
    main()
