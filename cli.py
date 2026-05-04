import argparse

from image_analysis import ImageAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze image colors.")
    parser.add_argument("image_uri", help="Local path or remote HTTP/HTTPS image URI")
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of dominant color clusters",
    )
    args = parser.parse_args()

    analyzer = ImageAnalyzer(args.image_uri)
    result = analyzer.analyze_colors(num_clusters=args.clusters)

    print("Average color (RGB):", result["average_color"])
    print("Median color (RGB):", result["median_color"])
    print(f"Brightness score: {result['brightness_score']:.2f}")
    print("Dominant colors (RGB):")
    for color, pct in zip(result["dominant_colors"], result["dominant_percentages"]):
        print(f"  {color} ({pct:.2%}) => {ImageAnalyzer.rgb_to_hex(color)}")


if __name__ == "__main__":
    main()
