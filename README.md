# Painting Analysis

A small Python project for image color analysis. The repository includes:

- `image_analysis.py`: core analysis logic for loading images and extracting color data.
- `cli.py`: a command-line interface for quick color analysis.
- `streamlit_app.py`: a Streamlit web UI to display the image, analysis results, swatches, and histogram.

## image_analysis.py

`ImageAnalyzer` is the main class in `image_analysis.py`.

### What it does

- Loads an image from a local path or a remote HTTP(S) URI.
- Converts the image from OpenCV's default BGR format to RGB.
- Computes color statistics.

### Analysis methods

#### `load_image()`

- Loads the image using OpenCV.
- Supports local file paths and remote URLs.
- Returns an RGB NumPy array.

#### `analyze_colors(num_clusters=5)`

Performs color analysis using K-means clustering.

Outputs:

- `dominant_colors`: the main RGB colors discovered in the image.
- `dominant_percentages`: the relative fraction of pixels assigned to each dominant color.
- `average_color`: the per-channel mean RGB value across all pixels.
- `median_color`: the per-channel median RGB value across all pixels.
- `histogram`: a simple distribution of red, green, and blue channel intensities.

#### `rgb_to_hex(rgb)`

Converts an RGB tuple to a hexadecimal color string (for example `#ffcc00`).

### Histogram explanation

The histogram represents the distribution of pixel intensity values for each color channel:

- `red`: counts of red values across the image.
- `green`: counts of green values.
- `blue`: counts of blue values.

Each channel histogram is computed using a small number of bins (default 8), so it summarizes whether the image is darker or brighter in each channel and where the channel values cluster.

## CLI (`cli.py`)

The CLI provides a quick command-line way to analyze image colors.

### Usage

```bash
python cli.py /path/to/image.jpg
python cli.py https://example.com/image.png --clusters 6
```

### What it shows

- Average color in RGB and HEX
- Median color in RGB and HEX
- Dominant colors with their percentage share

## Streamlit UI (`streamlit_app.py`)

The Streamlit app provides an interactive browser UI for comparing two images side by side.

### Features

- Two input fields for image URIs or local paths.
- One analysis button to compare both images in parallel.
- Slider to choose the number of dominant color clusters.
- Displays each loaded image in its own panel.
- Shows average and median colors with swatches, RGB, and HEX values.
- Renders dominant color swatches and percentage shares for each image.
- Shows a colored histogram for the red, green, and blue channels.

### Run the app

```bash
streamlit run streamlit_app.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Dependencies:

- `opencv-python`
- `numpy`
- `streamlit`

## Notes

- Remote images must be served over HTTP or HTTPS.
- The Streamlit app uses `width=700` for the image display to avoid deprecation warnings.
