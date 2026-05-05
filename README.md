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
- `dominant_distances`: the perceptual distance of each dominant color from the average color, computed in Lab color space. In the Streamlit UI, this distance is color-coded on each dominant color entry:
  - green for 0–85
  - yellow/orange for 86–170
  - red for greater than 170
- `average_color`: the per-channel mean RGB value across all pixels.
- `median_color`: the per-channel median RGB value across all pixels.
- `warmth_score`: a signed score equal to average red minus average blue, where positive values indicate a warmer tone and negative values indicate a cooler tone.
- `brightness_score`: the perceived luminance of the average color, calculated using the standard formula 0.299*R + 0.587*G + 0.114*B. Values range from 0 (black) to 255 (white), indicating overall image brightness.
- `histogram`: a simple distribution of red, green, and blue channel intensities.

#### `analyze_style()`

Analyzes whether the painting style exhibits more angular or flowing characteristics.

Uses edge detection (Canny) and straight-line detection (Hough transform) to evaluate the prevalence of straight lines and geometric edges versus curves and organic shapes.

Outputs:

- `angular_score`: ranges from 0.0 to 1.0, where higher values indicate more angular features (sharp lines, geometric shapes, structured patterns).
- `flowing_score`: ranges from 0.0 to 1.0, where higher values indicate more flowing features (curves, organic shapes, smooth transitions). Note that `flowing_score = 1.0 - angular_score`.

The scoring combines:
- **Line coverage** (70%): the fraction of detected edge pixels that align with straight lines.
- **Line density** (15%): the total length of detected lines relative to image dimensions.
- **Orientation concentration** (15%): how consistently edge orientations are distributed (lower diversity = more angular).


#### `rgb_to_hex(rgb)`

Converts an RGB tuple to a hexadecimal color string (for example `#ffcc00`).

### Histogram explanation

The histogram represents how the pixel brightness values are distributed for each color channel.

#### What is a channel?

Images are stored in RGB format, where each pixel contains three separate channels:

- `red`: the amount of red light in the pixel.
- `green`: the amount of green light.
- `blue`: the amount of blue light.

These channels combine to produce the final visible color of each pixel.

#### How are the values separated into bins?

Each channel value ranges from 0 to 255. The histogram groups these values into a fixed number of ranges called bins.

For example, with 8 bins:

- bin 0 covers values near 0–31
- bin 1 covers values near 32–63
- ...
- bin 7 covers values near 224–255

Each bin counts how many pixels have channel values in that range.

#### What the histogram tells you

- A tall red bin means many pixels have red values in that range.
- If the blue histogram is concentrated in the low bins, the image is darker or has little blue intensity.
- If all three color histograms are strong at high bins, the image is bright in all channels.

The app also computes a perceptual distance between each dominant color and the image average color.
This distance is measured in Lab color space, which reflects how humans perceive color differences more accurately than raw RGB distances.

In the Streamlit UI, each channel is drawn in its own color:

- red channel bars are colored red
- green channel bars are colored green
- blue channel bars are colored blue

This makes it easier to compare how the red, green, and blue intensities are distributed across the image.

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
- Angular and flowing style scores

## Streamlit UI (`streamlit_app.py`)

The Streamlit app provides an interactive browser UI for comparing two images side by side.

### Features

- Two input fields for image URIs or local paths.
- One analysis button to compare both images in parallel.
- Slider to choose the number of dominant color clusters.
- Displays each loaded image in its own panel.
- Shows average and median colors with swatches, RGB, and HEX values.
- Shows a warmth score for each image, where positive values indicate a warmer red bias and negative values indicate a cooler blue bias.
- Displays the perceptual distance of each dominant color from the average color.
- Renders dominant color swatches and percentage shares for each image.
- Shows a colored histogram for the red, green, and blue channels.
- Displays angular and flowing style scores to characterize whether the painting favors geometric/angular features or organic/flowing shapes.

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
