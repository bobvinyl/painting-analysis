import io
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from urllib.request import urlopen
except ImportError:
    urlopen = None


class ImageAnalyzer:
    """Analyze an image from a local path or a remote URI."""

    def __init__(self, image_uri: str) -> None:
        self.image_uri = image_uri
        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None

    def load_image(self) -> np.ndarray:
        if self.image_uri.startswith("http://") or self.image_uri.startswith("https://"):
            self.image_bgr = self._load_image_from_url(self.image_uri)
        else:
            self.image_bgr = self._load_image_from_path(self.image_uri)

        if self.image_bgr is None:
            raise ValueError(f"Unable to load image from URI: {self.image_uri}")

        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        return self.image_rgb

    def _load_image_from_path(self, path: str) -> Optional[np.ndarray]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image file: {path}")
        return image

    def _load_image_from_url(self, url: str) -> Optional[np.ndarray]:
        if urlopen is None:
            raise RuntimeError("urllib is not available to fetch remote images")

        with urlopen(url) as response:
            data = response.read()

        image_data = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image from URL: {url}")
        return image

    def analyze_colors(self, num_clusters: int = 5) -> Dict[str, object]:
        if self.image_rgb is None:
            self.load_image()

        pixels = self.image_rgb.reshape(-1, 3)
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels,
            num_clusters,
            None,
            criteria,
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS,
        )

        counts = np.bincount(labels.flatten(), minlength=num_clusters)
        total = counts.sum()
        sorted_idx = np.argsort(-counts)

        dominant_colors = [tuple(map(int, centers[i])) for i in sorted_idx]
        percentages = [float(counts[i]) / total for i in sorted_idx]

        average_color = tuple(map(int, np.mean(pixels, axis=0)))
        median_color = tuple(map(int, np.median(pixels, axis=0)))
        warmth_score = self.warmth_score(average_color)
        brightness_score = self.brightness_score(average_color)
        dominant_distances = [
            float(self.color_distance(average_color, dominant_color))
            for dominant_color in dominant_colors
        ]

        histogram = self._color_histogram(self.image_rgb)

        return {
            "dominant_colors": dominant_colors,
            "dominant_percentages": percentages,
            "dominant_distances": dominant_distances,
            "average_color": average_color,
            "median_color": median_color,
            "warmth_score": warmth_score,
            "brightness_score": brightness_score,
            "histogram": histogram,
        }

    def _color_histogram(self, image_rgb: np.ndarray, bins: int = 8) -> Dict[str, np.ndarray]:
        histogram = {
            "red": cv2.calcHist([image_rgb], [0], None, [bins], [0, 256]).flatten(),
            "green": cv2.calcHist([image_rgb], [1], None, [bins], [0, 256]).flatten(),
            "blue": cv2.calcHist([image_rgb], [2], None, [bins], [0, 256]).flatten(),
        }
        return histogram

    @staticmethod
    def rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
        rgb_arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
        lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
        return lab[0, 0]

    def color_distance(
        self,
        rgb_a: Tuple[int, int, int],
        rgb_b: Tuple[int, int, int],
        method: str = "lab",
    ) -> float:
        if method == "lab":
            lab_a = self.rgb_to_lab(rgb_a).astype(np.float32)
            lab_b = self.rgb_to_lab(rgb_b).astype(np.float32)
            return float(np.linalg.norm(lab_a - lab_b))
        if method == "rgb":
            diff = np.array(rgb_a, dtype=np.float32) - np.array(rgb_b, dtype=np.float32)
            return float(np.linalg.norm(diff))
        raise ValueError(f"Unsupported distance method: {method}")

    @staticmethod
    def warmth_score(rgb: Tuple[int, int, int]) -> int:
        return int(rgb[0] - rgb[2])

    @staticmethod
    def brightness_score(rgb: Tuple[int, int, int]) -> float:
        return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % rgb
