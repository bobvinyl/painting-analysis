import io
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
except ImportError:
    HTTPError = None
    URLError = None
    Request = None
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
        if urlopen is None or Request is None:
            raise RuntimeError("urllib is not available to fetch remote images")

        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": "https://art.thewalters.org/",
            },
        )

        try:
            with urlopen(request, timeout=20) as response:
                data = response.read()
        except HTTPError as exc:
            raise ValueError(f"Unable to fetch image from URL ({exc.code}): {url}") from exc
        except URLError as exc:
            raise ValueError(f"Unable to fetch image from URL: {url}") from exc

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

    def _color_histogram(self, image_rgb: np.ndarray, bins: int = 16) -> Dict[str, np.ndarray]:
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

    def analyze_style(self) -> Dict[str, float]:
        """Analyze whether the painting style is more angular or flowing based on straight lines and edge orientation."""
        if self.image_rgb is None:
            self.load_image()

        gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Detect straight lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=10,
        )

        # Create a mask for detected lines and measure their length.
        line_mask = np.zeros_like(edges)
        total_line_length = 0.0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
                total_line_length += np.hypot(x2 - x1, y2 - y1)

        covered_edge_pixels = np.sum((edges > 0) & (line_mask > 0))
        total_edge_pixels = np.sum(edges > 0)
        line_coverage = float(covered_edge_pixels) / total_edge_pixels if total_edge_pixels > 0 else 0.0

        # Compute orientation entropy of the edge pixels.
        entropy_norm = 0.0
        if total_edge_pixels > 0:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
            orientation = (orientation + 180) % 180
            orientations = orientation[edges > 0]

            hist, _ = np.histogram(orientations, bins=36, range=(0, 180))
            hist = hist.astype(float)
            hist /= hist.sum() + 1e-10

            entropy = -np.sum(hist * np.log(hist + 1e-10))
            entropy_norm = float(entropy / np.log(36))

        # Scale angular contribution by overall line density and orientation concentration.
        diag = np.hypot(*gray.shape[::-1])
        line_density = np.clip(total_line_length / max(1.0, 2.5 * diag), 0.0, 1.0)
        orientation_concentration = 1.0 - entropy_norm

        angular_score = (
            0.15 * line_density
            + 0.70 * line_coverage
            + 0.15 * orientation_concentration
        )
        angular_score = float(np.clip(angular_score, 0.0, 1.0))
        flowing_score = 1.0 - angular_score

        return {"angular_score": angular_score, "flowing_score": flowing_score}
