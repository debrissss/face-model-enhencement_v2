"""保留原工具 MediaPipe CPU 定位与双方向三维解剖检查。"""
from ...utils.logging import get_logger
from pathlib import Path
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw
from .automatic import nominal_rotation
ANCHOR_INDICES = (33, 133, 362, 263, 1, 152)

def create_face_landmarker(model_path: Path) -> vision.FaceLandmarker:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe model asset is missing: {model_path}. "
            "See this tool's README for the official download URL."
        )
    delegate = mp_python.BaseOptions.Delegate.CPU
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def detect_image(detector: vision.FaceLandmarker, pixels: np.ndarray) -> dict:
    """Return the six anchor pixels and image size, or a per-image skip reason."""
    image = pixels
    height, width = image.shape[:2]
    try:
        rgb = np.ascontiguousarray(image)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception as exc:
        return {
            "image_size": [width, height],
            "anchor_pixels": None,
            "reason": f"landmark_detection_failed: {type(exc).__name__}: {exc}",
        }

    if not result.face_landmarks:
        return {
            "image_size": [width, height],
            "anchor_pixels": None,
            "reason": "face_not_detected",
        }

    landmarks = result.face_landmarks[0]
    if max(ANCHOR_INDICES) >= len(landmarks):
        return {
            "image_size": [width, height],
            "anchor_pixels": None,
            "reason": "mediapipe_anchor_indices_unavailable",
        }

    pixels = np.asarray(
        [[landmarks[index].x * width, landmarks[index].y * height] for index in ANCHOR_INDICES],
        dtype=np.float64,
    )
    if not np.isfinite(pixels).all():
        return {
            "image_size": [width, height],
            "anchor_pixels": None,
            "reason": "non_finite_landmarks",
        }
    return {
        "image_size": [width, height],
        "anchor_pixels": pixels.tolist(),
        "reason": None,
    }


class FaceLocator:
    def __init__(self, model_path: Path):
        self.error = None
        self.detector = None
        try:
            self.detector = create_face_landmarker(model_path)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            get_logger().warning("Face locator unavailable; multi-region fallback: %s", self.error)

    def close(self):
        if self.detector is not None:
            self.detector.close()

    def locate(self, pixels: np.ndarray, points: np.ndarray, camera: dict, view: str):
        if self.error:
            return None, {"status": "unavailable", "error": self.error}, None
        original = Image.fromarray(pixels)
        rotated = np.asarray(original.transpose(Image.Transpose.ROTATE_180))
        results = [detect_image(self.detector, pixels), detect_image(self.detector, rotated)]
        candidates = []
        k = camera["Cointrinsics"]
        uv = points[:, :2] / points[:, 2:3] * [-k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]
        def patch_anchor(pixel):
            for radius in (8, 16, 32, 64):
                mask = (np.all(np.abs(uv - pixel) <= radius, axis=1)
                        & (points[:, 2] > -900) & (points[:, 2] < -150))
                patch = points[mask]
                if len(patch) >= 30:
                    front = patch[:, 2] >= np.percentile(patch[:, 2], 50)
                    return np.median(patch[front], axis=0), radius, len(patch)
            return None, None, 0
        for orientation, result in zip(("raw", "rot180"), results):
            pixels = result.get("anchor_pixels")
            if pixels is None:
                continue
            pixels = np.asarray(pixels)
            if orientation == "rot180":
                pixels = [original.width - 1, original.height - 1] - pixels
            nose = pixels[4]
            anchor, radius, count = patch_anchor(nose)
            if anchor is not None:
                eye_span = float(np.linalg.norm(pixels[0] - pixels[3]))
                anatomy = [patch_anchor(pixel)[0] for pixel in pixels]
                consistent = False
                if all(value is not None for value in anatomy):
                    model_oriented = np.array(anatomy) @ nominal_rotation(view).T
                    eye_y = np.mean(model_oriented[:4, 1])
                    nose_y, chin_y = model_oriented[4:, 1]
                    consistent = bool(eye_y > nose_y + 3 and nose_y > chin_y + 8)
                candidates.append((consistent, eye_span, anchor, pixels, orientation, radius, count))
        if not candidates:
            return None, {"status": "not_detected_or_no_3d_patch", "detections": results}, None
        consistent, _, anchor, pixels, orientation, radius, count = max(candidates, key=lambda x: (x[0], x[1]))
        if not consistent:
            return None, {"status": "inconsistent_3d_anatomy", "detections": results}, None
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        for u, v in pixels:
            draw.ellipse((u-4, v-4, u+4, v+4), fill="red")
        u, v = pixels[4]
        draw.rectangle((u-radius, v-radius, u+radius, v+radius), outline="yellow", width=2)
        return anchor, {"status": "detected", "orientation": orientation,
                        "nose_pixel": pixels[4].tolist(), "anchor_pixels": pixels.tolist(),
                        "anchor_view_mm": anchor.tolist(), "patch_radius_px": radius,
                        "patch_point_count": count, "projection": "native Cointrinsics; no sensor Extrinsics applied"}, np.asarray(overlay).copy()

