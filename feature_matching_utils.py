"""Helpers for the simple Harris-corner feature matching notebook."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ASSET_DIR = Path(__file__).resolve().parent / "feature_matching_assets"
IMAGE1_PATH = ASSET_DIR / "feature_match_1.png"
IMAGE2_PATH = ASSET_DIR / "feature_match_2.png"


def load_demo_images() -> tuple[np.ndarray, np.ndarray]:
    image1 = cv2.imread(str(IMAGE1_PATH), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(IMAGE2_PATH), cv2.IMREAD_GRAYSCALE)
    if image1 is None or image2 is None:
        raise FileNotFoundError("Feature matching demo images are missing.")
    return image1, image2


def detect_harris_corners(
    image: np.ndarray,
    max_corners: int = 60,
    threshold_rel: float = 0.01,
    nms_radius: int = 7,
    border: int = 8,
) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    response = cv2.cornerHarris(image_f, blockSize=2, ksize=3, k=0.04)

    response = response.copy()
    response[:border, :] = 0
    response[-border:, :] = 0
    response[:, :border] = 0
    response[:, -border:] = 0

    threshold = threshold_rel * float(response.max())
    candidates = np.argwhere(response > threshold)
    candidates = sorted(candidates, key=lambda rc: response[rc[0], rc[1]], reverse=True)

    selected: list[tuple[int, int]] = []
    for row, col in candidates:
        keep = True
        for prev_row, prev_col in selected:
            if (row - prev_row) ** 2 + (col - prev_col) ** 2 <= nms_radius**2:
                keep = False
                break
        if keep:
            selected.append((row, col))
        if len(selected) >= max_corners:
            break

    return np.array([(col, row) for row, col in selected], dtype=np.int32)


def extract_patch_descriptors(
    image: np.ndarray,
    corners: np.ndarray,
    patch_size: int = 11,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    half = patch_size // 2
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT)

    kept_corners: list[tuple[int, int]] = []
    descriptors: list[np.ndarray] = []
    for x, y in corners:
        patch = padded[y : y + patch_size, x : x + patch_size].astype(np.float32)
        vector = patch.reshape(-1)
        if normalize:
            vector = vector - vector.mean()
            std = vector.std()
            if std < 1e-6:
                continue
            vector = vector / std
        kept_corners.append((x, y))
        descriptors.append(vector)

    return np.array(kept_corners, dtype=np.int32), np.array(descriptors, dtype=np.float32)


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> list[tuple[int, int, float]]:
    distances = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    matches: list[tuple[int, int, float]] = []

    for i in range(distances.shape[0]):
        order = np.argsort(distances[i])
        if len(order) < 2:
            continue
        best, second = order[0], order[1]
        if distances[i, best] < ratio_thresh * distances[i, second]:
            matches.append((i, int(best), float(distances[i, best])))

    reverse_best = np.argmin(distances, axis=0)
    mutual_matches = [match for match in matches if reverse_best[match[1]] == match[0]]
    mutual_matches.sort(key=lambda match: match[2])
    return mutual_matches


def draw_corners(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int] = (255, 220, 0)) -> np.ndarray:
    panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in corners:
        cv2.circle(panel, (int(x), int(y)), 4, color, 1, cv2.LINE_AA)
    return panel


def draw_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    corners1: np.ndarray,
    corners2: np.ndarray,
    matches: list[tuple[int, int, float]],
    max_matches: int = 25,
) -> np.ndarray:
    left = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([left, right])
    offset_x = image1.shape[1]
    rng = np.random.default_rng(3)

    for i, j, _ in matches[:max_matches]:
        x1, y1 = corners1[i]
        x2, y2 = corners2[j]
        color = tuple(int(v) for v in rng.integers(80, 255, size=3))
        cv2.circle(canvas, (int(x1), int(y1)), 4, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x2 + offset_x), int(y2)), 4, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2 + offset_x), int(y2)), color, 1, cv2.LINE_AA)

    return canvas
