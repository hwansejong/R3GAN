import numpy as np

# Color palette for floor plan segmentation
PALETTE = {
    "LivingRoom": (255, 0, 0),
    "Hall": (0, 255, 0),
    "Bedroom": (0, 0, 255),
    "Kitchen": (255, 165, 0),
    "Dining": (255, 255, 0),
    "Storage": (128, 0, 128),
    "Utility Room": (165, 42, 42),
    "Entry": (255, 0, 127),
    "Balcony": (0, 128, 255),
    "Bath": (128, 128, 0),
    "walls": (0, 0, 0),
    "doors": (255, 0, 255),
    "windows": (0, 255, 255),
}

OUTSIDE_COLOR = (255, 255, 255)

# Ordered color lists
CLASS_NAMES = list(PALETTE.keys())
COLORS = [PALETTE[name] for name in CLASS_NAMES] + [OUTSIDE_COLOR]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}
NUM_CLASSES = len(CLASS_NAMES)
NUM_ALL = len(COLORS)

COLOR_MAP = np.array([PALETTE[name] for name in CLASS_NAMES], dtype=np.float32) / 255.0


def rgb_to_label(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to label indices."""
    h, w, _ = img.shape
    label = np.full((h, w), NUM_ALL - 1, dtype=np.int64)  # outside by default
    for idx, color in enumerate(COLORS):
        mask = np.all(img == color, axis=-1)
        label[mask] = idx
    return label


def extract_graph_features(img: np.ndarray) -> np.ndarray:
    """Return area ratios and adjacency matrix flattened."""
    label = rgb_to_label(img)
    h, w = label.shape
    n = NUM_ALL
    areas = np.bincount(label.ravel(), minlength=n).astype(np.float32)
    areas /= float(h * w)
    adjacency = np.zeros((n, n), dtype=np.float32)

    # horizontal neighbors
    l1 = label[:, :-1]
    l2 = label[:, 1:]
    mask = l1 != l2
    idx_a = l1[mask]
    idx_b = l2[mask]
    np.add.at(adjacency, (idx_a, idx_b), 1)
    np.add.at(adjacency, (idx_b, idx_a), 1)

    # vertical neighbors
    l1 = label[:-1, :]
    l2 = label[1:, :]
    mask = l1 != l2
    idx_a = l1[mask]
    idx_b = l2[mask]
    np.add.at(adjacency, (idx_a, idx_b), 1)
    np.add.at(adjacency, (idx_b, idx_a), 1)

    adjacency = (adjacency > 0).astype(np.float32)
    return np.concatenate([areas, adjacency.reshape(-1)])
