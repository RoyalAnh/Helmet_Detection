"""Pure-function geometry helpers. No dependencies on CV or model code."""


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def get_head_region(bbox: list[float], ratio: float = 0.30) -> list[float]:
    """Return the top `ratio` fraction of a person bbox as the head region."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1 + (y2 - y1) * ratio]
