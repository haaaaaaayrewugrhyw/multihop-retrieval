"""
Synthetic multi-object dataset for the slot-attention objective.

Each image is HxWx3 with 2-3 colored shapes (circle/square/triangle) on a dark
background. We also return a per-pixel integer mask (0 = background, 1..K =
objects) so we can score object discovery with FG-ARI. No class labels — the
task is unsupervised reconstruction.
"""

import numpy as np


def _shape_mask(shape, H, W, cx, cy, size, rng):
    yy, xx = np.ogrid[:H, :W]
    if shape == "circle":
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= size ** 2
    if shape == "square":
        return (np.abs(xx - cx) <= size) & (np.abs(yy - cy) <= size)
    # upward triangle (apex at top)
    t = (yy - (cy - size)) / (2 * size + 1e-6)
    half = np.clip(t, 0, 1) * size
    return (yy >= cy - size) & (yy <= cy + size) & (np.abs(xx - cx) <= half)


def make_dataset(n, H=64, W=64, min_obj=2, max_obj=3, seed=0, same_color=False):
    """same_color=True -> all objects in an image share one color, so they can
    only be separated by shape/position. Stresses the binding mechanism (color
    can no longer shortcut object discovery)."""
    rng = np.random.RandomState(seed)
    shapes = ["circle", "square", "triangle"]
    palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
    imgs = np.full((n, H, W, 3), 0.1, dtype=np.float32)   # dark gray bg
    masks = np.zeros((n, H, W), dtype=np.int64)
    for i in range(n):
        k = rng.randint(min_obj, max_obj + 1)
        img_color = palette[rng.randint(len(palette))]   # used if same_color
        for o in range(1, k + 1):
            shape = shapes[rng.randint(len(shapes))]
            color = img_color if same_color else palette[rng.randint(len(palette))]
            size = rng.randint(7, 12)
            cx = rng.randint(size, W - size)
            cy = rng.randint(size, H - size)
            m = _shape_mask(shape, H, W, cx, cy, size, rng)
            imgs[i][m] = color
            masks[i][m] = o
    imgs = np.transpose(imgs, (0, 3, 1, 2))               # N,3,H,W
    return imgs, masks


if __name__ == "__main__":
    x, m = make_dataset(4)
    print("imgs", x.shape, x.min(), x.max(), "masks", m.shape,
          "unique ids", np.unique(m))
