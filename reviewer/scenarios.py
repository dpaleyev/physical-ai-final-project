"""PRIVATE. Never distribute this module or its outputs to students."""

import numpy as np


def scenario(task, seed):
    rng = np.random.default_rng(seed)
    if task == "cup_plate":
        # Change cup cross-section, retaining height/support conventions.
        sx, sy = float(rng.uniform(1.10, 1.18)), float(rng.uniform(0.87, 0.94))
        geoms = {"cup0_bottom": {"size": [0.027 * min(sx, sy), 0.004]}}
        for j in range(12):
            theta = 2 * np.pi * j / 12
            geoms[f"cup0_wall{j}"] = {
                "pos": [0.026 * np.cos(theta) * sx, 0.026 * np.sin(theta) * sy, 0],
                "size": [0.004, 0.009, 0.032],
            }
        return {"geom": geoms}
    if task == "cup_shelf":
        z = 0.554 + float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.035))
        return {"body": {"target0": {"pos": [0.48, 0.17, z]}}}
    if task == "cup_distractor":
        return {
            "body": {
                "cup0": {
                    "pos": [
                        0.42 + float(rng.uniform(-0.025, 0.025)),
                        -0.14 + float(rng.uniform(-0.02, 0.02)),
                        0.438,
                    ]
                },
                "cup1": {
                    "pos": [
                        float(rng.uniform(0.60, 0.67)),
                        float(rng.uniform(-0.025, 0.035)),
                        0.438,
                    ]
                },
            }
        }
    if task == "color_match":
        palette = np.array(
            [[0.95, 0.58, 0.08, 1], [0.56, 0.15, 0.74, 1], [0.08, 0.72, 0.68, 1]]
        )
        colors = palette[rng.permutation(3)[:2]]
        geoms = {}
        for i, color in enumerate(colors):
            geoms[f"plate{i}"] = {"rgba": color.tolist()}
            for j in range(20):
                geoms[f"plate{i}_rim{j}"] = {"rgba": color.tolist()}
        # The environment samples cup color from the current plate palette;
        # its target is determined by visible equality, never by a fixed slot.
        return {"geom": geoms}
    raise ValueError(task)
