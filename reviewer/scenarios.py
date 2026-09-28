"""PRIVATE. Never distribute this module or its outputs to students."""

import numpy as np
from physical_ai.scenes import COLORS


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
    if task == "swap":
        return {
            "body": {
                f"cup{i}": {
                    "pos": [
                        0.43 + float(rng.uniform(-0.035, 0.035)),
                        y + float(rng.uniform(-0.025, 0.025)),
                        0.438,
                    ]
                }
                for i, y in enumerate([-0.16, 0.16])
            }
        }
    if task == "sort":
        # Distinct new colors and different assignment to spatial slots.
        palette = [[0.95, 0.58, 0.08, 1], [0.56, 0.15, 0.74, 1], [0.08, 0.72, 0.68, 1]]
        order = rng.permutation(3)
        if np.array_equal(order, np.arange(3)):
            order = np.array([1, 2, 0])
        geoms = {}
        for i, color in enumerate(palette):
            for suffix in (
                ["bottom"]
                + [f"wall{j}" for j in range(12)]
                + [f"handle{j}" for j in range(3)]
            ):
                geoms[f"cup{i}_{suffix}"] = {"rgba": color}
            geoms[f"plate{i}"] = {"rgba": palette[int(order[i])]}
            for j in range(20):
                geoms[f"plate{i}_rim{j}"] = {"rgba": palette[int(order[i])]}
        return {"geom": geoms}
    raise ValueError(task)
