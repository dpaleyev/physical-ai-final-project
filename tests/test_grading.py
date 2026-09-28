import importlib.util
from pathlib import Path
import pytest


def test_grading_tools_exist():
    assert importlib.util.find_spec("reviewer.grade") is not None
    assert importlib.util.find_spec("reviewer.export_students") is not None


def test_export_excludes_private_materials_and_history(tmp_path):
    from reviewer.export_students import export_students

    dest = tmp_path / "student"
    export_students(dest)
    assert (dest / "ASSIGNMENT.md").is_file()
    assert len(list((dest / "scenes").glob("*.xml"))) == 8
    assert not (dest / "reviewer").exists()
    assert not (dest / ".git").exists()
    assert not (dest / "docs/superpowers").exists()
    assert not (dest / "tests/test_grading.py").exists()
    with pytest.raises(FileExistsError):
        export_students(dest)


def test_advanced_scenarios_are_deterministic_and_physically_valid():
    from reviewer.scenarios import scenario
    from physical_ai.env import ManipulationEnv
    import numpy as np

    for robot in ["ur5e", "iiwa14"]:
        for task in ["cup_plate", "cup_shelf", "swap", "sort"]:
            params = scenario(task, 714)
            assert params == scenario(task, 714) and params
            with ManipulationEnv(robot, task, parameters=params) as env:
                state, _ = env.reset(seed=12)
                assert np.isfinite(state).all() and not env.success()
                for _ in range(5):
                    env.step([0, 0, 0, 1])
                assert all(env.data.body(n).xpos[2] > 0.4 for n in env.object_names)


def test_submission_rejects_missing_matrix_and_path_escape(tmp_path):
    from reviewer.grade import validate_submission

    with pytest.raises(ValueError, match="eight"):
        validate_submission({"models": []}, tmp_path)


def test_color_targets_follow_appearance_not_indices():
    from physical_ai.env import ManipulationEnv
    import numpy as np

    params = {
        "geom": {
            "plate0": {"rgba": [0.1, 0.3, 0.85, 1]},
            "plate1": {"rgba": [0.8, 0.12, 0.12, 1]},
        }
    }
    with ManipulationEnv("ur5e", "sort", parameters=params) as env:
        assert abs(env.goals[0, 0] - 0.5) < 1e-8
        assert abs(env.goals[1, 0] - 0.34) < 1e-8


def test_missing_model_artifacts_are_isolated_in_grading_report(tmp_path):
    import json
    from reviewer.grade import grade

    models = []
    for robot in ["ur5e", "iiwa14"]:
        for task in ["cup_plate", "cup_shelf", "swap", "sort"]:
            models.append(
                dict(
                    robot=robot,
                    task=task,
                    rl="absent.zip",
                    bc_baseline="absent.ts",
                    bc_proprio="absent.ts",
                    bc_robust="absent.ts",
                )
            )
    (tmp_path / "REPORT.md").write_text("test")
    (tmp_path / "submission.json").write_text(
        json.dumps(dict(models=models, report="REPORT.md"))
    )
    results = grade(
        tmp_path / "submission.json", tmp_path / "out", episodes=1, max_steps=1
    )
    assert len(results) == 64 and all("error" in row for row in results)
    assert (tmp_path / "out/results.json").is_file()
