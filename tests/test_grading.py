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
    assert not (dest / "docs").exists()
    assert not list(dest.rglob("*.npz"))
    assert "ORGANIZER_ONLY" not in (dest / "README.md").read_text()
    assert "reviewer/" not in (dest / "README.md").read_text()
    assert not (dest / "tests/test_grading.py").exists()
    with pytest.raises(FileExistsError):
        export_students(dest)


def test_advanced_scenarios_are_deterministic_and_physically_valid():
    from reviewer.scenarios import scenario
    from physical_ai.env import ManipulationEnv
    import numpy as np

    for robot in ["ur5e", "iiwa14"]:
        for task in ["cup_plate", "cup_shelf", "cup_distractor", "color_match"]:
            params = scenario(task, 714, robot=robot)
            assert params == scenario(task, 714, robot=robot) and params
            with ManipulationEnv(robot, task, parameters=params) as env:
                state, _ = env.reset(seed=12)
                assert np.isfinite(state).all() and not env.success()
                for _ in range(5):
                    env.step([0, 0, 0, 1])
                assert all(env.data.body(n).xpos[2] > 0.4 for n in env.object_names)


@pytest.mark.parametrize("robot", ["ur5e", "iiwa14"])
@pytest.mark.parametrize(
    "task", ["cup_plate", "cup_shelf", "cup_distractor", "color_match"]
)
def test_submission_accepts_any_single_supported_pair(tmp_path, robot, task):
    import json
    from reviewer.grade import validate_submission

    submission = json.loads(
        (Path(__file__).parents[1] / "submission.example.json").read_text()
    )
    submission["models"][0].update(robot=robot, task=task)
    (tmp_path / "REPORT.md").write_text("test")
    assert validate_submission(submission, tmp_path) == submission["models"]


def test_submission_rejects_wrong_count_unknown_pair_and_path_escape(tmp_path):
    import copy
    import json
    from reviewer.grade import validate_submission

    submission = json.loads(
        (Path(__file__).parents[1] / "submission.example.json").read_text()
    )
    (tmp_path / "REPORT.md").write_text("test")
    for entries in ([], submission["models"] * 2):
        with pytest.raises(ValueError, match="exactly one"):
            validate_submission(dict(submission, models=entries), tmp_path)
    for key in ("robot", "task"):
        invalid = copy.deepcopy(submission)
        invalid["models"][0][key] = "unknown"
        with pytest.raises(ValueError, match="Unsupported"):
            validate_submission(invalid, tmp_path)
    invalid = copy.deepcopy(submission)
    invalid["models"][0]["rl"] = "../outside.zip"
    with pytest.raises(ValueError, match="escapes"):
        validate_submission(invalid, tmp_path)


def test_color_targets_follow_appearance_not_indices():
    from physical_ai.env import ManipulationEnv
    import numpy as np

    params = {
        "geom": {
            "plate0": {"rgba": [0.1, 0.3, 0.85, 1]},
            "plate1": {"rgba": [0.8, 0.12, 0.12, 1]},
        }
    }
    with ManipulationEnv("ur5e", "color_match", parameters=params) as env:
        for seed in range(8):
            env.reset(seed=seed)
            color = env.model.geom("cup0_bottom").rgba
            match = next(
                i
                for i in range(2)
                if np.allclose(color, env.model.geom(f"plate{i}").rgba)
            )
            np.testing.assert_allclose(
                env.goals[0], env.data.body(f"target{match}").xpos + [0, 0, 0.034]
            )


def test_missing_model_artifacts_are_isolated_in_grading_report(tmp_path):
    import json
    from reviewer.grade import grade

    models = [
        dict(
            robot="iiwa14",
            task="color_match",
            rl="absent.zip",
            bc_baseline="absent.ts",
            bc_proprio="absent.ts",
            bc_robust="absent.ts",
        )
    ]
    (tmp_path / "REPORT.md").write_text("test")
    (tmp_path / "submission.json").write_text(
        json.dumps(dict(models=models, report="REPORT.md"))
    )
    results = grade(
        tmp_path / "submission.json", tmp_path / "out", episodes=1, max_steps=1
    )
    assert len(results) == 8 and all("error" in row for row in results)
    assert {(r["robot"], r["task"]) for r in results} == {("iiwa14", "color_match")}
    assert {(r["variant"], r["split"]) for r in results} == {
        (v, split)
        for v in ("rl", "bc_baseline", "bc_proprio", "bc_robust")
        for split in ("base", "advanced")
    }
    assert (tmp_path / "out/results.json").is_file()


def test_public_recipe_removes_only_model_equivalent_parameters():
    from reviewer.export_curriculum import public_parameters

    params = {"body": {"target0": {"pos": [0.48, 0.17, 0.454]}}}
    options, digest = public_parameters("iiwa14", "cup_shelf", params)
    assert options == {} and len(digest) == 64
    with pytest.raises(ValueError, match="different"):
        public_parameters(
            "iiwa14", "cup_shelf", {"body": {"target0": {"pos": [0.48, 0.17, 0.6]}}}
        )


def test_public_recipe_has_no_task_specific_parameter_overrides():
    import json

    recipe = json.loads(
        (Path(__file__).parents[1] / "configs/ppo_calibrated.json").read_text()
    )
    assert all(not node["train"].get("parameters") for node in recipe["stages"])
