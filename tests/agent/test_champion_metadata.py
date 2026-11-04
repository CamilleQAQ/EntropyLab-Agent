from __future__ import annotations

import json
from pathlib import Path

from entropy_agent.agent import ClassifierSpec, MethodSpec, build_pool, run_agent
from entropy_agent.agent.candidates import Candidate
from entropy_agent.agent.loop import champion_diagnosis, champion_model_id

METHOD = MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10})
CLASSIFIER = ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0})
CANDIDATE = Candidate(method=METHOD, classifier=CLASSIFIER)


def test_display_name_replaces_the_configured_combination(tiny_dataset) -> None:
    spec, _ = tiny_dataset
    configured = spec.model_copy(
        update={
            "diagnosis": spec.diagnosis.model_copy(
                update={"display_name": "Toy rig, MDE + SVM-RBF"}
            )
        }
    )

    diagnosis = champion_diagnosis(configured, CANDIDATE)

    assert diagnosis.display_name == "Toy rig — agent-selected MPE + EXTRA-TREES"
    assert "MDE" not in diagnosis.display_name
    assert diagnosis.profile_id == spec.diagnosis.profile_id


def test_display_name_falls_back_when_configuration_has_no_combination(tiny_dataset) -> None:
    spec, _ = tiny_dataset

    diagnosis = champion_diagnosis(spec, CANDIDATE)

    assert diagnosis.display_name == "tiny — agent-selected MPE + EXTRA-TREES"


def test_model_id_names_the_selected_combination(tiny_dataset) -> None:
    spec, _ = tiny_dataset

    model_id = champion_model_id(spec, CANDIDATE)

    assert model_id == "tiny_test_set-agent-mpe-extra-trees"


def test_champion_artifact_describes_the_selected_model(tiny_dataset, tmp_path: Path) -> None:
    spec, root = tiny_dataset
    output = tmp_path / "run"
    pool = build_pool(
        methods=(
            MethodSpec("MDE", {"m": 2, "nc": 4, "tau": 1, "scale": 10}),
            MethodSpec("MPE", {"m": 3, "tau": 1, "scale": 10}),
        ),
        classifiers=(
            ClassifierSpec("SVM-RBF", "scikit-learn", {"C": 1.0, "gamma": "scale"}),
            ClassifierSpec("EXTRA-TREES", "scikit-learn", {"n_estimators": 40, "random_state": 0}),
        ),
    )
    summary = run_agent(
        spec,
        root,
        output,
        rounds=1,
        per_round=4,
        arms=("agent",),
        pool=pool,
        run_id="metadata-run",
    )

    champion = summary["champion"]
    assert champion is not None
    config = json.loads((output / "champion_model" / "config.json").read_text(encoding="utf-8"))
    display_name = config["diagnosis"]["display_name"]
    assert champion["method"] in display_name
    assert champion["classifier"] in display_name
    assert config["model_id"] == f"tiny_test_set-agent-{champion['method'].lower()}-{champion['classifier'].lower()}"
    assert config["diagnosis"]["profile_id"] == spec.diagnosis.profile_id
