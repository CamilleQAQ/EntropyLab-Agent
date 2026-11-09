"""Command-line entry point for EntropyLab Agent.

Subcommands are thin wrappers: every behaviour lives in a library module, so the
CLI cannot drift from the API used by the tests and scripts.

    entropylab agent run ...        run the Agent (units or seconds budget)
    entropylab train ...            one supervised experiment (test opened once)
    entropylab models ...           list / register / show / delete artifacts
    entropylab diagnose ...         diagnose one signal file
    entropylab diagnose-batch ...   diagnose a directory of signal files
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ..io.dataset import load_dataset_spec
from ..io.input_signal import SignalInput, describe_input, load_input_signal
from ..training.experiment import run_supervised_experiment


def _add_signal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--signal", type=Path, help="signal file (.csv/.npy/.npz/.mat)")
    parser.add_argument("--channel", default=None, help="CSV column name or 0-based index")
    parser.add_argument("--key", default=None, help="NPZ/MAT array name")
    parser.add_argument(
        "--index", action="append", type=int, default=None,
        help="index for one non-sample axis; repeat once per axis",
    )
    parser.add_argument("--sample-axis", type=int, default=-1)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument(
        "--sampling-rate", type=float, default=None,
        help="verified sampling rate of the file, in Hz; required for diagnosis",
    )
    parser.add_argument(
        "--signal-type", default=None,
        help="verified signal type for a registered profile, e.g. ACCELERATION",
    )
    parser.add_argument(
        "--signal-role", default=None,
        help="verified signal role for a registered profile, e.g. GEARBOX_VIBRATION_CHANNEL_1",
    )


def _signal_input(args: argparse.Namespace, path: Path) -> SignalInput:
    channel: str | int | None = args.channel
    if isinstance(channel, str) and channel.isdigit():
        channel = int(channel)
    return SignalInput(
        path=path,
        channel=channel,
        key=args.key,
        sample_axis=args.sample_axis,
        indices=tuple(args.index or ()),
        start_sample=args.start_sample,
        max_samples=args.max_samples,
        has_header=not args.no_header,
        delimiter=args.delimiter,
        skip_rows=args.skip_rows,
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _cmd_train(args: argparse.Namespace) -> int:
    spec = load_dataset_spec(args.config)
    output = Path(args.output)
    model_id = args.model_id or output.name
    try:
        summary = run_supervised_experiment(
            spec, args.data_root, output, model_id=model_id, config_path=args.config
        )
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"dataset      : {summary['dataset_id']}")
    print(f"feature      : {summary['feature']['method']} (dim {summary['feature']['dimension']})")
    print(f"classifier   : {summary['classifier']['algorithm']}")
    print(f"records      : {summary['split']['records']}")
    print(f"validation   : accuracy {_fmt(summary['validation']['multiclass_accuracy'])} "
          f"macro-F1 {_fmt(summary['validation']['multiclass_macro_f1'])}")
    print(f"test         : accuracy {_fmt(summary['test']['multiclass_accuracy'])} "
          f"macro-F1 {_fmt(summary['test']['multiclass_macro_f1'])}")
    print(f"model        : {summary['model_dir']}")
    return 0


def _cmd_agent_run(args: argparse.Namespace) -> int:
    from ..agent import run_agent

    spec = load_dataset_spec(args.spec)
    summary = run_agent(
        spec,
        args.data_root,
        args.output,
        rounds=args.rounds,
        per_round=args.per_round,
        units=args.units,
        budget_seconds=args.budget_seconds,
        seeds=tuple(int(part) for part in args.seeds.split(",") if part.strip()),
        validation_repeats=args.validation_repeats,
        test_mode=args.test_mode,
        arms=tuple(part.strip() for part in args.arms.split(",") if part.strip()),
        cache_root=args.cache_root,
        run_test=not args.no_test,
    )
    budget = summary["budget"]
    mode = budget.get("mode")
    if mode == "seconds":
        print(f"budget       : {budget['budget_seconds']:.1f} s (estimated extraction seconds)")
    else:
        print(f"budget       : {mode}")
    for arm, stats in summary["aggregate"].items():
        print(f"{arm:12s} : validation {_fmt(stats['mean'])} ± {_fmt(stats['std'])}  wins {stats['wins']}")
    if summary["champion"]:
        print(f"champion     : {summary['champion']['method']} + {summary['champion']['classifier']}")
    test = summary.get("test")
    if test:
        if test.get("mode") == "per-seed":
            print(f"test         : {_fmt(test['mean_accuracy'])} ± {_fmt(test['std_accuracy'])} "
                  f"({test['seeds']} seeds)")
        else:
            print(f"test         : {_fmt(test['test']['multiclass_accuracy'])} (single split)")
    print(f"report       : {Path(args.output) / 'report.md'}")
    return 0


def _cmd_continual_run(args: argparse.Namespace) -> int:
    from ..agent.candidates import build_pool
    from ..continual.search_policy import SearchPolicy
    from ..continual import BatchPlan, run_continual

    spec = load_dataset_spec(args.spec)
    plan = BatchPlan.load(args.plan)
    summary = run_continual(
        spec,
        args.data_root,
        plan,
        args.output,
        model_prefix=args.model_prefix,
        run_holdout=not args.no_holdout,
        criterion=args.criterion,
        guard_tolerance=args.guard_tolerance,
        min_gain=args.min_gain,
        select=args.select,
        selection_budget=args.selection_budget,
        pool=build_pool() if args.select else None,
        search_policy=(
            SearchPolicy.adaptive(
                min_lead=args.stop_lead, patience=args.stop_patience
            )
            if args.search_policy == "adaptive"
            else SearchPolicy()
        ),
    )
    print(f"plan         : {args.plan}")
    print(f"batches      : {summary['plan']['groups']['batches']}")
    for row in summary["trajectory"]:
        decision = "promote" if row["promoted"] else "reject"
        def _m(value):
            return "—" if value is None else f"{float(value):.4f}"

        print(
            f"batch {row['batch_id']} ({row['note']}): training {row['training_records']} "
            f"challenger {_m(row['challenger_macro_f1'])} -> champion "
            f"{_m(row['champion_macro_f1'])} [{decision}]"
        )
    if summary.get("holdout"):
        for name in ("frozen", "continual", "joint"):
            entry = summary["holdout"].get(name)
            if entry:
                print(f"holdout {name:9s}: accuracy {entry['accuracy']:.4f} macro-F1 {entry['macro_f1']:.4f}")
    if summary.get("selection", {}).get("enabled"):
        champion = summary.get("champion_configuration") or {}
        print(
            f"selection    : pool {summary['selection']['pool_size']}, budget "
            f"{summary['selection']['budget_seconds']:.1f}s/round, "
            f"policy {summary['selection']['search_policy']['enabled']}, final "
            f"{champion.get('feature', '—')} + {champion.get('classifier', '—')}"
        )
    print(f"report       : {Path(args.output) / 'report.md'}")
    return 0


def _cmd_lab_run(args: argparse.Namespace) -> int:
    from ..lab import LabContext, run_lab

    spec = load_dataset_spec(args.spec)
    ctx = LabContext(
        spec=spec,
        data_root=args.data_root,
        workspace=args.output,
        goal=args.goal,
        budget_seconds=args.budget,
    )
    summary = run_lab(
        ctx,
        goal=args.goal,
        max_rounds=args.max_rounds,
        interactive=args.interactive,
    )
    print(f"status       : {summary['status']}")
    print(f"rounds       : {summary['rounds']}")
    print(f"spent        : {summary['spent_seconds']} s")
    print(f"hypotheses   : {len(summary['hypotheses'])}")
    for row in summary["hypotheses"]:
        print(f"  - r{row['round']}: {row['text'][:70]}")
    print(f"experiments  : {len(summary['experiments'])}")
    for review in summary["reviews"]:
        codes = [f["code"] for f in review["findings"]]
        print(f"  review r{review['round']}: {review['verdict']} {codes}")
    print(f"report       : {summary['report']}")
    return 0


def _cmd_lab_factory(args: argparse.Namespace) -> int:
    from ..lab import LabContext, chat
    from ..lab.factory import build, propose

    spec = load_dataset_spec(args.spec)
    ctx = LabContext(
        spec=spec,
        data_root=args.data_root,
        workspace=args.output,
        goal=args.goal,
        budget_seconds=args.budget,
    )
    candidate = propose(chat, args.goal)
    print(f"proposal     : {candidate.name}")
    print(f"rationale    : {candidate.rationale[:120]}")
    result = build(ctx, candidate, seeds=args.seeds, llm=chat)
    print(f"stage        : {result['stage']}")
    print(f"accepted     : {result['accepted']}")
    if result.get("metrics"):
        metrics = result["metrics"]
        baseline = result["baseline"]
        print(
            f"candidate    : {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}"
        )
        print(
            f"baseline     : {baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_std']:.4f}"
        )
        print(f"delta        : {result['delta']:+.4f} (needs > {result['threshold']:.4f})")
    for problem in result.get("problems") or []:
        print(f"problem      : {problem}")
    if result.get("registered"):
        print(f"registered   : {result['registered']}")
    return 0


def _registry(args: argparse.Namespace):
    from ..diagnosis import ModelRegistry

    return ModelRegistry(args.registry, trusted=args.trusted)


def _cmd_models(args: argparse.Namespace) -> int:
    registry = _registry(args)
    if args.models_command == "list":
        for entry in registry.entries():
            artifact = Path(entry["path"])
            missing = "  [MISSING]" if not artifact.is_dir() else ""
            profile_id = ""
            config = artifact / "config.json"
            if config.is_file():
                try:
                    profile_id = json.loads(config.read_text(encoding="utf-8"))[
                        "diagnosis"
                    ]["profile_id"]
                except (OSError, ValueError, KeyError):
                    profile_id = ""
            print(f"{entry['model_id']:60s} {profile_id}")
            print(f"    {artifact}{missing}")
        if not registry.model_ids():
            print("(registry is empty)")
        return 0
    if args.models_command == "show":
        print(json.dumps(registry.show(args.model_id), ensure_ascii=False, indent=2))
        return 0
    if args.models_command == "register":
        entry = registry.register(args.artifact, replace=args.replace)
        print(f"registered {entry['model_id']} -> {entry['path']}")
        return 0
    if args.models_command == "delete":
        path = registry.delete(args.model_id, delete_files=args.delete_files)
        print(f"unregistered {args.model_id} (files at {path}{'' if args.delete_files else ' kept'})")
        return 0
    raise ValueError(f"unknown models command {args.models_command!r}")


def _diagnose_one(args: argparse.Namespace, path: Path) -> dict:
    from ..diagnosis import MatchStatus, diagnose_artifact_signal

    loaded = load_input_signal(_signal_input(args, path))
    if args.artifact is not None:
        model_path = Path(args.artifact)
        runner = None
    else:
        # A registered profile is a contract: the metadata must be supplied
        # explicitly and is never inferred from the signal itself.
        registry = _registry(args).as_diagnosis_registry()
        match = registry.match(
            {
                "diagnostic_profile": args.profile,
                "sampling_rate_hz": args.sampling_rate,
                "signal_type": args.signal_type,
                "signal_role": args.signal_role,
            },
            sample_count=loaded.analyzed_samples,
        )
        if match.status is not MatchStatus.MATCHED or match.registration is None:
            detail = "; ".join(match.required_metadata or match.reasons or ())
            raise ValueError(f"profile match failed ({match.status.value}): {detail}")
        model_path = match.registration.model_path
        runner = match.registration.runner
    if runner is None:
        report = diagnose_artifact_signal(
            loaded.values,
            model_path,
            sampling_rate_hz=args.sampling_rate,
            trusted=args.trusted,
        )
    else:
        report = runner(loaded.values, model_path, sampling_rate_hz=args.sampling_rate)
    report["_signal"] = str(path)
    return report


def _print_diagnosis(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    ood = report.get("ood") or {}
    print(f"signal       : {report.get('_signal', '')}")
    print(f"status       : {report['status']}")
    print(f"model        : {report.get('model_id', '—')}")
    if report["status"] == "DIAGNOSED":
        print(f"diagnosis    : {report['diagnosis']} (class {report['class_id']})")
        print(f"vote share   : {report['vote_share']:.2f}")
    else:
        print(f"reason       : {report.get('reason', '—')}")
    if ood.get("enabled"):
        print(
            f"ood          : rejected={ood['rejected']} "
            f"outlier_windows={ood['outlier_windows']}/{ood['window_count']} "
            f"threshold={ood['threshold']:.4g}"
        )
    print(f"windows      : {report.get('window_count', 0)}")


def _cmd_diagnose(args: argparse.Namespace) -> int:
    if args.signal is None:
        print("error: --signal is required", file=sys.stderr)
        return 2
    if args.artifact is None and args.profile is None:
        print("error: pass --artifact DIR or --registry ROOT --profile ID", file=sys.stderr)
        return 2
    report = _diagnose_one(args, args.signal)
    _print_diagnosis(report, args.json)
    return 0 if report["status"] == "DIAGNOSED" else 1


def _cmd_diagnose_batch(args: argparse.Namespace) -> int:
    if args.input_dir is None:
        print("error: --input-dir is required", file=sys.stderr)
        return 2
    files = sorted(Path(args.input_dir).glob(args.pattern))
    if not files:
        print(f"error: no files match {args.pattern} in {args.input_dir}", file=sys.stderr)
        return 2
    rows = []
    for path in files:
        try:
            report = _diagnose_one(args, path)
        except Exception as error:  # a bad file must not stop the batch
            rows.append(
                {
                    "signal": str(path),
                    "status": "ERROR",
                    "diagnosis": "",
                    "class_id": "",
                    "vote_share": "",
                    "windows": "",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        rows.append(
            {
                "signal": str(path),
                "status": report["status"],
                "diagnosis": report.get("diagnosis", ""),
                "class_id": report.get("class_id", ""),
                "vote_share": round(float(report.get("vote_share") or 0.0), 4),
                "windows": report["window_count"],
                "reason": report.get("reason", ""),
            }
        )
    fieldnames = list(rows[0])
    handle = (
        sys.stdout
        if args.output is None
        else Path(args.output).open("w", encoding="utf-8", newline="")
    )
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if handle is not sys.stdout:
            handle.close()
    diagnosed = sum(1 for row in rows if row["status"] == "DIAGNOSED")
    print(f"diagnosed {diagnosed}/{len(rows)} files", file=sys.stderr)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Show what a signal file contains without guessing a selection."""
    print(describe_input(args.signal, has_header=not args.no_header, delimiter=args.delimiter))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entropylab",
        description="EntropyLab Agent: experiment orchestration, training, and diagnosis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- agent -----------------------------------------------------------
    agent = sub.add_parser("agent", help="agent commands")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)

    run = agent_sub.add_parser("run", help="run the Agent and its control arms")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--units", type=int, default=None)
    run.add_argument("--budget-seconds", type=float, default=None)
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--per-round", type=int, default=3)
    run.add_argument("--seeds", default="20261104")
    run.add_argument("--validation-repeats", type=int, default=3)
    run.add_argument("--test-mode", default="single", choices=("single", "per-seed"))
    run.add_argument("--arms", default="agent,random,fixed,exhaustive")
    run.add_argument("--cache-root", type=Path, default=None)
    run.add_argument("--no-test", action="store_true")
    run.set_defaults(handler=_cmd_agent_run)

    # -- train -----------------------------------------------------------
    train = sub.add_parser("train", help="one supervised experiment (test opened once)")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model-id", default=None)
    train.set_defaults(handler=_cmd_train)

    # -- models ----------------------------------------------------------
    models = sub.add_parser("models", help="model registry commands")
    models.add_argument(
        "--registry",
        type=Path,
        default=Path("models/registry/registry.json"),
    )
    models.add_argument("--trusted", action="store_true")
    models_sub = models.add_subparsers(dest="models_command", required=True)

    models_sub.add_parser("list", help="list registered models").set_defaults(
        handler=_cmd_models
    )

    show = models_sub.add_parser("show", help="show one registered model")
    show.add_argument("model_id")
    show.set_defaults(handler=_cmd_models)

    register = models_sub.add_parser("register", help="register an artifact directory")
    register.add_argument("--artifact", type=Path, required=True)
    register.add_argument("--replace", action="store_true")
    register.set_defaults(handler=_cmd_models)

    delete = models_sub.add_parser("delete", help="unregister (and optionally delete) a model")
    delete.add_argument("model_id")
    delete.add_argument("--delete-files", action="store_true")
    delete.set_defaults(handler=_cmd_models)

    # -- diagnose --------------------------------------------------------
    diagnose = sub.add_parser("diagnose", help="diagnose one signal file")
    diagnose.add_argument("--artifact", type=Path, default=None)
    diagnose.add_argument("--registry", type=Path, default=None)
    diagnose.add_argument("--profile", default=None)
    diagnose.add_argument("--trusted", action="store_true")
    diagnose.add_argument("--json", action="store_true")
    _add_signal_arguments(diagnose)
    diagnose.set_defaults(handler=_cmd_diagnose)

    batch = sub.add_parser("diagnose-batch", help="diagnose every matching file in a directory")
    batch.add_argument("--artifact", type=Path, default=None)
    batch.add_argument("--registry", type=Path, default=None)
    batch.add_argument("--profile", default=None)
    batch.add_argument("--trusted", action="store_true")
    batch.add_argument("--input-dir", type=Path, default=None)
    batch.add_argument("--pattern", default="*.csv")
    batch.add_argument("--output", type=Path, default=None)
    _add_signal_arguments(batch)
    batch.set_defaults(handler=_cmd_diagnose_batch)

    # -- continual -------------------------------------------------------
    continual = sub.add_parser("continual", help="continual learning commands")
    continual_sub = continual.add_subparsers(dest="continual_command", required=True)
    continual_run = continual_sub.add_parser("run", help="run the batch loop")
    continual_run.add_argument("--spec", type=Path, required=True)
    continual_run.add_argument("--data-root", type=Path, required=True)
    continual_run.add_argument("--plan", type=Path, required=True)
    continual_run.add_argument("--output", type=Path, required=True)
    continual_run.add_argument("--model-prefix", default="continual")
    continual_run.add_argument("--no-holdout", action="store_true")
    continual_run.add_argument(
        "--criterion", choices=("margin", "guardrail"), default="margin",
        help="promotion rule: fixed validation margin or held-out probe guardrail",
    )
    continual_run.add_argument("--guard-tolerance", type=float, default=0.02)
    continual_run.add_argument("--min-gain", type=float, default=0.0)
    continual_run.add_argument(
        "--select", action="store_true",
        help="re-select the feature family and classifier each round",
    )
    continual_run.add_argument("--selection-budget", type=float, default=12.0)
    continual_run.add_argument(
        "--search-policy", choices=("full", "adaptive"), default="full",
        help="full: spend the whole budget; adaptive: stop on lead/patience",
    )
    continual_run.add_argument("--stop-lead", type=float, default=0.02)
    continual_run.add_argument("--stop-patience", type=int, default=6)
    continual_run.set_defaults(handler=_cmd_continual_run)

    # -- lab -------------------------------------------------------------
    lab = sub.add_parser("lab", help="autonomous research lab")
    lab_sub = lab.add_subparsers(dest="lab_command", required=True)
    lab_run = lab_sub.add_parser("run", help="run the plan/execute/review loop")
    lab_run.add_argument("--spec", type=Path, required=True)
    lab_run.add_argument("--data-root", type=Path, required=True)
    lab_run.add_argument("--output", type=Path, required=True)
    lab_run.add_argument("--goal", required=True)
    lab_run.add_argument("--max-rounds", type=int, default=6)
    lab_run.add_argument("--budget", type=float, default=600.0, help="seconds")
    lab_run.add_argument("--interactive", action="store_true", help="pause for approval before risky actions")
    lab_run.set_defaults(handler=_cmd_lab_run)

    lab_factory = lab_sub.add_parser("factory", help="generate and validate a new feature extractor")
    lab_factory.add_argument("--spec", type=Path, required=True)
    lab_factory.add_argument("--data-root", type=Path, required=True)
    lab_factory.add_argument("--output", type=Path, required=True)
    lab_factory.add_argument("--goal", required=True)
    lab_factory.add_argument("--seeds", type=int, default=3)
    lab_factory.add_argument("--budget", type=float, default=600.0)
    lab_factory.set_defaults(handler=_cmd_lab_factory)

    # -- inspect ---------------------------------------------------------
    inspect = sub.add_parser("inspect", help="describe a signal file's columns/arrays")
    inspect.add_argument("--signal", type=Path, required=True)
    inspect.add_argument("--no-header", action="store_true")
    inspect.add_argument("--delimiter", default=",")
    inspect.set_defaults(handler=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
