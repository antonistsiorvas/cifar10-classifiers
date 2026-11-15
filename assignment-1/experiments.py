import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DEFAULTS, run

ASSIGNMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ASSIGNMENT_DIR / "results"
SMOKE_DIR = ASSIGNMENT_DIR / "results_smoke"

# Reference models; everything not listed uses the defaults in train.py
# (ReLU, Adam, lr 1e-3, batch 128, unit, no augmentation/dropout/batchnorm)
MLP_REF = {"model": "mlp", "hidden": [512, 512, 512], "norm": "unit"}
CNN_REF = {"model": "cnn", "channels": [32, 64, 128], "kernel_size": 3, "pool": "max", "norm": "unit"}
REFS = [("mlp", MLP_REF), ("cnn", CNN_REF)]

# Every experiment is a list of (run name, config); one factor changes at a time.
# A configuration that appears twice inside one experiment (the reference) is run once.


def e1_mlp_width():
    return [(f"E1_mlp_h{w}", {"model": "mlp", "hidden": [w]})
            for w in [32, 64, 128, 256, 512, 1024, 2048]]


def e2_mlp_depth():
    return [(f"E2_mlp_d{d}_w{w}", {"model": "mlp", "hidden": [w] * d})
            for d in [2, 3, 4] for w in [64, 128, 256, 512]]


def e3_activations():
    return [(f"E3_{tag}_{act}", {**ref, "activation": act})
            for tag, ref in REFS
            for act in ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu"]]


def e4_normalization():
    return [(f"E4_{tag}_{norm}", {**ref, "norm": norm})
            for tag, ref in REFS
            for norm in ["raw", "unit"]]


def e5_cnn_variations():
    return [
        ("E5_cnn_ref", CNN_REF),  # kernel 3, channels x1, 3 blocks, max-pool, no BN/dropout/augmentation
        ("E5_cnn_k5", {**CNN_REF, "kernel_size": 5}),
        ("E5_cnn_k7", {**CNN_REF, "kernel_size": 7}),
        ("E5_cnn_ch16-32-64", {**CNN_REF, "channels": [16, 32, 64]}),
        ("E5_cnn_ch64-128-256", {**CNN_REF, "channels": [64, 128, 256]}),
        ("E5_cnn_ch32-64", {**CNN_REF, "channels": [32, 64]}),
        ("E5_cnn_ch32-64-128-256", {**CNN_REF, "channels": [32, 64, 128, 256]}),
        ("E5_cnn_avgpool", {**CNN_REF, "pool": "avg"}),
        ("E5_cnn_batchnorm", {**CNN_REF, "batchnorm": True}),
        ("E5_cnn_dropout0.25", {**CNN_REF, "dropout": 0.25}),
        ("E5_cnn_dropout0.5", {**CNN_REF, "dropout": 0.5}),
        ("E5_cnn_augment", {**CNN_REF, "augment": True}),
    ]


def e6_training_parameters():
    runs = []
    for tag, ref in REFS:
        runs += [
            (f"E6_{tag}_ref", ref),  # Adam, lr 1e-3, batch 128
            (f"E6_{tag}_lr1e-4", {**ref, "lr": 1e-4}),
            (f"E6_{tag}_lr1e-2", {**ref, "lr": 1e-2}),
            (f"E6_{tag}_bs32", {**ref, "batch_size": 32}),
            (f"E6_{tag}_bs512", {**ref, "batch_size": 512}),
            (f"E6_{tag}_sgd", {**ref, "optimizer": "sgd"}),
        ]
    return runs


EXPERIMENTS = {
    "E1": e1_mlp_width,
    "E2": e2_mlp_depth,
    "E3": e3_activations,
    "E4": e4_normalization,
    "E5": e5_cnn_variations,
    "E6": e6_training_parameters,
}


def main():
    parser = argparse.ArgumentParser(description="Run the experiments E1-E6")
    parser.add_argument("--exp", nargs="+", required=True,
                        help="experiments to run, e.g. --exp E1 E3, or --exp all")
    parser.add_argument("--dry-run", action="store_true", help="only list the runs")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny runs (2,000 training images, 2 epochs) into results_smoke/")
    parser.add_argument("--force", action="store_true", help="rerun runs that already finished")
    parser.add_argument("--verbose", action="store_true", help="print every epoch")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.exp == ["all"] else [e.upper() for e in args.exp]
    for e in names:
        if e not in EXPERIMENTS:
            parser.error(f"unknown experiment {e}, choose from {list(EXPERIMENTS)} or all")
    out_dir = SMOKE_DIR if args.smoke else RESULTS_DIR

    all_runs = [(e, name, cfg) for e in names for name, cfg in EXPERIMENTS[e]()]
    print(f"{len(all_runs)} runs in {', '.join(names)} -> {out_dir}")
    if args.dry_run:
        for e in names:
            print(f"  {e}: {sum(r[0] == e for r in all_runs)} runs")
        # Runs are not deduplicated across experiments; this only counts distinct configurations
        unique = {repr(sorted({**DEFAULTS, **cfg}.items())) for _, _, cfg in all_runs}
        print(f"  total {len(all_runs)} runs, {len(unique)} unique configurations")

    for i, (experiment, name, cfg) in enumerate(all_runs, start=1):
        config = {**cfg, "name": name, "experiment": experiment,
                  "out_dir": str(out_dir), "verbose": args.verbose}
        if args.smoke:
            config.update(train_subset=2000, epochs=2)
        prefix = f"[{i}/{len(all_runs)}] {name:<26}"

        if args.dry_run:
            print(f"{prefix} {cfg}")
            continue
        # Resume: a run is finished once its metrics.json exists
        if (out_dir / "runs" / name / "metrics.json").exists() and not args.force:
            print(f"{prefix} already done, skipping")
            continue

        m = run(config)
        print(f"{prefix} val {m['val_acc']:.2%}  test {m['test_acc']:.2%}  "
              f"epochs {m['epochs_run']} (best {m['best_epoch']})  {m['train_time_s']:.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
