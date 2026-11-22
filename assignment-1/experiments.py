import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import DEFAULTS, run

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SMOKE_DIR = Path(__file__).resolve().parent / "results_smoke"

MLP_REF = {"model": "mlp", "hidden": [512, 512, 512]}
CNN_REF = {"model": "cnn", "channels": [32, 64, 128], "kernel_size": 3, "pool": "max"}

E1 = [
    {"name": "E1_mlp_h32", "model": "mlp", "hidden": [32]},
    {"name": "E1_mlp_h64", "model": "mlp", "hidden": [64]},
    {"name": "E1_mlp_h128", "model": "mlp", "hidden": [128]},
    {"name": "E1_mlp_h256", "model": "mlp", "hidden": [256]},
    {"name": "E1_mlp_h512", "model": "mlp", "hidden": [512]},
    {"name": "E1_mlp_h1024", "model": "mlp", "hidden": [1024]},
    {"name": "E1_mlp_h2048", "model": "mlp", "hidden": [2048]},
]

E2 = [
    {"name": "E2_mlp_d2_w64", "model": "mlp", "hidden": [64, 64]},
    {"name": "E2_mlp_d2_w128", "model": "mlp", "hidden": [128, 128]},
    {"name": "E2_mlp_d2_w256", "model": "mlp", "hidden": [256, 256]},
    {"name": "E2_mlp_d2_w512", "model": "mlp", "hidden": [512, 512]},
    {"name": "E2_mlp_d3_w64", "model": "mlp", "hidden": [64, 64, 64]},
    {"name": "E2_mlp_d3_w128", "model": "mlp", "hidden": [128, 128, 128]},
    {"name": "E2_mlp_d3_w256", "model": "mlp", "hidden": [256, 256, 256]},
    {"name": "E2_mlp_d3_w512", "model": "mlp", "hidden": [512, 512, 512]},
    {"name": "E2_mlp_d4_w64", "model": "mlp", "hidden": [64, 64, 64, 64]},
    {"name": "E2_mlp_d4_w128", "model": "mlp", "hidden": [128, 128, 128, 128]},
    {"name": "E2_mlp_d4_w256", "model": "mlp", "hidden": [256, 256, 256, 256]},
    {"name": "E2_mlp_d4_w512", "model": "mlp", "hidden": [512, 512, 512, 512]},
]

E3 = [
    {**MLP_REF, "name": "E3_mlp_sigmoid", "activation": "sigmoid"},
    {**MLP_REF, "name": "E3_mlp_tanh", "activation": "tanh"},
    {**MLP_REF, "name": "E3_mlp_relu", "activation": "relu"},
    {**MLP_REF, "name": "E3_mlp_leaky_relu", "activation": "leaky_relu"},
    {**MLP_REF, "name": "E3_mlp_elu", "activation": "elu"},
    {**MLP_REF, "name": "E3_mlp_gelu", "activation": "gelu"},
    {**CNN_REF, "name": "E3_cnn_sigmoid", "activation": "sigmoid"},
    {**CNN_REF, "name": "E3_cnn_tanh", "activation": "tanh"},
    {**CNN_REF, "name": "E3_cnn_relu", "activation": "relu"},
    {**CNN_REF, "name": "E3_cnn_leaky_relu", "activation": "leaky_relu"},
    {**CNN_REF, "name": "E3_cnn_elu", "activation": "elu"},
    {**CNN_REF, "name": "E3_cnn_gelu", "activation": "gelu"},
]

E4 = [
    {**MLP_REF, "name": "E4_mlp_raw", "norm": "raw"},
    {**MLP_REF, "name": "E4_mlp_unit", "norm": "unit"},
    {**CNN_REF, "name": "E4_cnn_raw", "norm": "raw"},
    {**CNN_REF, "name": "E4_cnn_unit", "norm": "unit"},
]

E5 = [
    {**CNN_REF, "name": "E5_cnn_ref"},
    {**CNN_REF, "name": "E5_cnn_k5", "kernel_size": 5},
    {**CNN_REF, "name": "E5_cnn_k7", "kernel_size": 7},
    {**CNN_REF, "name": "E5_cnn_ch16-32-64", "channels": [16, 32, 64]},
    {**CNN_REF, "name": "E5_cnn_ch64-128-256", "channels": [64, 128, 256]},
    {**CNN_REF, "name": "E5_cnn_ch32-64", "channels": [32, 64]},
    {**CNN_REF, "name": "E5_cnn_ch32-64-128-256", "channels": [32, 64, 128, 256]},
    {**CNN_REF, "name": "E5_cnn_avgpool", "pool": "avg"},
    {**CNN_REF, "name": "E5_cnn_batchnorm", "batchnorm": True},
    {**CNN_REF, "name": "E5_cnn_dropout0.25", "dropout": 0.25},
    {**CNN_REF, "name": "E5_cnn_dropout0.5", "dropout": 0.5},
    {**CNN_REF, "name": "E5_cnn_augment", "augment": True},
]

E6 = [
    {**MLP_REF, "name": "E6_mlp_ref"},
    {**MLP_REF, "name": "E6_mlp_lr1e-4", "lr": 1e-4},
    {**MLP_REF, "name": "E6_mlp_lr1e-2", "lr": 1e-2},
    {**MLP_REF, "name": "E6_mlp_bs32", "batch_size": 32},
    {**MLP_REF, "name": "E6_mlp_bs512", "batch_size": 512},
    {**MLP_REF, "name": "E6_mlp_sgd", "optimizer": "sgd"},
    {**CNN_REF, "name": "E6_cnn_ref"},
    {**CNN_REF, "name": "E6_cnn_lr1e-4", "lr": 1e-4},
    {**CNN_REF, "name": "E6_cnn_lr1e-2", "lr": 1e-2},
    {**CNN_REF, "name": "E6_cnn_bs32", "batch_size": 32},
    {**CNN_REF, "name": "E6_cnn_bs512", "batch_size": 512},
    {**CNN_REF, "name": "E6_cnn_sgd", "optimizer": "sgd"},
]

E7 = [
    {"name": "E7_mlp_512-256", "model": "mlp", "hidden": [512, 256]},
    {"name": "E7_mlp_256-512", "model": "mlp", "hidden": [256, 512]},
    {"name": "E7_mlp_256-128-64", "model": "mlp", "hidden": [256, 128, 64]},
    {"name": "E7_mlp_64-128-256", "model": "mlp", "hidden": [64, 128, 256]},
    {"name": "E7_mlp_512-256-128", "model": "mlp", "hidden": [512, 256, 128]},
    {"name": "E7_mlp_128-256-512", "model": "mlp", "hidden": [128, 256, 512]},
    {"name": "E7_mlp_1024-512-256", "model": "mlp", "hidden": [1024, 512, 256]},
    {"name": "E7_mlp_256-512-1024", "model": "mlp", "hidden": [256, 512, 1024]},
    {"name": "E7_mlp_176-88-44", "model": "mlp", "hidden": [176, 88, 44]},
]

E8 = [
    {**MLP_REF, "name": "E8_mlp_best", "activation": "elu", "lr": 1e-4, "batch_size": 512,
     "epochs": 100},
    {**CNN_REF, "name": "E8_cnn_best", "activation": "leaky_relu", "batchnorm": True,
     "dropout": 0.5, "augment": True, "epochs": 100},
]

EXPERIMENTS = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6": E6, "E7": E7, "E8": E8}


def main():
    parser = argparse.ArgumentParser(description="Run the experiments E1-E8")
    parser.add_argument("--exp", nargs="+", required=True, help="e.g. --exp E1 E3, or --exp all")
    parser.add_argument("--smoke", action="store_true",
                        help="quick test: 2,000 training images, 2 epochs, into results_smoke/")
    args = parser.parse_args()

    if args.exp == ["all"]:
        names = list(EXPERIMENTS.keys())
    else:
        names = args.exp
    if args.smoke:
        out_dir = SMOKE_DIR
    else:
        out_dir = RESULTS_DIR

    for exp in names:
        print(f"=== {exp}: {len(EXPERIMENTS[exp])} runs ===")
        for entry in EXPERIMENTS[exp]:
            config = {"experiment": exp, "name": entry["name"]}
            config.update(DEFAULTS)
            config.update(entry)
            if args.smoke:
                config["train_subset"] = 2000
                config["epochs"] = 2

            run_dir = out_dir / "runs" / config["name"]
            if (run_dir / "metrics.json").exists():
                print(f"{config['name']}: already done, skipping")
                continue

            print(f"{config['name']}:")
            results = run(config, run_dir, out_dir / "summary.csv")
            print(f"  -> val {results['val_acc']:.2%}  test {results['test_acc']:.2%}  "
                  f"(best epoch {results['best_epoch']}, {results['train_time_s']:.1f} s)")


if __name__ == "__main__":
    main()
