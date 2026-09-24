"""MAC experiment definitions. No tensor dependency: usable before model creation."""

import hashlib
import json
import math
from pathlib import Path


WINDOWS = {
    "none": (0.0, 0.0),
    "all": (0.0, 1.0),
    "early": (0.0, 0.5),
    "middle": (0.25, 0.75),
    "late": (0.5, 1.0),
}


def mac_bounds(args):
    """Return a zero-based, half-open [start, end) interval in optimizer steps."""
    timing = getattr(args, "mac_timing", "all")
    if timing == "custom":
        lo, hi = args.mac_start_fraction, args.mac_end_fraction
    elif timing in WINDOWS:
        lo, hi = WINDOWS[timing]
    else:
        raise ValueError(f"Unknown mac_timing: {timing}")
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi <= 1):
        raise ValueError("MAC fractions must satisfy 0 <= start <= end <= 1.")
    if timing != "none" and lo == hi:
        raise ValueError("MAC interval must have a positive length.")
    return int(args.total_iters * lo), int(args.total_iters * hi)


def validate_mac_args(args):
    if args.total_iters <= 0 or args.grad_accum <= 0 or args.batch_size <= 0:
        raise ValueError("total_iters, grad_accum and batch_size must be positive.")
    start, end = mac_bounds(args)
    if getattr(args, "mac_target", "both") not in ("both", "main", "aux"):
        raise ValueError("mac_target must be both, main or aux.")
    if getattr(args, "mac_selection", "model") not in ("model", "random"):
        raise ValueError("mac_selection must be model or random.")
    if getattr(args, "mac_score", "h0") not in ("h0", "h1", "mix"):
        raise ValueError("mac_score must be h0 (original MAC), h1 or mix.")
    if not 0 < args.mac_percent <= 1 or not math.isfinite(args.mac_percent):
        raise ValueError("mac_percent must be in (0, 1].")
    if not math.isfinite(args.mac_weight) or args.mac_weight < 0:
        raise ValueError("mac_weight must be finite and non-negative.")
    if args.mac_warmup_iters < 0:
        raise ValueError("mac_warmup_iters must be non-negative.")
    enabled = args.mac and getattr(args, "mac_timing", "all") != "none"
    if enabled:
        if end <= start or int(args.batch_size * args.mac_percent) == 0:
            raise ValueError("MAC must select at least one step and one sample.")
        if getattr(args, "mac_timing", "all") != "all" and args.mac_warmup_iters != 0:
            raise ValueError("Timing comparisons require mac_warmup_iters=0 (fixed selection fraction).")
        if getattr(args, "mac_target", "both") == "aux" and not args.v_head:
            raise ValueError("mac_target=aux requires v_head=True.")
        if args.method != "imf" and (
            getattr(args, "mac_target", "both") == "aux"
            or getattr(args, "mac_selection", "model") != "model"
            or getattr(args, "mac_normalize_weights", False)
        ):
            raise ValueError("Auxiliary/random/normalized MAC experiments require method=imf.")


def scheduled_mac_percentile(args, step):
    """None disables both scoring and weighting outside the selected window."""
    start, end = mac_bounds(args)
    if not args.mac or args.mac_weight == 0 or not start <= step < end:
        return None
    warmup = args.mac_warmup_iters
    if warmup <= 0 or step >= warmup:
        return args.mac_percent
    return 1.0 - step * (1.0 - args.mac_percent) / warmup


CONFIG_KEYS = (
    "method", "arch", "dataset", "model_channels", "batch_size", "grad_accum",
    "total_iters", "lr", "warmup_iters", "optimizer_betas", "dropout",
    "ema_decay", "ema_decays", "seed", "use_edm_aug", "ratio", "tr_sampler",
    "P_mean_t", "P_std_t", "P_mean_r", "P_std_r", "norm_p", "norm_eps",
    "v_head", "use_cfg", "use_cfg_interval", "num_classes", "class_dropout_prob",
    "cfg_s_max", "mac", "mac_timing", "mac_start_fraction", "mac_end_fraction",
    "mac_target", "mac_selection", "mac_percent", "mac_weight", "mac_warmup_iters",
    "mac_scorer", "mac_random_seed", "mac_normalize_weights", "deterministic",
    "mac_score",
)

# 分岐 (init_from) のとき、元の run と違っていてよいキー = MAC の設定だけ。
MAC_KEYS = tuple(key for key in CONFIG_KEYS if key.startswith("mac"))


def training_config(args):
    config = {key: getattr(args, key, None) for key in CONFIG_KEYS}
    # Include the implementation itself so a modified model cannot silently resume.
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(list((root / "models").glob("*.py")) +
                       [root / "runner.py", root / "mac_experiment.py",
                        root / "training/data_transform.py"]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    config["source_sha256"] = digest.hexdigest()
    config["experiment_version"] = 1
    return config


def experiment_id(args, preset="run"):
    config = training_config(args)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    timing = args.mac_timing if args.mac else "none"
    score = getattr(args, "mac_score", "h0")
    score_tag = "" if score == "h0" or not args.mac else f"-{score}"   # h0 (元の MAC) は従来どおりの名前
    return (f"{args.method}-{preset}-{timing}-{args.mac_target}-{args.mac_selection}{score_tag}"
            f"-seed{args.seed}-{digest}")


def check_resume_config(saved, current):
    if saved is None:
        raise ValueError("Legacy checkpoint has no experiment_config. Use a new ckpt_dir.")
    differences = [key for key in sorted(set(saved) | set(current))
                   if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError("Checkpoint configuration mismatch: " + ", ".join(differences)
                         + ". Use a new experiment/checkpoint directory.")


def mac_inactive_before(args, step):
    """step より前に MAC が一度も有効になっていない設定か (分岐の可否判定)。"""
    if not args.mac or args.mac_weight == 0:
        return True
    start, end = mac_bounds(args)
    return end <= start or start >= step


def check_branch_config(parent_args, parent_config, current_args, current_config, step):
    """
    前半を共有して後半だけ分岐させる (init_from) ための検査。
      1. MAC 以外の設定 (seed, 学習率, モデル, 総ステップ数, ソースコード) が完全に一致すること
      2. 親 run も新しい run も、分岐点 step より前に MAC が有効でないこと
    この 2 つを満たせば、分岐 run は「最初から通しで学習した run」と同じ計算になる。
    """
    if parent_config is None:
        raise ValueError("Parent checkpoint has no experiment_config.")
    differences = [key for key in sorted(set(parent_config) | set(current_config))
                   if key not in MAC_KEYS and parent_config.get(key) != current_config.get(key)]
    if differences:
        raise ValueError("Cannot branch: non-MAC settings differ from the parent run: "
                         + ", ".join(differences))
    if not mac_inactive_before(parent_args, step):
        raise ValueError(f"Cannot branch: the parent run already used MAC before step {step}.")
    if not mac_inactive_before(current_args, step):
        raise ValueError(f"Cannot branch at step {step}: this run's MAC window starts earlier "
                         f"({mac_bounds(current_args)}).")
