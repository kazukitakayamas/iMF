"""
単一 GPU (Colab) 用のランナー。

上流の train.py / training/ は torchrun + DDP + torch.compile 前提なので、
それらには一切触らずに、単一 GPU 用の学習ループ・FID 評価・wandb ログをここにまとめている。
ノートブックは args を組み立ててこのクラスを呼ぶだけのスイッチになる。

MF と iMF の両方を同じループで回せるので、同一条件でのベースライン比較ができる。
"""

import math
import os
import time
import functools
import json

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models.augment import AugmentPipe
from models.model_configs import instantiate_model
from mac_experiment import (validate_mac_args, scheduled_mac_percentile, mac_bounds,
                            training_config, check_resume_config)
from training.data_transform import get_transform_cifar

try:
    import wandb
except ImportError:  # wandb 無しでも動く
    wandb = None


def _log(metrics, step):
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def _isolated_rng(fn):
    """Evaluation must not advance the training RNG, including first FID initialization."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            return fn(self, *args, **kwargs)
    return wrapped


class TrainingBatchSampler:
    """Infinite epoch-shuffled batches, addressable by consumed microbatch count.

    Prefetch does not change restart position. Random image flips happen in the
    training process, not in workers, so their RNG can also be checkpointed.
    """
    def __init__(self, size, batch_size, seed, start_batch=0):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start_batch = start_batch
        self.batches_per_epoch = size // batch_size
        if self.batches_per_epoch == 0:
            raise ValueError("Training dataset must contain at least one full batch.")

    def __iter__(self):
        epoch, offset = divmod(self.start_batch, self.batches_per_epoch)
        while True:
            g = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.size, generator=g).tolist()
            for batch in range(offset, self.batches_per_epoch):
                start = batch * self.batch_size
                yield order[start:start + self.batch_size]
            epoch, offset = epoch + 1, 0

    def __len__(self):
        return self.batches_per_epoch


class Runner:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = torch.device(device)
        validate_mac_args(args)
        self.experiment_config = training_config(args)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(getattr(args, "deterministic", False))
        torch.backends.cudnn.benchmark = not getattr(args, "deterministic", False)
        self.ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
        self.start_step = 0
        ck = None
        if os.path.exists(self.ckpt_path):
            if not getattr(args, "resume_training", False):
                raise FileExistsError("Checkpoint exists. Set resume_training=True to resume, "
                                      "or use a new experiment directory.")
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
            check_resume_config(ck.get("experiment_config"), self.experiment_config)
            self.start_step = int(ck["step"])
            if not 0 <= self.start_step <= args.total_iters:
                raise ValueError("Checkpoint step is outside the training schedule.")
        elif getattr(args, "resume_training", False):
            raise FileNotFoundError(f"No checkpoint to resume: {self.ckpt_path}")

        # ---- data ----
        self.train_set = torchvision.datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=get_transform_cifar(True)
        )
        self.fid_set = torchvision.datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=get_transform_cifar(True)
        )
        workers = getattr(args, "num_workers", 2)
        self.train_loader = DataLoader(
            self.train_set,
            batch_sampler=TrainingBatchSampler(len(self.train_set), args.batch_size, args.seed,
                                               self.start_step * args.grad_accum),
            num_workers=workers, pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(args.seed + 100000),
        )
        self.fid_loader = DataLoader(
            self.fid_set, batch_size=500, shuffle=False, num_workers=workers,
            generator=torch.Generator().manual_seed(getattr(args, "eval_seed", 42)),
        )

        # ---- model ----
        self.model = instantiate_model(args).to(self.device)
        self.is_imf = getattr(args, "method", "mf") == "imf"
        if self.is_imf:
            self.model.mac_generator = torch.Generator(self.device).manual_seed(
                getattr(args, "mac_random_seed", 12345))
        n_params = sum(p.numel() for p in self.model.net.parameters())
        print(f"method={args.method} | arch params: {n_params / 1e6:.2f} M")

        self.opt = torch.optim.Adam(
            self.model.net.parameters(), lr=args.lr, betas=tuple(args.optimizer_betas)
        )
        self.augment_pipe = (
            AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)
            if args.use_edm_aug else None
        )

        # ---- checkpoint ----
        os.makedirs(args.ckpt_dir, exist_ok=True)
        if ck is not None:
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            torch.set_rng_state(ck["rng_cpu"])
            if self.device.type == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"], self.device)
            if self.is_imf and ck.get("mac_rng") is not None:
                self.model.mac_generator.set_state(ck["mac_rng"])
            print(f"resumed from step {self.start_step}")
        self.train_iter = iter(self.train_loader)
        with open(os.path.join(args.ckpt_dir, "config.json"), "w") as f:
            json.dump({"args": vars(args), "experiment_config": self.experiment_config},
                      f, indent=2, ensure_ascii=False)

        # ---- fixed noise for sample grids ----
        g = torch.Generator(self.device).manual_seed(1234)
        self.fixed_noise = torch.randn(64, 3, 32, 32, device=self.device, generator=g)
        self.fixed_labels = (
            torch.arange(64, device=self.device) % args.num_classes
            if args.num_classes > 0 else torch.zeros(64, dtype=torch.long, device=self.device)
        )
        self._fid = None

    # ------------------------------------------------------------------
    @staticmethod
    def _infinite(loader):
        while True:
            for batch in loader:
                yield batch

    def _lr_at(self, step):
        return self.args.lr * min(1.0, (step + 1) / max(1, self.args.warmup_iters))

    def save(self, step):
        state = {"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                 "step": step, "args": vars(self.args),
                 "experiment_config": self.experiment_config,
                 "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                 "mac_rng": self.model.mac_generator.get_state() if self.is_imf else None}
        temporary = self.ckpt_path + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, self.ckpt_path)

    # ------------------------------------------------------------------
    # 生成 / 可視化
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n, num_steps=1, omega=1.0, t_min=0.0, t_max=1.0, labels=None,
               net=None, generator=None):
        net = net if net is not None else self.model.net_ema
        net.eval()
        shape = (n, 3, 32, 32)
        if self.is_imf:
            return self.model.sample(shape, net=net, device=self.device, num_steps=num_steps,
                                     omega=omega, t_min=t_min, t_max=t_max,
                                     labels=labels, generator=generator)
        if num_steps != 1:
            raise ValueError("The MF baseline in this Runner supports only num_steps=1.")
        z = torch.randn(shape, device=self.device, generator=generator)
        t = torch.ones(n, device=self.device)
        return z - net(z, (t, t), aug_cond=None)

    @staticmethod
    def to_image(z):
        """[-1,1] -> [0,1] の 8bit 量子化 (上流 eval_loop.py と同じ処理)"""
        img = (z * 0.5 + 0.5).clamp(0.0, 1.0)
        return torch.floor(img * 255.0) / 255.0

    @torch.no_grad()
    @_isolated_rng
    def sample_grid(self, omega=1.0, nrow=8, save_path=None):
        z = self.fixed_noise.clone()
        if self.is_imf:
            n = z.shape[0]
            const = lambda v: torch.full((n, 1, 1, 1), float(v), device=self.device)
            self.model.net_ema.eval()
            u = self.model.u_fn(self.model.net_ema, z, const(1.0), const(0.0),
                                const(omega), const(0.0), const(1.0), self.fixed_labels)[0]
            z = z - u
        else:
            z = self.model.sample(z.shape, net=self.model.net_ema, device=self.device)
        grid = make_grid(self.to_image(z), nrow=nrow)
        if save_path:
            torchvision.utils.save_image(self.to_image(z), save_path, nrow=nrow)
        return grid

    # ------------------------------------------------------------------
    # FID (上流 eval_loop.py と同じく torchmetrics / CIFAR-10 train 50K が基準)
    # ------------------------------------------------------------------
    def _fid_metric(self):
        if self._fid is None:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self._fid = FrechetInceptionDistance(
                feature=2048, normalize=True, reset_real_features=False
            ).to(self.device)
            with torch.no_grad():
                for x, _ in self.fid_loader:
                    self._fid.update(x.to(self.device, non_blocking=True), real=True)
            print("real features:", int(self._fid.real_features_num_samples))
        return self._fid

    @torch.no_grad()
    @_isolated_rng
    def compute_fid(self, n_samples=None, num_steps=1, omega=1.0, t_min=0.0, t_max=1.0,
                    bs=None, eval_seed=None):
        n_samples = self.args.fid_samples if n_samples is None else n_samples
        bs = getattr(self.args, "eval_batch_size", 250) if bs is None else bs
        eval_seed = getattr(self.args, "eval_seed", 42) if eval_seed is None else eval_seed
        if n_samples <= 0 or bs <= 0 or num_steps <= 0:
            raise ValueError("n_samples, bs and num_steps must be positive.")
        noise_generator = torch.Generator(self.device).manual_seed(eval_seed)
        label_generator = torch.Generator(self.device).manual_seed(eval_seed + 1)
        fid = self._fid_metric()
        fid.reset()  # real 統計は保持される
        remain = n_samples
        while remain > 0:
            b = min(bs, remain)
            labels = None
            if self.is_imf:
                classes = self.args.num_classes
                labels = (torch.randint(classes, (b,), device=self.device, generator=label_generator)
                          if classes > 0 else torch.zeros(b, dtype=torch.long, device=self.device))
            z = self.sample(b, num_steps=num_steps, omega=omega, t_min=t_min, t_max=t_max,
                            labels=labels, generator=noise_generator)
            fid.update(self.to_image(z).float(), real=False)
            remain -= b
        return float(fid.compute())

    def evaluate_steps(self, steps=None, n_samples=None, omega=1.0):
        if steps is None:
            steps = getattr(self.args, "eval_steps", [1, 2, 4]) if self.is_imf else [1]
        steps = list(steps)
        results = {}
        for nfe in steps:
            value = self.compute_fid(n_samples=n_samples, num_steps=nfe, omega=omega)
            results[str(nfe)] = value
            print(f"{nfe}-NFE FID({n_samples or self.args.fid_samples}), "
                  f"eval_seed={self.args.eval_seed}: {value:.4f}")
        return results

    def fid_sweep(self, omegas, n_samples=None, num_steps=1):
        """柔軟な CFG の効果 (論文 Fig.4) を見るための omega スイープ"""
        results = {}
        for om in omegas:
            results[om] = self.compute_fid(n_samples=n_samples, num_steps=num_steps, omega=om)
            print(f"  omega={om}: FID = {results[om]:.3f}")
        return results

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def train(self):
        args = self.args
        self.model.train()
        
        run = {key: 0.0 for key in ("loss", "loss_u", "loss_v", "loss_u_unweighted",
                                    "loss_v_unweighted", "mac_err", "mac_err_sel",
                                    "mac_frac", "mac_weight_mean", "mac_active")}
        log_count = 0
        bounds = mac_bounds(args)
        print(f"MAC {args.mac_timing}: zero-based steps {bounds}, "
              f"target={args.mac_target}, selection={args.mac_selection}")
        t0 = time.time()

        for step in range(self.start_step, args.total_iters):
            mac_p = scheduled_mac_percentile(args, step)
            log_count += 1
            run["mac_active"] += float(mac_p is not None)
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)

            self.opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                x, y = next(self.train_iter)
                x = x.to(self.device, non_blocking=True) * 2.0 - 1.0
                y = y.to(self.device, non_blocking=True)
                # Same p=0.5 flip as the original transform, with checkpointable training RNG.
                flip = torch.rand((x.shape[0], 1, 1, 1), device=self.device) < 0.5
                x = torch.where(flip, x.flip(-1), x)

                aug_cond = None
                if self.augment_pipe is not None:
                    x, aug_cond = self.augment_pipe(x)

                if self.is_imf:
                    loss, parts = self.model.forward_with_loss(x, y, aug_cond, mac_percentile=mac_p)
                else:
                    loss, parts = self.model.forward_with_loss(x, aug_cond, mac_percentile=mac_p), {}

                (loss / args.grad_accum).backward()
                run["loss"] += float(loss.detach()) / args.grad_accum
                run["loss_u"] += parts.get("loss_u", 0.0) / args.grad_accum
                run["loss_v"] += parts.get("loss_v", 0.0) / args.grad_accum
                run["mac_err"] += parts.get("mac_err", 0.0) / args.grad_accum
                run["mac_err_sel"] += parts.get("mac_err_sel", 0.0) / args.grad_accum
                for key in ("loss_u_unweighted", "loss_v_unweighted", "mac_frac", "mac_weight_mean"):
                    run[key] += parts.get(key, 0.0) / args.grad_accum

            self.opt.step()
            self.model.update_ema()
            self.start_step = step + 1

            if not math.isfinite(run["loss"]):
                raise ValueError(f"loss diverged at step {step}")

            if (step + 1) % args.log_every == 0 or step + 1 == args.total_iters:
                dt, n = time.time() - t0, log_count
                metrics = {"train/loss": run["loss"] / n, "train/loss_u": run["loss_u"] / n,
                           "train/loss_v": run["loss_v"] / n,
                           "train/lr": self.opt.param_groups[0]["lr"],
                           "perf/sec_per_iter": dt / n}
                metrics.update({"train/loss_main_before_mac": run["loss_u_unweighted"] / n,
                                "train/loss_aux_before_mac": run["loss_v_unweighted"] / n,
                                "mac/active": float(mac_p is not None),
                                "mac/active_fraction": run["mac_active"] / n,
                                "mac/percentile": mac_p if mac_p is not None else 0.0})
                active_count = run["mac_active"]
                if active_count > 0 and self.is_imf:
                    metrics.update({"mac/endpoint_err": run["mac_err"] / active_count,
                                    "mac/endpoint_err_selected": run["mac_err_sel"] / active_count,
                                    "mac/selected_fraction": run["mac_frac"] / active_count,
                                    "mac/weight_mean": run["mac_weight_mean"] / active_count})
                _log(metrics, step + 1)
                with open(os.path.join(args.ckpt_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step + 1, **metrics}) + "\n")
                print(f"step {step+1:>7} | loss {run['loss']/n:.4f} "
                      f"(u {run['loss_u']/n:.3f} / v {run['loss_v']/n:.3f}) "
                      f"| MAC {int(mac_p is not None)} | lr {self.opt.param_groups[0]['lr']:.2e} | {dt/n:.3f} s/it")
                run = {k: 0.0 for k in run}
                log_count = 0
                t0 = time.time()

            if (step + 1) % args.sample_every == 0:
                if wandb is not None and wandb.run is not None:
                    imgs = {"samples/1nfe_ema": wandb.Image(self.sample_grid(omega=1.0))}
                    if self.is_imf and args.use_cfg:
                        imgs["samples/1nfe_ema_w2"] = wandb.Image(self.sample_grid(omega=2.0))
                    _log(imgs, step + 1)
                self.model.train()
                t0 = time.time()

            if (step + 1) % args.ckpt_every == 0:
                self.save(step + 1)
                t0 = time.time()

            if (step + 1) % args.eval_every == 0:
                scores = self.evaluate_steps()
                _log({f"eval/fid_{nfe}nfe": value for nfe, value in scores.items()}, step + 1)
                with open(os.path.join(args.ckpt_dir, "evaluations.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step + 1, "fid": scores,
                                        "n_samples": args.fid_samples, "eval_seed": args.eval_seed,
                                        "eval_batch_size": args.eval_batch_size}) + "\n")
                self.model.train()
                t0 = time.time()

        self.save(args.total_iters)
        print("done")
