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

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from models.augment import AugmentPipe
from models.model_configs import instantiate_model
from training.data_transform import get_transform_cifar

try:
    import wandb
except ImportError:  # wandb 無しでも動く
    wandb = None


def _log(metrics, step):
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


class Runner:
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = torch.device(device)
        torch.manual_seed(args.seed)

        # ---- data ----
        self.train_set = torchvision.datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=get_transform_cifar(False)
        )
        self.fid_set = torchvision.datasets.CIFAR10(
            args.data_path, train=True, download=True, transform=get_transform_cifar(True)
        )
        self.train_loader = DataLoader(
            self.train_set, batch_size=args.batch_size, shuffle=True, num_workers=2,
            pin_memory=True, drop_last=True, persistent_workers=True,
        )
        self.fid_loader = DataLoader(self.fid_set, batch_size=500, shuffle=False, num_workers=2)
        self.train_iter = self._infinite(self.train_loader)

        # ---- model ----
        self.model = instantiate_model(args).to(self.device)
        self.is_imf = getattr(args, "method", "mf") == "imf"
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
        self.ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
        self.start_step = 0
        if os.path.exists(self.ckpt_path):
            ck = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.start_step = ck["step"]
            print(f"resumed from step {self.start_step}")

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
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                    "step": step, "args": vars(self.args)}, self.ckpt_path)

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
        return self.model.sample(shape, net=net, device=self.device)   # MF は 1-NFE のみ

    @staticmethod
    def to_image(z):
        """[-1,1] -> [0,1] の 8bit 量子化 (上流 eval_loop.py と同じ処理)"""
        img = (z * 0.5 + 0.5).clamp(0.0, 1.0)
        return torch.floor(img * 255.0) / 255.0

    @torch.no_grad()
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
    def compute_fid(self, n_samples=None, num_steps=1, omega=1.0, t_min=0.0, t_max=1.0, bs=250):
        n_samples = n_samples or self.args.fid_samples
        fid = self._fid_metric()
        fid.reset()  # real 統計は保持される
        remain = n_samples
        while remain > 0:
            b = min(bs, remain)
            z = self.sample(b, num_steps=num_steps, omega=omega, t_min=t_min, t_max=t_max)
            fid.update(self.to_image(z).float(), real=False)
            remain -= b
        return float(fid.compute())

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
        run = {"loss": 0.0, "loss_u": 0.0, "loss_v": 0.0}
        t0 = time.time()

        for step in range(self.start_step, args.total_iters):
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)

            self.opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                x, y = next(self.train_iter)
                x = x.to(self.device, non_blocking=True) * 2.0 - 1.0
                y = y.to(self.device, non_blocking=True)

                aug_cond = None
                if self.augment_pipe is not None:
                    x, aug_cond = self.augment_pipe(x)

                if self.is_imf:
                    loss, parts = self.model.forward_with_loss(x, y, aug_cond)
                else:
                    loss, parts = self.model.forward_with_loss(x, aug_cond), {}

                (loss / args.grad_accum).backward()
                run["loss"] += float(loss.detach()) / args.grad_accum
                run["loss_u"] += parts.get("loss_u", 0.0) / args.grad_accum
                run["loss_v"] += parts.get("loss_v", 0.0) / args.grad_accum

            self.opt.step()
            self.model.update_ema()

            if not math.isfinite(run["loss"]):
                raise ValueError(f"loss diverged at step {step}")

            if (step + 1) % args.log_every == 0:
                dt, n = time.time() - t0, args.log_every
                _log({"train/loss": run["loss"] / n, "train/loss_u": run["loss_u"] / n,
                      "train/loss_v": run["loss_v"] / n,
                      "train/lr": self.opt.param_groups[0]["lr"],
                      "perf/sec_per_iter": dt / n}, step + 1)
                print(f"step {step+1:>7} | loss {run['loss']/n:.4f} "
                      f"(u {run['loss_u']/n:.3f} / v {run['loss_v']/n:.3f}) "
                      f"| lr {self.opt.param_groups[0]['lr']:.2e} | {dt/n:.3f} s/it")
                run = {k: 0.0 for k in run}
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
                fid = self.compute_fid(omega=1.0)
                _log({"eval/fid_1nfe": fid}, step + 1)
                print(f"step {step+1}: FID({args.fid_samples}) = {fid:.3f}")
                self.model.train()
                t0 = time.time()

        self.save(args.total_iters)
        print("done")
