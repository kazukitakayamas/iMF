"""
improved MeanFlow (iMF)

Geng et al., "Improved Mean Flows: On the Challenges of Fastforward Generative Models", CVPR 2026.

上流の models/meanflow.py と同じインターフェースを持つ。差分は 2 点。

  改変 1 (論文 Sec 4.1): u-loss -> v-loss
      MF : u_tgt = v_c - (t-r) * JVP(u; v_c)  を教師にして u を回帰   (教師がネットワーク依存)
      iMF: V = u + (t-r) * sg(JVP(u; v_theta)) を v_g に回帰          (教師がネットワーク非依存)
      JVP の接ベクトルを条件付き速度 (e-x) からネットワーク予測 v_theta に置き換えるのが本質。

  改変 2 (論文 Sec 4.2): 固定 CFG -> 柔軟な CFG
      guidance scale omega と guidance interval (t_min, t_max) を条件変数としてネットワークに入力し、
      学習時にランダムサンプルする。推論時に自由に変更できる。

参照: 公式 JAX 実装 https://github.com/Lyy-iiis/imeanflow (imf.py)
"""

import math

import torch
import torch.nn as nn

from models.ema import init_ema, update_ema_net
from models.time_sampler import sample_two_timesteps


class iMF(nn.Module):
    def __init__(self, arch, args, net_configs):
        super().__init__()
        self.net = arch(**net_configs)
        self.args = args
        self.num_classes = args.num_classes
        self.null_class = args.num_classes  # null クラスの id

        self.register_buffer("num_updates", torch.tensor(0))
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

    def update_ema(self):
        self.num_updates += 1
        update_ema_net(self.net, self.net_ema, self.num_updates)
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], self.num_updates)

    # ------------------------------------------------------------------
    # ネットワーク呼び出しラッパ
    # ------------------------------------------------------------------
    def u_fn(self, net, z, t, r, omega, t_min, t_max, y=None, aug_cond=None):
        """u_theta(z_t, r, t | c, Omega)。戻り値は (u, v_aux)"""
        return net(z, t, t - r, omega, t_min, t_max, y, aug_cond)

    def v_fn(self, net, z, t, omega, y=None, aug_cond=None):
        """v_theta(z_t, t | c, omega)。h=0 で呼び、補助ヘッド（または境界条件の u）を取る"""
        h = torch.zeros_like(t)
        return net(z, t, h, omega, torch.zeros_like(t), torch.ones_like(t), y, aug_cond)[1]

    # ------------------------------------------------------------------
    # 条件のサンプリング
    # ------------------------------------------------------------------
    def sample_cfg_scale(self, bsz, device):
        """omega ~ [1, 1 + s_max] のべき分布 (公式 imf.py の cfg_beta=1.0 の場合)"""
        u = torch.rand(bsz, 1, 1, 1, device=device)
        return torch.exp(u * math.log1p(self.args.cfg_s_max))

    def sample_cfg_interval(self, bsz, fm_mask, device):
        """guidance interval。t=r の Flow Matching サンプルには区間を適用しない"""
        t_min = torch.rand(bsz, 1, 1, 1, device=device) * 0.5
        t_max = 0.5 + torch.rand(bsz, 1, 1, 1, device=device) * 0.5
        return (
            torch.where(fm_mask, torch.zeros_like(t_min), t_min),
            torch.where(fm_mask, torch.ones_like(t_max), t_max),
        )

    # ------------------------------------------------------------------
    # 改変 2: ガイダンス付き教師信号
    # ------------------------------------------------------------------
    def guidance_target(self, z, t, v_t, y, fm_mask, omega, t_min, t_max, aug_cond=None):
        """
        v_g = v_t + (1 - 1/omega) * (v_c - v_u) を返す。
        あわせて JVP の接ベクトルに使う v_c も返す。
        """
        bsz = z.shape[0]

        # 条件付き / 無条件の v を 2B バッチでまとめて 1 回
        zz = torch.cat([z, z])
        tt = torch.cat([t, t])
        ww = torch.cat([omega, torch.ones_like(omega)])
        yy = torch.cat([y, torch.full_like(y, self.null_class)])
        aa = torch.cat([aug_cond, aug_cond]) if aug_cond is not None else None

        v_both = self.v_fn(self.net, zz, tt, ww, yy, aa)
        v_c_raw, v_u = v_both[:bsz], v_both[bsz:]
        v_g_fm = v_t + (1.0 - 1.0 / omega) * (v_c_raw - v_u)

        if not self.args.use_cfg_interval:
            return v_g_fm, v_c_raw

        # 区間外では CFG を無効化 (omega = 1)
        w = torch.where((t >= t_min) & (t <= t_max), omega, torch.ones_like(omega))
        v_c = self.v_fn(self.net, z, t, w, y, aug_cond)
        v_g = v_t + (1.0 - 1.0 / w) * (v_c - v_u)
        v_g = torch.where(fm_mask, v_g_fm, v_g)
        return v_g, v_c

    # ------------------------------------------------------------------
    # 学習
    # ------------------------------------------------------------------
    def forward_with_loss(self, x, y=None, aug_cond=None):
        device = x.device
        bsz = x.shape[0]

        t, r = sample_two_timesteps(self.args, num_samples=bsz, device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)
        fm_mask = t == r  # r = t の (= Flow Matching) サンプル

        e = torch.randn_like(x)
        z = (1.0 - t) * x + t * e
        v_t = e - x  # 条件付き瞬時速度

        if y is None:
            y = torch.zeros(bsz, dtype=torch.long, device=device)

        if self.args.use_cfg:
            omega = self.sample_cfg_scale(bsz, device)
            if self.args.use_cfg_interval:
                t_min, t_max = self.sample_cfg_interval(bsz, fm_mask, device)
            else:
                t_min, t_max = torch.zeros_like(t), torch.ones_like(t)

            with torch.no_grad():
                v_g, v_c = self.guidance_target(
                    z, t, v_t, y, fm_mask, omega, t_min, t_max, aug_cond
                )

            # class 条件のドロップ。ドロップしたサンプルは無ガイドの目標に戻す。
            drop = (torch.rand(bsz, device=device) < self.args.class_dropout_prob).view(-1, 1, 1, 1)
            y_in = torch.where(drop.view(-1), torch.full_like(y, self.null_class), y)
            v_g = torch.where(drop, v_t, v_g)
        else:
            omega = torch.ones_like(t)
            t_min, t_max = torch.zeros_like(t), torch.ones_like(t)
            y_in = torch.full_like(y, self.null_class) if self.num_classes > 0 else y
            with torch.no_grad():
                v_c = self.v_fn(self.net, z, t, omega, y_in, aug_cond)
            v_g = v_t

        # ---- 改変 1: 合成関数 V_theta = u + (t-r) * sg(du/dt) ----
        def fn(z_, t_, r_):
            return self.u_fn(self.net, z_, t_, r_, omega, t_min, t_max, y_in, aug_cond)

        with torch.amp.autocast("cuda", enabled=False):
            # 上流 README 推奨の非 compile 手順: u は通常 forward、JVP は no_grad 下で計算する。
            # dropout マスクを 2 回の forward で共有するため RNG 状態を退避・復元する。
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
            u, v_aux = fn(z, t, r)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)

            with torch.no_grad():
                _, (dudt, _) = torch.func.jvp(
                    fn, (z, t, r), (v_c, torch.ones_like(t), torch.zeros_like(r))
                )

            V = u + (t - r) * dudt.detach()
            v_g = v_g.detach()

            def adaptive_weight(loss):
                w = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
                return loss / w

            loss_u = adaptive_weight(((V - v_g) ** 2).sum(dim=(1, 2, 3)))
            loss = loss_u
            loss_v = torch.zeros((), device=device)
            if self.args.v_head:
                loss_v = adaptive_weight(((v_aux - v_g) ** 2).sum(dim=(1, 2, 3)))
                loss = loss + loss_v

            loss = loss.mean()

        return loss, {
            "loss_u": float(loss_u.mean().detach()),
            "loss_v": float(loss_v.mean().detach()),
        }

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        samples_shape,
        net=None,
        device=None,
        num_steps=1,
        omega=1.0,
        t_min=0.0,
        t_max=1.0,
        labels=None,
        generator=None,
    ):
        """num_steps=1 で 1-NFE 生成。omega / interval は推論時に自由に変えられる。"""
        net = net if net is not None else self.net_ema
        n = samples_shape[0]

        z = torch.randn(samples_shape, dtype=torch.float32, device=device, generator=generator)
        if labels is None:
            labels = (
                torch.randint(0, self.num_classes, (n,), device=device, generator=generator)
                if self.num_classes > 0
                else torch.zeros(n, dtype=torch.long, device=device)
            )

        def const(v):
            return torch.full((n, 1, 1, 1), float(v), device=device)

        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t, r = const(t_steps[i].item()), const(t_steps[i + 1].item())
            u = self.u_fn(net, z, t, r, const(omega), const(t_min), const(t_max), labels)[0]
            z = z - (t - r) * u
        return z
