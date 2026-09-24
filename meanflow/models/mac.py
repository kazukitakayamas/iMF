"""
Model-Aligned Coupling (MAC)

Lin et al., "Beyond Optimal Transport: Model-Aligned Coupling for Flow Matching",
CVPR Findings 2026.  公式実装: https://github.com/tmllab/2026_CVPR_MAC
  - wrappers/utils.py   select_low_loss_indices()  <- 本ファイルの endpoint_error / mac_weights
  - wrappers/meanflow.py MACWrapper.get_loss()       <- 重み付けの適用箇所 (imf.py / meanflow.py)
  - main.py             get_current_percentile()    <- 本ファイルの mac_percentile

アルゴリズム (論文 Algorithm 1, Eq. 6 / Eq. 8):
  1. ランダム結合 (x, e) ごとに、端点 2 点での瞬時速度の予測誤差
         L̂_pair = 1/2 ( ||v(データ側端点) - v_gt||^2 + ||v(ノイズ側端点) - v_gt||^2 )
     を EMA モデルで評価する (勾配なし)。
  2. 誤差の小さい上位 k 割 (percentile) を「学習しやすい結合」S_theta とする。
  3. S_theta に属するサンプルの損失を (1 + lambda) 倍する。それ以外は 1 倍。
     (S_theta だけで学習すると周辺分布が歪むので、重み付けにとどめる)

時間の向きの違いに注意:
  MAC 公式 : x_t = (1-t) z0 + t x1  (t=0 ノイズ, t=1 データ), 目標速度 x1 - z0
  iMF / MF : z   = (1-t) x  + t e   (t=0 データ, t=1 ノイズ), 目標速度 e - x
端点 {0, 1} の 2 点で評価する分には「データ側端点」「ノイズ側端点」の対で同じものなので、
本ファイルは iMF / MF の規約 (t=1 がノイズ) で書いてある。
"""

import torch
import torch.nn.functional as F


def dup(c):
    """条件テンソルを 2B バッチ用に複製する (None はそのまま)"""
    return None if c is None else torch.cat([c, c], dim=0)


def mac_percentile(step, warmup_iters, end_val, start_val=1.0):
    """公式 main.py の get_current_percentile。1.0 -> end_val へ線形に減らす。

    公式は warmup_steps=20000 (CIFAR-10, ~97k step 中)。total_iters が短い場合は
    2 割程度 (int(0.2 * total_iters)) にするのが目安。
    """
    if warmup_iters <= 0 or step >= warmup_iters:
        return end_val
    return start_val - step * (start_val - end_val) / warmup_iters


@torch.no_grad()
def endpoint_error(v_fn, x, e):
    """
    L̂_pair (Eq. 6) をバッチごとに返す。shape [B]。

    v_fn(z, t): 瞬時速度 v_theta(z, t) を返す関数 (omega=1, 無ガイダンス, EMA 推奨)。
                t は [2B,1,1,1] の float tensor。z, t は 2B バッチで渡されるので、
                ラベルや aug_cond などの条件は dup() で 2 倍に複製して渡すこと。
    x: データ [B,C,H,W], e: ノイズ [B,C,H,W]  (既に対応付いたランダム結合)

    公式 select_low_loss_indices と同じく MSE (要素平均) で比較し、2 端点を平均する。
    2B バッチにまとめて 1 回だけ forward する。
    """
    B = x.shape[0]
    v_gt = e - x
    t0 = torch.zeros(B, 1, 1, 1, device=x.device, dtype=x.dtype)  # データ側端点 z_0 = x
    t1 = torch.ones(B, 1, 1, 1, device=x.device, dtype=x.dtype)   # ノイズ側端点 z_1 = e

    zz = torch.cat([x, e], dim=0)
    tt = torch.cat([t0, t1], dim=0)
    v_pred = v_fn(zz, tt)

    err = F.mse_loss(v_pred, torch.cat([v_gt, v_gt], dim=0), reduction="none")
    err = err.mean(dim=tuple(range(1, err.ndim)))  # [2B]
    return 0.5 * (err[:B] + err[B:])               # [B]


MAC_SCORES = ("h0", "h1", "mix")


@torch.no_grad()
def onestep_error(u1_fn, x, e):
    """
    [MAC-h1] 1-NFE (平均速度, gap h = t - r = 1) での結合の採点。shape [B]。

        s1(x, e) = mean || u_theta(e, r=0, t=1) - (e - x) ||^2
                 = mean || x_hat(e) - x ||^2,   x_hat(e) = e - u_theta(e, 0, 1)  (1-NFE サンプル)

    つまり「ノイズ e から今のモデルが 1 ステップで作る画像」と、ランダムに組まれた
    データ x との距離。元の MAC (endpoint_error) は h=0 の瞬時速度しか見ないので、
    1-NFE が使う平均速度 u(e,0,1) とは採点の基準がずれている、という仮説を検証するための採点。

    u1_fn(e): u_theta(e, r=0, t=1) を返す関数 (omega=1, 無ガイダンス, EMA 推奨)。B バッチ。
    """
    u = u1_fn(e)
    err = F.mse_loss(u, e - x, reduction="none")
    return err.mean(dim=tuple(range(1, err.ndim)))  # [B]


@torch.no_grad()
def rank01(v):
    """スコア -> [0, 1] の順位 (小さいほど 0)。スケールの違う採点を平均するため。"""
    r = torch.empty_like(v)
    r[torch.argsort(v)] = torch.arange(len(v), device=v.device, dtype=v.dtype)
    return r / max(len(v) - 1, 1)


@torch.no_grad()
def pair_score(score, v_fn, u1_fn, x, e):
    """
    MAC の採点を切り替える。戻り値: (score [B], stats dict)。値が小さいほど「学習しやすい結合」。

      "h0"  : 元の MAC。瞬時速度 (h=0) の 2 端点誤差 = endpoint_error。 <- 既定・従来と完全に同じ
      "h1"  : 1-NFE 向け。平均速度 u(e,0,1) の誤差 = onestep_error
      "mix" : h0 と h1 をそれぞれ順位に直して平均 (スケールの違いで片方に引きずられないように)
    """
    if score not in MAC_SCORES:
        raise ValueError(f"Unknown mac_score: {score}")
    stats = {}
    s0 = s1 = None
    if score in ("h0", "mix"):
        s0 = endpoint_error(v_fn, x, e)
        stats["mac_err_h0"] = float(s0.mean())
    if score in ("h1", "mix"):
        s1 = onestep_error(u1_fn, x, e)
        stats["mac_err_h1"] = float(s1.mean())
    if score == "h0":
        return s0, stats
    if score == "h1":
        return s1, stats
    return 0.5 * (rank01(s0) + rank01(s1)), stats


@torch.no_grad()
def mac_weights(err, percentile, add_weight, selection="model", generator=None, normalize=False):
    """
    Eq. 8 の w(x0, x1)。誤差が小さい上位 percentile 割のサンプルに 1 + add_weight、他は 1。
    戻り値: (weights [B], selected_mask [B] bool)
    """
    B = err.shape[0]
    w = torch.ones_like(err)
    mask = torch.zeros(B, dtype=torch.bool, device=err.device)
    k = int(B * percentile)
    if k <= 0 or add_weight <= 0:
        return w, mask
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be in [0, 1].")
    if selection == "model":
        _, idx = torch.topk(err, k, largest=False)
    elif selection == "random":
        if generator is None:
            raise ValueError("Random selection requires an independent generator.")
        idx = torch.randperm(B, device=err.device, generator=generator)[:k]
    else:
        raise ValueError(f"Unknown MAC selection: {selection}")
    mask[idx] = True
    w[idx] = 1.0 + add_weight
    if normalize:
        w = w / w.mean()
    return w, mask


def apply_mac_losses(loss_main, loss_aux, weights, target="both"):
    """Apply MAC after adaptive weighting, before averaging the batch."""
    if target not in ("both", "main", "aux"):
        raise ValueError(f"Unknown mac_target: {target}")
    if weights is not None:
        if target in ("both", "main"):
            loss_main = loss_main * weights
        if target in ("both", "aux"):
            loss_aux = loss_aux * weights
    return loss_main, loss_aux
