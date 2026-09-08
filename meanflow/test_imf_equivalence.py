"""
論文 Sec 4.1 の主張「original MF は V_theta(z_t, e-x) に対する v-loss と等価」を数値で確認する。

iMF の合成関数 V = u + (t-r)*sg(du/dt) について、JVP の接ベクトルと回帰ターゲットを
どちらも条件付き速度 e-x に固定すると

    V - (e-x) = -( u_tgt - u_pred ),   u_tgt = (e-x) - (t-r)*du/dt

となり、二乗すれば MF の損失と完全に一致するはず。

    cd meanflow && python -m tests.test_imf_equivalence
"""

import sys

import torch

from models.model_configs import instantiate_model
from models.time_sampler import sample_two_timesteps
from train_arg_parser import get_args_parser


def build_args():
    args = get_args_parser().parse_args([])
    args.method = "imf"
    args.use_cfg = False
    args.use_cfg_interval = False
    args.num_classes = 0
    args.v_head = False
    args.ema_decays = []
    args.model_channels = 32
    args.dropout = 0.0          # dropout を切って両者の乱数消費を厳密に揃える
    args.use_edm_aug = False
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = build_args()
    model = instantiate_model(args).to(device).eval()
    net = model.net

    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32, device=device) * 2 - 1
    y = torch.zeros(8, dtype=torch.long, device=device)

    def make_batch(seed):
        torch.manual_seed(seed)
        t, r = sample_two_timesteps(args, num_samples=x.shape[0], device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)
        e = torch.randn_like(x)
        return t, r, e

    t, r, e = make_batch(123)
    z = (1 - t) * x + t * e
    v = e - x

    one, zero = torch.ones_like(t), torch.zeros_like(t)
    fn = lambda z_, t_, r_: model.u_fn(net, z_, t_, r_, one, zero, one, y)

    u_pred, _ = fn(z, t, r)
    with torch.no_grad():
        _, (dudt, _) = torch.func.jvp(fn, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))

    def adaptive(loss):
        return (loss / ((loss.detach() + args.norm_eps) ** args.norm_p)).mean()

    # original MF
    u_tgt = (v - (t - r) * dudt).detach()
    loss_mf = adaptive(((u_pred - u_tgt) ** 2).sum(dim=(1, 2, 3)))

    # iMF (接ベクトルとターゲットを e-x に固定した場合)
    V = u_pred + (t - r) * dudt.detach()
    loss_imf = adaptive(((V - v) ** 2).sum(dim=(1, 2, 3)))

    print(f"MF                = {float(loss_mf.detach()):.8f}")
    print(f"iMF(tangent=e-x)  = {float(loss_imf.detach()):.8f}")
    ok = torch.allclose(loss_mf, loss_imf, rtol=1e-5, atol=1e-6)
    print("equivalent:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
