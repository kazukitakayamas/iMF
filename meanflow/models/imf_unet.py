"""
improved MeanFlow (iMF) 用のバックボーン。

上流の SongUNet を継承し、条件埋め込みの作り方だけを差し替える。
Encoder / Decoder ブロックは一切変更していない。

追加した条件:
  - guidance scale omega
  - guidance interval (t_min, t_max)
  - class label c (null class を含む)
追加した出力:
  - auxiliary v-head (論文 Sec 4.1)。out_channels を 2 倍にして前半を u、後半を v とする。
"""

import numpy as np
import torch
from torch.nn.functional import silu

from .unet import SongUNet, UNetBlock, Linear


class iMFUNet(SongUNet):
    def __init__(self, num_classes=0, v_head=True, out_channels=3, **kwargs):
        # v-head を使う場合は出力チャネルを 2 倍にして (u, v) に分割する
        super().__init__(out_channels=out_channels * (2 if v_head else 1), **kwargs)

        self.n_out = out_channels
        self.v_head = v_head
        self.num_classes = num_classes

        noise_channels = kwargs["model_channels"] * kwargs.get("channel_mult_noise", 1)

        # 追加 1: guidance 条件 (omega, t_min, t_max) -> emb と同じ次元へ落として加算する。
        # map_layer0 の入力次元を変えずに済む。ゼロ初期化なので学習開始時は MF と同じ挙動。
        self.map_guidance = Linear(
            in_features=noise_channels * 3,
            out_features=noise_channels * 2,
            init_mode="xavier_uniform",
        )
        torch.nn.init.zeros_(self.map_guidance.weight)
        torch.nn.init.zeros_(self.map_guidance.bias)

        # 追加 2: class 条件。上流の map_label は出力次元が合わないため別に持つ。
        self.map_class = (
            torch.nn.Embedding(num_classes + 1, noise_channels * 2) if num_classes > 0 else None
        )
        if self.map_class is not None:
            torch.nn.init.normal_(self.map_class.weight, std=0.02)

    def _pos_emb(self, s):
        e = self.map_noise(s.reshape(-1))
        return e.reshape(e.shape[0], 2, -1).flip(1).reshape(*e.shape)  # swap sin/cos (上流と同じ)

    def forward(self, x, t, h, omega, t_min, t_max, y=None, aug_cond=None):
        # ---- Mapping (ここだけが上流 SongUNet.forward との差分) ----
        emb = torch.cat([self._pos_emb(t), self._pos_emb(h)], dim=1)
        emb = emb + self.map_guidance(
            torch.cat(
                [self._pos_emb(omega), self._pos_emb(t_min), self._pos_emb(t_max)], dim=1
            )
        )
        if self.map_class is not None:
            emb = emb + self.map_class(y.reshape(-1))
        if self.map_augment is not None and aug_cond is not None:
            emb = emb + self.map_augment(aug_cond)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # ---- 以下 Encoder / Decoder は上流 SongUNet.forward と完全に同一 ----
        skips, aux = [], x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        aux, tmp = None, None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)

        if self.v_head:
            return aux[:, : self.n_out], aux[:, self.n_out :]
        # v-head 無しの場合、v は境界条件 v(z,t) = u(z,t,t) で読み替える (h=0 で呼ぶ)
        return aux, aux
