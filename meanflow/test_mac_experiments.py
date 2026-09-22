"""CPU regression checks: python -m unittest test_mac_experiments -v."""
import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from train_arg_parser import get_args_parser
from mac_experiment import (mac_bounds, scheduled_mac_percentile, validate_mac_args,
                            experiment_id, training_config, check_resume_config)
from models.mac import mac_weights, apply_mac_losses
from models.imf import iMF
from runner import Runner, TrainingBatchSampler


def args_for_test():
    args = get_args_parser().parse_args([])
    args.method, args.num_classes = "imf", 0
    args.mac, args.mac_timing, args.mac_warmup_iters = True, "late", 0
    args.total_iters, args.batch_size, args.grad_accum = 30000, 4, 1
    args.model_channels, args.ema_decays, args.use_edm_aug = 32, [], False
    args.norm_p, args.num_workers, args.dropout = 0.0, 0, 0.0
    args.use_cfg, args.use_cfg_interval = False, False
    args.seed, args.eval_seed = 42, 123
    return args


class TinyNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.main_scale = nn.Parameter(torch.tensor(0.2))
        self.aux_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, z, t, h, omega, t_min, t_max, y=None, aug_cond=None):
        return self.main_scale * z + 0.1 * h, self.aux_scale * z


def tiny_model(args):
    return iMF(TinyNet, args, {})


class TinyData(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = torch.arange(12 * 3 * 2 * 2, dtype=torch.float32).reshape(12, 3, 2, 2) / 144

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], 0


class ScheduleTests(unittest.TestCase):
    def test_exact_windows_and_boundaries(self):
        args = args_for_test()
        expected = {"none": (0, 0), "early": (0, 15000), "middle": (7500, 22500),
                    "late": (15000, 30000), "all": (0, 30000)}
        for timing, interval in expected.items():
            args.mac_timing = timing
            self.assertEqual(mac_bounds(args), interval)
            active = [s for s in range(args.total_iters) if scheduled_mac_percentile(args, s) is not None]
            self.assertEqual(active, list(range(*interval)))
        args.mac = False
        self.assertIsNone(scheduled_mac_percentile(args, 20000))

    def test_invalid_conditions_fail(self):
        args = args_for_test()
        args.mac_warmup_iters = 6000
        with self.assertRaises(ValueError):
            validate_mac_args(args)
        args.mac_warmup_iters, args.mac_target, args.v_head = 0, "aux", False
        with self.assertRaises(ValueError):
            validate_mac_args(args)
        args.v_head, args.mac_timing = True, "custom"
        args.mac_start_fraction, args.mac_end_fraction = 0.8, 0.2
        with self.assertRaises(ValueError):
            validate_mac_args(args)

    def test_unique_run_and_resume_guard(self):
        args = args_for_test()
        original = experiment_id(args)
        changed = copy.deepcopy(args)
        changed.mac_target = "main"
        self.assertNotEqual(original, experiment_id(changed))
        with self.assertRaises(ValueError):
            check_resume_config(training_config(args), training_config(changed))
        with self.assertRaises(ValueError):
            check_resume_config(None, training_config(args))


class LossTests(unittest.TestCase):
    def test_selection_and_normalization(self):
        err = torch.tensor([4., 1., 3., 2.])
        w, mask = mac_weights(err, 0.5, 1.0)
        self.assertEqual(w.tolist(), [1., 2., 1., 2.])
        self.assertEqual(mask.sum(), 2)
        normalized, _ = mac_weights(err, 0.5, 1.0, normalize=True)
        self.assertEqual(normalized.mean(), 1)

    def test_random_control_does_not_advance_training_rng(self):
        torch.manual_seed(9)
        before = torch.get_rng_state().clone()
        w, mask = mac_weights(torch.arange(8.), 0.5, 1., selection="random",
                              generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(mask.sum(), 4)
        self.assertEqual(w.mean(), 1.5)

    def test_target_routes_gradients(self):
        for target, expected_main, expected_aux in [
            ("main", [1., 2.], [1., 1.]),
            ("aux", [1., 1.], [1., 2.]),
            ("both", [1., 2.], [1., 2.]),
        ]:
            main = torch.tensor([2., 3.], requires_grad=True)
            aux = torch.tensor([5., 7.], requires_grad=True)
            a, b = apply_mac_losses(main, aux, torch.tensor([1., 2.]), target)
            (a + b).sum().backward()
            self.assertEqual(main.grad.tolist(), expected_main)
            self.assertEqual(aux.grad.tolist(), expected_aux)

    def test_real_imf_forward_loss_and_gradient_routing(self):
        x = torch.arange(48., dtype=torch.float32).reshape(4, 3, 2, 2) / 48
        base_grads = {}
        args = args_for_test()
        # Select the whole batch so the expected gradient multiplier is exactly two.
        for target in ("none", "main", "aux", "both"):
            args.mac_target = "both" if target == "none" else target
            model = tiny_model(args)
            torch.manual_seed(19)
            loss, parts = model.forward_with_loss(x, mac_percentile=None if target == "none" else 1.0)
            loss.backward()
            base_grads[target] = (model.net.main_scale.grad.clone(), model.net.aux_scale.grad.clone())
            if target != "none":
                self.assertAlmostEqual(parts["loss_u"], parts["loss_u_unweighted"] * (2 if target in ("main", "both") else 1), places=5)
                self.assertAlmostEqual(parts["loss_v"], parts["loss_v_unweighted"] * (2 if target in ("aux", "both") else 1), places=5)
        for target in ("main", "aux", "both"):
            for j, name in enumerate(("main", "aux")):
                expected = base_grads["none"][j] * (2 if target in (name, "both") else 1)
                torch.testing.assert_close(base_grads[target][j], expected)

    def test_disabled_mac_skips_endpoint_scoring(self):
        model = tiny_model(args_for_test())
        with patch("models.imf.endpoint_error", side_effect=AssertionError("MAC must be skipped")):
            loss, _ = model.forward_with_loss(torch.ones(4, 3, 2, 2), mac_percentile=None)
        self.assertTrue(torch.isfinite(loss))


class RunnerTests(unittest.TestCase):
    def test_sampler_resume_matches_uninterrupted_batches(self):
        batches = list(itertools.islice(iter(TrainingBatchSampler(13, 4, 42)), 11))
        resumed = list(itertools.islice(iter(TrainingBatchSampler(13, 4, 42, 5)), 6))
        self.assertEqual(batches[5:], resumed)

    def test_fid_noise_labels_and_training_rng_are_repeatable(self):
        runner = Runner.__new__(Runner)
        runner.device, runner.args, runner.is_imf = torch.device("cpu"), args_for_test(), True
        runner.args.num_classes = 10
        captured = []
        class Metric:
            def reset(self): pass
            def update(self, x, real): pass
            def compute(self): return torch.tensor(0.)
        def initialize():
            torch.randn(7)  # Simulate random initialization of an evaluation network.
            return Metric()
        runner._fid_metric = initialize
        def sample(n, generator, labels, **kwargs):
            noise = torch.randn(n, 3, 2, 2, generator=generator)
            captured.append((noise.clone(), labels.clone()))
            return noise
        runner.sample = sample
        torch.manual_seed(11)
        before = torch.get_rng_state().clone()
        runner.compute_fid(n_samples=7, num_steps=1, bs=3)
        runner.compute_fid(n_samples=7, num_steps=4, bs=3)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for left, right in zip(captured[:3], captured[3:]):
            self.assertTrue(torch.equal(left[0], right[0]))
            self.assertTrue(torch.equal(left[1], right[1]))

    @patch("runner.torchvision.datasets.CIFAR10", TinyData)
    @patch("runner.instantiate_model", tiny_model)
    def test_training_resume_matches_uninterrupted_with_random_mac(self):
        with tempfile.TemporaryDirectory() as root:
            args = args_for_test()
            args.total_iters, args.log_every = 6, 2
            args.ckpt_every, args.sample_every, args.eval_every = 2, 100, 100
            args.mac_selection, args.mac_timing = "random", "middle"
            args.ckpt_dir = str(Path(root) / "full")
            full = Runner(args, device="cpu")
            full.train()
            expected = copy.deepcopy(full.model.state_dict())

            interrupted_args = copy.deepcopy(args)
            interrupted_args.ckpt_dir = str(Path(root) / "resume")
            interrupted = Runner(interrupted_args, device="cpu")
            original_save = interrupted.save
            def stop_at_two(step):
                original_save(step)
                if step == 2:
                    raise InterruptedError("simulated interruption")
            interrupted.save = stop_at_two
            with self.assertRaises(InterruptedError):
                interrupted.train()
            interrupted_args.resume_training = True
            resumed = Runner(interrupted_args, device="cpu")
            resumed.train()
            for key, value in expected.items():
                torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
            logs = [json.loads(line) for line in (Path(args.ckpt_dir) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["mac/active"] for row in logs], [1., 1., 0.])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
