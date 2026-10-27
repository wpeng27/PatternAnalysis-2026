"""Numerical EMA checks, skipped-update semantics and checkpoint identity."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modules import ModelEMA, checkpoint_weight_kind
from train import parse_args, run_epoch
from training_state import WarmupCosine


class EMATests(unittest.TestCase):
    def test_numerical_average_buffers_and_independent_storage(self):
        model = nn.Linear(2, 2)
        model.register_buffer("float_buffer", torch.tensor(2.0))
        model.register_buffer("counter", torch.tensor(0))
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(2)
        random_state = torch.get_rng_state().clone()
        ema = ModelEMA(model, 0.75)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        self.assertFalse(ema.model.training)
        self.assertTrue(all(not p.requires_grad for p in ema.model.parameters()))
        self.assertNotEqual(model.weight.data_ptr(), ema.model.weight.data_ptr())
        with torch.no_grad():
            model.weight.fill_(6)
            model.bias.fill_(6)
            model.float_buffer.fill_(6)
            model.counter.fill_(7)
        ema.update(model)
        self.assertTrue(torch.equal(ema.model.weight, torch.full_like(model.weight, 3)))
        self.assertEqual(ema.model.float_buffer.item(), 3)
        self.assertEqual(ema.model.counter.item(), 7)
        ema.update(model)
        self.assertEqual(ema.model.bias[0].item(), 3.75)
        self.assertEqual(ema.updates, 2)

    def test_shadow_is_fp32_and_invalid_decay_rejected(self):
        model = nn.Linear(2, 2).to(torch.bfloat16)
        ema = ModelEMA(model, 0)
        self.assertEqual(ema.model.weight.dtype, torch.float32)
        ema.update(model)
        self.assertTrue(torch.equal(ema.model.weight, model.weight.float()))
        for decay in (-1, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ModelEMA(model, decay)

    def test_skipped_optimizer_step_does_not_update_ema(self):
        class FakeScaler:
            def __init__(self, skip):
                self.skip = skip
                self.value = 8.0
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def is_enabled(self):
                return True
            def get_scale(self):
                return self.value
            def step(self, optimizer):
                if not self.skip:
                    optimizer.step()
            def update(self):
                if self.skip:
                    self.value /= 2

        model = nn.Linear(2, 2)
        ema = ModelEMA(model, 0.5)
        original = deepcopy(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmupCosine(optimizer, total_steps=6, warmup_steps=2)
        loader = DataLoader(TensorDataset(torch.tensor([[1., 2.], [3., 1.]]), torch.tensor([0, 1])), batch_size=2)
        with redirect_stdout(io.StringIO()):
            skipped = run_epoch(model, loader, torch.device("cpu"), optimizer, FakeScaler(True), ema=ema,
                                step_scheduler=scheduler)
        self.assertEqual(skipped['amp_skipped_updates'], 1)
        self.assertEqual(ema.updates, 0)
        self.assertEqual(scheduler.completed_updates, 0)
        for k, v in original.items():
            self.assertTrue(torch.equal(v, ema.model.state_dict()[k]))
        with redirect_stdout(io.StringIO()):
            successful = run_epoch(model, loader, torch.device("cpu"), optimizer, FakeScaler(False), ema=ema,
                                   step_scheduler=scheduler)
        self.assertEqual(successful['optimizer_steps'], 1)
        self.assertEqual(ema.updates, 1)
        self.assertEqual(scheduler.completed_updates, 1)
        for k, v in original.items():
            self.assertTrue(torch.allclose(ema.model.state_dict()[k], 0.5*v + 0.5*model.state_dict()[k]))

    def test_checkpoint_identity_and_update_count(self):
        self.assertEqual(checkpoint_weight_kind({'config': {}}), 'online')
        checkpoint = {'config': {'ema': True, 'ema_decay': 0.999, 'validation_weights': 'ema'},
                      'evaluation_weights': 'ema', 'training_model_state_dict': {}, 'ema_updates': 3,
                      'history': [{'train': {'optimizer_steps': 1}}, {'train': {'optimizer_steps': 2}}]}
        self.assertEqual(checkpoint_weight_kind(checkpoint), 'ema')
        for key, value in (('ema_updates', 4), ('evaluation_weights', 'online')):
            bad = deepcopy(checkpoint)
            bad[key] = value
            with self.assertRaises(ValueError):
                checkpoint_weight_kind(bad)
        del checkpoint['training_model_state_dict']
        with self.assertRaises(ValueError):
            checkpoint_weight_kind(checkpoint)

    def test_cli_defaults_and_evaluation_cannot_override(self):
        self.assertFalse(parse_args(['--data-root', 'unused']).ema)
        args = parse_args(['--data-root', 'unused', '--ema'])
        self.assertEqual(args.ema_decay, 0.999)
        for options in (['--ema-decay', '0.9'], ['--ema', '--ema-decay', '1'],
                        ['--mode', 'evaluate', '--checkpoint', 'unused', '--ema']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--data-root', 'unused', *options])


if __name__ == '__main__':
    unittest.main()
