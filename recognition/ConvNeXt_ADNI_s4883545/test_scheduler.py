"""Warmup/cosine endpoints, update order, state validation and CLI bounds."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import unittest

import torch

from train import parse_args
from training_state import WarmupCosine, verify_cosine_checkpoint


class SchedulerTests(unittest.TestCase):
    def make(self, total=6, warmup=2):
        params = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
        optimizer = torch.optim.SGD([{'params': [params[0]], 'weight_decay': 0.01},
                                     {'params': [params[1]], 'weight_decay': 0.0}], lr=0.1)
        return optimizer, WarmupCosine(optimizer, total, warmup, min_lr=0.01, start_factor=0.1)

    def test_known_sequence_no_duplicate_peak_and_exact_final_rate(self):
        optimizer, schedule = self.make()
        for expected in (0.01, 0.055, 0.1, 0.0775, 0.0325, 0.01):
            for group in optimizer.param_groups:
                self.assertAlmostEqual(group['lr'], expected)
            optimizer.step()
            schedule.step()
        self.assertEqual([g['weight_decay'] for g in optimizer.param_groups], [0.01, 0.0])
        self.assertEqual(schedule.completed_updates, 6)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            schedule.step()

    def test_zero_warmup_and_invalid_lengths(self):
        optimizer, schedule = self.make(total=3, warmup=0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.1)
        self.assertAlmostEqual(schedule.rates_at(1)[0], 0.055)
        self.assertEqual(schedule.rates_at(2)[0], 0.01)
        for total, warmup in ((0, 0), (1, 0), (4, -1), (4, 3), (3.5, 1)):
            with self.assertRaises(ValueError):
                self.make(total, warmup)

    def test_resume_keeps_next_lr_and_rejects_changed_horizon_or_count(self):
        optimizer, schedule = self.make()
        optimizer.step()
        schedule.step()
        saved = deepcopy(schedule.state_dict())
        other_optimizer, other = self.make()
        other_optimizer.load_state_dict(optimizer.state_dict())
        other.load_state_dict(saved)
        self.assertEqual(other.state_dict(), saved)
        schedule.step()
        other.step()
        self.assertEqual(other.state_dict(), schedule.state_dict())
        for key, value in (('completed_updates', 99), ('last_lrs', [0.1, 0.1])):
            bad = deepcopy(saved)
            bad[key] = value
            with self.assertRaises(ValueError):
                other.load_state_dict(bad)
        _, changed = self.make(total=7)
        with self.assertRaisesRegex(ValueError, 'horizon'):
            changed.load_state_dict(saved)

    def test_checkpoint_verifier_rejects_wrong_update_or_rate(self):
        optimizer, scheduler = self.make()
        first_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        config = {'scheduler_settings': dict(scheduler.settings)}
        checkpoint = {'config': config, 'scheduler_state_dict': scheduler.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'history': [{'learning_rate': first_lr, 'next_learning_rate': optimizer.param_groups[0]['lr'],
                                   'scheduler_updates': 1, 'train': {'optimizer_steps': 1}}]}
        self.assertEqual(verify_cosine_checkpoint(checkpoint, config), scheduler.settings)
        for field in ('learning_rate', 'next_learning_rate', 'scheduler_updates'):
            bad = deepcopy(checkpoint)
            bad['history'][0][field] = 99
            with self.assertRaises(ValueError):
                verify_cosine_checkpoint(bad, config)

    def test_cli_bounds_and_legacy_default(self):
        self.assertEqual(parse_args(['--data-root','unused']).lr_schedule, 'plateau')
        args = parse_args(['--data-root','unused','--lr-schedule','warmup_cosine','--epochs','40'])
        self.assertEqual((args.warmup_epochs, args.schedule_epochs), (3, 40))
        invalid = [ ['--warmup-epochs','3'], ['--schedule-epochs','40'],
                    ['--lr-schedule','warmup_cosine','--epochs','3'],
                    ['--lr-schedule','warmup_cosine','--epochs','40','--schedule-epochs','20'],
                    ['--lr-schedule','warmup_cosine','--epochs','40','--warmup-epochs','-1'],
                    ['--lr-schedule','warmup_cosine','--mode','evaluate','--checkpoint','unused'] ]
        for options in invalid:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--data-root','unused',*options])


if __name__ == '__main__':
    unittest.main()
