"""Parameter grouping, actual AdamW shrinkage and checkpoint provenance checks."""

import copy
import unittest

import torch
from torch import nn

from compare_validation import verify_optimizer_groups
from modules import build_model, weight_decay_groups
from train import parse_args


class WeightDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_convnext_membership_covers_each_trainable_parameter_once(self):
        model = build_model("convnext_tiny")
        groups, metadata = weight_decay_groups(model, .01, "exclude_1d_bias")
        self.assertEqual([g["weight_decay"] for g in groups], [.01, 0])
        names = [name for group in metadata for name in group["parameter_names"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(dict(model.named_parameters())))
        self.assertEqual(sum(g["parameter_count"] for g in metadata), 27_821_666)
        decayed = set(metadata[0]["parameter_names"])
        excluded = set(metadata[1]["parameter_names"])
        for name in ("downsample_layers.0.0.weight", "stages.0.0.dwconv.weight", "stages.0.0.pwconv1.weight", "head.weight"):
            self.assertIn(name, decayed)
        for name in ("downsample_layers.0.0.bias", "downsample_layers.0.1.weight", "stages.0.0.gamma",
                     "stages.0.0.norm.weight", "stages.0.0.dwconv.bias", "norm.weight", "head.bias"):
            self.assertIn(name, excluded)
        self.assertEqual({id(p) for group in groups for p in group["params"]}, {id(p) for p in model.parameters()})

    def test_zero_gradient_step_decays_matrix_but_not_bias_or_norm(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
        for p in model.parameters():
            nn.init.ones_(p)
            p.grad = torch.zeros_like(p)
        groups, _ = weight_decay_groups(model, .2, "exclude_1d_bias")
        torch.optim.AdamW(groups, lr=.1).step()
        torch.testing.assert_close(model[0].weight, torch.full_like(model[0].weight, .98))
        for p in (model[0].bias, model[1].weight, model[1].bias):
            torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)

    def test_legacy_group_order_and_frozen_parameter_exclusion(self):
        model = nn.Linear(3, 2)
        groups, _ = weight_decay_groups(model, .01)
        self.assertEqual([id(p) for p in groups[0]["params"]], [id(p) for p in model.parameters()])
        self.assertNotIn("param_names", groups[0])
        model.bias.requires_grad_(False)
        groups, metadata = weight_decay_groups(model, .01, "exclude_1d_bias")
        self.assertEqual(len(groups), 1)
        self.assertEqual(metadata[0]["parameter_names"], ["weight"])

    def make_checkpoint(self, policy):
        model = build_model("small_cnn")
        groups, metadata = weight_decay_groups(model, .01, policy)
        optimizer = torch.optim.AdamW(groups, lr=.001)
        config = {"model": "small_cnn", "model_kwargs": {"drop_path_rate": 0}, "weight_decay": .01,
                  "weight_decay_policy": policy, "optimizer_parameter_groups": metadata}
        return {"config": copy.deepcopy(config), "optimizer_state_dict": optimizer.state_dict()}, config

    def test_verifier_accepts_grouped_and_legacy_checkpoints(self):
        checkpoint, config = self.make_checkpoint("exclude_1d_bias")
        verify_optimizer_groups(checkpoint, config, .001)
        checkpoint, config = self.make_checkpoint("all")
        for record in (config, checkpoint["config"]):
            record.pop("weight_decay_policy")
            record.pop("optimizer_parameter_groups")
        verify_optimizer_groups(checkpoint, config, .001)

    def test_verifier_rejects_wrong_decay_membership_and_metadata(self):
        original, config = self.make_checkpoint("exclude_1d_bias")
        for fault in ("decay", "names", "duplicate", "metadata", "policy"):
            checkpoint = copy.deepcopy(original)
            groups = checkpoint["optimizer_state_dict"]["param_groups"]
            if fault == "decay":
                groups[1]["weight_decay"] = .01
            elif fault == "names":
                groups[0]["param_names"].reverse()
            elif fault == "duplicate":
                groups[1]["params"][0] = groups[0]["params"][0]
            elif fault == "metadata":
                checkpoint["config"].pop("optimizer_parameter_groups")
            else:
                checkpoint["config"]["weight_decay_policy"] = "all"
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                verify_optimizer_groups(checkpoint, config, .001)

    def test_cli_default_and_opt_in(self):
        self.assertEqual(parse_args(["--data-root", "fixture"]).weight_decay_policy, "all")
        args = parse_args(["--data-root", "fixture", "--weight-decay-policy", "exclude_1d_bias"])
        self.assertEqual(args.weight_decay_policy, "exclude_1d_bias")


if __name__ == "__main__":
    unittest.main()
