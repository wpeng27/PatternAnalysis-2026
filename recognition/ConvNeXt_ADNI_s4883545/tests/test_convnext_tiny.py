"""Full-model architecture checks on synthetic inputs, not ADNI performance."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from modules import ChannelLayerNorm, ConvNeXtBlock, ConvNeXtTiny, build_model


class ConvNeXtTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        torch.manual_seed(42)
        cls.model = build_model("convnext_tiny")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.set_num_threads(cls.previous_threads)

    def test_feature_shapes_and_full_backward(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        observed = {}
        def record(name):
            def hook(module, inputs, outputs):
                observed[name] = tuple(outputs.shape)
            return hook
        layers = {f"downsample_{i}": layer for i, layer in enumerate(model.downsample_layers)}
        layers.update({f"stage_{i + 1}": layer for i, layer in enumerate(model.stages)})
        layers.update(pool=model.avgpool, norm=model.norm, head=model.head)
        handles = [layer.register_forward_hook(record(name)) for name, layer in layers.items()]
        images = torch.randn(2, 3, 224, 224, requires_grad=True)
        try:
            logits = model(images)
        finally:
            for handle in handles:
                handle.remove()
        for i, (width, size) in enumerate(((96, 56), (192, 28), (384, 14), (768, 7))):
            self.assertEqual(observed[f"downsample_{i}"], (2, width, size, size))
            self.assertEqual(observed[f"stage_{i + 1}"], (2, width, size, size))
        self.assertEqual(observed["pool"], (2, 768, 1, 1))
        self.assertEqual(observed["norm"], (2, 768))
        self.assertEqual(observed["head"], (2, 2))
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(images.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_tiny_structure_and_scratch_initialization(self):
        model = self.model
        self.assertEqual([len(stage) for stage in model.stages], [3, 3, 9, 3])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 27_821_666)
        for stage, width in zip(model.stages, (96, 192, 384, 768)):
            for block in stage:
                self.assertIsInstance(block, ConvNeXtBlock)
                self.assertEqual(block.dwconv.groups, width)
                self.assertEqual(block.drop_path.p, 0)
                torch.testing.assert_close(block.gamma, torch.full((width,), 1e-6))
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                self.assertAlmostEqual(module.weight.std().item(), 0.02, delta=0.002)
            elif isinstance(module, nn.LayerNorm):
                self.assertEqual(module.eps, 1e-6)
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))

    def test_channels_first_norm_matches_per_pixel_reference(self):
        inputs = torch.randn(2, 8, 5, 7)
        norm = ChannelLayerNorm(8)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
            norm.bias.copy_(torch.linspace(-0.2, 0.2, 8))
        variance, mean = torch.var_mean(inputs, dim=1, correction=0, keepdim=True)
        expected = (inputs - mean) * torch.rsqrt(variance + 1e-6)
        expected = expected * norm.weight[None, :, None, None] + norm.bias[None, :, None, None]
        torch.testing.assert_close(norm(inputs), expected)

    def test_depth_schedule_and_invalid_configuration(self):
        model = ConvNeXtTiny(drop_path_rate=0.2)
        probabilities = [block.drop_path.p for stage in model.stages for block in stage]
        torch.testing.assert_close(torch.tensor(probabilities), torch.linspace(0, 0.2, 18))
        for kwargs in ({"drop_path_rate": -0.1}, {"drop_path_rate": float("nan")},
                       {"drop_path_rate": 1.1}, {"layer_scale_init_value": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConvNeXtTiny(**kwargs)


if __name__ == "__main__":
    unittest.main()
