"""Synthetic architecture checks, not ADNI training or performance evidence."""

import unittest

import torch
from torch.nn import functional as F

from modules import ConvNeXtBlock


class ConvNeXtBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)

    def check_backward(self, shape, device="cpu", amp=False):
        block = ConvNeXtBlock(shape[1], drop_path=0.2).to(device).train()
        inputs = torch.randn(shape, device=device, requires_grad=True)
        with torch.autocast(device_type=device, enabled=amp):
            outputs = block(inputs)
            loss = outputs.float().square().mean()
        loss.backward()
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for name, parameter in block.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_tiny_stage_shapes_and_finite_gradients(self):
        for shape in ((2, 96, 56, 56), (2, 192, 28, 28),
                      (2, 384, 14, 14), (2, 768, 7, 7), (1, 8, 13, 17)):
            with self.subTest(shape=shape):
                self.check_backward(shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_float32_and_amp_gradients(self):
        for amp in (False, True):
            with self.subTest(amp=amp):
                self.check_backward((2, 96, 56, 56), device="cuda", amp=amp)

    def test_matches_independent_nchw_reference(self):
        # Use 1x1 convolutions and explicit channel normalization to catch a
        # wrong permutation, LN axis, operation order or residual placement.
        block = ConvNeXtBlock(4, layer_scale_init_value=0.5).double().eval()
        inputs = torch.randn(2, 4, 5, 7, dtype=torch.float64, requires_grad=True)
        actual = block(inputs)
        branch = F.conv2d(inputs, block.dwconv.weight, block.dwconv.bias,
                          padding=3, groups=4)
        variance, mean = torch.var_mean(branch, dim=1, correction=0, keepdim=True)
        branch = (branch - mean) * torch.rsqrt(variance + 1e-6)
        branch = branch * block.norm.weight[None, :, None, None] + block.norm.bias[None, :, None, None]
        branch = F.conv2d(branch, block.pwconv1.weight[:, :, None, None], block.pwconv1.bias)
        branch = F.conv2d(F.gelu(branch), block.pwconv2.weight[:, :, None, None], block.pwconv2.bias)
        expected = inputs + branch * block.gamma[None, :, None, None]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        parameters = (inputs, *block.parameters())
        actual_grads = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().mean(), parameters)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)

    def test_zero_layer_scale_is_identity_with_identity_input_gradient(self):
        block = ConvNeXtBlock(8, layer_scale_init_value=0)
        inputs = torch.randn(2, 8, 9, 11, requires_grad=True)
        outputs = block(inputs)
        self.assertTrue(torch.equal(outputs, inputs))
        outputs.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_stochastic_depth_drops_whole_sample_branch_and_scales_survivors(self):
        block = ConvNeXtBlock(4, drop_path=0.5, layer_scale_init_value=1).eval()
        inputs = torch.randn(64, 4, 5, 7)
        with torch.no_grad():
            expected = block(inputs)
            self.assertTrue(torch.equal(expected, block(inputs)))
            block.train()
            torch.manual_seed(42)
            actual = block(inputs)
            dropped = (actual == inputs).flatten(1).all(1)
            self.assertTrue(dropped.any())
            self.assertTrue((~dropped).any())
            torch.testing.assert_close(actual[~dropped],
                                       inputs[~dropped] + 2 * (expected[~dropped] - inputs[~dropped]))
            block.drop_path.p = 1.0
            self.assertTrue(torch.equal(block(inputs), inputs))
            block.eval()
            torch.testing.assert_close(block(inputs), expected)
            block.drop_path.p = 0.0
            block.train()
            torch.testing.assert_close(block(inputs), expected)

    def test_default_scale_and_invalid_settings(self):
        torch.testing.assert_close(ConvNeXtBlock(8).gamma, torch.full((8,), 1e-6))
        for settings in ({"dim": 0}, {"dim": 2.5}, {"dim": True},
                         {"dim": 8, "drop_path": -0.1}, {"dim": 8, "drop_path": 1.1},
                         {"dim": 8, "drop_path": float("nan")},
                         {"dim": 8, "layer_scale_init_value": -1},
                         {"dim": 8, "layer_scale_init_value": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ConvNeXtBlock(**settings)


if __name__ == "__main__":
    unittest.main()
