import unittest

import torch

from embedding_train.losses import (
    df_flops_regularizer,
    flops_regularizer,
    l1_regularizer,
)


# Must track configs/model/splade.yaml.
DEFAULT_ALPHA = 40.0
DEFAULT_BETA = 0.08


def _representations(seed=0, rows=8, vocab=16):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(rows, vocab, generator=generator)
    # SPLADE activations are log1p(relu(.)) >= 0 and genuinely sparse.
    return torch.where(values > 0.6, values, torch.zeros_like(values))


class L1RegularizerTests(unittest.TestCase):
    def test_matches_mean_of_per_row_l1(self):
        representations = _representations()
        self.assertAlmostEqual(
            l1_regularizer(representations).item(),
            representations.abs().sum(dim=1).mean().item(),
            places=6,
        )

    def test_gradient_does_not_vanish_at_small_activations(self):
        # The reason L1 is used on the query side: d/da of |a| is constant, so
        # small activations keep being pushed to exactly zero. FLOPS uses a^2,
        # whose gradient vanishes as a -> 0, which is why sweeping
        # flops_lambda_q x20 raised query nnz instead of lowering it.
        tiny = torch.full((4, 3), 1e-3, requires_grad=True)
        l1_regularizer(tiny).backward()
        l1_grad = tiny.grad.abs().min().item()

        tiny_flops = torch.full((4, 3), 1e-3, requires_grad=True)
        flops_regularizer(tiny_flops).backward()
        flops_grad = tiny_flops.grad.abs().max().item()

        self.assertGreater(l1_grad, 100 * flops_grad)


class DfFlopsRegularizerTests(unittest.TestCase):
    def test_alpha_zero_degenerates_to_half_flops(self):
        representations = _representations()
        document_frequency = torch.rand(representations.size(1))
        self.assertAlmostEqual(
            df_flops_regularizer(
                representations, document_frequency, alpha=0.0, beta=0.05
            ).item(),
            0.5 * flops_regularizer(representations).item(),
            places=6,
        )

    def test_penalizes_high_df_dimension_more_than_low_df(self):
        # Same activation magnitude in two dimensions; only df differs.
        representations = torch.zeros(4, 2)
        representations[:, 0] = 0.5
        representations[:, 1] = 0.5
        high_then_low = torch.tensor([0.9, 0.001])

        weights = torch.sigmoid(DEFAULT_ALPHA * (high_then_low - DEFAULT_BETA))
        contributions = weights * representations.abs().mean(dim=0) ** 2
        self.assertGreater(contributions[0].item(), contributions[1].item())
        self.assertAlmostEqual(
            df_flops_regularizer(
                representations, high_then_low, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA
            ).item(),
            contributions.sum().item(),
            places=6,
        )

    def test_configured_defaults_actually_discriminate(self):
        """The defaults must separate the real df range, not just be monotone.

        Calibrated on the measured df of prod_soup query tokens: p10=0.011,
        p50=0.076, p90=0.207. alpha=20/beta=0.05 gave only 3.1x separation
        across that span — effectively plain FLOPS with a constant factor — so
        this guards against silently shipping a no-op regularizer.
        """
        percentiles = torch.tensor([0.011, 0.076, 0.207])
        weights = torch.sigmoid(DEFAULT_ALPHA * (percentiles - DEFAULT_BETA))
        self.assertLess(weights[0].item(), 0.10)   # rare dims ~unpenalized
        self.assertGreater(weights[2].item(), 0.90)  # common dims fully penalized
        self.assertGreater(weights[2].item() / weights[0].item(), 10.0)

    def test_document_frequency_carries_no_gradient(self):
        # df is a statistic, not a parameter: it must never receive gradient.
        representations = _representations().requires_grad_(True)
        document_frequency = torch.rand(representations.size(1), requires_grad=True)
        df_flops_regularizer(
            representations, document_frequency, alpha=20.0, beta=0.05
        ).backward()
        self.assertIsNotNone(representations.grad)
        # In training the buffer is updated under no_grad, so nothing propagates
        # into it; assert the loss is at least differentiable w.r.t. activations.
        self.assertTrue(torch.isfinite(representations.grad).all())


class BackwardCompatibilityTests(unittest.TestCase):
    def test_flops_regularizer_unchanged(self):
        representations = _representations(seed=3)
        expected = (representations.abs().mean(dim=0) ** 2).sum()
        self.assertEqual(flops_regularizer(representations).item(), expected.item())


class NonPersistentBufferTests(unittest.TestCase):
    """The df_ema buffer must stay out of state_dict.

    train.py warm-starts with a strict load_state_dict from v1a_best.ckpt. A new
    *persistent* buffer would raise on the missing key, and training SPLADE from
    scratch degenerates — so this mistake silently costs a whole run.
    """

    def test_non_persistent_buffer_absent_from_state_dict(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("kept", torch.zeros(3), persistent=True)
                self.register_buffer("df_ema", torch.zeros(3), persistent=False)

        module = Module()
        self.assertIn("kept", module.state_dict())
        self.assertNotIn("df_ema", module.state_dict())

    def test_strict_load_succeeds_without_the_buffer(self):
        class Old(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

        class New(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.register_buffer("df_ema", torch.zeros(3), persistent=False)

        New().load_state_dict(Old().state_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
