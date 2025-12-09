
import numpy as np

from src.functions.risk_rules import fuse_model_rule_votes


def test_rule_fusion_outputs_flags():
    model_scores = np.array([0.1, 0.6, 0.9])
    rule_scores = np.array([10.0, 70.0, 95.0])

    fused_scores, fused_flags, rules_triggered = fuse_model_rule_votes(
        model_scores,
        rule_scores,
        model_weight=0.5,
        rule_weight=0.5,
        threshold=0.5,
        trigger_threshold=60.0,
    )

    assert fused_scores.shape == model_scores.shape
    assert fused_flags.dtype == bool
    assert rules_triggered.dtype == bool
    assert fused_flags[-1], "High scores should exceed the fusion threshold"
    assert rules_triggered[1], "Mid rule score should trigger when above threshold"
