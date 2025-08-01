import pytest
from any_guardrail.utils.model_registry import model_regsitry


class TestModelRegistry:
    """Test cases for the model registry."""

    def test_model_registry_is_dict(self):
        """Test that model_registry is a dictionary."""
        assert isinstance(model_regsitry, dict)

    def test_model_registry_not_empty(self):
        """Test that model_registry is not empty."""
        assert len(model_regsitry) > 0

    def test_model_registry_contains_expected_guardrails(self):
        """Test that model_registry contains all expected guardrail classes."""
        from any_guardrail.guardrails.deepset import Deepset
        from any_guardrail.guardrails.duoguard import DuoGuard
        from any_guardrail.guardrails.flowjudge import FlowJudgeClass
        from any_guardrail.guardrails.glider import GLIDER
        from any_guardrail.guardrails.harmguard import HarmGuard
        from any_guardrail.guardrails.injecguard import InjecGuard
        from any_guardrail.guardrails.jasper import Jasper
        from any_guardrail.guardrails.pangolin import Pangolin
        from any_guardrail.guardrails.protectai import ProtectAI
        from any_guardrail.guardrails.sentinel import Sentinel
        from any_guardrail.guardrails.shield_gemma import ShieldGemma

        expected_classes = {
            Deepset,
            DuoGuard,
            FlowJudgeClass,
            GLIDER,
            HarmGuard,
            InjecGuard,
            Jasper,
            Pangolin,
            ProtectAI,
            Sentinel,
            ShieldGemma,
        }

        registry_classes = set(model_regsitry.values())
        assert registry_classes == expected_classes

    def test_model_registry_keys_are_strings(self):
        """Test that all keys in model_registry are strings."""
        for key in model_regsitry.keys():
            assert isinstance(key, str)

    def test_model_registry_values_are_classes(self):
        """Test that all values in model_registry are classes."""
        for value in model_regsitry.values():
            assert isinstance(value, type)

    def test_model_registry_contains_deepset(self):
        """Test that model_registry contains Deepset guardrail."""
        from any_guardrail.guardrails.deepset import Deepset
        assert "deepset/deberta-v3-base-injection" in model_regsitry
        assert model_regsitry["deepset/deberta-v3-base-injection"] == Deepset

    def test_model_registry_contains_duoguard(self):
        """Test that model_registry contains DuoGuard guardrails."""
        from any_guardrail.guardrails.duoguard import DuoGuard
        duoguard_keys = [
            "DuoGuard/DuoGuard-0.5B",
            "DuoGuard/DuoGuard-1B-Llama-3.2-transfer",
            "DuoGuard/DuoGuard-1.5B-transfer",
        ]
        for key in duoguard_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == DuoGuard

    def test_model_registry_contains_flowjudge(self):
        """Test that model_registry contains FlowJudge guardrail."""
        from any_guardrail.guardrails.flowjudge import FlowJudgeClass
        flowjudge_keys = ["Flowjudge", "flowjudge", "FlowJudge"]
        for key in flowjudge_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == FlowJudgeClass

    def test_model_registry_contains_glider(self):
        """Test that model_registry contains GLIDER guardrail."""
        from any_guardrail.guardrails.glider import GLIDER
        assert "PatronusAI/glider" in model_regsitry
        assert model_regsitry["PatronusAI/glider"] == GLIDER

    def test_model_registry_contains_harmguard(self):
        """Test that model_registry contains HarmGuard guardrail."""
        from any_guardrail.guardrails.harmguard import HarmGuard
        assert "hbseong/HarmAug-Guard" in model_regsitry
        assert model_regsitry["hbseong/HarmAug-Guard"] == HarmGuard

    def test_model_registry_contains_injecguard(self):
        """Test that model_registry contains InjecGuard guardrail."""
        from any_guardrail.guardrails.injecguard import InjecGuard
        assert "leolee99/InjecGuard" in model_regsitry
        assert model_regsitry["leolee99/InjecGuard"] == InjecGuard

    def test_model_registry_contains_jasper(self):
        """Test that model_registry contains Jasper guardrails."""
        from any_guardrail.guardrails.jasper import Jasper
        jasper_keys = [
            "JasperLS/deberta-v3-base-injection",
            "JasperLS/gelectra-base-injection",
        ]
        for key in jasper_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == Jasper

    def test_model_registry_contains_pangolin(self):
        """Test that model_registry contains Pangolin guardrails."""
        from any_guardrail.guardrails.pangolin import Pangolin
        pangolin_keys = [
            "dcarpintero/pangolin-guard-large",
            "dcarpintero/pangolin-guard-base",
        ]
        for key in pangolin_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == Pangolin

    def test_model_registry_contains_protectai(self):
        """Test that model_registry contains ProtectAI guardrails."""
        from any_guardrail.guardrails.protectai import ProtectAI
        protectai_keys = [
            "protectai/deberta-v3-base-prompt-injection",
            "protectai/deberta-v3-small-prompt-injection-v2",
            "protectai/deberta-v3-base-prompt-injection-v2",
        ]
        for key in protectai_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == ProtectAI

    def test_model_registry_contains_sentinel(self):
        """Test that model_registry contains Sentinel guardrail."""
        from any_guardrail.guardrails.sentinel import Sentinel
        assert "qualifire/prompt-injection-sentinel" in model_regsitry
        assert model_regsitry["qualifire/prompt-injection-sentinel"] == Sentinel

    def test_model_registry_contains_shield_gemma(self):
        """Test that model_registry contains ShieldGemma guardrails."""
        from any_guardrail.guardrails.shield_gemma import ShieldGemma
        shield_gemma_keys = [
            "google/shieldgemma-2b",
            "google/shieldgemma-9b",
            "google/shieldgemma-27b",
        ]
        for key in shield_gemma_keys:
            assert key in model_regsitry
            assert model_regsitry[key] == ShieldGemma

    def test_model_registry_no_duplicate_keys(self):
        """Test that model_registry has no duplicate keys."""
        keys = list(model_regsitry.keys())
        unique_keys = set(keys)
        assert len(keys) == len(unique_keys)

    def test_model_registry_values_are_callable(self):
        """Test that all values in model_registry are callable (classes)."""
        for value in model_regsitry.values():
            assert callable(value)
            