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

model_registry = {
    "deepset/deberta-v3-base-injection": Deepset,
    "DuoGuard/DuoGuard-0.5B": DuoGuard,
    "DuoGuard/DuoGuard-1B-Llama-3.2-transfer": DuoGuard,
    "DuoGuard/DuoGuard-1.5B-transfer": DuoGuard,
    "Flowjudge": FlowJudgeClass,
    "flowjudge": FlowJudgeClass,
    "FlowJudge": FlowJudgeClass,
    "PatronusAI/glider": GLIDER,
    "hbseong/HarmAug-Guard": HarmGuard,
    "leolee99/InjecGuard": InjecGuard,
    "JasperLS/deberta-v3-base-injection": Jasper,
    "JasperLS/gelectra-base-injection": Jasper,
    "dcarpintero/pangolin-guard-large": Pangolin,
    "dcarpintero/pangolin-guard-base": Pangolin,
    "protectai/deberta-v3-base-prompt-injection": ProtectAI,
    "protectai/deberta-v3-small-prompt-injection-v2": ProtectAI,
    "protectai/deberta-v3-base-prompt-injection-v2": ProtectAI,
    "qualifire/prompt-injection-sentinel": Sentinel,
    "google/shieldgemma-2b": ShieldGemma,
    "google/shieldgemma-9b": ShieldGemma,
    "google/shieldgemma-27b": ShieldGemma,
}
