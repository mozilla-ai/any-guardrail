from any_guardrail.utils.model_registry import model_regsitry
from any_guardrail.guardrails.guardrail import Guardrail
from typing import List

class CallGuardrail():
    """
    Allows users to dynamically call upon a guardrail using either their HuggingFace path, or, if not implemented with
    HuggingFace, the name of the model.
    """
    def __init__(self) -> None:
        self.model_registry = model_regsitry
    
    def list_all_supported_guardrails(self) -> List[str]:
        return list(self.model_registry.keys())
    
    def call_guardrail(self, guardrail_path: str, **kwargs) -> Guardrail:
        if guardrail_path in self.model_registry.keys():
            return self.model_registry[guardrail_path](modelpath=guardrail_path, **kwargs)
        else:
            raise ValueError("We do not support the model you are trying to instantiate. Please use list_all_support_guardrails" \
            "to see which guardrails are supported.")
