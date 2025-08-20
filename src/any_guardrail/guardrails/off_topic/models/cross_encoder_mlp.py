from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class CrossEncoderWithMLP(nn.Module, PyTorchModelHubMixin):
    """Defines the classification head for the [govtech/stsb-roberta-base-off-topic model](https://huggingface.co/govtech/stsb-roberta-base-off-topic).

    The classification head sits on top of the cross-encoder/stsb-roberta-base from Meta.

    Args:
        base_model: The base cross-encoder model. This model must be the cross-encoder/stsb-roberta-base from Meta in order to
        load the weights properly for the off-topic model.

    """

    def __init__(self, base_model: Any):
        """Classification head with base model cross-encoder/stsb-roberta-base."""
        super().__init__()

        # Existing cross-encoder model
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Input: a single sentence
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # Reduce the size of the layer
            nn.ReLU(),
        )
        self.num_labels = 2  # This must be 2 in order to load in the weights properly for the off-topic model
        self.classifier = nn.Linear(hidden_size // 4, self.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """Forward function for the CrossEncoderWithMLP. Outputs logits for postprocessing."""
        outputs = self.base_model(input_ids, attention_mask)  # Encode the pair of sentences in one pass
        pooled_output = outputs.pooler_output
        mlp_output = self.mlp(pooled_output)
        return self.classifier(mlp_output)
