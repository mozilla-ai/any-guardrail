from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class Adapter(nn.Module):
    def __init__(self, hidden_size: int):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(hidden_size // 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_project(x)
        activated = self.activation(down)
        up = self.up_project.forward(activated)
        return up + x
    

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(hidden_states, self.attention_weights)
        attention_weights = torch.softmax(scores, dim=0)
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=0)
        return weighted_sum

# Custom bi-encoder model with MLP layers for interaction
class CrossEncoderWithSharedBase(nn.Module, PyTorchModelHubMixin):
    """Defines the classification head for the [govtech/jina-embeddings-v2-small-en-off-topic](https://huggingface.co/govtech/jina-embeddings-v2-small-en-off-topic).
    The classification head sits on top of the jinaai/jina-embeddings-v2-small-en model.

    Args:
        base_model: The base embeddings model. This model must be the jinaai/jina-embeddings-v2-small-en in order to
        load the weights properly for the off-topic model.
    """
    def __init__(self, base_model: Any) -> None:
        super(CrossEncoderWithSharedBase, self).__init__()
        # Shared pre-trained model
        self.shared_encoder = base_model
        hidden_size = self.shared_encoder.config.hidden_size
        # Sentence-specific adapters
        self.adapter1 = Adapter(hidden_size)
        self.adapter2 = Adapter(hidden_size)
        # Cross-attention layers
        self.num_heads = 8 # This must be 8 in order to load in the weights properly for the off-topic model
        self.cross_attention_1_to_2 = nn.MultiheadAttention(hidden_size, self.num_heads)
        self.cross_attention_2_to_1 = nn.MultiheadAttention(hidden_size, self.num_heads)
        # Attention pooling layers
        self.attn_pooling_1_to_2 = AttentionPooling(hidden_size)
        self.attn_pooling_2_to_1 = AttentionPooling(hidden_size)
        # Projection layer with non-linearity
        self.projection_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        # Classifier with three hidden layers
        self.num_labels = 2 # This must be 2 in order to load in the weights properly for the off-topic model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, self.num_labels)
        )

    def forward(self, 
                input_ids1: torch.Tensor, 
                attention_mask1: torch.Tensor, 
                input_ids2: torch.Tensor, 
                attention_mask2: torch.Tensor) -> Any:
        # Encode sentences
        outputs1 = self.shared_encoder(input_ids1, attention_mask=attention_mask1)
        outputs2 = self.shared_encoder(input_ids2, attention_mask=attention_mask2)
        # Apply sentence-specific adapters
        embeds1 = self.adapter1(outputs1.last_hidden_state)
        embeds2 = self.adapter2(outputs2.last_hidden_state)
        # Transpose for attention layers
        embeds1 = embeds1.transpose(0, 1)
        embeds2 = embeds2.transpose(0, 1)
        # Cross-attention
        cross_attn_1_to_2, _ = self.cross_attention_1_to_2(embeds1, embeds2, embeds2)
        cross_attn_2_to_1, _ = self.cross_attention_2_to_1(embeds2, embeds1, embeds1)
        # Attention pooling
        pooled_1_to_2 = self.attn_pooling_1_to_2(cross_attn_1_to_2)
        pooled_2_to_1 = self.attn_pooling_2_to_1(cross_attn_2_to_1)
        # Concatenate and project
        combined = torch.cat((pooled_1_to_2, pooled_2_to_1), dim=1)
        projected = self.projection_layer(combined)
        # Classification
        logits = self.classifier(projected)
        return logits
    