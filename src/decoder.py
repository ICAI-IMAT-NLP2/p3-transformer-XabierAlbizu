import torch
import torch.nn as nn

try:
    from utils import MultiHeadAttention, FeedForward, Embeddings
except ModuleNotFoundError:
    from src.utils import MultiHeadAttention, FeedForward, Embeddings

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    This class implements a single layer of the Transformer decoder, consisting
    of a masked multi-head self-attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization before self-attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization before feed-forward network.
        self_attention (MultiHeadAttention): Masked multi-head self-attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_attention_heads)
        self.cross_attention = MultiHeadAttention(d_model,num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            enc_output (torch.Tensor): Output tensor from the Transformer encoder of shape (batch_size, seq_len, d_model).
            tgt_mask (torch.Tensor): Causal mask tensor for target tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply layer normalization and masked multi-head self-attention
        hidden_state = self.layer_norm_1(x)
        x = x + self.self_attention(hidden_state, hidden_state, hidden_state, tgt_mask)

        # Apply layer normalization and cross-attention
        hidden_state = self.layer_norm_2(x)
        x = x + self.cross_attention(hidden_state, enc_output, enc_output, mask=None)
        
        # Apply layer normalization and feed-forward network
        hidden_state = self.layer_norm_3(x)
        x = x + self.feed_forward(hidden_state)

        return x

class TransformerDecoder(nn.Module):
    """Transformer Decoder.

    This class implements the decoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer decoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer decoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer decoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerDecoder, self).__init__()
        # Capa de embeddings (tokens + posicionales)
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)

        # Pila de capas del decoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, input_ids: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer decoder.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            enc_output (torch.Tensor): Output tensor from the Transformer encoder of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Generate token embeddings
        x = self.embeddings(input_ids)
        batch_size, seq_len, _ = x.size()

        # Generate causal mask for target tensor
        tgt_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask)

        return x

