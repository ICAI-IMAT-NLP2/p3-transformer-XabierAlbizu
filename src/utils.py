
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    """Single Attention Head with Masking Support.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value, with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.

    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(d_model,d_q)
        self.wk = nn.Linear(d_model,d_k)
        self.wv = nn.Linear(d_model,d_v)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights with optional causal mask.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """

        # The dimension of the key tensor, used to scale the scores.
        dim_k = k.shape[-1]

        # Calculate the dot product between query and the transpose of key.
        # The result is then scaled by the square root of dim_k.
        scores = torch.bmm(q,k.transpose(1,2)) *(1/math.sqrt(dim_k))

        if mask is not None:
            mask_new = torch.zeros_like(mask)
            mask_new[mask < 1] = -1e6
            mask_new[mask == 1] = 0
            scores = scores + mask_new

        # Apply the softmax function to obtain the attention weights.
        weights = torch.softmax(scores, dim=-1)

        # Compute the output by performing a weighted sum of the value tensor
        # using the attention weights.
        output = torch.bmm(weights, v)
        return output, weights

    def forward(self, x_q, x_k, x_v, mask=None):
        """Forward pass for the attention head with optional causal mask.

        Args:
            x_q (Tensor): Input query tensor of shape (batch_size, seq_len, d_model).
            x_k (Tensor): Input key tensor of shape (batch_size, seq_len, d_model).
            x_v (Tensor): Input value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # Project input tensor to query, key, and value tensors.
        q = self.wq(x_q)
        k = self.wk(x_k)
        v = self.wv(x_v)

        output, _ = self.scaled_dot_product_attention(q,k,v,mask)
        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with Masking Support.

    This class implements the multi-head attention mechanism, allowing
    the model to focus on different parts of the input sequence at each layer,
    with support for an optional causal mask.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        d_kqv = (d_model/num_attention_heads)
        if int(d_kqv) % d_kqv != 0:
            raise RuntimeError (f"d_model ({d_model}) must be divisible by num_attention_heads ({num_attention_heads})")
        d_kqv = int(d_kqv)
        self.heads = nn.ModuleList(AttentionHead(d_model=d_model,d_k=d_kqv,d_q=d_kqv,d_v=d_kqv) for i in range(num_attention_heads))
        self.output_linear = nn.Linear(in_features=num_attention_heads*d_kqv, out_features=d_model)


    def forward(self, x_q, x_k, x_v, mask=None):
        """Forward pass for the multi-head attention layer with optional causal mask.

        Args:
            x_q (Tensor): Input query tensor of shape (batch_size, seq_len, d_model).
            x_k (Tensor): Input key tensor of shape (batch_size, seq_len, d_model).
            x_v (Tensor): Input value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Causal mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Concatenate the outputs from all attention heads.
        head_outputs = [head(x_q, x_k, x_v, mask) for head in self.heads]
        x = torch.cat(head_outputs, dim=-1)

        x = self.output_linear(x)
        return x
    
class FeedForward(nn.Module):
    """FeedForward module for the Transformer Decoder.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): First linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): Second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x
    
class Embeddings(nn.Module):
    """Embeddings module for the Transformer Decoder.

    This module combines token embeddings and positional embeddings and applies
    layer normalization.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token embeddings.
        position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
        layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)

        return embeddings

