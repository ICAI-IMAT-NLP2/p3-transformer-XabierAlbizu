import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

try:
    from decoder import TransformerDecoder
    from encoder import TransformerEncoder
except ModuleNotFoundError:
    from src.decoder import TransformerDecoder
    from src.encoder import TransformerEncoder


class Transformer(nn.Module):
    """Transformer model: Encoder + Decoder + proyección a vocabulario objetivo."""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 max_enc_position_embeddings: int, max_dec_position_embeddings: int,
                 enc_d_model: int, dec_d_model: int, num_attention_heads: int, 
                 enc_intermediate_size: int, dec_intermediate_size: int, 
                 num_enc_hidden_layers: int, num_dec_hidden_layers: int):
        super(Transformer, self).__init__()

        # Encoder y Decoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            max_position_embeddings=max_enc_position_embeddings,
            d_model=enc_d_model,
            num_attention_heads=num_attention_heads,
            intermediate_size=enc_intermediate_size,
            num_hidden_layers=num_enc_hidden_layers,
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            max_position_embeddings=max_dec_position_embeddings,
            d_model=dec_d_model,
            num_attention_heads=num_attention_heads,
            intermediate_size=dec_intermediate_size,
            num_hidden_layers=num_dec_hidden_layers,
        )

        # Proyección final a vocabulario objetivo
        self.output_linear = nn.Linear(dec_d_model, tgt_vocab_size)

    def forward(self, src_input: torch.Tensor, tgt_input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        src_input: (B, S_src) ids
        tgt_input: (B, S_tgt) ids
        attn_mask: (B, S_src, S_src) padding mask para el encoder (opcional)
        return: logits (B, S_tgt, tgt_vocab_size)
        """
        # 1) Encoder
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)
  # <- quitar attn_mask kwarg

        # 2) Decoder y proyección
        dec_hidden = self.decoder(tgt_input, enc_output)
        dec_output = self.output_linear(dec_hidden)
        return dec_output
    
    # --------- Decoding strategies ---------

    def generate(self, src_input: torch.Tensor, max_length: int = 50, decoding_strategy: str = 'greedy', **kwargs) -> torch.Tensor:
        if decoding_strategy == 'greedy':
            return self.__greedy_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'beam_search':
            return self.__beam_search_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'sampling':
            return self.__sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_k':
            return self.__top_k_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_p':
            return self.__top_p_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'contrastive':
            return self.__contrastive_decode(src_input, max_length, **kwargs)
        else:
            raise ValueError(f"Invalid decoding strategy: {decoding_strategy}")

    def __greedy_decode(self, src_input: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        """
        Greedy: en cada paso toma el argmax del último paso temporal.
        Devuelve ids sin el SOS inicial. Soporta batches.
        """
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        # (B, 1) con SOS
        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_hidden = self.decoder(tgt_input, enc_output)             # (B, T, d_model)
            logits = self.output_linear(dec_hidden)                      # (B, T, vocab)
            next_logits = logits[:, -1, :]                               # (B, vocab)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True) # (B, 1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)        # (B, T+1)

            if (next_token == EOS_token).all():
                break

        # Quita el SOS inicial
        generated_sequence = tgt_input[:, 1:]
        return generated_sequence

    def __beam_search_decode(self, src_input: torch.Tensor, max_length: int, beam_size: int = 3, **kwargs) -> torch.Tensor:
        """
        Beam search simple para batch_size=1.
        Devuelve secuencia sin SOS.
        """
        batch_size = src_input.size(0)
        if batch_size != 1:
            raise NotImplementedError("Beam search decoding currently only supports batch_size=1")
        device = src_input.device

        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)
        beam = [(tgt_input, 0.0)]  # (seq, log_prob)

        for _ in range(max_length):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == EOS_token:
                    candidates.append((seq, score))
                    continue
                dec_hidden = self.decoder(seq, enc_output)           # (1, T, d_model)
                logits = self.output_linear(dec_hidden)[:, -1, :]    # (1, vocab)
                log_probs = F.log_softmax(logits, dim=-1)            # (1, vocab)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)  # (1, k)

                for i in range(beam_size):
                    nt = topk_indices[0, i].view(1, 1)               # (1,1)
                    new_seq = torch.cat([seq, nt], dim=1)            # (1, T+1)
                    new_score = score + topk_log_probs[0, i].item()
                    candidates.append((new_seq, new_score))

            # elige top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]

            if all(seq[0, -1].item() == EOS_token for seq, _ in beam):
                break

        best_seq, _ = max(beam, key=lambda x: x[1])
        generated_sequence = best_seq[:, 1:]  # quitar SOS
        return generated_sequence

    def __sampling_decode(self, src_input: torch.Tensor, max_length: int, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Sampling con temperatura. Devuelve ids sin SOS.
        """
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_hidden = self.decoder(tgt_input, enc_output)
            logits = self.output_linear(dec_hidden)[:, -1, :]  # (B, vocab)
            scaled_logits = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        return tgt_input[:, 1:]

    def __top_k_sampling_decode(self, src_input: torch.Tensor, max_length: int, k: int = 10, **kwargs) -> torch.Tensor:
        """
        Top-k sampling: muestrea solo entre los k más probables del último paso.
        """
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_hidden = self.decoder(tgt_input, enc_output)
            logits = self.output_linear(dec_hidden)[:, -1, :]   # (B, vocab)
            log_probs = F.log_softmax(logits, dim=-1)           # (B, vocab)

            topk_log_probs, topk_indices = torch.topk(log_probs, k, dim=-1)  # (B, k)
            probs = F.softmax(topk_log_probs, dim=-1)                          # (B, k)
            sampled_idx_in_topk = torch.multinomial(probs, num_samples=1)      # (B,1)
            next_token = topk_indices.gather(1, sampled_idx_in_topk)           # (B,1)

            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        return tgt_input[:, 1:]
    
    def __top_p_sampling_decode(self, src_input: torch.Tensor, max_length: int, p: float = 0.9, **kwargs) -> torch.Tensor:
        """
        Nucleus (top-p) sampling.
        """
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_hidden = self.decoder(tgt_input, enc_output)
            logits = self.output_linear(dec_hidden)[:, -1, :]  # (B, vocab)
            probs = F.softmax(logits, dim=-1)                  # (B, vocab)

            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)  # (B, vocab)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # máscara para cortar por p
            cutoff = (cumulative_probs > p)
            # asegura mantener al menos el primer token
            cutoff[..., 0] = False
            filtered_probs = sorted_probs.masked_fill(cutoff, 0.0)
            # renormaliza
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled_idx_in_sorted = torch.multinomial(filtered_probs, num_samples=1)  # (B,1)
            next_token = sorted_indices.gather(1, sampled_idx_in_sorted)              # (B,1)

            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        return tgt_input[:, 1:]
    
    def __contrastive_decode(self, src_input: torch.Tensor, max_length: int, k: int = 5, alpha: float = 0.6, **kwargs) -> torch.Tensor:
        """
        Contrastive search (versión simplificada y vectorizada por batch).
        """
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = enc_output = self.encoder(src_input, None if attn_mask is None else attn_mask)


        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            # logits actuales
            dec_hidden = self.decoder(tgt_input, enc_output)
            logits = self.output_linear(dec_hidden)[:, -1, :]                   # (B, vocab)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)             # (B, k)

            # candidatos
            expanded_tgt_input = tgt_input.unsqueeze(1).expand(-1, k, -1)       # (B, k, T)
            next_tokens = topk_indices.unsqueeze(-1)                             # (B, k, 1)
            y_candidates = torch.cat([expanded_tgt_input, next_tokens], dim=-1) # (B, k, T+1)

            # pasa candidatos por el decoder
            B, K, T1 = y_candidates.shape
            flat_cands = y_candidates.reshape(B*K, T1)                           # (B*K, T+1)
            dec_outputs_cand = self.decoder(flat_cands, enc_output.repeat_interleave(K, dim=0))
            h_j = dec_outputs_cand[:, :-1, :]    # (B*K, T, d)
            h_v = dec_outputs_cand[:, -1, :]     # (B*K, d)

            # normaliza y calcula penalización
            h_v_norm = F.normalize(h_v, dim=-1)                    # (B*K, d)
            h_j_norm = F.normalize(h_j, dim=-1)                    # (B*K, T, d)
            cos_sim = torch.einsum('bd,btd->bt', h_v_norm, h_j_norm)  # (B*K, T)
            max_sim = cos_sim.max(dim=-1).values                   # (B*K,)

            P_LM_v = topk_probs.reshape(B*K)                       # (B*K,)
            scores = (1 - alpha) * P_LM_v - alpha * max_sim        # (B*K,)

            # elige mejor por cada batch
            scores = scores.reshape(B, K)
            best_idx = scores.argmax(dim=-1)                       # (B,)
            next_token = topk_indices.gather(1, best_idx.unsqueeze(-1))  # (B,1)

            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        return tgt_input[:, 1:]
