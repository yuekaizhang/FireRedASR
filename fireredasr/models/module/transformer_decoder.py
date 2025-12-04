from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Assuming 'whisper' is in the python path
from whisper.model import (
    LayerNorm,
    MultiHeadAttention,
    Linear,
    sinusoids,
)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head)
        self.attn_ln = LayerNorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn_ln = LayerNorm(d_model)


        n_mlp = d_model * 4
        self.mlp = nn.Sequential(
            Linear(d_model, n_mlp),
            nn.GELU(),
            Linear(n_mlp, d_model),
        )
        self.mlp_ln = LayerNorm(d_model)

    def forward(self, x, xa, mask, cross_attn_mask, cache=None):
        residual = x
        x_norm = self.attn_ln(x)
        if cache is not None:
            xq = x_norm[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = mask[:, -1:, :]
        else:
            xq = x_norm

        # Whisper's MultiHeadAttention returns a tuple (output, attn_weights)
        # We only need the output, which is the first element.
        # Correctly calling with query=xq, key/value=x_norm.
        x_attn, _ = self.attn(xq, xa=x_norm, mask=mask)
        x = residual + x_attn

        residual = x
        x_norm = self.cross_attn_ln(x)
        # Correctly calling with query=x_norm, key/value=xa (from encoder).
        x_cross_attn, _ = self.cross_attn(x_norm, xa=xa, mask=cross_attn_mask)
        x = residual + x_cross_attn

        residual = x
        x = residual + self.mlp(self.mlp_ln(x))

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x

class TransformerDecoder(nn.Module):
    def __init__(
            self, sos_id, eos_id, pad_id, odim,
            n_layers, n_head, d_model,
            pe_maxlen=5000):
        super().__init__()
        self.INF = 1e10
        # parameters
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        # Components
        self.token_embedding = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_embedding = nn.Parameter(torch.empty(pe_maxlen, d_model))

        self.blocks = nn.ModuleList()
        for l in range(n_layers):
            block = ResidualAttentionBlock(d_model, n_head)
            self.blocks.append(block)

        self.output_projection = Linear(d_model, odim, bias=False)
        self.ln = LayerNorm(d_model)

        self.output_projection.weight = self.token_embedding.weight
        self.scale = (d_model ** 0.5)
        print(self.scale, 2333333)
        # self.scale = 1

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0 and eos_penalty <= 1.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        caches: List[Optional[Tensor]] = []
        for _ in range(self.n_layers):
            caches.append(None)
        scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
        scores = scores.repeat(N).view(N*B, 1)
        is_finished = torch.zeros_like(scores)

        # Autoregressive Prediction
        for t in range(maxlen):
            tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

            dec_output = (
                self.token_embedding(ys) * self.scale +
                self.positional_embedding[:ys.shape[1]])

            i = 0
            for block in self.blocks:
                dec_output = block.forward(
                    dec_output, encoder_outputs,
                    tgt_mask, src_mask,
                    cache=caches[i])
                caches[i] = dec_output
                i += 1

            dec_output = self.ln(dec_output)

            t_logit = self.output_projection(dec_output[:, -1])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # Accumulated
            scores = scores + t_topB_scores

            # Pruning
            scores = scores.view(N, B*B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)

            topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
            stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

            # Update ys
            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            # Update caches
            new_caches: List[Optional[Tensor]] = []
            for cache in caches:
                if cache is not None:
                    new_caches.append(cache[topB_row_number_in_ys])
            caches = new_caches

            # Update finished state
            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == N*B:
                break

        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]]
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        mask = torch.ne(padded_targets, ignore_id)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T).unsqueeze(0).to(mask.dtype)
        upper_tri_0_mask = upper_tri_0_mask.to(mask.dtype).to(mask.device)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size):
        ones = torch.ones(size, size)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished):
        NB, B = scores.size()
        is_finished = is_finished.float()
        mask_score = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(scores.device)
        mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()
