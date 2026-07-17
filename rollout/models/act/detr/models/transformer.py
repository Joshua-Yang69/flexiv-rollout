# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor



class Transformer(nn.Module):

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 vitac_cross_attn_layers=None,
                 vitacdreamer_fusion_mode=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
            cross_attn_layers=_resolve_cross_attn_layers(vitac_cross_attn_layers, num_encoder_layers),
        )

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers,
                                          decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self.vitacdreamer_fusion_mode = vitacdreamer_fusion_mode

        # "feature_query_policy_kv" fusion: vitacdreamer feature acts as Q,
        # policy encoder memory acts as K/V.  These modules are only created
        # when this mode is active so that checkpoints trained without it are
        # not affected.
        if vitacdreamer_fusion_mode == "feature_query_policy_kv":
            # Learnable positional embedding for the single vitacdreamer token
            self.vitacdreamer_feature_pos_embed = nn.Parameter(torch.zeros(1, 1, d_model))
            # Project encoder memory to keys and values (LayerNorm + Linear)
            self.vitacdreamer_policy_key_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            )
            self.vitacdreamer_policy_value_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
            )
            # Cross-attention: vitacdreamer feature token attends to policy memory
            self.vitacdreamer_feature_cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout
            )
            self.vitacdreamer_feature_norm1 = nn.LayerNorm(d_model)
            # Post-attention FFN
            self.vitacdreamer_feature_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
            self.vitacdreamer_feature_norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src,
                mask,
                query_embed,
                pos_embed,
                latent_input=None,
                proprio_input=None,
                additional_pos_embed=None,
                vitac_memory=None,
                vitac_gate=None):
        # TODO flatten only when input has H and W
        # if len(src.shape) == 4:  # has H and W
        #     # flatten NxCxHxW to HWxNxC
        #     bs, c, h, w = src.shape
        #     src = src.flatten(2).permute(2, 0, 1)
        #     pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        #     query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        #     # mask = mask.flatten(1)

        #     additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
        #     pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        #     addition_input = torch.stack([latent_input, proprio_input], axis=0)
        #     src = torch.cat([addition_input, src], axis=0)
        # else:
        #     assert len(src.shape) == 3
        #     # flatten NxHWxC to HWxNxC
        #     bs, hw, c = src.shape
        #     src = src.permute(1, 0, 2)
        #     pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
        #     query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        bs, c, hw = src.shape # [N, D, HW]
        src = src.permute(2, 0, 1) # [N, D, HW] -> [HW, N, D]
        pos_embed = pos_embed.permute(2, 0, 1).repeat(1, bs, 1) # [N, D, HW] -> [HW, N, D]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # seq, bs, dim
        pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        addition_input = torch.stack([latent_input, proprio_input], axis=0)
        src = torch.cat([addition_input, src], axis=0)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            pos=pos_embed,
            cross_memory=vitac_memory,
            cross_attn_gate=vitac_gate,
        )

        # feature_query_policy_kv: vitacdreamer feature (Q) cross-attends to
        # the policy encoder output (K, V), then the attended token is appended
        # to encoder memory so the decoder can also attend to it.
        if self.vitacdreamer_fusion_mode == "feature_query_policy_kv" and vitac_memory is not None:
            # vitac_memory: (1, B, d_model) – single projected vitacdreamer token
            vitac_q = vitac_memory + self.vitacdreamer_feature_pos_embed  # (1, B, D)

            # Project encoder memory to K and V (operates on last dim, seq-first ok)
            policy_k = self.vitacdreamer_policy_key_proj(memory)   # (L, B, D)
            policy_v = self.vitacdreamer_policy_value_proj(memory)  # (L, B, D)

            # Cross-attention
            vitac_attn_out, _ = self.vitacdreamer_feature_cross_attn(vitac_q, policy_k, policy_v)
            # Apply gate (same convention as per-layer cross-attn in encoder layers)
            if vitac_gate is not None:
                vitac_attn_out = vitac_gate * vitac_attn_out
            vitac_out = self.vitacdreamer_feature_norm1(vitac_q + vitac_attn_out)  # (1, B, D)

            # FFN + residual
            vitac_out = self.vitacdreamer_feature_norm2(
                vitac_out + self.vitacdreamer_feature_ffn(vitac_out)
            )  # (1, B, D)

            # Append vitacdreamer token to memory so the decoder can attend to it.
            # mask is None in practice (detr_vae passes None), so no mask adjustment
            # needed. Extend pos_embed with zeros for the new token slot.
            vitac_pos = pos_embed.new_zeros(1, bs, self.d_model)
            memory = torch.cat([memory, vitac_out], dim=0)
            pos_embed = torch.cat([pos_embed, vitac_pos], dim=0)

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return hs


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, cross_attn_layers=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.cross_attn_layers = set(cross_attn_layers or [])

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                cross_memory: Optional[Tensor] = None,
                cross_memory_key_padding_mask: Optional[Tensor] = None,
                cross_pos: Optional[Tensor] = None,
                cross_attn_gate: Optional[Tensor] = None):
        output = src

        for layer_idx, layer in enumerate(self.layers):
            layer_cross_memory = cross_memory if layer_idx in self.cross_attn_layers else None
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                cross_memory=layer_cross_memory,
                cross_memory_key_padding_mask=cross_memory_key_padding_mask,
                cross_pos=cross_pos,
                cross_attn_gate=cross_attn_gate,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     cross_memory: Optional[Tensor] = None,
                     cross_memory_key_padding_mask: Optional[Tensor] = None,
                     cross_pos: Optional[Tensor] = None,
                     cross_attn_gate: Optional[Tensor] = None):
        q = self.with_pos_embed(src, pos)
        if cross_memory is None:
            k = q
            value = src
            attn_mask = src_mask
            key_padding_mask = src_key_padding_mask
        else:
            k = self.with_pos_embed(cross_memory, cross_pos)
            value = cross_memory
            attn_mask = None
            key_padding_mask = cross_memory_key_padding_mask
        src2 = self.self_attn(q, k, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if cross_memory is not None and cross_attn_gate is not None:
            src2 = cross_attn_gate * src2
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    cross_memory: Optional[Tensor] = None,
                    cross_memory_key_padding_mask: Optional[Tensor] = None,
                    cross_pos: Optional[Tensor] = None,
                    cross_attn_gate: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = self.with_pos_embed(src2, pos)
        if cross_memory is None:
            k = q
            value = src2
            attn_mask = src_mask
            key_padding_mask = src_key_padding_mask
        else:
            k = self.with_pos_embed(cross_memory, cross_pos)
            value = cross_memory
            attn_mask = None
            key_padding_mask = cross_memory_key_padding_mask
        src2 = self.self_attn(q, k, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if cross_memory is not None and cross_attn_gate is not None:
            src2 = cross_attn_gate * src2
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                cross_memory: Optional[Tensor] = None,
                cross_memory_key_padding_mask: Optional[Tensor] = None,
                cross_pos: Optional[Tensor] = None,
                cross_attn_gate: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(
                src,
                src_mask,
                src_key_padding_mask,
                pos,
                cross_memory,
                cross_memory_key_padding_mask,
                cross_pos,
                cross_attn_gate,
            )
        return self.forward_post(
            src,
            src_mask,
            src_key_padding_mask,
            pos,
            cross_memory,
            cross_memory_key_padding_mask,
            cross_pos,
            cross_attn_gate,
        )


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                                    pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos,
                                 query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _resolve_cross_attn_layers(layer_spec, num_layers):
    if layer_spec is None:
        return []
    if isinstance(layer_spec, (list, tuple, set)):
        return [int(layer_idx) for layer_idx in layer_spec if 0 <= int(layer_idx) < num_layers]

    spec = str(layer_spec).strip().lower()
    if spec in ("", "none", "false", "0"):
        return []
    if spec in ("middle", "mid", "middle2"):
        if num_layers <= 1:
            return [0]
        start = max(num_layers // 2 - 1, 0)
        end = min(start + 2, num_layers)
        return list(range(start, end))

    layers = []
    for raw_idx in spec.split(","):
        raw_idx = raw_idx.strip()
        if not raw_idx:
            continue
        layer_idx = int(raw_idx)
        if layer_idx < 0:
            layer_idx += num_layers
        if 0 <= layer_idx < num_layers:
            layers.append(layer_idx)
    return layers


def build_transformer(args):
    use_vitac = getattr(args, "use_vitacdreamer_feature", False)
    fusion_mode = getattr(args, "vitacdreamer_fusion_mode", None)
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        vitac_cross_attn_layers=getattr(args, "vitacdreamer_cross_attn_layers", "middle")
        if use_vitac and fusion_mode != "feature_query_policy_kv"
        else None,
        vitacdreamer_fusion_mode=fusion_mode if use_vitac else None,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
