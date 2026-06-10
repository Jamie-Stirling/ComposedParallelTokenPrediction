import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, H):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head
        self.causal = True if H.sampler == "autoregressive" else False
        if self.causal:
            block_size = np.prod(H.latent_shape)
            if H.dataset == "clevr_pos":
                block_size = (
                    256 + 9
                )  # add 9 for the "position token" which is tiled 9 times to match the relation dataset
            if H.dataset == "clevr_rel":
                block_size = 256 + 9  # add 9 for the relation tokens
            if H.dataset == "ffhq":
                block_size = (
                    256 + 9
                )  # add 9 for the attribute token which is tiled 9 times to match the relation dataset
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal and layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        self.attn = CausalSelfAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


class Transformer(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, H):
        super().__init__()

        if H.dataset == "clevr":
            self.vocab_size = H.codebook_size + 1 + 15
        else:
            self.vocab_size = H.codebook_size + 1

        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == "autoregressive"
        if self.causal:
            self.vocab_size = H.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.rel_tok_emb = nn.Embedding(3 + 2 + 8 + 2 + 7, self.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.rel_pos_emb = nn.Parameter(torch.zeros(1, 9, self.n_embd))

        self.clevr_position_proj = torch.nn.Linear(2, self.n_embd)

        self.ffhq_attr_emb = nn.Embedding(6, self.n_embd)

        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size - 1, bias=False)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, t=None, cond_dict=None):
        # each index maps to a (learnable) vector
        token_embeddings = self.tok_emb(idx)

        if self.causal:
            token_embeddings = torch.cat(
                (
                    self.start_tok.repeat(token_embeddings.size(0), 1, 1),
                    token_embeddings,
                ),
                dim=1,
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings

        x = self.drop(x)

        cond_embeddings = torch.zeros(x.shape[0], 0, x.shape[2]).to(x.device)

        # here we can iteratively build up multiple condition embeddings, however in practice we only use one for each dataset
        if "clevr_pos" in cond_dict:
            if cond_dict["clevr_pos"] is None:
                pos = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)

            else:
                pos = cond_dict["clevr_pos"].float()
                pos = self.clevr_position_proj(pos).unsqueeze(1)
            cond_embeddings = torch.cat((cond_embeddings, pos), dim=1)
            if self.causal:
                # tile cond_embeddings in dim 1 9 times to match the relation dataset
                cond_embeddings = cond_embeddings.repeat(1, 9, 1)
        elif "clevr_rel" in cond_dict:
            if cond_dict["clevr_rel"] is None:
                rel = self.rel_pos_emb.repeat(x.shape[0], 1, 1)
            else:
                rel = cond_dict["clevr_rel"]  # currently [b, 11] and long
                # cut out tokens at index 4 and 9 (they encode position)
                rel = torch.cat((rel[:, :4], rel[:, 5:9], rel[:, 10:]), dim=1)
                # add cumulative offsets since they currently start at 0 [0, 2, 10, 12]
                ## shape, size, color, material, pos = label[i * 5:i * 5 + 5]
                cumulative_offsets = torch.tensor([0, 3, 5, 13], device=rel.device)
                final_offset = 15  # 15 is the total number of attributes describing individual objects, so the relation embedding needs to start at 15
                rel[:, :4] += cumulative_offsets.unsqueeze(0)
                rel[:, 4:8] += cumulative_offsets.unsqueeze(0)

                rel[:, 8] += final_offset

                rel = self.rel_tok_emb(rel)
                rel += self.rel_pos_emb
            cond_embeddings = torch.cat((cond_embeddings, rel), dim=1)
        elif "ffhq" in cond_dict:
            if cond_dict["ffhq"] is None:
                attr = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
            else:
                attr = cond_dict["ffhq"].squeeze(1)

                # find -1s in the attribute tensor and remember their indices

                indices_missing = attr == -1
                indices_not_missing = attr != -1

                # split
                attr_missing = attr[indices_missing]
                attr_not_missing = attr[indices_not_missing]

                attr_not_missing = self.ffhq_attr_emb(attr_not_missing)
                attr_missing = torch.zeros(
                    attr_missing.shape[0], self.n_embd, device=x.device
                )

                # put back together in the right order
                attr = torch.zeros(attr.shape[0], self.n_embd, device=x.device)
                attr[indices_missing] = attr_missing
                attr[indices_not_missing] = attr_not_missing
                attr = attr.unsqueeze(1)

            cond_embeddings = torch.cat((cond_embeddings, attr), dim=1)

            if self.causal:
                # tile cond_embeddings in dim 1 9 times to match the relation dataset
                cond_embeddings = cond_embeddings.repeat(1, 9, 1)
        if self.training:
            cond_dropout = 0.1
            keep_mask = (
                torch.rand(
                    cond_embeddings.shape[0], 1, 1, device=cond_embeddings.device
                )
                > cond_dropout
            )
            keep_mask = keep_mask.float()
            cond_embeddings = cond_embeddings * keep_mask

        x = torch.cat((cond_embeddings, x), dim=1)

        n_cond_embeddings = cond_embeddings.shape[1]

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # remove condition embeddings, we don't want to output them
        x = x[:, n_cond_embeddings:]

        logits = self.head(x)

        return logits

    # TODO: allow cross-attention for conditional info
