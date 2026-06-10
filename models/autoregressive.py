import torch
import torch.nn.functional as F
from .sampler import Sampler
from .transformer import Transformer
import numpy as np
import math


class AutoregressiveTransformer(Sampler):
    def __init__(self, H, embedding_weight):
        super().__init__(H, embedding_weight)
        self.net = Transformer(H)
        self.n_samples = H.batch_size
        self.seq_len = np.prod(H.latent_shape)

    def train_iter(self, x, cond_dict):
        x_in = x[:, :-1]  # x is already flattened
        logits = self.net(x_in, cond_dict=cond_dict)
        loss = F.cross_entropy(logits.permute(0, 2, 1), x, reduction="none")
        loss = loss.sum(1).mean() / (math.log(2) * x.shape[1:].numel())

        stats = {"loss": loss}
        return stats

    def sample(self, temp=1.0):
        b, device = self.n_samples, "cuda"
        x = torch.zeros(b, 0).long().to(device)
        for _ in range(self.seq_len):
            logits = self.net(x)[:, -1]
            probs = F.softmax(logits / temp, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        return x

    def sample_compositional(
        self,
        temp=1.0,
        dataset="clevr_pos",
        cond_list=[],
        weight_list=[],
        batch=False,
        verbose=False,
    ):
        b, device = self.n_samples, "cuda"
        x = torch.zeros(b, 0).long().to(device)
        for _ in range(self.seq_len):
            if verbose:
                print(f"Autoregressive sample step {i+1}/{self.seq_len}", end="\r")
            logits_uncond = self.net(x, cond_dict={dataset: None})[:, -1]
            logprobs_uncond = torch.log_softmax(logits_uncond, dim=-1)
            logits_cond_diff = 0

            for i in range(len(cond_list)):

                cond = cond_list[i]
                if not batch:
                    cond = torch.tensor(cond, device=device).unsqueeze(0).repeat(b, 1)
                weight = weight_list[i]
                logits_cond = self.net(x, cond_dict={dataset: cond})[:, -1]
                logprobs_cond = torch.log_softmax(logits_cond, dim=-1)
                logits_cond_diff += weight * (logprobs_cond - logprobs_uncond)

            logits = logits_uncond + logits_cond_diff

            probs = F.softmax(logits / temp, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
        return x
