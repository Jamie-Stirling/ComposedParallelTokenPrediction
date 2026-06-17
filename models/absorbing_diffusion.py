import math
import torch
import torch.nn as nn
import torch.distributions as dists
import torch.nn.functional as F


class ComposedAbsorbingDiffusion(nn.Module):
    def __init__(self, H, transformer, mask_id, embedding_weight):
        """
        Args:
            H: A config data class containing the fields referenced below
            transformer (nn.Module): Encoder-only transformer implementing a forward() which accepts integer tokens of (batch, seq) along with conditioning information,
        and outputs logits of shape (batch, seq, logits)
            embedding_weight (nn.Module): Set of embeddings inherited from the trained VQ model
        """
        super().__init__()

        self.latent_shape = H.latent_shape
        self.emb_dim = H.emb_dim
        self.codebook_size = H.codebook_size
        self.mask_id = mask_id
        self.embedding_weight = embedding_weight
        self.embedding_weight.requires_grad = (
            False  # Freeze embeddings as they are in the VQ codebook
        )
        self.n_samples = H.n_samples

        self.dataset = H.dataset
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self.num_classes = H.codebook_size

        self.mask_id = self.num_classes

        self.transformer = transformer
        self.n_samples = H.batch_size
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        
        self.register_buffer(
            "loss_history", torch.zeros(self.num_timesteps + 1, device="cuda")
        )
        self.register_buffer(
            "Lt_history", torch.zeros(self.num_timesteps + 1, device="cuda")
        )
        self.register_buffer(
            "Lt_count", torch.zeros(self.num_timesteps + 1, device="cuda")
        )

        assert self.mask_schedule in ["random", "fixed"]

    def sample_time(self, b, device, method="uniform"):
        """
        Sample a batch of time indices for training, determining how much "noise" to add.

        We always use "uniform" sampling for our experiments.
        """
        if method == "uniform":
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, t):
        """
        Given a batch of "full" token sequences x_0 and time indices t, sample a batch of randomly masked indices x_t
        """
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (
            t.float().unsqueeze(-1) / self.num_timesteps
        )
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def _train_loss(self, x_0, cond_dict=None):
        """
        Given a batch of "full" token sequences representing images and
        optionally a cond_dict containing dataset-specific conditioning information, compute the absorbing diffusion training loss.
        """
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, "uniform")

        # make x noisy and denoise
        x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = self.transformer(x_t, t=t, cond_dict=cond_dict).permute(
            0, 2, 1
        )

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = F.cross_entropy(
            x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction="none"
        ).sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == "elbo":
            loss = vb_loss
        elif self.loss_type == "mlm":
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == "reweighted_elbo":
            weight = 1 - (t / self.num_timesteps)
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (
            (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        )

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (
            (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        )
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(
            dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype)
        )

        return loss.mean(), vb_loss.mean()

    def train_iter(self, x, cond_dict):
        loss, vb_loss = self._train_loss(x, cond_dict=cond_dict)
        stats = {"loss": loss, "vb_loss": vb_loss}
        return stats

    def sample_compositional(
        self,
        temp=1.0,
        sample_steps=None,
        dataset="clevr_pos",
        cond_list=[],
        weight_list=[],
        batch=False,
        verbose=False,
    ):
        """
        Given a set of conditions and corresponding weights, perform compositional absorbing diffusion sampling.
        """
        b, device = self.n_samples, "cuda"
        x_t = torch.ones((b, self.num_timesteps), device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            if verbose:
                print(f"Sample timestep {t:4d}", end="\r")

            t = torch.full((b,), t, device=device, dtype=torch.long)
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # compute log P(z_0 | z_t)
            z_0_z_t_logits = self.transformer(x_t, t=t, cond_dict={dataset: None})
            z_0_z_t_logprobs = torch.log_softmax(z_0_z_t_logits, dim=-1)

            sum_conditional_logprobs = torch.zeros_like(z_0_z_t_logprobs)

            for cond, weight in zip(cond_list, weight_list):
                # compute log P(z_0 | z_t,c)
                # repeat cond over the batch
                if not batch:
                    cond = torch.tensor(cond, device=device).unsqueeze(0).repeat(b, 1)
                # else it's already in the right shape
                z_0_given_c_logits = self.transformer(
                    x_t, t=t, cond_dict={dataset: cond}
                )
                z_0_given_c_logprobs = torch.log_softmax(z_0_given_c_logits, dim=-1)

                sum_conditional_logprobs += weight * (
                    z_0_given_c_logprobs - z_0_z_t_logprobs
                )
                # print(z_0_z_t_logprobs.square().mean().item(), sum_conditional_logprobs.square().mean().item())

            x_0_logits = z_0_z_t_logprobs + sum_conditional_logprobs
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        return x_t
