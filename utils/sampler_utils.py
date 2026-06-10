import os
import torch
from tqdm import tqdm
from .log_utils import save_latents, log
from models import Transformer, AbsorbingDiffusion, AutoregressiveTransformer

from text_helper import clevr_rel_to_text


def get_sampler(H, embedding_weight):

    if H.sampler == "absorbing":
        denoise_fn = Transformer(H).cuda()

        if H.dataset == "clevr":
            sampler = AbsorbingDiffusion(
                H, denoise_fn, H.codebook_size + 15, embedding_weight
            )
        else:
            sampler = AbsorbingDiffusion(
                H, denoise_fn, H.codebook_size, embedding_weight
            )

    elif H.sampler == "autoregressive":
        sampler = AutoregressiveTransformer(H, embedding_weight)

    return sampler


@torch.no_grad()
def get_samples(
    H,
    generator,
    sampler,
    training=True,
    cond_dict=None,
    batch=False,
    verbose=False,
    **kwargs,
):
    sample_batch = 25 if not training else H.batch_size
    if H.sampler == "absorbing":
        if H.sample_type == "diffusion":
            if "do_xor" in kwargs and kwargs["do_xor"]:
                print("generating logically")
                latents = sampler.sample_xor(
                    temp=H.temp,
                    sample_steps=H.sample_steps,
                    dataset=H.dataset,
                    cond_list=cond_dict[H.dataset],
                    weight_list=kwargs["weights"],
                    batch=batch,
                )
            else:
                latents = sampler.sample_compositional(
                    temp=H.temp,
                    sample_steps=H.sample_steps,
                    dataset=H.dataset,
                    cond_list=cond_dict[H.dataset],
                    weight_list=kwargs["weights"],
                    batch=batch,
                )

        else:
            latents = sampler.sample_mlm(temp=H.temp, sample_steps=H.sample_steps)

    if H.dataset == "clevr_comp":
        img_latents = latents[:, :-4]
    else:
        img_latents = latents
    # clip to img vocab
    img_latents = torch.clamp(img_latents, 0, H.codebook_size - 1)

    latents_one_hot = latent_ids_to_onehot(img_latents, H.latent_shape, H.codebook_size)
    q = sampler.embed(latents_one_hot)
    images = generator(q.float())

    return images, latents


def latent_ids_to_onehot(latent_ids, latent_shape, codebook_size):
    min_encoding_indices = latent_ids.reshape(-1).unsqueeze(1)
    encodings = torch.zeros(min_encoding_indices.shape[0], codebook_size).to(
        latent_ids.device
    )
    encodings.scatter_(1, min_encoding_indices, 1)
    one_hot = encodings.view(
        latent_ids.shape[0], latent_shape[1], latent_shape[2], codebook_size
    )
    return one_hot.reshape(one_hot.shape[0], -1, codebook_size)


@torch.no_grad()
def generate_latent_ids(H, ae, train_loader, val_loader=None):

    if H.dataset == "clevr":
        train_latent_ids, train_positions, train_ns = generate_latents_from_loader(
            H, ae, train_loader
        )
    elif H.dataset == "clevr_pos":
        train_latent_ids, train_positions = generate_latents_from_loader(
            H, ae, train_loader
        )
    elif H.dataset == "clevr_rel":
        train_latent_ids, train_relations = generate_latents_from_loader(
            H, ae, train_loader
        )
    elif H.dataset == "ffhq":
        train_latent_ids, train_attributes = generate_latents_from_loader(
            H, ae, train_loader
        )
    else:
        train_latent_ids = generate_latents_from_loader(H, ae, train_loader)

    if val_loader is not None:
        if H.dataset == "clevr":
            val_latent_ids, val_positions, val_ns = generate_latents_from_loader(
                H, ae, val_loader
            )
        elif H.dataset == "clevr_pos":
            val_latent_ids, val_positions = generate_latents_from_loader(
                H, ae, val_loader
            )
        elif H.dataset == "clevr_rel":
            val_latent_ids, val_relations = generate_latents_from_loader(
                H, ae, val_loader
            )
        elif H.dataset == "ffhq":
            val_latent_ids, val_attributes = generate_latents_from_loader(
                H, ae, val_loader
            )
        else:
            val_latent_ids = generate_latents_from_loader(H, ae, val_loader)
            val_positions = None
            val_ns = None
    else:
        val_latent_ids = None

    save_latents(H, train_latent_ids, val_latent_ids)

    if H.dataset == "clevr":
        # save train and val positions and ns using torch.save
        train_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        val_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
        torch.save({"positions": train_positions, "ns": train_ns}, train_path)
        if val_loader is not None:
            torch.save({"positions": val_positions, "ns": val_ns}, val_path)
    elif H.dataset == "clevr_pos":
        train_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        val_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
        torch.save({"positions": train_positions}, train_path)
        if val_loader is not None:
            torch.save({"positions": val_positions}, val_path)
    elif H.dataset == "clevr_rel":
        train_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        val_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
        torch.save({"relations": train_relations}, train_path)
        if val_loader is not None:
            torch.save({"relations": val_relations}, val_path)
    elif H.dataset == "ffhq":
        train_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        val_path = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
        torch.save({"attributes": train_attributes}, train_path)
        if val_loader is not None:
            torch.save({"attributes": val_attributes}, val_path)


def generate_latents_from_loader(H, autoencoder, dataloader):
    latent_ids = []
    positions = []
    relations = []
    attributes = []
    ns = []
    if H.dataset == "clevr":
        for x, pos, n in tqdm(dataloader):
            x = x.cuda()
            latents = autoencoder.encoder(x)

            latents = latents.permute(0, 2, 3, 1).contiguous()
            latents_flattened = latents.view(-1, H.emb_dim)

            distances = (
                (latents_flattened**2).sum(dim=1, keepdim=True)
                + (autoencoder.quantize.embedding.weight**2).sum(1)
                - 2
                * torch.matmul(
                    latents_flattened, autoencoder.quantize.embedding.weight.t()
                )
            )

            min_encoding_indices = torch.argmin(distances, dim=1)

            reshaped = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()

            latent_ids.append(reshaped)
            positions.append(pos)
            ns.append(n)
    elif H.dataset == "clevr_pos":
        for x, d in tqdm(dataloader):
            pos = d["y"]
            x = x.cuda()
            latents = autoencoder.encoder(x)

            latents = latents.permute(0, 2, 3, 1).contiguous()
            latents_flattened = latents.view(-1, H.emb_dim)

            distances = (
                (latents_flattened**2).sum(dim=1, keepdim=True)
                + (autoencoder.quantize.embedding.weight**2).sum(1)
                - 2
                * torch.matmul(
                    latents_flattened, autoencoder.quantize.embedding.weight.t()
                )
            )

            min_encoding_indices = torch.argmin(distances, dim=1)

            reshaped = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()

            latent_ids.append(reshaped)
            positions.append(pos)
    elif H.dataset == "clevr_rel":
        for x, d in tqdm(dataloader):
            rel = d["y"]
            x = x.cuda()
            latents = autoencoder.encoder(x)

            latents = latents.permute(0, 2, 3, 1).contiguous()
            latents_flattened = latents.view(-1, H.emb_dim)

            distances = (
                (latents_flattened**2).sum(dim=1, keepdim=True)
                + (autoencoder.quantize.embedding.weight**2).sum(1)
                - 2
                * torch.matmul(
                    latents_flattened, autoencoder.quantize.embedding.weight.t()
                )
            )

            min_encoding_indices = torch.argmin(distances, dim=1)

            reshaped = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()

            latent_ids.append(reshaped)
            relations.append(rel)
    elif H.dataset == "ffhq":
        for x, d in tqdm(dataloader):
            attr = d
            if isinstance(x, list):
                x = x[0]
            x = x.cuda()
            latents = autoencoder.encoder(x)

            latents = latents.permute(0, 2, 3, 1).contiguous()
            latents_flattened = latents.view(-1, H.emb_dim)

            distances = (
                (latents_flattened**2).sum(dim=1, keepdim=True)
                + (autoencoder.quantize.embedding.weight**2).sum(1)
                - 2
                * torch.matmul(
                    latents_flattened, autoencoder.quantize.embedding.weight.t()
                )
            )

            min_encoding_indices = torch.argmin(distances, dim=1)

            reshaped = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()

            latent_ids.append(reshaped)
            attributes.append(attr)
    else:
        for x, _ in tqdm(dataloader):
            x = x.cuda()
            latents = autoencoder.encoder(x)  # B, emb_dim, H, W

            latents = latents.permute(0, 2, 3, 1).contiguous()  # B, H, W, emb_dim
            latents_flattened = latents.view(-1, H.emb_dim)  # B*H*W, emb_dim

            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            distances = (
                (latents_flattened**2).sum(dim=1, keepdim=True)
                + (autoencoder.quantize.embedding.weight**2).sum(1)
                - 2
                * torch.matmul(
                    latents_flattened, autoencoder.quantize.embedding.weight.t()
                )
            )

            min_encoding_indices = torch.argmin(distances, dim=1)

            reshaped = min_encoding_indices.reshape(x.shape[0], -1).cpu().contiguous()
            latent_ids.append(reshaped)

    if H.dataset == "clevr":
        return (
            torch.cat(latent_ids, dim=0),
            torch.cat(positions, dim=0),
            torch.cat(ns, dim=0),
        )
    elif H.dataset == "clevr_pos":
        return torch.cat(latent_ids, dim=0), torch.cat(positions, dim=0)
    elif H.dataset == "clevr_rel":
        return torch.cat(latent_ids, dim=0), torch.cat(relations, dim=0)
    elif H.dataset == "ffhq":
        return torch.cat(latent_ids, dim=0), torch.cat(attributes, dim=0)
    else:
        return torch.cat(latent_ids, dim=0)


@torch.no_grad()
def get_latent_loaders(H, get_validation_loader=True, shuffle=True):
    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""

    train_latents_fp = (
        f"latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}"
    )
    train_latent_ids = torch.load(train_latents_fp)

    if H.dataset == "clevr":
        train_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        train_meta = torch.load(train_meta_fp)
        train_positions = train_meta["positions"]
        train_ns = train_meta["ns"]

        tensor_dataset = torch.utils.data.TensorDataset(
            train_latent_ids, train_positions, train_ns
        )
        train_latent_loader = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
        )
    elif H.dataset == "clevr_pos":
        train_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        train_meta = torch.load(train_meta_fp)
        train_positions = train_meta["positions"]

        tensor_dataset = torch.utils.data.TensorDataset(
            train_latent_ids, train_positions
        )
        train_latent_loader = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
        )
    elif H.dataset == "clevr_rel":
        train_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        train_meta = torch.load(train_meta_fp)
        train_relations = train_meta["relations"]

        tensor_dataset = torch.utils.data.TensorDataset(
            train_latent_ids, train_relations
        )
        train_latent_loader = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
        )
    elif H.dataset == "ffhq":
        train_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_train_meta"
        train_meta = torch.load(train_meta_fp)
        train_attributes = train_meta["attributes"]

        tensor_dataset = torch.utils.data.TensorDataset(
            train_latent_ids, train_attributes
        )
        train_latent_loader = torch.utils.data.DataLoader(
            tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
        )
    else:
        train_latent_loader = torch.utils.data.DataLoader(
            train_latent_ids, batch_size=H.batch_size, shuffle=shuffle
        )

    if get_validation_loader:
        val_latents_fp = (
            f"latents/{H.dataset}_{H.latent_shape[-1]}_val_latents{latents_fp_suffix}"
        )
        val_latent_ids = torch.load(val_latents_fp)
        if H.dataset == "clevr":
            val_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
            val_meta = torch.load(val_meta_fp)
            val_positions = val_meta["positions"]
            val_ns = val_meta["ns"]

            tensor_dataset = torch.utils.data.TensorDataset(
                val_latent_ids, val_positions, val_ns
            )
            val_latent_loader = torch.utils.data.DataLoader(
                tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
            )
        elif H.dataset == "clevr_pos":
            val_meta_fp = f"latents/{H.dataset}_{H.latent_shape[-1]}_val_meta"
            val_meta = torch.load(val_meta_fp)
            val_positions = val_meta["positions"]

            tensor_dataset = torch.utils.data.TensorDataset(
                val_latent_ids, val_positions
            )
            val_latent_loader = torch.utils.data.DataLoader(
                tensor_dataset, batch_size=H.batch_size, shuffle=shuffle
            )
        else:
            val_latent_loader = torch.utils.data.DataLoader(
                val_latent_ids, batch_size=H.batch_size, shuffle=shuffle
            )
    else:
        val_latent_loader = None

    return train_latent_loader, val_latent_loader


# TODO: rethink this whole thing - completely unnecessarily complicated
def retrieve_autoencoder_components_state_dicts(
    H, components_list, remove_component_from_key=False
):
    state_dict = {}
    # default to loading ema models first
    ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_ema_{H.ae_load_step}.th"
    if not os.path.exists(ae_load_path):
        ae_load_path = f"logs/{H.ae_load_dir}/saved_models/vqgan_{H.ae_load_step}.th"
    log(f"Loading VQGAN from {ae_load_path}")
    full_vqgan_state_dict = torch.load(ae_load_path, map_location="cpu")

    for key in full_vqgan_state_dict:
        for component in components_list:
            if component in key:
                new_key = key[3:]  # remove "ae."
                if remove_component_from_key:
                    new_key = new_key[len(component) + 1 :]  # e.g. remove "quantize."

                state_dict[new_key] = full_vqgan_state_dict[key]

    return state_dict
