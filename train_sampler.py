from numpy.core.fromnumeric import mean
import torch
import numpy as np
import copy
import time
import os
from tqdm import tqdm
from models import VQAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders, cycle
from utils.sampler_utils import (
    generate_latent_ids,
    get_latent_loaders,
    retrieve_autoencoder_components_state_dicts,
    get_samples,
    get_sampler,
)
from utils.train_utils import EMA, optim_warmup
from utils.log_utils import (
    log,
    log_stats,
    set_up_visdom,
    config_log,
    start_training_log,
    save_stats,
    load_stats,
    save_model,
    load_model,
    save_images,
    display_images,
)

# torch.backends.cudnn.benchmark = True


def attribs_to_tokens(x, num_latents, codebook_size):
    x = x.clone().detach()
    x[:, num_latents] += codebook_size  # smile
    x[:, num_latents + 1] += codebook_size + 2  # gender
    x[:, num_latents + 2] += codebook_size + 4  # glasses
    return x


def tokens_to_attribs(x, num_latents, codebook_size):
    x = x.clone().detach()
    x[:, num_latents] -= codebook_size  # smile
    x[:, num_latents + 1] -= codebook_size + 2
    x[:, num_latents + 2] -= codebook_size + 4
    # clip each attrib to its range
    x[:, num_latents] = x[:, num_latents].clamp(0, 1)
    x[:, num_latents + 1] = x[:, num_latents + 1].clamp(0, 1)
    x[:, num_latents + 2] = x[:, num_latents + 2].clamp(0, 3)
    return x


def attribs_to_str(x, num_latents, codebook_size):
    smile = x[num_latents].item()
    gender = x[num_latents + 1].item()
    glasses = x[num_latents + 2].item()

    """
    inverse of    smile_int = 1 if smile else 0
        gender_int = {"female":0,"male":1}[gender]
        glasses_int = {"NoGlasses":0,"ReadingGlasses":1,"Sunglasses":2,"SwimmingGoggles":3}[glasses]

    """
    smile = "smile" if smile else "no smile"
    gender = "male" if gender else "female"
    glasses = ["no glasses", "reading glasses", "sunglasses", "swimming goggles"][
        glasses
    ]

    return f"{smile}, {gender}, {glasses}"


def clevr_tokens_to_str(x, num_latents, codebook_size):
    x = x.clone().detach()
    color_offset = 1024
    color_range = 8
    material_offset = color_offset + color_range
    material_range = 2
    shape_offset = material_offset + material_range
    shape_range = 3
    size_offset = shape_offset + shape_range

    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    materials = ["rubber", "metal"]
    shapes = ["sphere", "cube", "cylinder"]
    sizes = ["large", "small"]

    color_int = x[num_latents].item() - color_offset
    material_int = x[num_latents + 1].item() - material_offset
    shape_int = x[num_latents + 2].item() - shape_offset
    size_int = x[num_latents + 3].item() - size_offset

    # clip
    color_int = min(max(color_int, 0), len(colors) - 1)
    material_int = min(max(material_int, 0), len(materials) - 1)
    shape_int = min(max(shape_int, 0), len(shapes) - 1)
    size_int = min(max(size_int, 0), len(sizes) - 1)

    color = colors[color_int]
    material = materials[material_int]
    shape = shapes[shape_int]
    size = sizes[size_int]

    return f"{color}, {material}, {shape}, {size}"


def main(H, vis):

    latents_fp_suffix = "_flipped" if H.horizontal_flip else ""
    latents_filepath = (
        f"latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}"
    )

    train_with_validation_dataset = False
    if H.steps_per_eval:
        train_with_validation_dataset = True

    if not os.path.exists(latents_filepath):
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H, ["encoder", "quantize", "generator"]
        )
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict, strict=False)
        # val_loader will be assigned to None if not training with validation dataest
        train_loader, val_loader = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            drop_last=False,
            shuffle=False,
            get_flipped=False,
            extra_augs=False,
            get_val_dataloader=train_with_validation_dataset,
        )

        log("Transferring autoencoder to GPU to generate latents...")
        ae = ae.cuda()  # put ae on GPU for generating
        generate_latent_ids(H, ae, train_loader, val_loader)
        log("Deleting autoencoder to conserve GPU memory...")
        ae = ae.cpu()
        ae = None

    train_latent_loader, val_latent_loader = get_latent_loaders(
        H, get_validation_loader=train_with_validation_dataset
    )

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H, ["quantize", "generator"], remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop("embedding.weight")
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    sampler = get_sampler(H, embedding_weight).cuda()

    log(str(sampler))

    optim = torch.optim.Adam(sampler.parameters(), lr=H.lr)

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    # initialise before loading so as not to overwrite loaded stats
    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0
    if H.load_step > 0:
        start_step = H.load_step + 1

        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f"{H.sampler}_ema", H.load_step, H.load_dir
                )
            except Exception:
                ema_sampler = copy.deepcopy(sampler)
        if H.load_optim:
            optim = load_model(optim, f"{H.sampler}_optim", H.load_step, H.load_dir)
            # only used when changing learning rates and reloading from checkpoint
            for param_group in optim.param_groups:
                param_group["lr"] = H.lr

        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = (train_stats["losses"],)
            mean_losses = (train_stats["mean_losses"],)
            val_losses = (train_stats["val_losses"],)
            val_elbos = train_stats["val_elbos"]
            elbo = (train_stats["elbo"],)
            H.steps_per_log = train_stats["steps_per_log"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            # val_elbos = val_elbos[0]
            elbo = elbo[0]

            # initialise plots
            vis.line(
                mean_losses,
                list(range(log_start_step, start_step, H.steps_per_log)),
                win="loss",
                opts=dict(title="Loss"),
            )
            """vis.line(
                elbo,
                list(range(log_start_step, start_step, H.steps_per_log)),
                win='ELBO',
                opts=dict(title='ELBO')
            )
            vis.line(
                val_losses,
                list(range(H.steps_per_eval, start_step, H.steps_per_eval)),
                win='Val_loss',
                opts=dict(title='Validation Loss')
            )"""
        else:
            log(
                "No stats file found for loaded model, displaying stats from load step only."
            )
            log_start_step = start_step

    scaler = torch.cuda.amp.GradScaler()
    train_iterator = cycle(train_latent_loader)
    # val_iterator = cycle(val_latent_loader)

    log(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")

    for step in range(start_step, H.train_steps):
        step_start_time = time.time()
        # lr warmup
        if H.warmup_iters:
            if step <= H.warmup_iters:
                optim_warmup(H, step, optim)

        x = next(train_iterator)

        if H.dataset == "clevr":
            x, pos, n = x
            pos = pos.cuda()
            n = n.cuda()

        if H.dataset == "clevr_pos":
            x, pos = x
            pos = pos.cuda()

        if H.dataset == "clevr_rel":
            x, rel = x
            rel = rel.cuda()

        if H.dataset == "ffhq":
            x, a = x
            a = a.cuda()

        x = x.cuda()

        if H.amp:
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                stats = sampler.train_iter(x, pos, n)
            scaler.scale(stats["loss"]).backward()
            scaler.step(optim)
            scaler.update()
        else:
            cond_dict = None
            if H.dataset == "clevr_pos":
                cond_dict = {"clevr_pos": pos}
            if H.dataset == "clevr_rel":
                cond_dict = {"clevr_rel": rel}
            if H.dataset == "ffhq":
                cond_dict = {"ffhq": a}
            stats = sampler.train_iter(x, cond_dict=cond_dict)

            if torch.isnan(stats["loss"]).any():
                log(f"Skipping step {step} with NaN loss")
                continue
            optim.zero_grad()
            stats["loss"].backward()
            optim.step()

        losses = np.append(losses, stats["loss"].item())

        if step % H.steps_per_log == 0:
            step_time_taken = time.time() - step_start_time
            stats["step_time"] = step_time_taken
            mean_loss = np.mean(losses)
            stats["mean_loss"] = mean_loss
            mean_losses = np.append(mean_losses, mean_loss)
            losses = np.array([])

            vis.line(
                np.array([mean_loss]),
                np.array([step]),
                win="loss",
                update=("append" if step > 0 else "replace"),
                opts=dict(title="Loss"),
            )
            log_stats(step, stats)

            if H.sampler == "absorbing":
                elbo = np.append(elbo, stats["vb_loss"].item())
                vis.bar(
                    sampler.loss_history,
                    list(range(sampler.loss_history.size(0))),
                    win="loss_bar",
                    opts=dict(title="loss_bar"),
                )
                vis.line(
                    np.array([stats["vb_loss"].item()]),
                    np.array([step]),
                    win="ELBO",
                    update=("append" if step > 0 else "replace"),
                    opts=dict(title="ELBO"),
                )

        if H.ema and step % H.steps_per_update_ema == 0 and step > 0:
            ema.update_model_average(ema_sampler, sampler)

        images = None
        if step % H.steps_per_display_output == 0 and step > 0:
            generator.eval()
            sampler.eval()
            if H.dataset == "clevr_pos":
                images, z = get_samples(
                    H,
                    generator,
                    ema_sampler if H.ema else sampler,
                    cond_dict={"clevr_pos": [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]},
                    weights=[3.0, 3.0],
                    batch=False,
                )
            if H.dataset == "clevr_rel":
                images, z = get_samples(
                    H,
                    generator,
                    ema_sampler if H.ema else sampler,
                    cond_dict={
                        "clevr_rel": [
                            [1, 0, 2, 1, 0, 1, 1, 0, 0, 1, 4],
                            [2, 1, 2, 1, 0, 2, 1, 3, 0, 1, 3],
                            [2, 1, 7, 0, 0, 1, 0, 5, 0, 1, 4],
                        ]
                    },
                    weights=[3, 3, 3],
                    batch=False,
                )
            if H.dataset == "ffhq":
                images, z = get_samples(
                    H,
                    generator,
                    ema_sampler if H.ema else sampler,
                    cond_dict={"ffhq": [[0], [2], [4]]},
                    weights=[3, 3, 3],
                    batch=False,
                )  # not smiling, female, no glasses
            sampler.train()
            display_images(vis, images, H, win_name=f"{H.sampler}_samples")
            if H.dataset == "clevr_comp":
                for i in range(len(z)):
                    print(clevr_tokens_to_str(z[i], 256, H.codebook_size))

        if step % H.steps_per_save_output == 0 and step > 0:
            if images is None:
                images = get_samples(H, generator, ema_sampler if H.ema else sampler)
            save_images(images, "samples", step, H.log_dir, H.save_individually)

        if H.steps_per_eval and step % H.steps_per_eval == 0 and step > 0:
            # calculate validation loss
            valid_loss, valid_elbo, num_samples = 0.0, 0.0, 0
            eval_repeats = 5
            log("Evaluating")
            for _ in tqdm(range(eval_repeats)):
                for x in val_latent_loader:
                    with torch.no_grad():
                        stats = sampler.train_iter(x.cuda())
                        valid_loss += stats["loss"].item()
                        if H.sampler == "absorbing":
                            valid_elbo += stats["vb_loss"].item()
                        num_samples += x.size(0)
            valid_loss = valid_loss / num_samples
            if H.sampler == "absorbing":
                valid_elbo = valid_elbo / num_samples

            val_losses = np.append(val_losses, valid_loss)
            val_elbos = np.append(val_elbos, valid_elbo)
            vis.line(
                np.array([valid_loss]),
                np.array([step]),
                win="Val_loss",
                update=("append" if step > 0 else "replace"),
                opts=dict(title="Validation Loss"),
            )
            if H.sampler == "absorbing":
                vis.line(
                    np.array([valid_elbo]),
                    np.array([step]),
                    win="Val_elbo",
                    update=("append" if step > 0 else "replace"),
                    opts=dict(title="Validation ELBO"),
                )

        if step % H.steps_per_checkpoint == 0 and step > H.load_step:
            save_model(sampler, H.sampler, step, H.log_dir)
            save_model(optim, f"{H.sampler}_optim", step, H.log_dir)

            if H.ema:
                save_model(ema_sampler, f"{H.sampler}_ema", step, H.log_dir)

            train_stats = {
                "losses": losses,
                "mean_losses": mean_losses,
                "val_losses": val_losses,
                "elbo": elbo,
                "val_elbos": val_elbos,
                "steps_per_log": H.steps_per_log,
                "steps_per_eval": H.steps_per_eval,
            }
            save_stats(H, train_stats, step)


if __name__ == "__main__":
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Setting up training for {H.sampler}")
    start_training_log(H)
    main(H, vis)
