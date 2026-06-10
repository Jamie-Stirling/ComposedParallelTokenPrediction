import sys

sys.path.append(".")
import torch
from tqdm import tqdm

from models import Generator

from train_sampler import attribs_to_str, tokens_to_attribs, clevr_tokens_to_str
from hparams import get_sampler_hparams

from utils.log_utils import (
    save_images,
    set_up_visdom,
    config_log,
    log,
    start_training_log,
    display_images,
    load_model,
)
from utils.sampler_utils import (
    get_sampler,
    get_samples,
    retrieve_autoencoder_components_state_dicts,
)


from utils.data_utils import ClevrDataset


def main(H, vis):

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

    sampler = load_model(sampler, f"{H.sampler}_ema", H.load_step, H.load_dir)
    # print modules of the model

    sampler.n_samples = 25  # get samples in 5x5 grid

    sampler.eval()
    generator.eval()  # ensure we don't drop out important conditioning information

    dataset = ClevrDataset(
        128,
        f"/datasets/clevr_generation_{H.n_components}_relations.npz",
        False,
        False,
        False,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)

    batch_count = 0
    for batch in tqdm(dataloader):
        _, ann = batch

        rel = ann["y"].cuda()
        print(rel.shape)
        n = rel.shape[1]
        # shape is [batch, cond, d] and we want cond lists of [batch, d]
        rel = [rel[:, i] for i in range(n)]
        weight = H.c_weight if hasattr(H, "c_weight") else 3.0

        images, z = get_samples(
            H,
            generator,
            sampler,
            training=False,
            cond_dict={"clevr_rel": rel},
            weights=[weight for _ in range(len(rel))],
            batch=True,
        )

        if H.dataset == "clevr_comp":
            for i in range(len(z)):
                print(clevr_tokens_to_str(z[i], 256, 1024))
        elif H.dataset == "ffhq":
            z = tokens_to_attribs(z, 256, H.codebook_size)
            for i in range(len(z)):
                print(attribs_to_str(z[i], 256, H.codebook_size))
        display_images(vis, images, H, win_name=f"{H.sampler}_samples")
        id = "samples_" + str(H.load_step) + "_" + str(weight) + "_" + str(n)
        save_images(images, id, batch_count, H.log_dir, save_individually=True)
        batch_count += 1


if __name__ == "__main__":
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Setting up training for {H.sampler}")
    start_training_log(H)
    main(H, vis)
