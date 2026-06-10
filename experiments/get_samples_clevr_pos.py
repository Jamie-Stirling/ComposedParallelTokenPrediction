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


from utils.data_utils import Clevr2DPosDataset


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

    dataset = Clevr2DPosDataset(
        128, f"/datasets/clevr_pos_5000_{H.n_components}.npz", False, False, False
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)

    batch_count = 0
    for batch in tqdm(dataloader):
        _, ann = batch

        pos = ann["y"].cuda()
        n = pos.shape[1]

        # shape is [batch, cond, d] and we want cond lists of [batch, d]
        pos = [pos[:, i] for i in range(pos.shape[1])]
        weight = H.c_weight if hasattr(H, "c_weight") else 3.0

        images, z = get_samples(
            H,
            generator,
            sampler,
            training=False,
            cond_dict={"clevr_pos": pos},
            weights=[weight for _ in range(len(pos))],
            batch=True,
        )

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
