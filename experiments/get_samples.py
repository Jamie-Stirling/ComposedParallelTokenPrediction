import sys

sys.path.append(".")
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


def clevr_comp():
    # TODO: fully conditional method (slow but should work best)
    # NOTE: interesting point on bias, the stronger we weight particular attributes, the closer it is to the dataset average
    # the fully conditional method is the only one that can get around this

    images, z = get_samples(
        H,
        generator,
        sampler,
        training=False,
        sample_method="multi_fully_compositional",
        attrib_list=[
            {"color": 7, "material": 2, "shape": 1, "size": 0},
            {"color": 6, "material": 2, "shape": 1, "size": 0},
            {"color": 5, "material": 2, "shape": 1, "size": 0},
            {"color": 4, "material": 2, "shape": 1, "size": 0},
            {"color": 3, "material": 2, "shape": 1, "size": 0},
            {"color": 2, "material": 2, "shape": 1, "size": 0},
            {"color": 1, "material": 2, "shape": 1, "size": 0},
            {"color": 0, "material": 2, "shape": 1, "size": 0},
            # {"color": 1, "material": 1, "shape": 2, "size": 1},
        ],
        weight_list=[
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": -1, "material": 0, "shape": 0, "size": 0},
            {"color": 10, "material": 0, "shape": 0, "size": 0},
            # {"color": 20, "material": 20, "shape": 20, "size": 20},
        ],
        self_ensemble=False,
        sample_steps=15,
    )

    """images, z = get_samples(H, generator, sampler, training=False, sample_method="conditional_naive", attribs=
        {"color": 1, "material": 1, "shape": 2, "size": 1}, weights={"color": 1, "material": 1.8, "shape": 1.8, "size": 1})"""
    """images, z = get_samples(H, generator, sampler, training=False, sample_method="compositional", attribs=
        {"color": 1, "material": 1, "shape": 0, "size": 0}, weights={"color": 10, "material": 10, "shape": 10, "size": 10}, self_ensemble=False, one_by_one=True)
    """


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

    for weight in [4, 5, 6]:
        for n in [3, 4, 5, 6, 7, 8]:
            images, z = get_samples(
                H,
                generator,
                sampler,
                training=False,
                pos=[[0.5, 0.33]],
                n=[n],
                weights=[weight],
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
            save_images(images, id, 0, H.log_dir)


if __name__ == "__main__":
    H = get_sampler_hparams()
    vis = set_up_visdom(H)
    config_log(H.log_dir)
    log("---------------------------------")
    log(f"Setting up training for {H.sampler}")
    start_training_log(H)
    main(H, vis)
