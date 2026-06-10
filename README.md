# Controllable Image Generation with Composed Parallel Token Prediction

**Paper (CVPRW):** [Controllable Image Generation with Composed Parallel Token Prediction](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Stirling_Controllable_Image_Generation_with_Composed_Parallel_Token_Prediction_CVPRW_2026_paper.html)

**Paper (arXiv):** [Controllable Image Generation with Composed Parallel Token Prediction](https://arxiv.org/abs/2604.05730)


## Environment Setup
First, ensure conda is installed. Then, initialise a conda environment using:

```
conda env create --name controllable --file requirements.yml
```

## Classifiers

Pre-trained evaluation classifiers are included in `./classifier/classifiers/`.

## Dataset Downloads and Setup

FFHQ can be obtained [here](https://github.com/NVlabs/ffhq-dataset) and FFHQ annotations can be obtained [here](https://github.com/DCGM/ffhq-features-dataset).

Positional CLEVR and Relational CLEVR can be obtained [here](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/tree/main/classifier).

After downloading the relevant files, place the following in `/datasets/` (alternatively, modify `./utils/data_utils.py` to use the directory of choice).


* `clevr_pos_data_128_30000.npz`
* `clevr_training_data_128.npz`
* `clevr_pos_5000_1.npz`
* `clevr_pos_5000_2.npz`
* `clevr_pos_5000_3.npz`
* `clevr_generation_1_relations.npz`
* `clevr_generation_2_relations.npz`
* `clevr_generation_3_relations.npz`
* `FFHQ/`
* `ffhq-features-dataset/`

The data partitions (image IDs) used for evaluation on FFHQ are in `ffhq_<N_COMPONENTS>_partition.txt`. The corresponding images are used for computing accuracy and FID. These were chosen (random uniform without replacement) from the FFHQ dataset.


## VQ-VAE/VQ-GAN Training
The following scripts train a VQ-VAE (for `clevr_pos` and `clevr_rel`) or VQ-GAN (for `FFHQ`). Settings are those used for results reported in the paper.

```
./train_vqgan_clevr_pos.sh
```

```
./train_vqgan_clevr_rel.sh
```

```
./train_vqgan_ffhq.sh
```

## Sampler Training
The following scripts train conditional samplers for each dataset of interest. Settings are those used for results reported in the paper.

```
./train_sampler_clevr_pos.sh
```

```
./train_sampler_clevr_rel.sh
```

```
./train_sampler_ffhq.sh
```

## Accuracy and FID Evaluation
Before running evaluations on FFHQ specifically, first run `python3 utils/prepare_ffhq_npz.py` to convert the images into the correct format.

The following scripts evaluate compositional generation (accuracy and FID).

```
./eval_clevr_pos.sh
```
```
./eval_clevr_rel.sh
```
```
./eval_ffhq.sh
```

STDOUTs are written to `./logs/experiments_<DATASET>/acc_<N_COMPONENTS>.txt` and `./logs/experiments_<DATASET>/FID_<N_COMPONENTS>.txt`, results are contained therein.

## Sample Time Evaluation

```
./run_time_batch.sh
```

STDOUTs are written to `./logs/experiments_time/time_<BATCH_SIZE>_<N_COMPONENTS>.txt`, results are contained therein.

## Citation

If you use this research, please cite:

```bibtex
@InProceedings{Stirling_2026_CVPRW,
    author    = {Stirling, Jamie and Al-Moubayed, Noura and Willcocks, Chris and Shum, Hubert},
    title     = {Controllable Image Generation with Composed Parallel Token Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2026},
    pages     = {5074--5083}
}
```

## Attribution
Code for this project is derived from [Unleashing Transformers](https://github.com/samb-t/unleashing-transformers) by Sam Bond-Taylor and Peter Hessey et al., licensed under the MIT License.

This repository has been substantially modified and extended to implement the training and evaluation code required to reproduce the quantitative results of [Controllable Image Generation with Composed Parallel Token Prediction](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/html/Stirling_Controllable_Image_Generation_with_Composed_Parallel_Token_Prediction_CVPRW_2026_paper.html).