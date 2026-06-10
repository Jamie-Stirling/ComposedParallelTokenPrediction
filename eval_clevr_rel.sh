for c in 1 2 3
do
    python experiments/get_samples_clevr_rel.py --ema --ae_load_dir vqgan_clevr_rel --ae_load_step 40000 --dataset clevr_rel \
        --log_dir experiments_clevr_rel --load_dir absorbing_clevr_rel --load_step 300000 --sampler absorbing  --temp 0.95 --sample_steps 30 --n_components $c

    python experiments/calc_FID.py --dataset clevr_rel --ema --ae_load_dir vqgan_clevr_rel --ae_load_step 40000 \
    --log_dir experiments_clevr_rel --load_dir absorbing_clevr_rel --load_step 300000 --sampler absorbing  --temp 0.95 --sample_steps 30 --n_samples 5000 --n_components $c > logs/experiments_clevr_rel/FID_$c.txt

    cd classifier
    python eval.py --dataset clevr_rel --checkpoint_dir classifiers --npy_path /datasets/clevr_generation_${c}_relations.npz --generated_img_folder ../logs/experiments_clevr_rel/images > ../logs/experiments_clevr_rel/acc_$c.txt

    cd ..

    mv logs/experiments_clevr_rel/images logs/experiments_clevr_rel/images_$c
done

