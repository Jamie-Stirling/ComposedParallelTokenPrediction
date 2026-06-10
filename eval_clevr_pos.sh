for c in 3
do
    python experiments/get_samples_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos \
    --log_dir experiments_clevr_pos --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing  --temp 0.9 --sample_steps 30 --n_components $c

    python experiments/calc_FID.py --dataset clevr_pos --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 \
    --log_dir experiments_clevr_pos --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing  --temp 0.9 --sample_steps 30 --n_samples 5000 --n_components $c > logs/experiments_clevr_pos/FID_$c.txt

    cd classifier
    python eval.py --dataset clevr_pos --checkpoint_dir classifiers --npy_path /datasets/clevr_pos_5000_$c.npz --generated_img_folder ../logs/experiments_clevr_pos/images > ../logs/experiments_clevr_pos/acc_$c.txt

    cd ..

    mv logs/experiments_clevr_pos/images logs/experiments_clevr_pos/images_$c
done

