for c in 2 3
do
    python experiments/get_samples_ffhq.py --ema --ae_load_dir vqgan_ffhq --ae_load_step 100000 --dataset ffhq \
        --log_dir experiments_ffhq --load_dir absorbing_ffhq --load_step 300000 --sampler absorbing  --temp 0.9 --sample_steps 30 --n_components $c

    python experiments/calc_FID.py --dataset ffhq --ema --ae_load_dir vqgan_ffhq --ae_load_step 20000 \
    --log_dir experiments_ffhq --load_dir absorbing_ffhq --load_step 300000 --sampler absorbing  --temp 0.9 --sample_steps 20 --n_samples 5000 --n_components $c > logs/experiments_ffhq/FID_$c.txt

    cd classifier
    python eval.py --dataset ffhq --checkpoint_dir classifiers --npy_path /datasets/ffhq_256_dataset.npz --generated_img_folder ../logs/experiments_ffhq/images --num_rels $c > ../logs/experiments_ffhq/acc_$c.txt

    cd ..
    
    mv logs/experiments_ffhq/images logs/experiments_ffhq/images_$c
done

