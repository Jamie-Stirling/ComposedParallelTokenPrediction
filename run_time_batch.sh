mkdir -p logs/experiments_time
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 1 \
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 1 > logs/experiments_time/time_1_1.txt
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 1 \
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 2 > logs/experiments_time/time_1_2.txt
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 1\
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 3 > logs/experiments_time/time_1_3.txt
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 25\
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 1 > logs/experiments_time/time_25_1.txt
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 25\
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 2 > logs/experiments_time/time_25_2.txt
python experiments/time_sampler_clevr_pos.py --ema --ae_load_dir vqgan_clevr_pos --ae_load_step 20000 --dataset clevr_pos --log_dir experiments_time --batch_size 25\
    --load_dir absorbing_clevr_pos --load_step 300000 --sampler absorbing --sample_steps 30  --temp 0.8 --n_components 3 > logs/experiments_time/time_25_3.txt