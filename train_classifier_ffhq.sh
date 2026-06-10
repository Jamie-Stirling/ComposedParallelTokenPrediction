cd classifier

python train.py --spec_norm --norm --im_size 128 --batch_size 16 --dataset ffhq --lr 1e-5 --checkpoint_dir results --cond_idx 0
python train.py --spec_norm --norm --im_size 128 --batch_size 16 --dataset ffhq --lr 1e-5 --checkpoint_dir results --cond_idx 1
python train.py --spec_norm --norm --im_size 128 --batch_size 16 --dataset ffhq --lr 1e-5 --checkpoint_dir results --cond_idx 3

