wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-training" \