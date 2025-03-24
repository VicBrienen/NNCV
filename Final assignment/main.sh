wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.001 \
    --weight-decay 0.0 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "Unet" \