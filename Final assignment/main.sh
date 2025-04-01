wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --annotation "coarse" \
    --batch-size 32 \
    --epochs 30 \
    --lr 0.0001 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer" \