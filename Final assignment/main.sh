wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --accumulation_steps 8 \
    --epochs 50 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer b5 ADE20k pretrained" \