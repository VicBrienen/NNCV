wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --resume-checkpoint "None" \
    --batch-size 8 \
    --epochs 30 \
    --lr 0.0001 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer b5 ADE20k pretrained" \