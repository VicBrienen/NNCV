wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --resume-checkpoint "None" \
    --batch-size 3 \
    --accumulation_steps 4 \
    --epochs 50 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 9 \
    --seed 42 \
    --experiment-id "segformer b5 ADE20k pretrained" \