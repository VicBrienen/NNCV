wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --resume-checkpoint "None" \
    --batch-size 3 \
    --accumulation_steps 4 \
    --epochs 30 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer b4 ADE20k pretrained" \