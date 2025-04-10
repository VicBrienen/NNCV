wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --resume-checkpoint "checkpoints/model.pth" \
    --batch-size 1 \
    --accumulation_steps 16 \
    --epochs 10 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer b5 ADE20k finetuning no augmentations" \