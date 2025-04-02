wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --annotation "coarse" \
    --resume-checkpoint "./checkpoints/b5 coarse/coarse_annotation_pretrained.pth" \
    --batch-size 8 \
    --epochs 3 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "b5 coarse" \