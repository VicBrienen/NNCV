wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --annotation "fine" \
    --resume-checkpoint "./checkpoints/b5 coarse/coarse_annotation_pretrained.pth" \
    --batch-size 8 \
    --epochs 30 \
    --lr 0.0001 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "mask2former" \