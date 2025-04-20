# Peak Performance on Cityscapes Dataset using Segformer-B5

- **Author:** Vic Brienen
- **Codalab Username:** VicBrienen
- **TU/e Email Address:** v.a.w.brienen@student.tue.nl

### 1. Cloning the Repository
```bash
git clone https:https://github.com/VicBrienen/NNCV
```

### 2. SLURM Setup
For setup on SLURM cluster, consult ``README-Slurm.md``.

### 3. Local Setup
The dataset can be downloaded here [Cityscapes download page](https://www.cityscapes-dataset.com/downloads/). Organize it as follows:

```plaintext
data/
├── cityscapes/
│   ├── leftImg8bit/
│       ├── train/
│           ├── aachen/
│           ├── .../
│       ├── val/
│           ├── .../
│   ├── gtFine/
│       ├── train/
│           ├── aachen/
│           ├── .../
│       ├── val/
│           ├── .../
```

Run the following to install the required libraries:
```bash
pip install -r requirements.txt
```

## File Descriptions
- `train.py`: Runs the training loop.
- `model.py`: Contains the model to be loaded for training and inference.
- `losses.py`: Contains the code for the mean dice loss.
- `mappings.py`: Contains the mappings code that was in `train.py` initially, created to maintain readability.
- `main.sh`: Script executed via `jobscript_slurm.sh` to run train.py inside the container.
- `jobscript_slurm.sh`: Submission script that runs `main.sh`.
- `download_docker_and_data.sh`: Contains downloading script for the container and the Cityscapes dataset.
- `.env`: Contains WandB api key and paths.

## Running the code
Running via SLURM is explained in `README-Slurm.md`.

Running locally can be done with the following command:

```bash
python train.py \
    --data-dir /path/to/your/local/cityscapes \
    --batch-size 2 \
    --accumulation_steps 8 \
    --epochs 50 \
    --lr 0.00006 \
    --weight-decay 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer b5 ADE20k pretrained" \
```
Make sure to set up WandB paths and API keys in the `.env` file.