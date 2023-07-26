# Set the path to save checkpoints and logs
OUTPUT_DIR='ssd_scratch/cvit/sarthak395/mvva_output/mvva_videomae_vit_base_k400_pretrain'
# path to pretrain model
# Google Drive Link: https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1
#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --output=out.txt
#SBATCH -c 9
#SBATCH -n 4

MODEL_PATH='Checkpoints/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python3 mvva_videomae_model.py \
      --finetune ${MODEL_PATH} \
      --batch_size 2 \
      --opt adamw \
      --lr 2.5e-4 \
      --layer_decay 0.75 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 1 \
      --data_set "mvva" \
      --cc_coeff 0 --sim_coeff 0 --l1_coeff 0 \

echo "done"

# # python3 mvva_videomae_model.py --finetune 'Checkpoints/checkpoint.pth' --batch_size 1 --opt adamw --lr 2.5e-4 --layer_decay 0.75 --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs 1 --data_set "mvva" --cc_coeff 0 --sim_coeff 0 --l1_coeff 0
