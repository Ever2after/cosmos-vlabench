# /home/jusang/cosmos-policy

export BASE_DATASETS_DIR=/NHNHOME/WORKSPACE/0426030040_A/data/vlabench-hdf5-if

BASE_DIR=/NHNHOME/WORKSPACE/0426030040_A

CKPT_ROOT=$BASE_DIR/cosmos-policy-vlabench-if/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_vlabench_primitives_if

# CKPT_DIR="$CKPT_ROOT/checkpoints/$(cat "$CKPT_ROOT/checkpoints/latest_checkpoint.txt")"
CKPT_DIR="$CKPT_ROOT/checkpoints/iter_000060000"

echo "$CKPT_DIR"

python -m cosmos_policy.experiments.robot.vlabench.deploy \
  --config cosmos_predict2_2b_480p_vlabench_primitives__inference_only \
  --ckpt_path "$CKPT_DIR" \
  --config_file cosmos_policy/config/config.py \
  --dataset_stats_path $BASE_DIR/data/vlabench-hdf5-if/vlabench_dataset_statistics.json \
  --t5_text_embeddings_path $BASE_DIR/data/vlabench-hdf5-if/t5_embeddings.pkl \
  --use_wrist_image True \
  --num_wrist_images 2 \
  --use_third_person_image True \
  --num_third_person_images 2 \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --trained_with_image_aug True \
  --chunk_size 8 \
  --num_open_loop_steps 8 \
  --num_denoising_steps_action 10 \
  --port 8777