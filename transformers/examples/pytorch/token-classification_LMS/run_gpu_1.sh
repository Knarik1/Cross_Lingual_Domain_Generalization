# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES=1
export TASK_NAME=ner
export META_SPLIT_SEED=8
export SEED=8
export FOLDER="ner_3e-5_lr" # ner_3e-5_lr / ner_1e-5_lr
export TARGET_LANG="es" # de/es/nl/zh


accelerate launch run_ner_no_trainer.py \
  --experiment_description "LMS predict target=$TARGET_LANG folder=$FOLDER" \
  --output_dir /mnt/xtb/knarik/outputs/$FOLDER \
  --data_dir data/$TASK_NAME \
  --task_name $TASK_NAME \
  --model_name_or_path bert-base-multilingual-cased \
  --max_length 128 \
  --weight_decay 0.01 \
  --seed $SEED \
  --warmup_proportion 0.4 \
  --per_device_train_batch_size 32 \
  --num_train_epochs 4 \
  --learning_rate 3e-5 \
  --meta_models_split_seed $META_SPLIT_SEED \
  --lms_num_train_epochs 4 \
  --lms_hidden_size 512 \
  --lms_target_lang $TARGET_LANG \
  --lms_model_seed $SEED \
  --do_lms_predict 