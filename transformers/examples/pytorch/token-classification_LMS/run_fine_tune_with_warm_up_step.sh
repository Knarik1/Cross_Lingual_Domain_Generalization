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

export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=ner
export BATCH_SIZE=32

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
  for LR in 1e-4 1e-5 1e-6
  do
    for SAVED_EPOCH in 3 4 5 6
    do
      accelerate launch run_ner_no_trainer_fine_tune.py \
          --experiment_description 'Fine tune lr=1e-4 1e-5 1e-6' \
          --output_dir /mnt/xtb/knarik/outputs/ner_old \
          --data_dir data/$TASK_NAME \
          --task_name $TASK_NAME \
          --model_name_or_path bert-base-multilingual-cased \
          --max_length 128 \
          --weight_decay 0.01 \
          --num_warmup_steps 1000 \
          --per_device_train_batch_size $BATCH_SIZE \
          --num_train_epochs $SAVED_EPOCH \
          --learning_rate $LR \
          --seed $RANDOM \
          --do_fine_tune
    done  
  done  
done            


