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
# export CUDA_LAUNCH_BLOCKING=1
# bert-base-multilingual-cased / facebook/xlm-roberta-xl / xlm-roberta-large
# en fr es de el bg ru tr ar vi th zh hi sw ur


accelerate launch run_xnli.py \
  --model_name_or_path xlm-roberta-base \
  --language "en fr es de el bg ru tr ar vi th zh hi sw ur" \
  --train_language en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1 \
  --overwrite_output_dir 