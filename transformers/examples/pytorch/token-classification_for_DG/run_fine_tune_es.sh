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
export PRE_TRAINED_MODEL=domain_es_epoch_50 # domain_en_de_zh_epoch_50 / domain_es_epoch_50
export LR=1e-5 # fine_tune -> 1e-5, classifier -> 1e-2

# DG Fine-tune domain=en_de_zh epoch=50 / DG Classifier domain=es epoch=50 pretrained=domain_en_de_zh_epoch_50 


accelerate launch run_ner_no_trainer.py \
    --experiment_description "DG Fine-tune domain=es epoch=50" \
    --output_dir /mnt/xtb/knarik/outputs/DG \
    --data_dir data/ner \
    --task_name ner \
    --model_name_or_path bert-base-multilingual-cased \
    --pre_trained_model $PRE_TRAINED_MODEL \
    --max_length 96 \
    --weight_decay 0.01 \
    --warmup_proportion 0.4 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 50 \
    --learning_rate $LR \
    --seed 8 \
    --train_langs "es" \
    --do_fine_tune         # do_fine_tune / do_train_classifier 