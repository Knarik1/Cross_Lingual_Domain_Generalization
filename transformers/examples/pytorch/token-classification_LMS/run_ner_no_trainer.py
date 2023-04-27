#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import argparse
import logging
import math
import os
import sys
import glob
import random
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from aim import Run
from functools import cache
import lang2vec.lang2vec as l2v

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ClassLabel, DatasetDict, load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ner_target_langs = {
    'de': 'nl',
    'es': 'nl', 
    'nl': 'de',
    'zh': 'de'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")


    # local arguments
    parser.add_argument("--data_dir", type=str, default='data', help='Path to your task data.')
    parser.add_argument("--experiment_description", type=str, help="Experiment.")
    parser.add_argument('--meta_models_split_seed', type=int, help="Meta train/dev/test fine tuned models' split seed.")
    parser.add_argument('--saved_epoch', type=int, default=2, help="Which epochs of the model to save.")
    parser.add_argument("--do_fine_tune", action="store_true", help="Fine tune")
    parser.add_argument("--do_lms_hyperparam_search", action="store_true", help="LMS hyperparameter ")
    parser.add_argument("--do_lms_train", action="store_true", help="LMS train")
    parser.add_argument("--do_lms_predict", action="store_true", help="Predict")
    parser.add_argument("--do_en_cls", action="store_true", help="En feature extraction")
    parser.add_argument('--warmup_proportion', type=float, help="Fine tune warm up.")
    parser.add_argument("--lms_num_train_epochs", type=int, help="Total number of training epochs to perform in LMS.")
    parser.add_argument("--lms_learning_rate", type=float, help="Learning rate in LMS.")
    parser.add_argument('--lms_learning_rates', nargs='+', help="Learning rates in LMS.")
    parser.add_argument("--lms_hidden_size", type=int, help="Hidden size in LMS.")
    parser.add_argument("--lms_batch_size", type=int, help="Batch size in LMS.")
    parser.add_argument("--lms_batch_sizes", nargs='+', help="Batch sizes in LMS.")
    parser.add_argument("--lms_target_lang", type=str, help="Target language in LMS.")
    parser.add_argument("--lms_model_seed", type=int, help="Model seed in LMS.")

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def kendall_rank_correlation(y_true, y_score):
    '''
    Kendall Rank Correlation Coefficient
    r = [(number of concordant pairs) - (number of discordant pairs)] / [n(n-1)/2]
    :param y_true:
    :param y_score:
    :return:
    '''

    # create labels
    golden_label = []
    for i in range(len(y_true) - 1):
        for j in range(i + 1, len(y_true)):
            if y_true[i] > y_true[j]:
                tmp_label = 1
            elif y_true[i] < y_true[j]:
                tmp_label = -1
            else:
                tmp_label = 0
            golden_label.append(tmp_label)

    # evaluate
    pred_label = []
    for i in range(len(y_score) - 1):
        for j in range(i + 1, len(y_score)):
            if y_score[i] > y_score[j]:
                tmp_label = 1
            elif y_score[i] < y_score[j]:
                tmp_label = -1
            else:
                tmp_label = 0
            pred_label.append(tmp_label)

    # res
    n_concordant_pairs = sum([1 if i == j else 0 for i, j in zip(golden_label, pred_label)])
    n_discordant_pairs = sum(
        [1 if ((i == 1 and j == -1) or (i == -1 and j == 1)) else 0 for i, j in zip(golden_label, pred_label)])

    N = len(y_score)
    res = (n_concordant_pairs - n_discordant_pairs) / (N * (N - 1) / 2)
    return res


def read_ner_data(path: str) -> dict:
    encoding = 'latin-1' if 'deu' in path else 'utf-8'

    with open(path, 'r', encoding=encoding) as f:
        data_dict = {'tokens': [], 'ner_tags': []}
        new_senetence_tokens = []
        new_senetence_ner_tags = []

        for line in f:
            if line.startswith('-DOCSTART') or line == '' or line == '\n':
                if new_senetence_tokens:
                    data_dict['tokens'].append(new_senetence_tokens)
                    data_dict['ner_tags'].append(new_senetence_ner_tags)
                    new_senetence_tokens = []
                    new_senetence_ner_tags = []
            else:
                token = line.split()[0]
                label = line.split()[-1]
                new_senetence_tokens.append(token)
                new_senetence_ner_tags.append(label)

        assert len(data_dict['tokens']) == len(data_dict['ner_tags'])   

    return data_dict


@cache
def load_vector(path):
    return torch.from_numpy(np.load(path)).view(-1)        


def get_labels(predictions, references, label_list):
    y_pred = predictions.detach().cpu().clone().numpy()
    y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels


def compute_metrics(metric, args):
    results = metric.compute(zero_division=1)
    if args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


def evaluate(model, dataloader, args, accelerator, label_list, return_CLS=False):
    model.eval()
    metric = load_metric("seqeval")
    avg_dev_loss = 0
    avg_cls_tokens = []

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            dev_loss = outputs.loss
            dev_loss = dev_loss / args.gradient_accumulation_steps
            avg_dev_loss += dev_loss.item()

            if return_CLS:
                batch_cls_tokens = outputs.hidden_states[-1][:, 0]
                avg_cls_tokens.append(torch.mean(batch_cls_tokens, dim=0, keepdim=True).detach().cpu().numpy())

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered, label_list)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    
    eval_metric = compute_metrics(metric, args)
    avg_cls_token = np.mean(avg_cls_tokens, axis=0, keepdims=True)[0] 

    return eval_metric, avg_dev_loss / len(dataloader), avg_cls_token


def get_meta_splits(output_dir, seed):
    model_folder_paths = np.array(glob.glob(os.path.join(output_dir, f'FineTuned_models/model-seed_*')))
    model_folder_paths_idx = np.random.RandomState(seed=seed).permutation(len(model_folder_paths))

    meta_train_folder_paths = model_folder_paths[model_folder_paths_idx[:120]]
    meta_dev_folder_paths = model_folder_paths[model_folder_paths_idx[120:180]]
    meta_test_folder_paths = model_folder_paths[model_folder_paths_idx[180:]]

    return {'train': meta_train_folder_paths, 'dev': meta_dev_folder_paths, 'test': meta_test_folder_paths}


def seed_everything(seed=42):
    # system
    random.seed(seed)
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    set_seed(seed)
    # cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def main():
    args = parse_args()
    seed_everything(0 if args.seed is None else args.seed)

    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    METRIC = 'f1' if args.task_name == 'ner' else 'accuracy'
    TARGET_LANGS = ner_target_langs.keys() if args.task_name == 'ner' else None

    # Initialize a new run
    run = Run(experiment=f'{args.experiment_description} for {args.task_name}.')

    # Log run parameters
    run["hparams"] = {
        "cli": sys.argv
    }

    for arg in vars(args):
        try:
            run["hparams", arg] = getattr(args, arg)
        except   TypeError:
            run["hparams", arg] = str(getattr(args, arg))


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    # Use the device given by the `accelerator` object.
    device = accelerator.device

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if args.do_fine_tune:
        logger.info("======================================= Fine Tuning =======================================")
        # Output folder for fine-tuned models on En train data
        models_dir = os.path.join(OUTPUT_DIR, 'FineTuned_models')
        os.makedirs(models_dir, exist_ok=True)

        # a little trick to seed all 240 models
        # seed = 1000000 + len(glob.glob(os.path.join(models_dir, f'model-*' )))
        seed = args.seed
        run["hparams", 'seed'] = seed
        seed_everything(seed)

        raw_datasets = DatasetDict()
            
        if args.task_name == 'ner':
            # en
            raw_datasets['train'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'train.txt')))
            raw_datasets['en_dev'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'valid.txt')))
            raw_datasets['en_test'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'test.txt')))

            # de
            raw_datasets['de_dev'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'deu.testa.txt')))
            raw_datasets['de_test'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'deu.testb.txt')))

            rand_100_idxs = np.random.randint(len(raw_datasets['de_dev']), size=100)
            raw_datasets['de_100_dev'] = raw_datasets['de_dev'].select(rand_100_idxs)

            # es
            raw_datasets['es_dev'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'esp.testa.txt')))
            raw_datasets['es_test'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'esp.testb.txt')))

            rand_100_idxs = np.random.randint(len(raw_datasets['es_dev']), size=100)
            raw_datasets['es_100_dev'] = raw_datasets['es_dev'].select(rand_100_idxs)

            # nl
            raw_datasets['nl_dev'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'ned.testa.txt')))
            raw_datasets['nl_test'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'ned.testb.txt')))

            rand_100_idxs = np.random.randint(len(raw_datasets['nl_dev']), size=100)
            raw_datasets['nl_100_dev'] = raw_datasets['nl_dev'].select(rand_100_idxs)

            # zh
            raw_datasets['zh_dev'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'msra_train_bio.txt')))
            # for zh lang there is no seperate dev data, so we sample it from the train, and according to the paper zh ALL-Target data is 4499
            rand_4499_idxs = np.random.randint(len(raw_datasets['zh_dev']), size=4499)
            raw_datasets['zh_dev'] = raw_datasets['zh_dev'].select(rand_4499_idxs)
            raw_datasets['zh_test'] = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'msra_test_bio.txt')))

            rand_100_idxs = np.random.randint(len(raw_datasets['zh_dev']), size=100)
            raw_datasets['zh_100_dev'] = raw_datasets['zh_dev'].select(rand_100_idxs)
            
        print(raw_datasets)

        # Trim a number of training examples
        if args.debug:
            for split in raw_datasets.keys():
                raw_datasets[split] = raw_datasets[split].select(range(100))
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        if raw_datasets["train"] is not None:
            column_names = raw_datasets["train"].column_names
            features = raw_datasets["train"].features
        else:
            column_names = raw_datasets["en_dev"].column_names
            features = raw_datasets["en_dev"].features

        if args.text_column_name is not None:
            text_column_name = args.text_column_name
        elif "tokens" in column_names:
            text_column_name = "tokens"
        else:
            text_column_name = column_names[0]

        if args.label_column_name is not None:
            label_column_name = args.label_column_name
        elif f"{args.task_name}_tags" in column_names:
            label_column_name = f"{args.task_name}_tags"
        else:
            label_column_name = column_names[1]

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(raw_datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)

        # Map that sends B-Xxx label to its I-Xxx counterpart
        b_to_i_label = []
        for idx, label in enumerate(label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in label_list:
                b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
            else:
                b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        else:
            config = CONFIG_MAPPING[args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
        if not tokenizer_name_or_path:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list),
                                                                output_hidden_states=True)  

        model.resize_token_embeddings(len(tokenizer))

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        padding = "max_length" if args.pad_to_max_length else False

        # Tokenize all texts and align the labels with them.

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                max_length=args.max_length,
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )

            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the labelrun for the first token of each word.
                    elif word_idx != previous_word_idx:
                        try:
                            label_ids.append(label_to_id[label[word_idx]])
                        except KeyError:
                            print('Key Error', label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if args.label_all_tokens:
                            label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        with accelerator.main_process_first():
            processed_raw_datasets = raw_datasets.map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )  

        # Log a few random samples from the training set:
        for index in random.sample(range(len(processed_raw_datasets['train'])), 3):
            logger.info(f"Sample {index} of the training set: {processed_raw_datasets['train'][index]}.")

        # DataLoaders creation:
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForTokenClassification(
                tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
            )

        dataloaders = {}    
        dataloaders['train'] = DataLoader(processed_raw_datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        dataloaders['en_dev'] = DataLoader(processed_raw_datasets['en_dev'], shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        dataloaders['en_test'] = DataLoader(processed_raw_datasets['en_test'], shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)

        for lang in TARGET_LANGS:
            dataloaders[f'{lang}_dev'] = DataLoader(processed_raw_datasets[f'{lang}_dev'], shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
            dataloaders[f'{lang}_100_dev'] = DataLoader(processed_raw_datasets[f'{lang}_100_dev'], shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
            dataloaders[f'{lang}_test'] = DataLoader(processed_raw_datasets[f'{lang}_test'], shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
            
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        model.to(device)

        # Prepare everything with our `accelerator`.
        model, optimizer = accelerator.prepare(model, optimizer)

        for i_dataloader in dataloaders.keys():
            dataloaders[i_dataloader] = accelerator.prepare(dataloaders[i_dataloader])  

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(dataloaders['train']) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        num_train_optimization_steps = int(len(raw_datasets['train']) / args.per_device_train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs    
        if args.warmup_proportion:
            warm_up_steps = int(args.warmup_proportion * num_train_optimization_steps)
        else:
            warm_up_steps = args.num_warmup_steps    
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=args.max_train_steps,
        )

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("============= Running training =============")
        logger.info(f"  Num train examples = {len(raw_datasets['train'])}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")
        logger.info(f"  Seed = {seed}")
        logger.info(f"  Learning rate = {args.learning_rate}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Warm up steps = {warm_up_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        iter_num = len(dataloaders['train'])
        model_evals_table = {}

        for epoch in range(args.num_train_epochs + 1):
            model.train()
            avg_train_loss = 0
            
            for step, batch in enumerate(dataloaders['train']):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                
                if epoch != 0:
                    accelerator.backward(loss)

                    if step % args.gradient_accumulation_steps == 0 or step == iter_num - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                avg_train_loss += loss.item()

                track_step = epoch * iter_num + step
                run.track(loss.item(), name="Loss", step=track_step, context={"subset": "train", "lang": "en"})
            
            # Track En-dev
            eval_metric_en_dev, loss_dev, _ = evaluate(model, dataloaders['en_dev'], args, accelerator, label_list)
            run.track(eval_metric_en_dev[METRIC], name='F1', step=epoch, context={"lang": "en"})
            run.track(loss_dev / len(dataloaders[f'en_dev']), name='Loss', step=epoch, context={"subset": "dev", "lang": "en"})  
            accelerator.print(f"epoch {epoch}: Train loss {avg_train_loss / iter_num} Dev loss {loss_dev}")

            logger.info(f"======================================= Dev Evaluations =======================================")
            logger.info(f"======================================= Lang en {eval_metric_en_dev[METRIC]} =======================================")

            # Track pivot-dev
            for lang in TARGET_LANGS:
                # ALL-Target evals
                eval_metric, loss_dev, cls_dev_token = evaluate(model, dataloaders[f'{lang}_dev'], args, accelerator, label_list, return_CLS=True)  
                run.track(eval_metric[METRIC], name='F1', step=epoch, context={"lang": lang})
                run.track(loss_dev / len(dataloaders[f'{lang}_dev']), name='Loss', step=epoch, context={"subset": "dev", "lang": lang}) 
            
                logger.info(f"======================================= Lang {lang} {eval_metric[METRIC]} =======================================")

        if eval_metric_en_dev[METRIC] > 0.5:
            # Track model name    
            model_name = f'model-seed_{seed}-lr_{args.learning_rate}-ep_{epoch}'
            run["hparams", 'model_name'] = model_name

            logger.info(f"======================================= Save model =======================================")
            accelerator.wait_for_everyone()
            checkpoint_path = os.path.join(models_dir, model_name)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_path, save_function=accelerator.save)               


            logger.info(f"======================================= Save tokenizer =======================================")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


            logger.info(f"======================================= Prepare all_models_evals.json file =======================================")           
            model_evals_table[model_name] = {
                'En_Dev': eval_metric_en_dev[METRIC]
            }

            for lang in TARGET_LANGS:
                # save eval metric
                model_evals_table[model_name][f'All-Target_{lang}'] = eval_metric[METRIC]
                # save cls
                np.save(checkpoint_path + '/cls_' + lang + '_dev', cls_dev_token)

                # 100-Target evals
                eval_metric, loss_dev, _ = evaluate(model, dataloaders[f'{lang}_100_dev'], args, accelerator, label_list)
                model_evals_table[model_name][f'100-Target_{lang}'] = eval_metric[METRIC]

                # Pivot-Dev evals
                pivot_lang = eval(args.task_name + '_target_langs')[lang]
                eval_metric, loss_dev, _ = evaluate(model, dataloaders[f'{pivot_lang}_dev'], args, accelerator, label_list)
                model_evals_table[model_name][f'Pivot_{lang}'] = eval_metric[METRIC]

                # Test evals
                eval_metric, loss_test, cls_test_token = evaluate(model, dataloaders[f'{lang}_test'], args, accelerator, label_list, return_CLS=True)  
                model_evals_table[model_name][f'Test_{lang}'] = eval_metric[METRIC]
                # save cls
                np.save(checkpoint_path + '/cls_' + lang + '_test', cls_test_token)
            
        saved_path = os.path.join(OUTPUT_DIR, 'all_models_evals.json')
        isExists = os.path.exists(saved_path)

        if not isExists:
            with open(saved_path, "w") as f:
                json.dump(model_evals_table, f)
        else:    
            with open(saved_path, 'r+') as f:
                all_models_evals_dict = json.load(f)
                all_models_evals_dict.update(model_evals_table)
                f.seek(0)
                json.dump(all_models_evals_dict, f)

        logger.info(f"Json file is saved at {saved_path}")          

        logger.info("======================================= Fine Tuning Done! =======================================") 


    class Model(nn.Module):
        def __init__(self, input_size, hidden_size, emb_dim, activation='relu', weighted_sum=0):
            super(Model, self).__init__()

            act_dict = {'relu': nn.ReLU(),
            'gelu': nn.GELU()}
            self.act = act_dict[activation]

            # all features together
            if weighted_sum == 1:
                self.n_feature = input_size // 768
                # weighted sum feature
                self.softmax_weight = nn.Parameter(torch.empty(768, self.n_feature))
            self.weighted_sum = weighted_sum

            self.fc1 = nn.Linear(768, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

            self.fc3 = nn.Linear(emb_dim, hidden_size)
            self.fc4 = nn.Linear(hidden_size, hidden_size)
            self.bilinear = nn.Parameter(torch.empty((hidden_size, hidden_size)))
            self.dropout = nn.Dropout(0.1)
            self.initialize()

        def initialize(self):
            nn.init.xavier_normal_(self.bilinear)
            if self.weighted_sum == 1:
                nn.init.xavier_normal_(self.softmax_weight)

        def forward(self, repr, emb):
            # (786) --> (hidden_size)
            # (512) --> (hidden_size)
            bz, _ = repr.size()
            if self.weighted_sum == 1:
                weight = F.softmax(self.softmax_weight, dim=-1)
                repr = repr.view(bz, self.n_feature, -1).transpose(1, 2)
                repr = weight * repr
                repr = torch.sum(repr, dim=-1)

            out = self.fc1(repr)
            out = self.dropout(out)
            out = self.act(out)
            out = self.fc2(out)
            out = self.dropout(out)
            out = self.act(out)

            emb = self.fc3(emb)
            out = self.dropout(out)
            emb = self.act(emb)
            emb = self.fc4(emb)
            out = self.dropout(out)
            emb = self.act(emb).unsqueeze(2)

            res =  out.matmul(self.bilinear).unsqueeze(1)
            res = res.matmul(emb).squeeze(2)

            return res


    class RankNet(nn.Module):
        def __init__(self, input_size, hidden_size, embedding, activation='gelu', emb_dim=768, weighted_sum=0):
            super(RankNet, self).__init__()
            self.lang_embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
            # load pretrained embedding
            self.f = Model(input_size, hidden_size, emb_dim, activation, weighted_sum)
            self.loss = nn.BCELoss()

            
        def forward(self, repr, repr2, label, lang=None):
            batchsize, _ = repr.size()
            lang_emb = self.lang_embedding(lang)
           
            si = self.f(repr, lang_emb)
            sj = self.f(repr2, lang_emb)

            oij = torch.sigmoid(si - sj)
            label = label.unsqueeze(-1)
            loss = self.loss(oij, label)

            return loss.mean(), oij.view(-1)

        def evaluate(self, repr, lang=None):
            lang_emb = self.lang_embedding(lang)
            out = self.f(repr, lang_emb).view(-1)

            return out    


    class LMS_Dataset(Dataset):
        def __init__(self, split_meta: str, langs: list = None, kendall: bool = False) -> None:
            self.split_meta = split_meta
            self.langs = langs 
            self.kendall = kendall
            self.lang2id = {'de': 0, 'es': 1, 'nl': 2, 'zh': 3}

            # load model performance table for each target lang on dev set
            with open(os.path.join(OUTPUT_DIR, 'all_models_evals.json')) as f:
                self.model_performance_dict = json.load(f)    

            meta_dict = get_meta_splits(OUTPUT_DIR, args.meta_models_split_seed)
            self.model_paths = meta_dict[self.split_meta]

            paths = []                           

            if kendall == True:
                lang = langs[0]
                for model_path in self.model_paths:
                    paths.append((model_path, lang))

            else:        
                for model_path_1 in self.model_paths:
                    for model_path_2 in self.model_paths:
                        if model_path_1 != model_path_2:
                            for lang in langs:
                                paths.append((model_path_1, model_path_2, lang)) 

            
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx: int):

            if self.split_meta == 'test':
                model_path, lang = self.paths[idx]

                # model_name
                model_name = model_path.split('/')[-1]

                # [CLS] vector
                cls_vector = load_vector(model_path + '/cls_' + lang + '_dev.npy')
                # cls_vector = load_vector(model_path + '/cls_' + lang + '_test.npy')

                # Gold rankings
                dev_metric = torch.tensor(self.model_performance_dict[model_name]['All-Target_' + lang], dtype=torch.float32)
                # test_metric = torch.tensor(self.model_performance_dict[model_name]['Test_' + lang], dtype=torch.float32)

                # lang id
                lang = torch.tensor(self.lang2id[lang], dtype=torch.int)

                return cls_vector, dev_metric, lang, model_name


            if self.kendall == True:
                model_path, lang = self.paths[idx]

                # model_name
                model_name = model_path.split('/')[-1]

                # [CLS] vector
                cls_vector = load_vector(model_path + '/cls_' + lang + '_dev.npy')

                # Gold rankings
                dev_metric = torch.tensor(self.model_performance_dict[model_name]['All-Target_' + lang], dtype=torch.float32)

                # lang id
                lang = torch.tensor(self.lang2id[lang], dtype=torch.int)

                return cls_vector, dev_metric, lang

            else:
                model_path_1, model_path_2, lang = self.paths[idx]

                # model names
                model_1_name = model_path_1.split('/')[-1]
                model_2_name = model_path_2.split('/')[-1]

                # [CLS] vectors
                cls_vector_1 = load_vector(model_path_1 + '/cls_' + lang + '_dev.npy')
                cls_vector_2 = load_vector(model_path_2 + '/cls_' + lang + '_dev.npy')

                # Gold rankings
                dev_metric_1 = self.model_performance_dict[model_1_name]['All-Target_' + lang]
                dev_metric_2 = self.model_performance_dict[model_2_name]['All-Target_' + lang]

                if dev_metric_1 == dev_metric_2:
                    gold_ranking = 0.5
                elif dev_metric_1 > dev_metric_2:
                    gold_ranking = 1
                else:
                    gold_ranking = 0

                gold_ranking = torch.tensor(gold_ranking, dtype=torch.float32)

                # lang id
                lang = torch.tensor(self.lang2id[lang], dtype=torch.int)

                return cls_vector_1, cls_vector_2, gold_ranking, lang

           
    class En_Dev_Dataset(Dataset):
        def __init__(self, split_meta: str = 'train', langs: list = None, kendall: bool = False) -> None:
            self.split_meta = split_meta
            self.langs = langs
            self.kendall = kendall

            # load model performance table for each target lang on dev set
            with open(os.path.join(OUTPUT_DIR, 'all_models_evals.json')) as f:
                self.model_performance_dict = json.load(f)    

            meta_dict = get_meta_splits(OUTPUT_DIR, args.meta_models_split_seed)
            self.model_paths = meta_dict[self.split_meta]

            paths = []                          

            if kendall == True:
                lang = langs[0]
                for model_path in self.model_paths:
                    paths.append((model_path, lang))

            else:        
                for model_path_1 in self.model_paths:
                    for model_path_2 in self.model_paths:
                        if model_path_1 != model_path_2:
                            for lang in langs:
                                paths.append((model_path_1, model_path_2, lang))        

            self.paths = paths

        def __len__(self):
            return len(self.paths)


        def __getitem__(self, idx: int):
            if self.kendall == True:
                model_path, lang = self.paths[idx]

                # model names
                model_name = model_path.split('/')[-1]

                # dev lang metric
                dev_metric =  self.model_performance_dict[model_name]['All-Target_' + lang]

                # en lang metric
                en_dev_metric =  self.model_performance_dict[model_name]['En_Dev']

                return en_dev_metric, dev_metric  

            else:
                model_path_1, model_path_2, lang = self.paths[idx]

                # model names
                model_1_name = model_path_1.split('/')[-1]
                model_2_name = model_path_2.split('/')[-1]

                # gold ranking
                dev_metric_1 = self.model_performance_dict[model_1_name]['All-Target_' + lang]
                dev_metric_2 = self.model_performance_dict[model_2_name]['All-Target_' + lang]

                if dev_metric_1 == dev_metric_2:
                    gold_ranking = 0.5
                elif dev_metric_1 > dev_metric_2:
                    gold_ranking = 1
                else:
                    gold_ranking = 0 

                # en dev ranking
                en_dev_metric_1 = self.model_performance_dict[model_1_name]['En_Dev']
                en_dev_metric_2 = self.model_performance_dict[model_2_name]['En_Dev']

                if en_dev_metric_1 == en_dev_metric_2:
                    en_dev_ranking = 0.5
                elif en_dev_metric_1 > en_dev_metric_2:
                    en_dev_ranking = 1
                else:
                    en_dev_ranking = 0

                return en_dev_ranking, gold_ranking        


    def evaluate_en_dev(LANGS, PIVOT_LANGS, batch_size):
        # calculates en-dev f1 against target-dev f1 with kendall and accuracy
        en_dev_datasets = {}
        en_dev_dataloaders = {}
        en_dev_baselines = {}

        for lang in tqdm(LANGS):
            en_dev_datasets[f'kendall_{lang}'] = En_Dev_Dataset(split_meta='dev', langs=[lang], kendall=True)
            en_dev_dataloaders[f'kendall_{lang}'] = DataLoader(en_dev_datasets[f'kendall_{lang}'], shuffle=False, batch_size=batch_size)

            en_dev_datasets[f'accuracy_{lang}'] = En_Dev_Dataset(split_meta='dev', langs=[lang], kendall=False)
            en_dev_dataloaders[f'accuracy_{lang}'] = DataLoader(en_dev_datasets[f'accuracy_{lang}'], shuffle=False, batch_size=batch_size)

            ### Baseline Accuracy
            correct_en_dev = 0
            
            for step, batch in enumerate(tqdm(en_dev_dataloaders[f'accuracy_{lang}'] )):
                # Data
                en_dev_ranking, gold_ranking = batch

                # Accuracy
                correct_en_dev += (en_dev_ranking == gold_ranking).float().sum().item()   
            
            en_dev_baselines[f'accuracy_{lang}'] = 100 * correct_en_dev / len(en_dev_datasets[f'accuracy_{lang}'])

            # Aim track
            run.track(en_dev_baselines[f'accuracy_{lang}'], name=f'Accuracy', step=0, context={"subset": "dev", "lang": lang, 'selected_by': 'en-dev-f1'})
            run.track(en_dev_baselines[f'accuracy_{lang}'], name=f'Accuracy', step=args.lms_num_train_epochs-1, context={"subset": "dev", "lang": lang, 'selected_by': 'en-dev-f1'})


            ### Baseline Kendall
            en_dev_metric_arr = []
            dev_metric_arr = []

            for step, batch in enumerate(tqdm(en_dev_dataloaders[f'kendall_{lang}'])):
                # Data
                en_dev_metric, dev_metric = batch

                # make list 
                en_dev_metric = en_dev_metric.detach().cpu().numpy().tolist()
                dev_metric = dev_metric.detach().cpu().numpy().tolist()

                # append
                en_dev_metric_arr.extend(en_dev_metric)
                dev_metric_arr.extend(dev_metric) 
        
            en_dev_baselines[f'kendall_{lang}'] = kendall_rank_correlation(en_dev_metric_arr, dev_metric_arr)

            # Aim track
            run.track(en_dev_baselines[f'kendall_{lang}'], name=f'Kendall', step=0, context={"subset": "dev", "lang": lang, 'selected_by': 'en-dev-f1'})
            run.track(en_dev_baselines[f'kendall_{lang}'], name=f'Kendall', step=args.lms_num_train_epochs-1, context={"subset": "dev", "lang": lang, 'selected_by': 'en-dev-f1'})

        # Avg accuracy and kendall 
        accuracy_avg = sum(en_dev_baselines[f'accuracy_{lang}'] for lang in PIVOT_LANGS) / len(PIVOT_LANGS)
        kendall_score_avg = sum(en_dev_baselines[f'kendall_{lang}'] for lang in PIVOT_LANGS) / len(PIVOT_LANGS)

        # Aim track
        run.track(accuracy_avg, name=f'Accuracy', step=0, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'en-dev-f1'})
        run.track(accuracy_avg, name=f'Accuracy', step=args.lms_num_train_epochs-1, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'en-dev-f1'})
        run.track(kendall_score_avg, name=f'Kendall', step=0, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'en-dev-f1'})
        run.track(kendall_score_avg, name=f'Kendall', step=args.lms_num_train_epochs-1, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'en-dev-f1'})
        
        logger.info("===================== En baseline evaluations ===================== ")
        logger.info(en_dev_baselines) 
        print("average Accuracy", accuracy_avg)
        print("average Kendall", kendall_score_avg)     


    def train_LMS(batch_size, learning_rate, target_lang, LANGS, train=False):
        logger.info(f"======================================== Target lang ====== {target_lang} =======================================") 

        # Model
        seed_everything(args.lms_model_seed)
        lms_model = RankNet(input_size=768, hidden_size=args.lms_hidden_size, embedding=embedding, emb_dim=512, activation='relu')
        lms_model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(lms_model.parameters(), lr=learning_rate)

        # Datasets
        #############################################  LMS  ############################################   
        lms_datasets = {}
        lms_dataloaders = {}
        PIVOT_LANGS = [l for l in LANGS if l != target_lang]

        # meta-train split models with all languages without target-lang (pivot languages)
        lms_dataset_train = LMS_Dataset(split_meta='train', langs=PIVOT_LANGS, kendall=False)
        lms_dataloader_train = DataLoader(lms_dataset_train, shuffle=True, batch_size=batch_size)

        # meta-dev split models with the one language
        for lang in tqdm(LANGS):
            lms_datasets[f'kendall_{lang}'] = LMS_Dataset(split_meta='dev', langs=[lang], kendall=True)
            lms_dataloaders[f'kendall_{lang}'] = DataLoader(lms_datasets[f'kendall_{lang}'], shuffle=False, batch_size=batch_size)

            lms_datasets[f'accuracy_{lang}'] = LMS_Dataset(split_meta='dev', langs=[lang], kendall=False)
            lms_dataloaders[f'accuracy_{lang}'] = DataLoader(lms_datasets[f'accuracy_{lang}'], shuffle=False, batch_size=batch_size)

        
        #############################################  En Dev  ############################################
        evaluate_en_dev(LANGS, PIVOT_LANGS, batch_size)
    
        iter_num = len(lms_dataloader_train)
        best_score = -1
        best_model = None

        for epoch in tqdm(range(args.lms_num_train_epochs)):
            logger.info(f"Epoch {epoch}")
            lms_model.train() 
            correct = 0
            loss_avg = 0

            for step, batch in enumerate(tqdm(lms_dataloader_train)):
                # Data
                lms_model.train() 
                cls_1, cls_2, gold_rankings, lang = batch
                cls_1 = cls_1.to(device)
                cls_2 = cls_2.to(device)
                gold_rankings = gold_rankings.to(device)
                lang = lang.to(device)

                # Forward
                loss, outputs = lms_model(cls_1, cls_2, gold_rankings, lang)

                # Loss
                loss_avg += loss.item()

                # Accuracy
                outputs = (outputs > 0.5).float()
                correct += (outputs == gold_rankings).float().sum().item()

                # Aim track
                track_step = epoch * iter_num + step
                run.track(loss.item(), name='Loss', step=track_step, context={"subset": "train"})

                if epoch != 0:
                    # Backprop and update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Avg accuracy and kendall 
            accuracy_avg = 100 * correct / len(lms_dataset_train)  

            # Aim track
            run.track(accuracy_avg, name=f'Accuracy', step=epoch, context={"subset": "train", "lang": PIVOT_LANGS, 'selected_by': 'lms'})

            
            lms_model.eval()

            ######################################## LMS Evaluations ####################################################    
            lms_evals = {}

            for lang in tqdm(LANGS):
                ### Accuracy
                correct_dev = 0

                for step, batch in enumerate(tqdm(lms_dataloaders[f'accuracy_{lang}'])):
                    # Data
                    cls_1, cls_2, gold_rankings, lang_tensor = batch
                    cls_1 = cls_1.to(device)
                    cls_2 = cls_2.to(device)
                    lang_tensor = lang_tensor.to(device)
                    gold_rankings = gold_rankings.to(device)

                    # Forward
                    _, outputs = lms_model(cls_1, cls_2, gold_rankings, lang_tensor)

                    # Accuracy
                    outputs = (outputs > 0.5).float()
                    correct_dev += (outputs == gold_rankings).float().sum().item()

                lms_evals[f'accuracy_{lang}'] = 100 * correct_dev / len(lms_datasets[f'accuracy_{lang}'])    

                # Aim track
                run.track(lms_evals[f'accuracy_{lang}'], name=f'Accuracy', step=epoch, context={"subset": "dev", "lang": lang, 'selected_by': 'lms'})


                ### Kendall
                outputs_arr = []
                dev_arr = []

                for step, batch in enumerate(tqdm(lms_dataloaders[f'kendall_{lang}'])):
                    # Data
                    cls, dev, lang_tensor = batch
                    cls = cls.to(device)
                    lang_tensor = lang_tensor.to(device)
                    dev = dev.to(device)

                    # Predict
                    with torch.no_grad():
                        outputs = lms_model.evaluate(cls, lang_tensor)

                    outputs = outputs.detach().cpu().numpy().tolist()
                    dev = dev.detach().cpu().numpy().tolist()

                    outputs_arr.extend(outputs)
                    dev_arr.extend(dev) 

                
                lms_evals[f'kendall_{lang}']  = kendall_rank_correlation(dev_arr, outputs_arr)

                # Aim track
                run.track(lms_evals[f'kendall_{lang}'], name=f'Kendall', step=epoch, context={"subset": "dev", "lang": lang, 'selected_by': 'lms'})

            # Avg accuracy and kendall 
            accuracy_avg = sum(lms_evals[f'accuracy_{lang}'] for lang in PIVOT_LANGS) / len(PIVOT_LANGS)
            kendall_score_avg = sum(lms_evals[f'kendall_{lang}'] for lang in PIVOT_LANGS) / len(PIVOT_LANGS)

            # Aim track
            run.track(accuracy_avg, name=f'Accuracy', step=epoch, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'lms'})
            run.track(kendall_score_avg, name=f'Kendall', step=epoch, context={"subset": "dev", "lang": PIVOT_LANGS, 'selected_by': 'lms'})

            logger.info("===================== LMS evaluations ===================== ")
            logger.info(lms_evals)
            print("average Accuracy", accuracy_avg)
            print("average Kendall", kendall_score_avg)

            # best score across epochs
            if train:
                if kendall_score_avg > best_score:
                    best_score = kendall_score_avg
                    best_model = lms_model

            else:    
                if lms_evals[f'kendall_{target_lang}'] > best_score:
                    best_score = lms_evals[f'kendall_{target_lang}']
                    best_model = lms_model
                

        return best_score, best_model        


    if args.do_lms_hyperparam_search:
        logger.info("======================================= Hyperparameter search =======================================")
        logger.info(f"Hyperparameters")
        logger.info(f'Target language {args.lms_target_lang}')
        logger.info(f'Learning rate {args.lms_learning_rate}')
        logger.info(f'Batch size {args.lms_batch_size}')
        logger.info(f'Hidden size {args.lms_hidden_size}')
        logger.info(f'Train epochs {args.lms_num_train_epochs}')
        logger.info(f'LMS Model seed {args.lms_model_seed}')

        # Output folder for LMS models
        LMS_models_dir = os.path.join(OUTPUT_DIR, 'LMS_models')
        os.makedirs(LMS_models_dir, exist_ok=True)    

        ##################### Lang2vec embeddings #####################
        lang_dic = {'deu': 0, 'spa': 1, 'nld': 2, 'zho': 3}

        embedding = [[] for _ in range(len(lang_dic))]
        emb = np.load("lang_vecs.npy", allow_pickle=True, encoding='latin1')

        for k, v in lang_dic.items():
            embedding[v] = emb.item()['optsrc' + k]

        embedding = torch.FloatTensor(np.array(embedding))
        ###############################################################

        # For cross-validation we choose one lang besides target language as fake target language
        FAKE_TARGET_LANGS = [l for l in TARGET_LANGS if l != args.lms_target_lang]

        cross_val_score = 0

        for fake_target_lang in tqdm(FAKE_TARGET_LANGS):
            run["hparams", 'lms_fake_target_lang'] = fake_target_lang

            # return the best score for each fake target language through epochs
            best_score, _ = train_LMS(args.lms_batch_size, args.lms_learning_rate, fake_target_lang, FAKE_TARGET_LANGS)

            # aggregate best score for the unseen target language (args.lms_target_lang)
            cross_val_score += best_score 
        
        cross_val_score = cross_val_score / len(FAKE_TARGET_LANGS)

        logger.info(f'Cross validation {args.lms_target_lang} language score {cross_val_score} with bs={args.lms_batch_size} and lr={args.lms_learning_rate}')

        csv_row = [args.lms_batch_size, args.lms_learning_rate, cross_val_score]

        with open(LMS_models_dir + f'/meta_dev_scores_{args.lms_target_lang}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)  

        
        logger.info("======================================= Hyperparam choose Done! =======================================")


    if args.do_lms_train:
        logger.info("======================================= Train LMS =======================================")
        logger.info(f"Hyperparameters")
        logger.info(f'Target language {args.lms_target_lang}')
        logger.info(f'Learning rate {args.lms_learning_rate}')
        logger.info(f'Batch size {args.lms_batch_size}')
        logger.info(f'Hidden size {args.lms_hidden_size}')
        logger.info(f'Train epochs {args.lms_num_train_epochs}')
        logger.info(f'LMS Model seed {args.lms_model_seed}')

        # Output folder for LMS models
        LMS_models_dir = os.path.join(OUTPUT_DIR, 'LMS_models')
        os.makedirs(LMS_models_dir, exist_ok=True)

        ##################### Lang2vec embeddings #####################
        lang_dic = {'deu': 0, 'spa': 1, 'nld': 2, 'zho': 3}

        embedding = [[] for _ in range(len(lang_dic))]
        emb = np.load("lang_vecs.npy", allow_pickle=True, encoding='latin1')
        for k, v in lang_dic.items():
            embedding[v] = emb.item()['optsrc' + k]

        embedding = torch.FloatTensor(np.array(embedding))
        ###############################################################

        target_langs = [args.lms_target_lang] if args.lms_target_lang else TARGET_LANGS

        for target_lang in tqdm(target_langs):
            if args.lms_batch_size is not None and args.lms_learning_rate is not None:
                lms_batch_size = args.lms_batch_size
                lms_learning_rate = args.lms_learning_rate
            else:
                # Getting the best hyperparameters
                df = pd.read_csv(LMS_models_dir + f'/meta_dev_scores_{target_lang}.csv', names=['batch_size', 'learning_rate', 'cv_score'])
                
                # get the first row even if there are more with the same score
                df_max = df[df['cv_score'] == df['cv_score'].max()].iloc[:1]
                print(df_max)

                lms_batch_size = df_max['batch_size'].item()
                lms_learning_rate = df_max['learning_rate'].item()
 
            # return the best score and model for each target language through epochs
            best_score, best_model = train_LMS(lms_batch_size, lms_learning_rate, target_lang, TARGET_LANGS, train=True)
            logger.info(f"Target {args.lms_target_lang} Dev split best LMS score {best_score}")

            torch.save(best_model.state_dict(), LMS_models_dir + f'/model_{target_lang}.pth')

        logger.info("======================================= Train LMS Done! =======================================")


    if args.do_lms_predict:
        logger.info("======================================= Predict score with LMS =======================================")  
        # Output folder for LMS models
        LMS_models_dir = os.path.join(OUTPUT_DIR, 'LMS_models')
        
        ##################### Lang2vec embeddings #####################
        lang_dic = {'deu': 0, 'spa': 1, 'nld': 2, 'zho': 3}

        embedding = [[] for _ in range(len(lang_dic))]
        emb = np.load("lang_vecs.npy", allow_pickle=True, encoding='latin1')
        for k, v in lang_dic.items():
            embedding[v] = emb.item()['optsrc' + k]

        embedding = torch.FloatTensor(np.array(embedding))
        ###############################################################

        target_lang = args.lms_target_lang
        lms_evals = {}
        
        for target_lang in [target_lang]:
        # for target_lang in TARGET_LANGS:
            print(f"======================================= Target lang ====== {target_lang} =======================================")
            # Getting the best hyperparameters
            df = pd.read_csv(LMS_models_dir + f'/meta_dev_scores_{target_lang}.csv', names=['batch_size', 'learning_rate', 'cv_score'])
            df_max = df[df['cv_score'] == df['cv_score'].max()].iloc[:1]
            print(df_max)

            lms_batch_size = df_max['batch_size'].item()

            # Model
            seed_everything(args.lms_model_seed)
            lms_model_path = LMS_models_dir + f'/model_{target_lang}.pth'
            lms_model = RankNet(input_size=512, hidden_size=args.lms_hidden_size, embedding=embedding, emb_dim=512, activation='relu')
            lms_model.to(device)

            checkpoint = torch.load(lms_model_path, map_location=device)
            lms_model.load_state_dict(checkpoint)
            lms_model.eval()

            # Dataset
            lms_dataset_test = LMS_Dataset(split_meta='test', langs=[target_lang], kendall=True)
            lms_test_dataloader = DataLoader(lms_dataset_test, shuffle=False, batch_size=lms_batch_size)

            ### Kendall
            outputs_arr = []
            dev_arr = []
            model_names = []

            for step, batch in enumerate(tqdm(lms_test_dataloader)):
                # Data
                cls, dev_metric, lang_tensor, model_name = batch
                cls = cls.to(device)
                lang_tensor = lang_tensor.to(device)
                dev_metric = dev_metric.to(device)

                # Predict
                with torch.no_grad():
                    outputs = lms_model.evaluate(cls, lang_tensor)

                outputs = outputs.detach().cpu().numpy().tolist()
                dev_metric = dev_metric.detach().cpu().numpy().tolist()

                outputs_arr.extend(outputs)
                dev_arr.extend(dev_metric) 
                model_names += list(model_name)
                
            lms_evals[f'kendall_{target_lang}']  = kendall_rank_correlation(dev_arr, outputs_arr)
            logger.info(f"Target {args.lms_target_lang} Test split LMS score {lms_evals[f'kendall_{target_lang}']}")

            # Aim track
            run.track(lms_evals[f'kendall_{target_lang}'], name=f'Kendall', step=0, context={"subset": "test", "lang": target_lang, 'selected_by': 'lms'})

            scores_df = pd.DataFrame({'model_name': model_names, 'score': outputs_arr, f'test_{METRIC}': dev_arr})
            scores_df = scores_df.sort_values('score')
            scores_df.to_csv(LMS_models_dir + f'/meta_test_scores_{target_lang}.csv', index=False)


        logger.info("======================================= Predict score with LMS Done! =======================================")     


if __name__ == "__main__":
    main()