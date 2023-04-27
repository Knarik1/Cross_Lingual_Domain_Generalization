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
import random
from pathlib import Path
import numpy as np
import pandas as pd
from aim import Run

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ClassLabel, DatasetDict, load_dataset, load_metric, Dataset, concatenate_datasets
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
    parser.add_argument('--saved_epoch', type=int, default=2, help="Which epochs of the model to save.")
    parser.add_argument("--do_fine_tune", action="store_true", help="Fine tune")
    parser.add_argument("--do_train_classifier", action="store_true", help="Train Classifier.")
    parser.add_argument('--warmup_proportion', type=float, help="Fine tune warm up.")
    parser.add_argument('--train_langs', nargs='+', default=["en", "de"], help="Train languages.")
    parser.add_argument('--dev_lang', help="Validation language.")
    parser.add_argument('--test_lang', help="Test language.")
    parser.add_argument('--pre_trained_model', type=str, help="Pre-trained bert.")
    parser.add_argument('--classifier_model', type=str, help="Classifier.")

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
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
    results = metric.compute(zero_division=1,  )
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


def evaluate(model, dataloader, args, accelerator, label_list):
    model.eval()
    metric = load_metric("seqeval")
    avg_dev_loss = 0

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            dev_loss = outputs.loss
            dev_loss = dev_loss / args.gradient_accumulation_steps
            avg_dev_loss += dev_loss.item()

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

    return eval_metric, avg_dev_loss / len(dataloader)


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
    METRIC = 'f1'
    TRAIN_LANGS = args.train_langs
    TRAIN_LANGS_STR = "_".join(TRAIN_LANGS)
    LANGS = ['en', 'de', 'es', 'nl', 'zh']

    # Output folder for fine-tuned models
    models_dir = os.path.join(OUTPUT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)

    classifiers_dir = os.path.join(OUTPUT_DIR, 'classifiers')
    os.makedirs(classifiers_dir, exist_ok=True)

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
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
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

    # we've chosen these numbers according to min size of each split
    train_size = 8323

    lang_datasets = {
        'en_train':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'train.txt'))).select(range(train_size)),
        'en_dev':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'valid.txt'))),
        'en_test':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'test.txt'))),
        'de_train':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'deu.train.txt'))).select(range(train_size)),
        'de_dev':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'deu.testa.txt'))),
        'de_test':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'deu.testb.txt'))),
        'es_train':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'esp.train.txt'))).select(range(train_size)),
        'es_dev':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'esp.testa.txt'))),
        'es_test':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'esp.testb.txt'))),
        'nl_train':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'ned.train.txt'))).select(range(train_size)),
        'nl_dev':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'ned.testa.txt'))),
        'nl_test':  Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'ned.testb.txt'))),
    }

    # zh
    zh_dataset = Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'msra_train_bio.txt')))
    # for zh lang there is no seperate dev data, so we sample it from the train
    rand_14499_idxs = np.random.randint(len(zh_dataset), size=14499)
    zh_data = {
        'zh_train': zh_dataset.select(rand_14499_idxs[:train_size]),
        'zh_dev': zh_dataset.select(rand_14499_idxs[train_size:]),
        'zh_test': Dataset.from_dict(read_ner_data(os.path.join(DATA_DIR, 'msra_test_bio.txt'))),
    }

    lang_datasets.update(zh_data)


    ############################################# Load Datasets #############################################

    raw_datasets = DatasetDict()
    raw_datasets.update(lang_datasets)

    if len(TRAIN_LANGS) > 1:
        raw_datasets[f'{TRAIN_LANGS_STR}_train'] = concatenate_datasets([lang_datasets[f'{d}_train'] for d in TRAIN_LANGS])
        raw_datasets[f'{TRAIN_LANGS_STR}_dev'] = concatenate_datasets([lang_datasets[f'{d}_dev'] for d in TRAIN_LANGS])
        raw_datasets[f'{TRAIN_LANGS_STR}_test'] = concatenate_datasets([lang_datasets[f'{d}_test'] for d in TRAIN_LANGS])


    print(raw_datasets)

    ###########################################################################################################

    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    column_names = lang_datasets["en_train"].column_names
    features = lang_datasets["en_train"].features

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

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(lang_datasets["en_train"][label_column_name])
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

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    print(raw_datasets)

    def tokenize_and_align_labels(examples, lang=None):
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
                # We set the label for the first token of each word.
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
            remove_columns=raw_datasets["en_train"].column_names,
            desc="Running tokenizer on dataset",
            )  

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

    for k in processed_raw_datasets:
        shuffle = True if 'train' in k else False   
        dataloaders[k] = DataLoader(processed_raw_datasets[k], shuffle=shuffle, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    
    for i_dataloader in dataloaders:
        dataloaders[i_dataloader] = accelerator.prepare(dataloaders[i_dataloader])  

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    def train(model, optimizer, classifier=True):
        # Prepare everything with our `accelerator`.
        model, optimizer = accelerator.prepare(model, optimizer)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(dataloaders[f'{TRAIN_LANGS_STR}_train']) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        num_train_optimization_steps = int(len(raw_datasets[f'{TRAIN_LANGS_STR}_train']) / args.per_device_train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs    
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
        logger.info(f"  Num train examples = {len(raw_datasets[f'{TRAIN_LANGS_STR}_train'])}")
        logger.info(f"  Num epochs = {args.num_train_epochs}")
        logger.info(f"  Seed = {args.seed}")
        logger.info(f"  Learning rate = {args.learning_rate}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Warm up steps = {warm_up_steps}")
        logger.info(f"  Train langs = {TRAIN_LANGS_STR}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        iter_num = len(dataloaders[f'{TRAIN_LANGS_STR}_train'])

        for epoch in range(args.num_train_epochs + 1):
            model.train()
            avg_train_loss = 0
            
            print("Training...")
            for step, batch in enumerate(dataloaders[f'{TRAIN_LANGS_STR}_train']):
                model.train()
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
                avg_train_loss += loss.item()
                
                # Track train loss
                track_step = epoch * iter_num + step
                run.track(loss.item(), name="Loss", step=track_step, context={"subset": "train", "domain": TRAIN_LANGS_STR})

            # Track all test data F1
            print("Evaluating...")

            for k in tqdm(dataloaders):
                if 'test' in k:
                    domain_name = k.replace("_test", "")
                    eval_metric_test, _, = evaluate(model, dataloaders[k], args, accelerator, label_list)
                    run.track(eval_metric_test[METRIC], name='F1', step=epoch, context={"subset": "test", "domain": domain_name})

            accelerator.print(f"Epoch {epoch}: Train loss {avg_train_loss / iter_num}")

            if epoch%10==0 and epoch!=0:
                print(f"======================================= Save model =======================================")
                accelerator.wait_for_everyone()
                if classifier:
                    saved_model_name = f'{args.pre_trained_model}_{TRAIN_LANGS_STR}_epoch_{epoch}'
                    checkpoint_path = os.path.join(classifiers_dir, saved_model_name)
                else:
                    saved_model_name = f'domain_{TRAIN_LANGS_STR}_epoch_{epoch}'
                    checkpoint_path = os.path.join(models_dir, saved_model_name)

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_path, save_function=accelerator.save)    


    if args.do_fine_tune:
        logger.info("======================================= Fine Tuning =======================================")

        # Model
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list))

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

        # Train
        train(model, optimizer, classifier=False)
       
        logger.info("======================================= Fine Tuning Done! =======================================")  


    if args.do_train_classifier:
        logger.info("======================================= Classifier Training =======================================")
        
        # Model                      
        # load config from base to init classifier layer (not to reuse it)
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_list)) 

        checkpoint_path = os.path.join(models_dir, args.pre_trained_model, 'pytorch_model.bin')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)  

        # Optimizer
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate)

        # Train
        train(model, optimizer, classifier=True) 

        logger.info("======================================= Classifier Training Done! =======================================")    

if __name__ == "__main__":
    main()