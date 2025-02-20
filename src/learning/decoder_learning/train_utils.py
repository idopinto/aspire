from __future__ import print_function

import codecs
import json
import os
import logging
import copy
import pprint
from datetime import datetime, timedelta
from pathlib import Path

import torch
import numpy as np
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import transformers
from src.learning import batchers
from src.learning.facetid_models import disent_models
from src.learning.decoder_learning import decoder_models, decoder_batchers
from transformers import AutoTokenizer, AutoModel
import sys
import wandb


WANDB_PROJECT_NAME = 'aspire-experiments'
ROOT = Path('/cs/labs/tomhope/idopinto12/aspire')
TRAIN_DATA_DIR = ROOT / 'datasets' / 'train'
CONFIG_DIR = ROOT / 'my_configs'
RUN_DIR = ROOT / 'runs' / 'decoder_models'
RUN_DIR.mkdir(parents=True, exist_ok=True)



def conditional_log(logger, process_rank, message):
    """
    Helper to log only from one process when using DDP.
    -- logger is entirely unused. Dint seem to work when used in conjuncton with cometml.
    """
    if process_rank == 0:
        logger.info(message)

def calculate_num_batches(train_hparams,num_epochs, num_gpus=1):
    """
    Calculate the number of samples per GPU, batches per epoch, and total iterations.

    :param train_hparams: Hyperparameters for the training process containing:
        - 'batch_size': Size of each training batch.
        - 'train_size': Total number of training examples.
    :type train_hparams: dict
    :param num_epochs: Total number of epochs for training.
    :type num_epochs: int
    :param num_gpus: Number of GPUs used for training (default: 1).
    :type num_gpus: int, optional

    :return: A tuple containing:
        - num_samples_per_gpu: Number of training examples per GPU.
        - batches_per_epoch: Number of batches in one epoch.
        - total_iterations: Total iterations over all epochs.
    :rtype: tuple
    """
    batch_size = train_hparams['batch_size']
    train_size = train_hparams['train_size']
    num_samples_per_gpu = train_size // num_gpus
    batches_per_epoch = int(np.ceil(num_samples_per_gpu / batch_size)) if num_samples_per_gpu > batch_size else 1
    total_iterations = num_epochs * batches_per_epoch
    return num_samples_per_gpu, batches_per_epoch, total_iterations

def get_optimizer(model, update_rule, learning_rate):
    # Initialize optimizer.
    if update_rule == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer
    elif update_rule == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
        return optimizer
    else:
        raise ValueError(f'Unknown update rule: {update_rule}')

def get_lr_scheduler(optimizer, lr_decay_method,train_hparams, num_batches, num_gpus=1):
    """
    Creates a learning rate scheduler based on the specified decay method.

    :param optimizer: Optimizer for which the scheduler is created.
    :param lr_decay_method: Decay method ('exponential', 'warmuplin', 'warmupcosine').
    :param train_hparams: Dictionary with relevant hyperparameters:
        - 'decay_lr_by' for 'exponential'.
        - 'num_warmup_steps' and 'num_epochs' for warmup methods.
    :param num_batches: Number of batches per epoch.
    :param num_gpus: Number of GPUs used in training (default: 1).
    :return: Configured learning rate scheduler.
    :rtype: torch.optim.lr_scheduler._LRScheduler or transformers.get_scheduler
    """
    if lr_decay_method == 'exponential':
        decay_lr_by = train_hparams['decay_lr_by']
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=decay_lr_by)
        return scheduler
    elif lr_decay_method == 'warmuplin':
        num_warmup_steps = train_hparams['num_warmup_steps'] // num_gpus
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 # Total number of training batches.
                                                                 num_training_steps=train_hparams['num_epochs'] * num_batches)
        return scheduler
    elif lr_decay_method == 'warmupcosine':
        num_warmup_steps = train_hparams['num_warmup_steps'] // num_gpus
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=train_hparams['num_epochs'] * num_batches)
        return scheduler
    else:
        raise ValueError(f'Unknown lr_decay_method: {train_hparams["lr_decay_method"]}')

def validate_accumulate_gradients(train_hparams):
    """
       Validates and calculates parameters for gradient accumulation.

       Gradient accumulation allows effective batch size enlargement by accumulating gradients
       over multiple mini-batches before updating model parameters.

       :param train_hparams: Dict containing training hyperparameters:
           - 'batch_size': Required, positive integer.
           - 'accumulated_batch_size': Optional, positive integer greater than and a multiple of
             'batch_size'. If 0 or -1, gradient accumulation is disabled.
       :type train_hparams: dict

       :return: Tuple (accumulate_gradients, accumulated_batch_size, update_params_every):
           - accumulate_gradients: Whether gradient accumulation is enabled (bool).
           - accumulated_batch_size: Effective accumulated batch size (int).
           - update_params_every: Steps before updating parameters (int).
       :rtype: tuple
    """
    batch_size = train_hparams['batch_size']
    accumulated_batch_size = train_hparams.get("accumulated_batch_size", -1)
    if accumulated_batch_size > 0:
        # It should be bigger and an exact multiple of the batch size.
        if accumulated_batch_size <= batch_size:
            raise ValueError("'accumulated_batch_size' must be greater than 'batch_size'.")
        if accumulated_batch_size % batch_size != 0:
            raise ValueError("'accumulated_batch_size' must be an exact multiple of 'batch_size'.")
        update_params_every = accumulated_batch_size // batch_size
        accumulate_gradients = True
    else:
        accumulate_gradients = False
        update_params_every = 1

    return accumulate_gradients, accumulated_batch_size, update_params_every


def save_and_push_best_model(model_name, all_hparams, trained_model_path, repo_name,tokenizer):
    try:
        is_lora = all_hparams.get('lora', False)
        # Create model
        model, _ = ModelFactory.create(model_name=model_name, hparams=all_hparams)

        # Load best checkpoint
        best_model_path = os.path.join(
            trained_model_path,
            "model_lora_cur_best" if is_lora else "model_cur_best.pt"
        )

        # Load weights
        if is_lora:
            print("Loading LoRA adapters...")
            # model.encoder.load_adapter(best_model_path)
            model.encoder.load_adapter(best_model_path, adapter_name="cur_best")  # Add adapter_name

            # model.load_state_dict(torch.load(best_model_path))

        else:
            print("Loading full model...")
            model.load_state_dict(torch.load(best_model_path))

        # Create repository path
        repo_path = os.path.join(trained_model_path, repo_name)
        os.makedirs(repo_path, exist_ok=True)  # Ensure directory exists

        # Save and push
        model.encoder.save_pretrained(repo_path)
        tokenizer.save_pretrained(repo_path)
        model.encoder.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)

        print(f"✅ Model & tokenizer pushed to: https://huggingface.co/idopinto/{repo_name}")

    except Exception as e:
        raise Exception(f"Error in saving and pushing model to Hugging Face Hub: {str(e)}")

def save_model(model, save_path, model_suffix="cur_best"):
    """
    Save LoRA fine-tuned adapters, handling both regular and DDP-trained models.
    :param model: LoRA fine-tuned model (could be wrapped in DDP)
    :param save_path: Directory where LoRA adapters should be saved
    :param model_suffix: Suffix for the saved model (e.g., 'best', 'latest')
    """
    os.makedirs(save_path, exist_ok=True)

    if isinstance(model, DDP):
        print("Model is wrapped in DDP. Unwrapping before saving...")
        model = model.module

    if isinstance(model.encoder, PeftModel):
        print("Saving LoRA fine-tuned adapters...")
        model.encoder.save_pretrained(os.path.join(save_path, f'model_lora_{model_suffix}'), adapter_name=f"{model_suffix}")
        print("Saving also full model...")
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{model_suffix}.pt"))
    else:
        print("Model is not a LoRA model. Saving full model instead.")
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{model_suffix}.pt"))

    print(f"Model saved at {save_path}")

def load_model(model, load_path, model_suffix):
    """
    Load LoRA fine-tuned adapters or full model state dict.
    :param model: Base model instance
    :param load_path: Directory where model/adapters are saved
    :param model_suffix: Suffix of the saved model (e.g., 'best', 'latest')
    :return: Loaded model
    """
    if isinstance(model, DDP):
        print("Model is wrapped in DDP. Unwrapping before loading...")
        model = model.module

    if isinstance(model, PeftModel):
        print("Loading LoRA fine-tuned adapters...")
        lora_path = os.path.join(load_path, f'model_lora_{model_suffix}.pt')
        model.load_adapter(lora_path)
    else:
        print("Loading full model state dict...")
        model_path = os.path.join(load_path, f"model_{model_suffix}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    print(f"Model loaded from {load_path}")
    return model


class ModelFactory:
    @staticmethod
    def create(model_name: str, hparams: dict):
        model_map = {
            'cocite-aspire-gte-qwen2-1.5b-instruct-biomed': (decoder_models.CoQwen, decoder_batchers.AbsTripleBatcher),
            # 'ts-aspire-gte-qwen2-1.5b-instruct-biomed': (decoder_models.TSQwen, decoder_batchers.AbsSentTokBatcherPreAlign),
            # 'ot-aspire-gte-qwen2-1.5b-instruct-biomed':(decoder_models.OTQwen, decoder_batchers.AbsSentTokBatcher)
        }

        if model_name not in model_map:
            raise ValueError(f'Unknown model: {model_name}')

        model_cls, batcher_cls = model_map[model_name]
        model = model_cls(model_hparams=hparams)
        model.model_name = model_name # for backward compatibility
        # batcher_cls.bert_config_str = hparams['base-pt-layer']
        # if 'gte-qwen2-' in model_name:
        #     batcher_cls.padding_side = 'left' # when using flash attention it's essential
        #     print("Padding side is left")
        if 'ts' in model_name:
            batcher_cls.align_type = hparams.get('align_type', 'cc_align')
        print(f"{model_name} loaded. Batcher: {batcher_cls.__name__}")
        return model, batcher_cls

def get_config(config_path):
    # Load hyperparameters.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)

    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    return all_hparams

def save_config(all_hparams, run_path):
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)


def init_wandb(all_hparams, run_name):
    wandb.login()
    wandb.init(project=WANDB_PROJECT_NAME, config=all_hparams, name=run_name, tags=['training'])
    print(f"Initialized Weights & Biases run: {wandb.run.name}")  # Debug print
    return

def get_logger():
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.pop()
    # Handlers.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter())
    logger.addHandler(
        handler
    )
    logger.setLevel(logging.INFO)

    return logger

def get_run_path(cl_args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_id = cl_args.run_id or f"run_{timestamp}"  # Default to timestamp if no run_id provided
    run_path = RUN_DIR / f"{cl_args.model_name}-{run_id}"
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def get_vram_usage():
    """ Get total VRAM usage in GB """
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        vram_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
        return vram_allocated, vram_cached
    else:
        return 0, 0  # If no CUDA device is available

def setup_logging(log_fname: str = None):
    """Set up logging configuration."""
    if log_fname is not None:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            filename=log_fname,
            filemode='w'  # Overwrite by default
        )
    else:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            stream=sys.stdout
        )
    # Log the full command-line call for reference
    logging.info(' '.join(sys.argv))

