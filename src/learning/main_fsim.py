"""
For the fine-grained similarity models:
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
import wandb
import argparse, os, sys
import codecs, pprint, json
import datetime
import logging
import torch
import torch.multiprocessing as torch_mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from src.learning import batchers, trainer
from src.learning.facetid_models import disent_models
# Copying from: https://discuss.pytorch.org/t/why-do-we-have-to-create-logger-in-process-for-correct-logging-in-ddp/102164/3
# Had double printing errors, solution finagled from:
# https://stackoverflow.com/q/6729268/3262406
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

PROJECT_NAME = 'aspire-experiments'
def init_wandb(all_hparams, run_name):
    wandb.login()
    wandb.init(project=PROJECT_NAME, config=all_hparams, name=run_name, tags=['training', all_hparams['model_name']])
    # wandb.init(project=PROJECT_NAME, config=all_hparams, name=run_name,tags=['training'])
    print(f"Initialized Weights & Biases run: {wandb.run.name}")  # Debug print

def setup_ddp(rank, world_size,master_port='12355'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port #'29500'

    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(0, 3600)
    )


def cleanup_ddp():
    dist.destroy_process_group()


def ddp_train_model(process_rank, args):
    """
    Read the int training and dev data, initialize and train the model.
    :param model_name: string; says which model to use.
    :param data_path: string; path to the directory with unshuffled data
        and the test and dev json files.
    :param config_path: string; path to the directory json config for model
        and trainer.
    :param run_path: string; path for shuffled training data for run and
        to which results and model gets saved.
    :param cl_args: argparse command line object.
    :return: None.
    """
    print(f"Process {process_rank} started with {args.num_gpus} GPUs")
    cl_args = args
    model_name, data_path, config_path, run_path = \
        cl_args.model_name, cl_args.data_path, cl_args.config_path, cl_args.run_path
    run_name = os.path.basename(run_path)
    # Load label maps and configs.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)
    setup_ddp(rank=process_rank, world_size=cl_args.num_gpus, master_port=cl_args.master_port)

    # Setup logging and experiment tracking.
    if process_rank == 0:
        # Print the called script and its args to the log.
        if wandb.run is None:  # Ensure initialization before logging
            init_wandb(all_hparams,run_name)
        logger = get_logger()
        print(' '.join(sys.argv))
        # Unpack hyperparameter settings.
        print('All hyperparams:')
        print(pprint.pformat(all_hparams))
        # Save hyperparams to disk from a single process.
        os.makedirs(run_path, exist_ok=True)  # Creates the directory if it doesn't exist
        run_info = {'all_hparams': all_hparams}
        with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
            json.dump(run_info, fp)
    else:
        os.environ["WANDB_MODE"] = "offline"
        logger = None

    # Initialize model.
    if model_name in {'cospecter'}:
        model = disent_models.MySPECTER(model_hparams=all_hparams)
    elif model_name in {'miswordbienc'}:
        model = disent_models.WordSentAlignBiEnc(model_hparams=all_hparams)
    elif model_name in {'sbalisentbienc'}:
        model = disent_models.WordSentAbsSupAlignBiEnc(model_hparams=all_hparams)
    elif model_name in {'miswordpolyenc'}:
        model = disent_models.WordSentAlignPolyEnc(model_hparams=all_hparams)
    else:
        sys.exit(1)
    # Model class internal logic uses the names at times so set this here so it
    # is backward compatible.
    model.model_name = model_name
    if process_rank == 0:
        # Save an untrained model version.
        trainer.generic_save_function_ddp(model=model, save_path=run_path, model_suffix='init')
        print(model)

    # Move model to the GPU.
    torch.cuda.set_device(process_rank)
    if torch.cuda.is_available():
        model.cuda(process_rank)
        if process_rank == 0: print('Running on GPU.')
    model = DistributedDataParallel(model, device_ids=[process_rank], find_unused_parameters=True)

    # Initialize the trainer.
    if model_name in ['cospecter']:
        batcher_cls = batchers.AbsTripleBatcher
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    elif model_name in ['miswordbienc', 'miswordpolyenc']:
        batcher_cls = batchers.AbsSentTokBatcher
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    elif model_name in ['sbalisentbienc']:
        batcher_cls = batchers.AbsSentTokBatcherPreAlign
        # Use the context based alignment by default.
        batcher_cls.align_type = all_hparams.get('align_type', 'cc_align')
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    else:
        sys.exit(1)

    if model_name in ['cospecter', 'miswordbienc',
                      'miswordpolyenc', 'sbalisentbienc']:
        model_trainer = trainer.BasicRankingTrainerDDP(
            logger=logger, process_rank=process_rank, num_gpus=cl_args.num_gpus,
            model=model, batcher=batcher_cls, data_path=data_path, model_path=run_path,
            early_stop=True, verbose=True,dev_score='loss', train_hparams=all_hparams)
        model_trainer.save_function = trainer.generic_save_function_ddp
    # Train and save the best model to model_path.
    model_trainer.train()
    print(f"Process {process_rank} finished training.")
    cleanup_ddp()
    if process_rank == 0:
        save_and_push_best_model(model_name, all_hparams, run_path, cl_args.repo_name)


def train_model(model_name, data_path, config_path, run_path, cl_args):
    """
    Read the int training and dev data, initialize and train the model.
    :param model_name: string; says which model to use.
    :param data_path: string; path to the directory with unshuffled data
        and the test and dev json files.
    :param config_path: string; path to the directory json config for model
        and trainer.
    :param run_path: string; path for shuffled training data for run and
        to which results and model gets saved.
    :param cl_args: argparse command line object.
    :return: None.
    """
    run_name = os.path.basename(run_path)
    # Load label maps and configs.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)
    
    # Unpack hyperparameter settings.
    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    init_wandb(all_hparams, run_name)
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    os.makedirs(run_path, exist_ok=True)  # Creates the directory if it doesn't exist
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
    
    # Initialize model.
    if model_name in {'cospecter'}:
        model = disent_models.MySPECTER(model_hparams=all_hparams)
        # Save an untrained model version.
        trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    elif model_name in {'miswordbienc'}:
        model = disent_models.WordSentAlignBiEnc(model_hparams=all_hparams)
        # Save an untrained model version.
        # trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    elif model_name in {'sbalisentbienc'}:
        model = disent_models.WordSentAbsSupAlignBiEnc(model_hparams=all_hparams)
        trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    # Model class internal logic uses the names at times so set this here so it
    # is backward compatible.
    model.model_name = model_name
    logging.info(model)
    
    # Move model to the GPU.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    
    # Initialize the trainer.
    if model_name in ['cospecter']:
        batcher_cls = batchers.AbsTripleBatcher
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    elif model_name in ['miswordbienc']:
        batcher_cls = batchers.AbsSentTokBatcher
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    elif model_name in ['sbalisentbienc']:
        batcher_cls = batchers.AbsSentTokBatcherPreAlign
        # Use the context based alignment by default.
        batcher_cls.align_type = all_hparams.get('align_type', 'cc_align')
        batcher_cls.bert_config_str = all_hparams['base-pt-layer']
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    
    if model_name in ['cospecter', 'miswordbienc','sbalisentbienc']:
        model_trainer = trainer.BasicRankingTrainer(model=model, batcher=batcher_cls, data_path=data_path, model_path=run_path,
                                                    early_stop=True, dev_score='loss', train_hparams=all_hparams)
        model_trainer.save_function = trainer.generic_save_function
    # Train and save the best model to model_path.
    model_trainer.train()

    save_and_push_best_model(model_name, all_hparams, run_path, cl_args.repo_name)

def save_and_push_best_model(model_name,all_hparams, trained_model_path, repo_name):
    # Initialize model.
    if model_name in {'cospecter'}:
        model = disent_models.MySPECTER(model_hparams=all_hparams)
    elif model_name in {'miswordbienc'}:
        model = disent_models.WordSentAlignBiEnc(model_hparams=all_hparams)
    elif model_name in {'sbalisentbienc'}:
        model = disent_models.WordSentAbsSupAlignBiEnc(model_hparams=all_hparams)
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    best_model_path = os.path.join(trained_model_path, "model_cur_best.pt")
    model.load_state_dict(torch.load(best_model_path))

    # Save model & tokenizer
    repo_path = os.path.join(trained_model_path,repo_name)
    model.bert_encoder.save_pretrained(repo_path)
    tokenizer = AutoTokenizer.from_pretrained(all_hparams["base-pt-layer"])  # Replace with your tokenizer
    tokenizer.save_pretrained(repo_path)

    # Push to Hugging Face Hub
    model.bert_encoder.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print(f"✅ Model & tokenizer pushed to: https://huggingface.co/idopinto/{repo_name}")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['cospecter','miswordbienc',
                                     'miswordpolyenc', 'sbalisentbienc'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['s2orcscidocs', 's2orccompsci', 's2orcbiomed', 'relish', 'treccovid'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--data_path', required=True,
                            help='Path to the jsonl dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    train_args.add_argument('--repo_name',
                            help='name of the huggingface repo to push the model and tokenizer to after training.')
    train_args.add_argument('--master_port',default="29500", help='master port to use in ddp')
    # train_args.add_argument('--log_fname',
    #                         help='Path to directory to save log files.')
    cl_args = parser.parse_args()
    # If a log file was passed then write to it.

    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")
    print(f"NCCL Available: {torch.distributed.is_nccl_available()}")  # Should return True
    print(f"Training {cl_args.model_name} on {cl_args.dataset} dataset with {cl_args.num_gpus} GPUs.")
    print(f"Huggingface repo: {cl_args.repo_name}")
    if cl_args.subcommand == 'train_model':
        if cl_args.num_gpus > 1:
            torch_mp.spawn(ddp_train_model, nprocs=cl_args.num_gpus, args=(cl_args,))
        else:
            train_model(model_name=cl_args.model_name, data_path=cl_args.data_path,
                        run_path=cl_args.run_path, config_path=cl_args.config_path, cl_args=cl_args)


if __name__ == '__main__':
    main()
