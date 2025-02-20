import argparse
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from src.learning.decoder_learning import trainer, trainer_ddp, train_utils
from src.learning.decoder_learning.ddp_setup import setup_ddp, cleanup_ddp

from src.learning.decoder_learning.decoder_batchers import BatcherConfig
from src.learning.decoder_learning.train_utils import *
import wandb

def ddp_train_model(process_rank, cl_args):
    """
    Read the int training and dev data, initialize and train the model.
    """
    run_path = get_run_path(cl_args)
    run_name = run_path.name
    config_path = CONFIG_DIR / str(cl_args.config_path)
    all_hparams = get_config(config_path) # Load hyperparameters and configs
    model_name = all_hparams['model_name']
    setup_ddp(rank=process_rank, world_size=cl_args.num_gpus)
    try:
        if  process_rank > 0:
            os.environ["WANDB_MODE"] = "offline"
            logger = None
        elif process_rank == 0:
            init_wandb(all_hparams, run_name)
            logger = get_logger()
            # Save hyperparams to disk from a single process.
            save_config(all_hparams, run_path)

        model, batcher_cls = ModelFactory.create(model_name, all_hparams)
        model = model.to(process_rank)

        if process_rank == 0:
            # Save an untrained model version.
            # train_utils.save_model(model=model, save_path=run_path, model_suffix='init')
            logger.info(f"Process rank {process_rank},pid={os.getpid()},checkpoint: initial model saved")
        assert torch.cuda.current_device() == process_rank, "Incorrect GPU assignment"
        model = ddp(model, device_ids=[process_rank], find_unused_parameters=True)


        model_trainer = (trainer_ddp.BasicRankingTrainerDDP(logger=logger,
                                                       process_rank=process_rank,
                                                       num_gpus=cl_args.num_gpus,
                                                       model=model,
                                                       batcher=batcher_cls,
                                                       data_path=TRAIN_DATA_DIR,
                                                       model_path=run_path,
                                                       early_stop=True,
                                                       verbose=True,
                                                       dev_score='loss',
                                                       train_hparams=all_hparams))
        # model_trainer.save_function = trainer_ddp.generic_save_function_ddp
        model_trainer.save_function = train_utils.save_model
        model_trainer.train()
        # Synchronize before cleanup
        dist.barrier()
        if process_rank == 0:
            train_utils.save_and_push_best_model(model_name,
                                                 all_hparams,
                                                 run_path,
                                                 cl_args.repo_name,
                                                 tokenizer=batcher_cls.get_tokenizer())
    finally:
        # Cleanup DDP
        cleanup_ddp()
        if process_rank == 0:
            # save_and_push_best_model(model_name, all_hparams, run_path, cl_args.repo_name)
            wandb.finish()


def train_model(cl_args):
    """
    Read training and dev data, initialize and train the model.
    Save the trained model and results.
    """
    run_path = get_run_path(cl_args)
    run_name = run_path.name
    config_path = CONFIG_DIR / str(cl_args.config_path)
    all_hparams = get_config(config_path)
    save_config(all_hparams, run_path)
    model_name = all_hparams['model_name']
    logger = get_logger()
    init_wandb(all_hparams, run_name)
    model, batcher_cls = ModelFactory.create(model_name, all_hparams)
    # Move model to GPU if available.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    logger.info(model)

    # Save initial (untrained) model.
    # trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    # train_utils.save_model(model=model, save_path=run_path, model_suffix='init')
    batcher_config = BatcherConfig(
        num_examples=all_hparams["train_size"],     # Total number of examples
        batch_size=all_hparams["batch_size"],       # Batch size
        max_length=all_hparams["max_seq_len"],      # Max token length per sequence
        padding_side=all_hparams["padding_side"],   # Padding direction
        model_name=model_name,                      # Model to use for tokenization
        base_pt_layer=all_hparams["base-pt-layer"],
        query_instruct=all_hparams["query_instruct"]  # Whether to prepend an instruction to queries
    )
    print(f"Train Batcher config: {batcher_config}")

    dev_batcher_config = BatcherConfig(
        num_examples=all_hparams["dev_size"],  # Total number of examples
        batch_size=all_hparams["batch_size"],  # Batch size
        max_length=all_hparams["max_seq_len"],  # Max token length per sequence
        padding_side=all_hparams["padding_side"],  # Padding direction
        model_name=model_name,  # Model to use for tokenization
        base_pt_layer=all_hparams["base-pt-layer"],
        query_instruct=all_hparams["query_instruct"]  # Whether to prepend an instruction to queries
    )
    print(f"Dev Batcher config: {batcher_config}")
    # Initialize the trainer.
    model_trainer = trainer.BasicRankingTrainer(
        logger=logger,
        model=model,
        batcher=batcher_cls,
        batcher_config=batcher_config,
        dev_batcher_config=dev_batcher_config,
        data_path=TRAIN_DATA_DIR,
        model_path=run_path,
        early_stop=True,
        verbose=True,
        dev_score='loss',
        train_hparams=all_hparams,
    )
    model_trainer.save_function = train_utils.save_model
    vram_allocated, vram_cached = get_vram_usage()

    logger.info(f"VRAM Allocated: {vram_allocated:.2f} GB")
    logger.info(f"VRAM Cached: {vram_cached:.2f} GB")
    # Train the model.
    model_trainer.train()
    train_utils.save_and_push_best_model(model_name,
                                         all_hparams,
                                         run_path,
                                         cl_args.repo_name,
                                         batcher_cls.get_tokenizer())

def main():
    """
      Train a specified model on a given dataset.
      """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')

    train_args = subparsers.add_parser('train_model')
    train_args.add_argument('--model_name', required=True,
                            choices=[
                                      'cocite-gte-qwen2-1.5b-instruct-biomed',  # CoQwen
                                     'ts-aspire-gte-qwen2-1.5b-instruct-biomed', # TSQwen
                                     'ot-aspire-gte-qwen2-1.5b-instruct-biomed' # OTQwen
                                     ],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['s2orcscidocs',
                                     's2orccompsci',
                                     's2orcbiomed'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    train_args.add_argument('--log_fname',
                            help='Path to directory to save log files.')
    train_args.add_argument('--repo_name',
                            help='name of the huggingface repo to push the model and tokenizer to after training.')
    train_args.add_argument('--run_id',
                            help='run_id in the format {run_001, run_002,...}')

    cl_args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA is available! Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is NOT available.")
    print(f"NCCL Available: {torch.distributed.is_nccl_available()}")  # Should return True
    print(f"Training {cl_args.model_name} on {cl_args.dataset} dataset with {cl_args.num_gpus} GPUs.")
    print(f"Huggingface repo: {cl_args.repo_name}")
    setup_logging(cl_args.log_fname)
    if cl_args.num_gpus > 1:
        torch_mp.spawn(ddp_train_model, nprocs=cl_args.num_gpus, args=(cl_args,))
    else:
        train_model(cl_args)
if __name__ == '__main__':
    main()