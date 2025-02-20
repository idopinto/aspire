from __future__ import print_function
import os
import logging
import time, copy
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import transformers

from src.learning.decoder_learning import predict_utils_dec as pu, train_utils
from src.learning.decoder_learning.decoder_batchers import BatcherConfig
from src.pre_process import data_utils as du
from src.learning.decoder_learning.train_utils import *
import wandb
from src.learning.decoder_learning.trainer import *

LOG_EVERY = 5
class GenericTrainerDDP:
    # If this isnt set outside of here it crashes with: "got multiple values for argument"
    # todo: Look into who this happens --low-pri.s
    save_function = train_utils.save_model

    def __init__(self, logger, process_rank, num_gpus, model, batcher, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss'):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other facetid_models should subclass this and define the data that the facetid_models
        being trained consume.
        :param logger: a logger to write logs with.
        :param process_rank: int; which process this is.
        :param num_gpus: int; how many gpus are being used to train.
        :param model: pytorch model.
        :param batcher: a model_utils.Batcher class.
        :param model_path: string; directory to which model should get saved.
        :param early_stop: boolean;
        :param verbose: boolean;
        :param dev_score: string; {'loss'/'f1'} How dev set evaluation should be done.
        # train_hparams dict elements.
        :param train_size: int; number of training examples.
        :param dev_size: int; number of dev examples.
        :param batch_size: int; number of examples per batch.
        :param accumulated_batch_size: int; number of examples to accumulate gradients
            for in smaller batch size before computing the gradient. If this is not present
            in the dictionary or is smaller than batch_size then assume no gradient
            accumulation.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param es_check_every: int; check some metric on the dev set every check_every iterations.
        :param lr_decay_method: string; {'exponential', 'warmuplin', 'warmupcosine'}
        :param decay_lr_by: float; decay the learning rate exponentially by the following
            factor.
        :param num_warmup_steps: int; number of steps for which to do warm up.
        :param decay_lr_every: int; decay learning rate every few iterations.

        """
        # Book keeping
        self.process_rank = process_rank
        self.logger = logger
        self.dev_score = dev_score
        self.verbose = verbose
        self.num_epochs = train_hparams['num_epochs']

        self.es_check_every = train_hparams['es_check_every'] // num_gpus
        self.num_train, self.num_batches, self.total_iters = calculate_num_batches(train_hparams, self.num_epochs, num_gpus)
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        self.accumulate_gradients, self.accumulated_batch_size, self.update_params_every = validate_accumulate_gradients(train_hparams)
        if self.accumulate_gradients:
            conditional_log(self.logger, self.process_rank,
                            f'Accumulating gradients for: {self.accumulated_batch_size}; updating params every: {self.update_params_every}; with batch size: {self.batch_size}')
        self.model_path = model_path  # Save model and checkpoints.
        self.iteration = 0

        # Model, batcher and the data.
        self.model = model
        # self.bert_like = train_hparams.get('bert_like', True) # for models that are not based on BERT, e.g. gte-qwen2-1.5B-instruct, for compatible batcher
        # self.query_instruct = train_hparams.get('query_instruct', False) # whether or not to wrap the query with instruction. for instruction-tuned models.
        self.use_bfloat16 = train_hparams.get('use_bfloat16', False) # whether or not to wrap the query with instruction. for instruction-tuned models.

        self.batcher_config = BatcherConfig(
            num_examples=self.num_train,  # Total number of examples
            batch_size=train_hparams["batch_size"],  # Batch size
            max_length=train_hparams.get("max_seq_len", 512),  # Max token length per sequence
            padding_side=train_hparams.get("padding_side","left"),  # Padding direction
            model_name=train_hparams['model_name'],  # Model to use for tokenization
            base_pt_layer=train_hparams["base-pt-layer"],
            query_instruct=train_hparams.get("query_instruct", True),  # Whether to prepend an instruction to queries
            task_description=train_hparams.get("task_description", "Represent the core scientific contributions and findings of this paper.")
        )

        self.dev_batcher_config = copy.deepcopy(self.batcher_config)
        self.dev_batcher_config.num_examples = self.num_dev
        print("Train batcher config", self.batcher_config)
        print("Dev batcher config", self.dev_batcher_config)

        # self.batcher_config = batcher_config
        # self.dev_batcher_config = dev_batcher_config
        self.batcher = batcher
        self.time_per_batch = 0
        self.time_per_dev_pass = 0

        # Different trainer classes can add this based on the data that the model they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']

        # Initialize optimizer.
        self.optimizer = get_optimizer(self.model, self.update_rule, self.learning_rate)


        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.scheduler = get_lr_scheduler(self.optimizer, self.lr_decay_method, train_hparams, self.num_batches,
                                          num_gpus)
        self.log_every = LOG_EVERY

        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []

        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss


    def dev_step(self):
        dev_start = time.time()

        self.model.eval()
        # Using the module as it is for eval:
        # https://discuss.pytorch.org/t/distributeddataparallel-barrier-doesnt-work-as-expected-during-evaluation/99867/11
        dev_score = -1.0 * pu.batched_loss_ddp(model=self.model.module,
                                               batcher=self.batcher,
                                               batcher_config=self.dev_batcher_config,
                                               ex_fnames=self.dev_fnames,
                                               loss_helper=self.loss_function_cal,
                                               logger=self.logger,
                                               use_bfloat16=self.use_bfloat16)
        dev_end = time.time()
        dev_time = dev_end - dev_start
        return dev_score, dev_time

    def train(self):
        """
        Trains the model over a specified number of epochs and iterations, performs optimization, evaluates
        on the development dataset, and manages early stopping if applicable. The method additionally tracks
        and logs relevant training metrics and saves both the final and best-performing facetid_models.

        :param self: Represents the instance of the class containing this method.

        :raises AnyErrorType: Raised when there are underlying issues during training (e.g., batch creation,
            model optimization). Replace `AnyErrorType` with specific exceptions if determinable.

        :param self.model: The model being trained.
        :param self.model_path: Path where the trained facetid_models (final and best) will be stored.
        :param self.num_epochs: Number of epochs for the training process.
        :param self.total_iters: Total iterations performed during training.
        :param self.batch_size: Batch size used for data sampling during training and evaluation.
        :param self.num_train: Number of training examples available.
        :param self.num_dev: Number of development examples available.
        :param self.dev_score_history: Holds the score values from past evaluations on the dev dataset, used for early stopping.
        """
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf
        total_time_per_batch = 0
        total_time_per_dev = 0
        conditional_log(self.logger, self.process_rank,
                        f'num_train: {self.num_train}; num_dev: {self.num_dev}')
        conditional_log(self.logger, self.process_rank,
                        f'Training {self.num_epochs} epochs, {self.total_iters} iterations')
        # print(f"pid: {os.getpid()}, checkpoint: {self.train_fnames}")

        train_start = time.time()

        for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
            # Initialize batcher. Shuffle one time before the start of every epoch.
            epoch_batcher = self.batcher(config=self.batcher_config, ex_fnames=ex_fnames)
            iters_start = time.time()
            best_params, best_epoch, best_iter, best_dev_score= self.train_step(epoch=epoch,
                                                                 epoch_batcher=epoch_batcher,
                                                                 total_time_per_batch=total_time_per_batch,
                                                                 total_time_per_dev=total_time_per_dev,
                                                                 best_dev_score=best_dev_score,
                                                                 best_params=best_params,
                                                                 best_epoch=best_epoch,
                                                                 best_iter=best_iter)
            epoch_time = time.time() - iters_start
            conditional_log(self.logger, self.process_rank, f'Epoch {epoch} time: {epoch_time:.4f}s')

        train_time = time.time() - train_start
        # Log time stats
        conditional_log(self.logger, self.process_rank, f'Training time: {train_time}s')
        self.time_per_batch = float(total_time_per_batch) / self.total_iters if self.total_iters > 0 else 0.0
        conditional_log(self.logger, self.process_rank, f'Time per batch: {self.time_per_batch:.4f}s')

        if self.early_stop and self.dev_score_history:
            self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history) if len(self.dev_score_history) > 0 else 0
            conditional_log(self.logger, self.process_rank, f'Time per dev pass: {self.time_per_dev_pass:4f}s')

        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        if self.process_rank == 0:
            self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
            self.model.load_state_dict(best_params)
            self.save_function(model=self.model, save_path=self.model_path, model_suffix='best')
            conditional_log(self.logger, self.process_rank, f'Best model; Epoch {best_epoch}; Iteration {best_iter}; Dev loss: {best_dev_score:.4f}')

    def train_step(self,epoch, epoch_batcher, total_time_per_batch, total_time_per_dev, best_dev_score, best_params, best_epoch, best_iter):
        for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
            self.model.train()
            batch_start = time.time()
            dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                if not self.accumulate_gradients:
                    self.optimizer.zero_grad()
                ret_dict = self.model.forward(batch_dict=batch_dict)
                objective = self.compute_loss(loss_components=ret_dict)
                objective.backward()
                if self.accumulate_gradients:
                    if (self.iteration + 1) % self.update_params_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()

                # The decay_lr_every doesnt need to be a multiple of self.log_every
                if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                    self.scheduler.step()

            # Log metrics
            if self.iteration % self.log_every == 0:
                if self.process_rank == 0:
                    wandb.log({
                        "train_loss": objective.item(),
                        "epoch": epoch,
                        "iteration": self.iteration,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                # # Save every loss component separately.
                loss_str = []
                for key in ret_dict:
                    if torch.cuda.is_available():
                        loss_comp = float(ret_dict[key].data.cpu().numpy())
                    else:
                        loss_comp = float(ret_dict[key].data.numpy())
                    self.loss_history[key].append(loss_comp)
                    loss_str.append(f'{key}: {loss_comp}')
                self.loss_checked_iters.append(self.iteration)
                if self.verbose:
                    log_str = (f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}; '
                               + '; '.join(loss_str))
                    conditional_log(self.logger, self.process_rank, log_str)
            elif self.verbose:
                log_str = f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}'
                conditional_log(self.logger, self.process_rank, log_str)

            batch_end = time.time()
            total_time_per_batch += batch_end - batch_start
            # Check every few iterations how you're doing on the dev set.
            if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop and self.process_rank == 0:

                # Save the loss at this point too.
                for key in ret_dict:
                    loss_comp = ret_dict[key].float().detach().cpu().numpy()
                    self.loss_history[key].append(loss_comp)
                self.loss_checked_iters.append(self.iteration)
                # Switch to eval model and check loss on dev set.
                dev_score, dev_time = self.dev_step()
                total_time_per_dev += dev_time
                wandb.log({"dev_loss": dev_score})
                self.dev_score_history.append(dev_score)
                self.dev_checked_iters.append(self.iteration)
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    # Deep copy so you're not just getting a reference.
                    best_params = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_iter = self.iteration
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                    wandb.log({"best_dev_loss": dev_score, "best_epoch": epoch, "best_iteration": self.iteration})
                    self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')

                else:
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'
                                         .format(*everything))
            dist.barrier()
            self.iteration += 1
        return best_params, best_epoch, best_iter,best_dev_score


    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError

class BasicRankingTrainerDDP(GenericTrainerDDP):
    def __init__(self, logger, process_rank, num_gpus, model, batcher,model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainerDDP.__init__(self, logger=logger, process_rank=process_rank, num_gpus=num_gpus,
                                   model=model, batcher=batcher, model_path=model_path,
                                   train_hparams=train_hparams, early_stop=early_stop, verbose=verbose,
                                   dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the dataset as there are epochs and a negative examples file.
        self.train_fnames = []

        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives facetid_models, different alignment facetid_models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = 'train-{:s}'.format(suffix)
            dev_basename = 'dev-{:s}'.format(suffix)
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself
            # Each process gets a part of the data to consume in training.
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', f'{train_basename}-{process_rank}-{i}.jsonl'),
            }
            self.train_fnames.append(ex_fname)
            # print(self.train_fnames)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(dev_basename)),
        }
        # The split command in bash is asked to make exactly equal sized splits with the remainder in a final file which is unused

        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss

    def train_step(self,epoch, epoch_batcher, total_time_per_batch, total_time_per_dev, best_dev_score, best_params, best_epoch, best_iter):
        for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
            self.model.train()
            batch_start = time.time()
            dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                if not self.accumulate_gradients:
                    self.optimizer.zero_grad()
                ret_dict = self.model.forward(batch_dict=batch_dict)
                objective = self.compute_loss(loss_components=ret_dict)
                objective.backward()
                if self.accumulate_gradients:
                    if (self.iteration + 1) % self.update_params_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()

                # The decay_lr_every doesnt need to be a multiple of self.log_every
                if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                    self.scheduler.step()

            # Log metrics
            if self.iteration % self.log_every == 0:
                if self.process_rank == 0:
                    wandb.log({
                        "train_loss": objective.item(),
                        "epoch": epoch,
                        "iteration": self.iteration,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                # # Save every loss component separately.
                loss_str = []
                for key in ret_dict:
                    if torch.cuda.is_available():
                        loss_comp = float(ret_dict[key].data.cpu().numpy())
                    else:
                        loss_comp = float(ret_dict[key].data.numpy())
                    self.loss_history[key].append(loss_comp)
                    loss_str.append(f'{key}: {loss_comp}')
                self.loss_checked_iters.append(self.iteration)
                if self.verbose:
                    log_str = (f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}; '
                               + '; '.join(loss_str))
                    conditional_log(self.logger, self.process_rank, log_str)
            elif self.verbose:
                log_str = f'Epoch: {epoch}; Iteration: {self.iteration}/{self.total_iters}'
                conditional_log(self.logger, self.process_rank, log_str)

            batch_end = time.time()
            total_time_per_batch += batch_end - batch_start
            # Check every few iterations how you're doing on the dev set.
            if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop and self.process_rank == 0:

                # Save the loss at this point too.
                for key in ret_dict:
                    loss_comp = ret_dict[key].float().detach().cpu().numpy()
                    self.loss_history[key].append(loss_comp)
                self.loss_checked_iters.append(self.iteration)
                # Switch to eval model and check loss on dev set.
                dev_score, dev_time = self.dev_step()
                total_time_per_dev += dev_time
                wandb.log({"dev_loss": dev_score})
                self.dev_score_history.append(dev_score)
                self.dev_checked_iters.append(self.iteration)
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    # Deep copy so you're not just getting a reference.
                    best_params = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                    best_iter = self.iteration
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                    wandb.log({"best_dev_loss": dev_score, "best_epoch": epoch, "best_iteration": self.iteration})
                    self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')

                else:
                    everything = (epoch, self.iteration, self.total_iters, dev_score)
                    if self.verbose:
                        self.logger.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'
                                         .format(*everything))
            dist.barrier()
            self.iteration += 1
        return best_params, best_epoch, best_iter,best_dev_score

    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']

