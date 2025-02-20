"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import time
from collections import defaultdict
import numpy as np
from src.learning.decoder_learning import predict_utils_dec as pu, train_utils
from src.learning import data_utils as du
from src.learning.decoder_learning.train_utils import *
import wandb
# from src.learning.decoder_learning.lora_utils import save_model_function
LOG_EVERY = 5

class GenericTrainer:
    save_function = train_utils.save_model

    def __init__(self,logger, model, batcher,batcher_config,dev_batcher_config, model_path, train_hparams,
                 early_stop=True, verbose=True, dev_score='loss',rank:int=-1):
        """
        A generic trainer class that defines the training procedure. Trainers
        for other facetid_models should subclass this and define the data that the facetid_models
        being trained consume.
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
        self.logger = logger
        self.dev_score = dev_score
        self.verbose = verbose
        self.es_check_every = train_hparams['es_check_every']
        self.num_dev = train_hparams['dev_size']
        self.batch_size = train_hparams['batch_size']
        self.num_epochs = train_hparams['num_epochs']
        self.accumulate_gradients,self.accumulated_batch_size ,self.update_params_every = validate_accumulate_gradients(train_hparams)

        if self.accumulate_gradients:
            logging.info(f'Accumulating gradients for: {self.accumulated_batch_size}; updating params every: {self.update_params_every}; with batch size: {self.batch_size}')

        self.num_train, self.num_batches, self.total_iters = calculate_num_batches(train_hparams, self.num_epochs)
        self.model_path = model_path  # Save model and checkpoints.
        self.iteration = 0

        # Model, batcher and the data.
        self.model = model
        self.bert_like = train_hparams.get('bert_like', True) # for models that are not based on BERT, e.g. gte-qwen2-1.5B-instruct, for compatible batcher
        # self.query_instruct = train_hparams.get('query_instruct', False) # whether or not to wrap the query with instruction. for instruction-tuned models.
        self.use_bfloat16 = train_hparams.get('use_bfloat16', False) # whether or not to model params are in bfloat16
        self.batcher_config = batcher_config
        self.dev_batcher_config = dev_batcher_config
        self.rank = rank
        self.is_main_process = rank in [-1, 0]  # True for non-DDP or rank 0
        # self.save_function = save_model_function

        self.batcher = batcher
        self.time_per_batch, self.time_per_dev_pass = 0, 0
        # Different trainer classes can add this based on the data that the model they are training needs.
        self.train_fnames = []
        self.dev_fnames = {}

        # Optimizer args.
        self.early_stop = early_stop
        self.update_rule = train_hparams['update_rule']
        self.learning_rate = train_hparams['learning_rate']
        self.optimizer = get_optimizer(self.model, self.update_rule, self.learning_rate)

        # Reduce the learning rate every few iterations.
        self.lr_decay_method = train_hparams['lr_decay_method']
        self.decay_lr_every = train_hparams['decay_lr_every']
        self.log_every = LOG_EVERY
        # self.wandb_callback = WandbCallback(log_every=5)
        self.scheduler =get_lr_scheduler(self.optimizer, self.lr_decay_method, train_hparams, self.num_batches)

        # Train statistics.
        self.loss_history = defaultdict(list)
        self.loss_checked_iters = []
        self.dev_score_history = []
        self.dev_checked_iters = []

        # Every subclass needs to set this.
        self.loss_function_cal = GenericTrainer.compute_loss

    def train(self):
        """
        Make num_epoch passes through the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_score = -np.inf

        total_time_per_batch = 0
        total_time_per_dev = 0
        train_start = time.time()
        logging.info('num_train: {:d}; num_dev: {:d}'.format(
            self.num_train, self.num_dev))
        logging.info('Training {:d} epochs, {:d} iterations'.
                     format(self.num_epochs, self.total_iters))
        for epoch, ex_fnames in zip(range(self.num_epochs), self.train_fnames):
            # Initialize batcher. Shuffle one time before the start of every
            # epoch.

            epoch_batcher = self.batcher(config=self.batcher_config, ex_fnames=ex_fnames)
            # Get the next training batch.
            iters_start = time.time()
            for batch_doc_ids, batch_dict in epoch_batcher.next_batch():
                self.model.train()
                batch_start = time.time()
                dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
                with torch.amp.autocast('cuda:0', dtype=dtype):
                    if self.accumulate_gradients:
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters
                        objective.backward()
                        if (self.iteration + 1) % self.update_params_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    else:
                        # Clear all gradient buffers.
                        self.optimizer.zero_grad()
                        # Compute objective.
                        ret_dict = self.model.forward(batch_dict=batch_dict)
                        objective = self.compute_loss(loss_components=ret_dict)
                        # Gradients wrt the parameters.
                        objective.backward()
                        # Step in the direction of the gradient.
                        self.optimizer.step()

                if self.iteration % self.log_every == 0:
                    wandb.log({
                        "train_loss": objective.item(),
                        "epoch": epoch,
                        "iteration": self.iteration,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                    # Save every loss component separately.
                    loss_str = []
                    for key in ret_dict:
                        if torch.cuda.is_available():
                            loss_comp = float(ret_dict[key].data.cpu().numpy())
                        else:
                            loss_comp = float(ret_dict[key].data.numpy())
                        self.loss_history[key].append(loss_comp)
                        loss_str.append('{:s}: {:.4f}'.format(key, loss_comp))
                    self.loss_checked_iters.append(self.iteration)
                    if self.verbose:
                        log_str = 'Epoch: {:d}; Iteration: {:d}/{:d}; '.format(epoch, self.iteration, self.total_iters)
                        logging.info(log_str + '; '.join(loss_str))
                elif self.verbose:
                    logging.info('Epoch: {:d}; Iteration: {:d}/{:d}'.
                                 format(epoch, self.iteration, self.total_iters))
                # The decay_lr_every doesnt need to be a multiple of self.log_every
                if self.iteration > 0 and self.iteration % self.decay_lr_every == 0:
                    self.scheduler.step()
                    # logging.info('Decayed learning rates: {}'.
                    #              format([g['lr'] for g in self.optimizer.param_groups]))
                batch_end = time.time()
                total_time_per_batch += batch_end - batch_start
                # Check every few iterations how you're doing on the dev set.
                if self.iteration % self.es_check_every == 0 and self.iteration != 0 and self.early_stop:
                    # Save the loss at this point too.
                    for key in ret_dict:
                        if torch.cuda.is_available():
                            loss_comp = float(ret_dict[key].data.cpu().numpy())
                        else:
                            loss_comp = float(ret_dict[key].data.numpy())
                        self.loss_history[key].append(loss_comp)
                    self.loss_checked_iters.append(self.iteration)
                    # Switch to eval model and check loss on dev set.
                    self.model.eval()
                    dev_start = time.time()
                    if self.dev_score == 'loss':
                        dev_score = -1.0 * pu.batched_loss(
                            model=self.model,
                            batcher=self.batcher,
                            batcher_config=self.dev_batcher_config,
                            loss_helper=self.loss_function_cal,
                            ex_fnames=self.dev_fnames,
                            use_bfloat16=self.use_bfloat16)
                        wandb.log({"dev_loss": dev_score})

                    dev_end = time.time()
                    total_time_per_dev += dev_end - dev_start
                    self.dev_score_history.append(dev_score)
                    self.dev_checked_iters.append(self.iteration)
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        # Deep copy so you're not just getting a reference.
                        best_params = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_iter = self.iteration
                        everything = (epoch, self.iteration, self.total_iters, dev_score)
                        wandb.log({"best_dev_loss": dev_score, "best_epoch": epoch, "best_iteration": self.iteration})

                        if self.verbose:
                            logging.info('Current best model; Epoch {:d}; '
                                         'Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))
                        self.save_function(model=self.model, save_path=self.model_path, model_suffix='cur_best')
                        # wandb.save(os.path.join(self.model_path, 'best_model.pt'))
                    else:
                        everything = (epoch, self.iteration, self.total_iters, dev_score)
                        if self.verbose:
                            logging.info('Epoch {:d}; Iteration {:d}/{:d}; Dev score: {:.4f}'.format(*everything))

                self.iteration += 1
            epoch_time = time.time() - iters_start
            logging.info('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
            logging.info('\n')

        # Say how long things took.
        train_time = time.time() - train_start
        logging.info('Training time: {:.4f}s'.format(train_time))
        if self.total_iters > 0:
            self.time_per_batch = float(total_time_per_batch) / self.total_iters
        else:
            self.time_per_batch = 0.0
        logging.info('Time per batch: {:.4f}s'.format(self.time_per_batch))
        if self.early_stop and self.dev_score_history:
            if len(self.dev_score_history) > 0:
                self.time_per_dev_pass = float(total_time_per_dev) / len(self.dev_score_history)
            else:
                self.time_per_dev_pass = 0
            logging.info('Time per dev pass: {:4f}s'.format(self.time_per_dev_pass))

        # Save the learnt model: save both the final model and the best model.
        # https://stackoverflow.com/a/43819235/3262406
        self.save_function(model=self.model, save_path=self.model_path, model_suffix='final')
        logging.info('Best model; Epoch {:d}; Iteration {:d}; Dev loss: {:.4f}'
                     .format(best_epoch, best_iter, best_dev_score))

    @staticmethod
    def compute_loss(loss_components):
        """
        Models will return dict with different loss components, use this and compute batch loss.
        :param loss_components: dict('str': Variable)
        :return:
        """
        raise NotImplementedError


class BasicRankingTrainer(GenericTrainer):
    def __init__(self,logger, model, batcher,batcher_config,dev_batcher_config, model_path, data_path,
                 train_hparams, early_stop=True, verbose=True, dev_score='loss'):
        """
        Trainer for any model returning a ranking loss. Uses everything from the
        generic trainer but needs specification of how the loss components
        should be put together.
        :param data_path: string; directory with all the int mapped data.
        """
        GenericTrainer.__init__(self,
                                logger=logger,
                                model=model,
                                batcher=batcher,
                                batcher_config=batcher_config,
                                dev_batcher_config=dev_batcher_config,
                                model_path=model_path,
                                train_hparams=train_hparams,
                                early_stop=early_stop,
                                verbose=verbose,
                                dev_score=dev_score)
        # Expect the presence of a directory with as many shuffled copies of the
        # dataset as there are epochs and a negative examples file.
        self.train_fnames = []
        # Expect these to be there for the case of using diff kinds of training data for the same
        # model; hard negatives facetid_models, different alignment facetid_models and so on.
        if 'train_suffix' in train_hparams:
            suffix = train_hparams['train_suffix']
            train_basename = f'train-{suffix}'
            dev_basename = f'dev-{suffix}'
        else:
            train_basename = 'train'
            dev_basename = 'dev'
        for i in range(self.num_epochs):
            # Each run contains a copy of shuffled data for itself.
            ex_fname = {
                'pos_ex_fname': os.path.join(data_path, 'shuffled_data', f'{train_basename}-{i}.jsonl'),
                # 'pos_ex_fname': os.path.join(data_path, '{:s}.jsonl'.format(train_basename)),
            }
            logger.info(f"training data located in: {ex_fname}")
            self.train_fnames.append(ex_fname)
        self.dev_fnames = {
            'pos_ex_fname': os.path.join(data_path, f'{dev_basename}.jsonl'),
        }
        logger.info(f"development data located in: {self.dev_fnames}")
        # Every subclass needs to set this.
        self.loss_function_cal = BasicRankingTrainer.compute_loss


    @staticmethod
    def compute_loss(loss_components):
        """
        Simply add loss components.
        :param loss_components: dict('rankl': rank loss value)
        :return: Variable.
        """
        return loss_components['rankl']
