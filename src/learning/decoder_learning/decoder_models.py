from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from src.learning.facetid_models import pair_distances as pair_dist
from src.learning.facetid_models.loss_functions import CustomTripletMarginWithDistanceLoss

rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
rep_len_ali_tup = namedtuple('RepLenAli', ['embed', 'abs_lens', 'align_idxs'])

def get_dist_function(score_agg_type: str, model_hparams: dict = None):
    """
    Returns the appropriate distance function based on the specified score aggregation type.

    :param score_agg_type: str, type of distance function ('l2max' or 'l2wasserstein').
    :param model_hparams: dict, optional model hyperparameters for Wasserstein distance.
    :return: Callable distance function.
    """
    if score_agg_type == 'l2max':
        dist_function = pair_dist.allpair_masked_dist_l2max
        return dist_function

    elif score_agg_type == 'l2wasserstein':
        ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
        dist_function = ot_distance.compute_distance
        return dist_function
    else:
        raise ValueError(f'Unknown aggregation: {score_agg_type}')

class CoQwen(nn.Module):
    """
    A Transformer-based model for representing scientific abstracts using gte-qwen2-1.5b-instruct.

    This model:
    - Uses the final layer EOS token representation.
    - Employs a triplet loss function.
    - Does not insert SEP tokens between abstract sentences.
    """

    def __init__(self, model_hparams):
        """
        Initializes the CoQwen model.
        :param model_hparams: dict, containing model hyperparameters.
        """
        super(CoQwen, self).__init__()
        self.embedding_dim = model_hparams['embedding_dim']
        self.use_bfloat16 = model_hparams.get('use_bfloat16', False)
        self.dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        self.attn_implementation = model_hparams.get('attn_implementation', 'sdpa')
        print(f"Loading CoQwen model.\n dtype={self.dtype}\n attn_implementations={self.attn_implementation}")

        # Load pretrained Transformer model
        self.encoder = AutoModel.from_pretrained(
            model_hparams['base-pt-layer'],
            torch_dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation
        )
        self.encoder.config.output_hidden_states = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handle fine-tuning and LoRA
        if model_hparams['fine_tune']:
            # Apply LoRA if lora_config is provided
            lora = model_hparams.get('lora', False)
            if lora:
                print(f"Trainable parameters (before LoRA): {self._count_trainable_parameters():,}")
                lora_config = model_hparams.get('lora_config', None)
                self.encoder = self._apply_lora(lora_config)
                print(f"Trainable parameters (after LoRA): {self._count_trainable_parameters():,}")
        else:
            # If fine tune is False then freeze the params
            self.encoder.requires_grad_(False)

        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')

    def _count_trainable_parameters(self):
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def _apply_lora(self, lora_config):
        """
        Applies LoRA (Low-Rank Adaptation) to the model for efficient fine-tuning.

        :param lora_config: dict, LoRA configuration.
        :return: Model with LoRA applied.
        """
        if not lora_config:
            return self.encoder

        lora_configuration = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', "none")  # Options: 'none', 'all', 'lora_only'
        )
        return get_peft_model(self.encoder, lora_configuration)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Extracts the representation of the last token in the sequence.

        :param last_hidden_states: Tensor, Transformer hidden states.
        :param attention_mask: Tensor, attention mask.
        :return: Tensor, last token representation.
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def forward(self, batch_dict):
        """
        Computes the triplet loss during training.

        :param batch_dict: dict, containing input batches for training.
        :return: dict, loss dictionary.
        """
        batch_loss = self.forward_rank(batch_dict['batch_rank'])
        return {'rankl': batch_loss}

    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Compute scores and return.
        query_encode_ret_dict: {'sent_reps': numpy.array, 'doc_cls_reps': numpy.array} (EOS)
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array}) (EOS)
        """
        # Pack representations as padded gpu tensors.
        query_cls_rep = query_encode_ret_dict['doc_cls_reps']
        cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
        query_cls_reps = []
        for bi in range(len(cand_cls_reps)):
            query_cls_reps.append(query_cls_rep)
        query_cls_reps = torch.tensor(np.vstack(query_cls_reps), dtype=torch.bfloat16, device=self.device)
        cand_cls_reps = torch.tensor(np.vstack(cand_cls_reps), dtype=torch.bfloat16, device=self.device)
        # Compute scores as at train time.
        doc_sims = (-1 * functional.pairwise_distance(query_cls_reps, cand_cls_reps, p=2.0)).squeeze()
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = doc_sims.float().cpu().data.numpy()
        else:
            batch_scores = doc_sims.data.numpy()
        # Return the same thing as batch_scores and pair_scores because the pp_gen_nearest class expects it.
        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': batch_scores
        }
        return ret_dict

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the negative examples.
        :return: ret_dict
        """
        doc_bert_batch, batch_size = batch_dict['bert_batch'], len(batch_dict['bert_batch']['seq_lens'])
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.float().cpu().detach().numpy()
        else:
            doc_cls_reps = doc_cls_reps.detach().numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i in range(batch_size):
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :]})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch = batch_dict['bert_batch']
        # Get the representations from the model.
        doc_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        doc_reps = doc_reps.float().cpu().detach().numpy()
        ret_dict = {
            'doc_cls_reps': doc_reps,  # batch_size x encoding_dim
        }
        return ret_dict

    def forward_rank(self, batch_rank):
        """
        Computes the triplet loss for a batch of (query, positive, negative) samples.

        :param batch_rank: dict, containing tokenized inputs for query, positive, and negative examples.
         {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: torch.Tensor, computed loss.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']

        # Get representations from model
        q_sent_reps = self.partial_forward(qbert_batch)
        p_context_reps = self.partial_forward(pbert_batch)

        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.partial_forward(nbert_batch)
        else:
            # Use in-batch negatives by shuffling positives
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]

        return self.criterion(q_sent_reps, p_context_reps, n_context_reps)

    def partial_forward(self, bert_batch):
        """
        Encodes a batch of inputs using the Transformer model.

        :param bert_batch: dict, containing tokenized input sequences.
        :return: torch.Tensor, document-level representations.
        """
        # batch_size x bert_encoding_dim
        cls_doc_reps = self.get_document_representations(bert_batch=bert_batch)
        if len(cls_doc_reps.size()) == 1:
            cls_doc_reps = cls_doc_reps.unsqueeze(0)
        return cls_doc_reps

    def get_document_representations(self, bert_batch):
        """
        Extracts document representations using the Transformer decoder by taking the last token EOS embedding..

       :param bert_batch: dict, containing tokenized sequences and attention masks.
       :return: torch.Tensor, document-level representations.
       """

        tokid_tt, attnmask_tt = bert_batch['tokid_tt'].to(self.device), bert_batch['attnmask_tt'].to(self.device)
        model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        doc_reps = self.last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()
        return doc_reps

# class OTQwen(CoQwen):
#     def __init__(self, model_hparams=None, decoder_config=None):
#         super(OTQwen, self).__init__(model_hparams)
#         self.score_agg_type = model_hparams['score_aggregation']
#         self.dist_function = get_dist_function(self.score_agg_type, model_hparams) # should be l2wasserstein
#         self.criterion = CustomTripletMarginWithDistanceLoss(distance_function=self.dist_function,
#                                                              margin=1.0,
#                                                              reduction='sum')
#         self.sent_loss_prop = 1.0
#         self.abs_loss_prop = 0.0
#
#     def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
#         query_cls_rep, query_sent_reps = query_encode_ret_dict['doc_cls_reps'], query_encode_ret_dict['sent_reps']
#         cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
#         cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
#         batch_size = len(cand_sent_reps)
#         cand_lens = [r.shape[0] for r in cand_sent_reps]
#         cmax_sents = max(cand_lens)
#         qmax_sents, encoding_dim = query_sent_reps.shape[0], query_sent_reps.shape[1]
#         query_lens = [qmax_sents] * batch_size
#         padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
#         padded_query_sent_reps = np.zeros((batch_size, qmax_sents, encoding_dim))
#
#         query_cls_reps = []
#         for bi, ex_reps in enumerate(cand_sent_reps):
#             padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
#             # Just repeat the query sents for now.
#             padded_query_sent_reps[bi, :qmax_sents, :] = query_sent_reps
#             query_cls_reps.append(query_cls_rep)
#         padded_query_sent_reps = torch.tensor(padded_query_sent_reps, dtype=torch.bfloat16, device=self.device)
#         padded_cand_sent_reps = torch.tensor(padded_cand_sent_reps, dtype=torch.bfloat16, device=self.device)
#         query_cls_reps = torch.tensor(np.vstack(query_cls_reps), dtype=torch.bfloat16, device=self.device)
#         cand_cls_reps = torch.tensor(np.vstack(cand_cls_reps), dtype=torch.bfloat16, device=self.device)
#         # Compute scores as at train time.
#         qt = rep_len_tup(embed=padded_query_sent_reps.permute(0, 2, 1), abs_lens=query_lens)
#         ct = rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens)
#         batch_sent_sims, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
#         # TSQwen also uses this function, so:
#         # if sent_loss_prop is zero use the supervised sent prop instead.
#         sent_loss_prop = max(self.sent_loss_prop, getattr(self, "sentsup_loss_prop", 0))
#         batch_scores = sent_loss_prop * batch_sent_sims
#         if self.abs_loss_prop > 0.0:
#             batch_doc_sims = -1 * functional.pairwise_distance(query_cls_reps, cand_cls_reps, p=2.0)
#             batch_scores += self.abs_loss_prop * batch_doc_sims
#
#         # Make numpy arrays and return.
#         batch_scores = batch_scores.float().cpu().detach().numpy()
#         if isinstance(pair_sims, list):
#             pair_sims = [t.float().cpu().detach().numpy() for t in pair_sims]
#         else:
#             pair_sims = pair_sims.cpu().detach().numpy()
#
#         unpadded_pair_sm = []
#         for i, (clen, qlen) in enumerate(zip(cand_lens, query_lens)):
#             # Happens in the case of wasserstein distance.
#             if len(pair_sims) == 5:
#                 upsm = [pair_sims[0][i, :qlen], pair_sims[1][i, :clen],
#                         pair_sims[2][i, :qlen, :clen], pair_sims[3][i, :qlen, :clen],
#                         pair_sims[4][i, :qlen, :clen]]
#             else:
#                 # encoding_dim x num_sents
#                 upsm = pair_sims[i, :qlen, :clen]
#             # return: # num_sents x encoding_dim
#             unpadded_pair_sm.append(upsm)
#
#         ret_dict = {
#             'batch_scores': batch_scores,
#             'pair_scores': unpadded_pair_sm
#         }
#         return ret_dict
#
#     def caching_encode(self, batch_dict):
#         """
#         Function used at test time.
#         batch_dict: dict of the form accepted by forward_rank but without any of the
#             negative examples.
#         :return: ret_dict
#         """
#         doc_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
#         doc_query_senttoki = batch_dict['senttok_idxs']
#         # Get the representations from the model; batch_size x encoding_dim x max_sents
#         doc_cls_reps, sent_reps = self.partial_forward(bert_batch=doc_batch,
#                                                        abs_lens=doc_abs_lens,
#                                                        sent_tok_idxs=doc_query_senttoki)
#         # Ensure tensors are converted to float32 if they are bfloat16 before converting to NumPy
#         sent_reps = sent_reps.float().cpu().numpy()
#         doc_cls_reps = doc_cls_reps.float().cpu().numpy()
#
#         # Return a list of reps instead of reps collated as one np array.
#         batch_reps = []
#         for i, num_sents in enumerate(doc_abs_lens):
#             # encoding_dim x num_sents
#             upsr = sent_reps[i, :, :num_sents]
#             # return: # num_sents x encoding_dim
#             batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :], # its EOS actually but for consistency
#                                'sent_reps': upsr.transpose(1, 0)})
#         return batch_reps
#
#     def encode(self, batch_dict):
#         """
#         Function used at test time.
#         batch_dict: dict of the form accepted by forward_rank but without any of the
#             negative examples.
#         :return: ret_dict
#         """
#         doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
#         doc_query_senttoki = batch_dict['senttok_idxs']
#         # Get the representations from the model; batch_size x encoding_dim x max_sents
#         doc_cls_reps, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
#                                          sent_tok_idxs=doc_query_senttoki)
#         sent_reps = sent_reps.float().cpu().detach().numpy()
#
#         unpadded_sent_reps = []
#         for i, num_sents in enumerate(doc_abs_lens):
#             # encoding_dim x num_sents
#             upsr = sent_reps[i, :, :num_sents]
#             # return: # num_sents x encoding_dim
#             unpadded_sent_reps.append(upsr.transpose(1, 0))
#         ret_dict = {
#             'sent_reps': unpadded_sent_reps,
#         }
#         return ret_dict
#
#     def forward_rank(self, batch_rank):
#         """
#         Function used at training time.
#         batch_dict: dict of the form:
#         {
#             'query_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'query_abs_lens': list(int); Number of sentences in query abs.
#             'query_senttoki': list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
#             'pos_abs_lens': list(int);
#             'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from positive abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
#             'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'neg_abs_lens': list(int);
#             'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
#         }
#         :return: loss_val; torch as tensor
#         """
#         qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
#         pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
#         query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
#         # Get the representations from the model.
#         _, q_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens, sent_tok_idxs=query_senttoki)
#         _, p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens, sent_tok_idxs=pos_senttoki)
#
#         # Happens when running on the dev set.
#         if 'neg_bert_batch' in batch_rank:
#             nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
#             neg_senttoki = batch_rank['neg_senttok_idxs']
#             _, n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens,
#                                                   sent_tok_idxs=neg_senttoki)
#             # Bundle the lengths with the embeds so the similarity
#             # function can use the lens for masking.
#             query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
#             pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
#             neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)
#
#             loss_val = self.criterion(query_sents, pos_sents, neg_sents)
#             return loss_val
#         else:
#             # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#             random_idxs = torch.randperm(p_sent_reps.size()[0])
#             n_sent_reps = p_sent_reps[random_idxs]
#             nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
#             # Bundle the lengths with the embeds so the similarity function can use the lens for masking.
#             query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
#             pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
#             neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)
#
#             loss_val = self.criterion(query_sents, pos_sents, neg_sents)
#
#             return loss_val
#
#     def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
#         """
#         Pass a batch of sentences through BERT and read off sentence
#         representations based on SEP idxs.
#         :return:
#             sent_reps: batch_size x encoding_dim x num_sents
#         """
#         # batch_size x num_sents x encoding_dim
#         doc_cls_reps, sent_reps = self.get_doc_and_sent_reps(bert_batch=bert_batch,
#                                                             num_sents=abs_lens,
#                                                             batch_senttok_idxs=sent_tok_idxs)
#         if len(sent_reps.size()) == 2:
#             sent_reps = sent_reps.unsqueeze(0)
#         if len(doc_cls_reps.size()) == 1:
#             doc_cls_reps = doc_cls_reps.unsqueeze(0)
#         # Similarity function expects: batch_size x encoding_dim x q_max_sents;
#         return doc_cls_reps, sent_reps.permute(0, 2, 1)
#
#     def get_doc_and_sent_reps(self, bert_batch, num_sents, batch_senttok_idxs):
#         """
#         Pass the concated abstract through BERT, and average token reps to get sentence reps.
#         -- NO weighted combine across layers.
#         :param batch:
#         :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
#             representations. The sentence mapped to BERT vocab and appropriately padded.
#         :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
#         :param num_sents: list(int); number of sentences in each example in the batch passed.
#         :return:
#             doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
#             sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
#         """
#         seq_lens, max_sents = bert_batch['seq_lens'], max(num_sents)
#         batch_size, max_seq_len = len(seq_lens), max(seq_lens)
#         # Pass input through Qwen2 and return all layer hidden outputs. (seg_tt not needed)
#         tokid_tt = bert_batch['tokid_tt'].to(self.device, non_blocking=True)
#         attnmask_tt = bert_batch['attnmask_tt'].to(self.device, non_blocking=True)
#         model_outputs = self.encoder(tokid_tt, attention_mask=attnmask_tt)
#         final_hidden_state = model_outputs.last_hidden_state
#         doc_cls_reps = self.last_token_pool(final_hidden_state, attention_mask=attnmask_tt).squeeze()
#
#         # Average token reps for every sentence to get sentence representations.
#         # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
#         sent_reps = self.get_sentence_representations(final_hidden_state, batch_senttok_idxs=batch_senttok_idxs,
#                                                       num_sents=num_sents,max_seq_len=max_seq_len)
#         return doc_cls_reps, sent_reps
#
#     def get_sentence_representations(self, final_hidden_state, batch_senttok_idxs, num_sents, max_seq_len):
#         """
#         Compute sentence representations by averaging token embeddings.
#
#         :param final_hidden_state: Tensor [batch_size x seq_len x embedding_dim]
#         :param batch_senttok_idxs: list(list(list(int))); token indices for each sentence in batch
#         :param num_sents: list(int); number of sentences per example
#         :param max_seq_len: int; maximum sequence length
#         :return: sent_reps: FloatTensor [batch_size x max_sents x embedding_dim]
#         """
#         batch_size, max_sents = len(num_sents), max(num_sents)
#         sent_reps = []
#
#         for sent_i in range(max_sents):
#             cur_sent_mask = np.zeros((batch_size, max_seq_len, self.embedding_dim))
#
#             # Build sentence masks
#             for batch_abs_i in range(batch_size):
#                 abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
#                 sent_i_tok_idxs = abs_sent_idxs[sent_i] if sent_i < len(abs_sent_idxs) else []
#                 cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
#
#             sent_mask = torch.tensor(cur_sent_mask, dtype=torch.float32, device=self.device)
#
#             # Mask hidden states and compute sentence representations
#             sent_tokens = final_hidden_state * sent_mask
#             cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
#                             torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
#
#             sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
#
#         # Concatenate all sentence representations (batch_size x max_sents x encoding_dim)
#         sent_reps = torch.cat(sent_reps, dim=1)
#         return sent_reps
#
# class TSQwen(OTQwen):
#     def __init__(self, model_hparams=None, decoder_config=None):
#         """
#         :param model_hparams: dict(string:int); model hyperparams.
#         """
#         super(TSQwen, self).__init__(model_hparams, decoder_config)
#         self.score_agg_type = model_hparams['score_aggregation']
#         self.dist_function = get_dist_function(self.score_agg_type, model_hparams) # can be l2wasserstein for ts+ot aspire, for ts only is l2max (which not actually used)
#         weighted_sup = model_hparams.get('weighted_sup', False)
#         self.sentsup_dist_function = pair_dist.allpair_masked_dist_l2sup_weighted if weighted_sup else pair_dist.allpair_masked_dist_l2sup
#
#         self.criterion_sent = CustomTripletMarginWithDistanceLoss(distance_function=self.dist_function,
#                                                                   margin=1.0,
#                                                                   reduction='sum')
#
#
#         self.criterion_sentsup = CustomTripletMarginWithDistanceLoss(distance_function=self.sentsup_dist_function,
#                                                                      margin=1.0,
#                                                                      reduction='sum')
#
#         self.criterion_abs = nn.TripletMarginLoss(margin=1,
#                                                   p=2,
#                                                   reduction='sum')
#
#         self.abs_loss_prop = float(model_hparams.get('abs_loss_prop', 0.0))
#         self.sent_loss_prop = float(model_hparams.get('sent_loss_prop', 0.0))
#         self.sentsup_loss_prop = float(model_hparams['sentsup_loss_prop'])
#
#
#
#     def _prepare_random_in_batch_negatives(self, pos_sent_reps, pos_rep, pos_abs_lens, pos_align_idxs):
#         # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#         random_idxs = torch.randperm(pos_sent_reps.size()[0])
#         neg_sent_reps = pos_sent_reps[random_idxs]
#         neg_reps = pos_rep[random_idxs]
#         neg_abs_len = [pos_abs_lens[i] for i in random_idxs.tolist()]
#         neg_align_idxs = [pos_align_idxs[i] for i in random_idxs.tolist()]
#         # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
#         return neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs
#
#     def forward_rank(self, batch_rank):
#         """
#         Function used at training time.
#         batch_dict: dict of the form:
#         {
#             'query_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'query_abs_lens': list(int); Number of sentences in query abs.
#             'query_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#             'pos_abs_lens': list(int);
#             'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from positive abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'pos_align_idxs': list([int int]); query align sent idx, cand align sent idx
#             'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#             'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
#                 concated sentences from query abstracts; Tokenized and int mapped
#                 sentences and other inputs to BERT.
#             'neg_abs_lens': list(int);
#             'neg_align_idxs': list([int int]); query align sent idx, cand align sent idx
#             'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
#                     num_sent_tokens(ints)))
#         }
#         :return: loss_val; torch Variable.
#         """
#         # Unpack batch_rank
#         query_batch, query_abs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
#         pos_batch, pos_abs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
#         query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
#         pos_align_idxs = batch_rank['pos_align_idxs']
#
#         # Get query paper rep and sentences reps from the model
#         query_rep, query_sent_reps = self.partial_forward(bert_batch=query_batch,
#                                                         abs_lens=query_abs_lens,
#                                                         sent_tok_idxs = query_senttoki)
#         # Get positive paper rep and sentences reps from the model
#
#         pos_rep, pos_sent_reps = self.partial_forward(bert_batch=pos_batch,
#                                                           abs_lens=pos_abs_lens,
#                                                           sent_tok_idxs=pos_senttoki)
#         # Happens when running on the dev set.
#         if 'neg_bert_batch' in batch_rank:
#             neg_batch, neg_abs_len = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
#             neg_senttoki = batch_rank['neg_senttok_idxs']
#             neg_reps, neg_sent_reps = self.partial_forward(bert_batch=neg_batch, abs_lens=neg_abs_len,
#                                                              sent_tok_idxs=neg_senttoki)
#             query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
#             pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
#             neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
#             # Dev set based on "predictions" not the pre-alignments. (they can be noisy!)
#             loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
#             if self.abs_loss_prop > 0:
#                 abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
#                 loss_val += self.abs_loss_prop * abs_loss_val
#             return loss_val
#         else:
#             # Use a shuffled set of positives as the negatives. -- in-batch negatives.
#             neg_sent_reps, neg_reps, neg_abs_len, neg_align_idxs = self._prepare_random_in_batch_negatives(pos_sent_reps,
#                                                                                                            pos_rep,
#                                                                                                            pos_abs_lens,
#                                                                                                            pos_align_idxs)
#             # Bundle the lengths with the embeds so the similarity function can use the lengths for masking.
#             query_sents = rep_len_tup(embed=query_sent_reps, abs_lens=query_abs_lens)
#             pos_sents = rep_len_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens)
#             neg_sents = rep_len_tup(embed=neg_sent_reps, abs_lens=neg_abs_len)
#             pos_sents_ali = rep_len_ali_tup(embed=pos_sent_reps, abs_lens=pos_abs_lens, align_idxs=pos_align_idxs)
#             neg_sents_ali = rep_len_ali_tup(embed=neg_sent_reps, abs_lens=neg_abs_len, align_idxs=neg_align_idxs)
#
#             loss_val = self.sentsup_loss_prop * self.criterion_sentsup(query_sents, pos_sents_ali, neg_sents_ali)
#
#             if self.sent_loss_prop > 0:
#                 sent_loss_val = self.criterion_sent(query_sents, pos_sents, neg_sents)
#                 loss_val += self.sent_loss_prop * sent_loss_val
#             if self.abs_loss_prop > 0:
#                 abs_loss_val = self.criterion_abs(query_rep, pos_rep, neg_reps)
#                 loss_val += self.abs_loss_prop * abs_loss_val
#
#             return loss_val
