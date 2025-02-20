import json
from dataclasses import dataclass
from typing import Optional, Iterator, Any, Dict, List, Tuple
import numpy as np
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


@dataclass
class BatcherConfig:
   num_examples: int
   batch_size: int
   max_length: int = 512
   padding_side: str = 'left'
   model_name: Optional[str] = None
   base_pt_layer: str = None
   query_instruct: bool = False
   task_description: str = "Represent the core scientific contributions and findings of this paper."

class GenericBatcher(ABC):
    def __init__(self, config: BatcherConfig):
       self.config = config
       self._initialize_batch_tracking()

    def _initialize_batch_tracking(self):
        if self.config.num_examples <= 0  or self.config.batch_size <= -1:
            return
        self.full_len = self.config.num_examples
        self.batch_size = self.config.batch_size
        self.num_batches = max(1, int(np.ceil(self.full_len / self.batch_size)))
        self.batch_start = 0
        self.batch_end = self.batch_size

    def next_batch(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement next_batch")

    @staticmethod
    def raw_batch_from_file(ex_file: Any, to_read_count: int) -> Iterator[Any]:
        raise NotImplementedError("Subclasses must implement raw_batch_from_file")

    @property
    def has_next_batch(self) -> bool:
        return self.batch_start < self.full_len

class SentTripleBatcher(GenericBatcher):
    _config: Optional[BatcherConfig] = None

    def __init__(self, config: BatcherConfig, ex_fnames: Dict[str, str]):
        super().__init__(config)
        SentTripleBatcher._config = config  # Set class-level config
        self.tokenizer = self._setup_tokenizer()
        self.pos_ex_file = open(ex_fnames['pos_ex_fname'], 'r') if ex_fnames else None

    def _setup_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(SentTripleBatcher._config.base_pt_layer)
        tokenizer.padding_side = self.config.padding_side
        print(f"Tokenizer setup completed.\n tokenizer config: {tokenizer}")
        return tokenizer

    @classmethod
    def get_tokenizer(cls):
        if cls._config is None:
            raise ValueError("Batcher not initialized, config is None")
        tokenizer = AutoTokenizer.from_pretrained(cls._config.base_pt_layer)
        tokenizer.padding_side = cls._config.padding_side
        return tokenizer

    def next_batch(self) -> Iterator[Tuple[List[int], Dict]]:
        for _ in range(self.num_batches):
           # Get current batch size
           cur_batch_size = min(self.batch_size, self.full_len - self.batch_start)

           # Get batch data
           batch_ids, queries, pos, neg = next(self.raw_batch_from_file(self.pos_ex_file, cur_batch_size))
           # Prepare feed based on available data
           feed = self._prepare_feed(queries, pos, neg)

           # Tokenize and format batch
           batch_dict = {'batch_rank': self.make_batch(raw_feed=feed, tokenizer=self.tokenizer)}

           # Update batch indices
           self.batch_start = self.batch_end
           self.batch_end += self.batch_size

           yield batch_ids, batch_dict

    def _prepare_feed(self, queries: List[str], pos: List[str], neg: List[str]) -> Dict:
       prep_pos = all(x is not None for x in pos)
       prep_neg = all(x is not None for x in neg)
       if prep_neg and prep_pos:
           return {'query_texts': queries, 'pos_texts': pos, 'neg_texts': neg}
       elif prep_pos:
           return {'query_texts': queries, 'pos_texts': pos}
       return {'query_texts': queries}

    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count: int) -> Iterator[Tuple]:
           read_count, ids, queries, pos, neg = 0, [], [], [], []
           for ex in ex_file:
               ex = json.loads(ex)
               ids.append(read_count)
               queries.append(ex['query'])
               pos.append(ex.get('pos_context'))
               neg.append(ex.get('neg_context'))
               read_count += 1
               if read_count == to_read_count:
                   yield ids, queries, pos, neg
                   read_count, ids, queries, pos, neg = 0, [], [], [], []

    @staticmethod
    def make_batch(raw_feed: Dict, tokenizer) -> Dict:
        batch_dict = {}
        # query_texts, pos_texts, neg_texts: actually sents is batch of abstracts
        for key, sents in raw_feed.items():
            query_instruct = SentTripleBatcher._config.query_instruct if "query" in key else False
            batch,_, _ = SentTripleBatcher.prepare_bert_sentences(sents, tokenizer, query_instruct)
            batch_type = key.split("_")[0] # query, pos or neg (not all exist together)
            batch_dict[f"{batch_type}_bert_batch"] = {
               'tokid_tt': batch['input_ids'],
               'attnmask_tt': batch['attention_mask'],
               'seq_lens': batch['attention_mask'].sum(1).tolist()
            }
        return batch_dict


    @staticmethod
    def prepare_bert_sentences(sents: List[str], tokenizer, query_instruct) -> Tuple[Dict[str, torch.Tensor], List[List[str]], List[List[int]]]:
        max_len = SentTripleBatcher._config.max_length if query_instruct else 500
        tokenized= tokenizer(sents,
                             padding=True,
                             truncation=True,
                             max_length=max_len,
                             return_tensors='pt')

        tokenized = add_eos_token(tokenized, tokenizer.eos_token_id)

        batch = {
           'tokid_tt': tokenized['input_ids'],
           'attnmask_tt': tokenized['attention_mask'],
           'seq_lens': tokenized['attention_mask'].sum(1).tolist()
        }

        # For backwards compatibility
        tokenized_text = [tokenizer.tokenize(s)[:max_len] for s in sents]
        tokenized_batch = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_text]

        return batch, tokenized_text, tokenized_batch

class AbsTripleBatcher(SentTripleBatcher):
    _config: Optional[BatcherConfig] = None

    def __init__(self, config: BatcherConfig, ex_fnames: Dict[str, str]):
        super().__init__(config, ex_fnames)
        AbsTripleBatcher._config = config  # Set class-level config

    @staticmethod
    def make_batch(raw_feed: Dict[str, List[Dict]], tokenizer) -> Dict:
        query_batch = AbsTripleBatcher.prepare_abstracts(raw_feed['query_texts'], tokenizer, AbsTripleBatcher._config.query_instruct)

        batch_dict = {'query_bert_batch': query_batch}
        if 'pos_texts' in raw_feed:
            batch_dict['pos_bert_batch'] = AbsTripleBatcher.prepare_abstracts(raw_feed['pos_texts'], tokenizer,query_instruct=False)
        if  'neg_texts' in raw_feed:
            batch_dict['neg_bert_batch'] = AbsTripleBatcher.prepare_abstracts(raw_feed['neg_texts'], tokenizer,query_instruct=False)

        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs: List[Dict], tokenizer, query_instruct:bool=False) -> Dict:
        batch_seqs = []
        for abstract in batch_abs:
            if query_instruct:
                text = get_detailed_instruct(BatcherConfig.task_description, abstract['TITLE'])
            else:
                text = f"{abstract['TITLE']}. "
            text += " ".join(abstract['ABSTRACT']) # transform from title, [s1,..,sn] -> instruct(title)+s1+..+sn
            batch_seqs.append(text)
        tokenized = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length= AbsTripleBatcher._config.max_length if AbsTripleBatcher._config.query_instruct else 500,
            return_tensors="pt"
        )

        tokenized = add_eos_token(tokenized, tokenizer.eos_token_id)

        return {
            'tokid_tt': tokenized['input_ids'],
            'attnmask_tt': tokenized['attention_mask'],
            'seq_lens': tokenized['attention_mask'].sum(1).tolist()
        }

def add_eos_token(tokenized: Dict[str, torch.Tensor], eos_id: int) -> Dict[str, torch.Tensor]:
    batch_tokens = tokenized['input_ids']
    batch_attn_mask = tokenized['attention_mask']
    eos_attns = torch.ones((tokenized['attention_mask'].shape[0], 1))
    eos_tokens = torch.full((tokenized['input_ids'].shape[0], 1), eos_id)
    tokenized['input_ids'] = torch.cat([batch_tokens, eos_tokens], dim=1)
    tokenized['attention_mask'] = torch.cat([batch_attn_mask, eos_attns], dim=1)
    return tokenized

def get_detailed_instruct(task_description: str, title: str) -> str:
    return f'Instruct: {task_description}\nQuery: {title}'


class AbsSentTokBatcher(SentTripleBatcher):
    """
    Feeds a model which inputs query, positive and negative abstracts and sentence
    TOKEN indices for the abstracts. Negatives only at dev time, else the model uses in-batch
    negatives.
    """
    _config: Optional[BatcherConfig] = None
    def __init__(self, config: BatcherConfig, ex_fnames: Dict[str, str]):
        super().__init__(config, ex_fnames)
        AbsSentTokBatcher._config = config  # Set class-level config
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer, query_instruct:bool=False):
        """
        :param bert_like:
        :param query_instruct:
        :param pt_lm_tokenizer:åå
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'query_abs_lens': list(int); Number of sentences in query abs.
                'query_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_abs_lens': list(int);
                'pos_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_abs_lens': list(int);
                'neg_senttok_idxs': list(list(list(int))); batch_size(
                    num_abs_sents(num_sent_tokens(ints)))
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, qabs_len, qabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
            query_texts, pt_lm_tokenizer, query_instruct=query_instruct)

        # Happens in the dev set.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, nabs_len, nabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
                neg_texts, pt_lm_tokenizer, query_instruct=False)
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
                pos_texts, pt_lm_tokenizer, query_instruct=False)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'neg_bert_batch': nbert_batch, 'neg_abs_lens': nabs_len, 'neg_senttok_idxs': nabs_senttok_idxs
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(pos_texts,
                                                                                           pt_lm_tokenizer,
                                                                                           query_instruct=False)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch, 'abs_lens': qabs_len, 'senttok_idxs': qabs_senttok_idxs
            }
        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs, tokenizer, query_instruct: bool = False):
        """
        Given a batch of documents (each as a list of sentences with the first sentence as the title),
        prepare a batch for the model while tracking token indices for each sentence.
        The indices are adjusted to account for left padding.

        :param batch_abs: List[List[str]]; each document is a list of sentences (title + abstract sentences).
        :param tokenizer: Initialized tokenizer.
        :param query_instruct: If True, modify the title with detailed instructions.
        :return:
            batch: dict; tokenized batch with "input_ids" and "attention_mask".
            num_sents_per_abstract: List[int]; number of abstract sentences (excluding title) per document.
            batch_sent_token_idxs: List[List[List[int]]]; for each document, a list (per abstract sentence) of token indices.
        """
        max_length = AbsSentTokBatcher._config.max_length if query_instruct else 500
        batch_tokenized_text = []
        batch_sent_token_idxs = []
        num_sents_per_abstract = []

        for abs_sents in batch_abs:
            # Optionally modify the title with detailed instructions.
            if query_instruct:
                abs_sents[0] = get_detailed_instruct(AbsSentTokBatcher._config.task_description, abs_sents[0])

            tokenized_sentences = []  # List of token lists for each sentence.
            sent_token_indices = []  # List of token index lists for each sentence.
            current_index = 0

            # Tokenize each sentence and record its indices (pre-padding)
            for sent in abs_sents:
                tokens = tokenizer.tokenize(sent)
                tokenized_sentences.append(tokens)
                indices = list(range(current_index, current_index + len(tokens)))
                sent_token_indices.append(indices)
                current_index += len(tokens)

            # Flatten tokens from all sentences.
            all_tokens = [tok for sent in tokenized_sentences for tok in sent]
            # Truncate to max_length.
            truncated_tokens = all_tokens[:max_length]
            # If there's room, append the EOS token.
            if len(truncated_tokens) < max_length:
                truncated_tokens.append(tokenizer.eos_token)

            # After truncation, the length of our tokens (before padding)
            current_len = len(truncated_tokens)
            # For left padding, the encoder will pad on the left to reach max_length.
            pad_offset = max_length - current_len

            # Adjust the sentence indices:
            adjusted_sent_indices = []
            for indices in sent_token_indices:
                # Only keep indices that fall within the truncated tokens.
                adjusted = [i for i in indices if i < current_len]
                # Shift the indices by pad_offset so they refer to positions in the final padded tensor.
                adjusted = [i + pad_offset for i in adjusted]
                adjusted_sent_indices.append(adjusted)

            # Exclude the title (first sentence) from the returned sentence indices.
            batch_sent_token_idxs.append(adjusted_sent_indices[1:])
            num_sents_per_abstract.append(len(adjusted_sent_indices) - 1)

            batch_tokenized_text.append(truncated_tokens)

        # Encode the pre-tokenized lists. We use is_split_into_words=True to prevent re-tokenization.
        encoded_inputs = tokenizer.batch_encode_plus(
            batch_tokenized_text,
            is_split_into_words=True,
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",  # This will add left padding since tokenizer.padding_side is "left".
            truncation=False,  # Already truncated manually.
            return_tensors="pt",
        )

        batch = {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
        }

        return batch, num_sents_per_abstract, batch_sent_token_idxs

class AbsSentTokBatcherPreAlign(AbsSentTokBatcher):
    """
    Feeds a model which inputs query, positive and negative abstracts and sentence
    TOKEN indices for the abstracts. Negatives only at dev time, else the model uses in-batch
    negatives.
    """
    # Which pre-aligned index to read. Can be: {'cc_align', 'abs_align'}
    align_type = 'cc_align'

    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer, query_instruct=False, bert_like=True):
        """
        :param bert_like:
        :param query_instruct:
        :param pt_lm_tokenizer:
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'query_abs_lens': list(int); Number of sentences in query abs.
                'query_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_abs_lens': list(int);
                'pos_align_idxs': list([int int]); query align sent idx, cand align sent idx
                'pos_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_abs_lens': list(int);
                'neg_align_idxs': list([int int]); query align sent idx, cand align sent idx
                'neg_senttok_idxs': list(list(list(int))); batch_size(
                    num_abs_sents(num_sent_tokens(ints)))
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, qabs_len, qabs_senttok_idxs = AbsSentTokBatcher.prepare_abstracts(
            query_texts, pt_lm_tokenizer, query_instruct=query_instruct)

        # Happens in the dev set.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, nabs_len, nabs_senttok_idxs, neg_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                neg_texts, pt_lm_tokenizer, query_instruct=False)
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs, pos_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                pos_texts, pt_lm_tokenizer, query_instruct=False)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'neg_bert_batch': nbert_batch, 'neg_abs_lens': nabs_len, 'neg_senttok_idxs': nabs_senttok_idxs,
                'pos_align_idxs': pos_align_idxs, 'neg_align_idxs': neg_align_idxs
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch, pabs_len, pabs_senttok_idxs, pos_align_idxs = AbsSentTokBatcherPreAlign.prepare_abstracts(
                pos_texts, pt_lm_tokenizer, query_instruct=False, bert_like=bert_like)
            batch_dict = {
                'query_bert_batch': qbert_batch, 'query_abs_lens': qabs_len, 'query_senttok_idxs': qabs_senttok_idxs,
                'pos_bert_batch': pbert_batch, 'pos_abs_lens': pabs_len, 'pos_senttok_idxs': pabs_senttok_idxs,
                'pos_align_idxs': pos_align_idxs
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch, 'abs_lens': qabs_len, 'senttok_idxs': qabs_senttok_idxs
            }
        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer, query_instruct=False, bert_like=True):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param bert_like:
        :param query_instruct:
        :param pt_lm_tokenizer:
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
            pre_computed_alignments: list(list(int)); batch_size([q_idx, c_idx])
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        pre_computed_alignments = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        sep = " [SEP] " if bert_like else "\n"
        for ex_abs in batch_abs:
            seqs = [ex_abs['TITLE'] + sep]
            seqs.extend([s for s in ex_abs['ABSTRACT']])
            batch_abs_seqs.append(seqs)
            if AbsSentTokBatcherPreAlign.align_type in ex_abs:
                assert(len(ex_abs[AbsSentTokBatcherPreAlign.align_type]) == 2)
                pre_computed_alignments.append(ex_abs[AbsSentTokBatcherPreAlign.align_type])
        bert_batch, tokenized_abs, sent_token_idxs = AbsSentTokBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer,query_instruct=False)

        # Get SEP indices from the sentences; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for abs_sent_tok_idxs in sent_token_idxs:
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)

        if pre_computed_alignments:
            assert(len(pre_computed_alignments) == len(abs_lens))
            return bert_batch, abs_lens, sent_token_idxs, pre_computed_alignments
        else:
            return bert_batch, abs_lens, sent_token_idxs


# @dataclass
# class BatchOutput:
#     input_ids: torch.Tensor
#     attention_mask: torch.Tensor
#     sequence_lengths: List[int]
#
# @dataclass
# class SentenceTokenBatch:
#    batch: BatchOutput
#    sentence_lengths: List[int]
#    sentence_token_indices: List[List[List[int]]]
#
# @dataclass
# class TripletBatch:
#    query: SentenceTokenBatch
#    positive: Optional[SentenceTokenBatch] = None
#    negative: Optional[SentenceTokenBatch] = None