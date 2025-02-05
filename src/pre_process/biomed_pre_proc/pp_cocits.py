"""
Functions to work with co-citations in each area.
"""
import os
import random
import argparse
import time
import collections
import itertools
import re
import pprint
import pickle
import codecs, json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Paths for saving/loading the vectors
context_vectors_path = "/cs/labs/tomhope/idopinto12/aspire/datasets/train/context_reps.npy"
abstract_vectors_path = "/cs/labs/tomhope/idopinto12/aspire/datasets/train/abstract_reps.npy"

def save_representations(filepath, vectors):
    """Save vectors to a file."""
    np.save(filepath, vectors)
    logger.info(f"Saved vectors to {filepath}")

def load_representations(filepath):
    """Load vectors from a file."""
    if os.path.exists(filepath):
        logger.info(f"Loading vectors from {filepath}")
        return np.load(filepath)
    else:
        return None

#####################
class AbsSentenceStream:
    """
    Given a list of pids,  returns their sentences.
    """

    def __init__(self, in_pids, pid2abstract):
        """
        :param in_pids:
        :param pid2abstract:
        """
        self.in_pids = in_pids
        self.pid2abstract = pid2abstract
        self.num_sents = self.count_sents()

    def __len__(self):
        return self.num_sents

    def count_sents(self):
        nsents = 0
        for pid in self.in_pids:
            doc = self.pid2abstract[int(pid)]['abstract']
            nsents += len(doc)
        return nsents

    def __iter__(self):
        return self.next()

    def next(self):
        # In each loop iteration return one example.
        for pid in self.in_pids:
            doc = self.pid2abstract[int(pid)]['abstract']
            for sent in doc:
                yield sent


class ContextSentenceStream:
    """
    Given a list of pids,  returns their sentences.
    """

    def __init__(self, listofcontexts):
        """
        :param listofcontexts: list(list(tuple(pid, sent)))
        """
        self.listofcontexts = listofcontexts
        self.num_sents = self.count_sents()

    def __len__(self):
        return self.num_sents

    def count_sents(self):
        nsents = 0
        for clist in self.listofcontexts:
            nsents += len(clist)
        return nsents

    def __iter__(self):
        return self.next()

    def next(self):
        # In each loop iteration return one example.
        for clist in self.listofcontexts:
            for c in clist:
                yield c[1]


def filter_cocitation_papers(run_path, dataset):
    """
    Read in the absfilt co-cotations and filter out co-citations using:
    - the number of cocited papers.
    - the number of tokens in the citation context.
    - if the citation context was supriously tagged as a citation context:
        - The heuristic for this is when the sentence doesnt contain any [] or ().
          This is  more important in biomed papers than in compsci papers.
    This is used to train the abstract level similarity modeSSSls.
    """
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    with open(os.path.join(run_path, f'cocitpids2contexts-{area}-absfilt.pickle'), 'rb') as fp:
        cocitpids2contexts = pickle.load(fp)

    # Filter out noise.
    cocitedpids2contexts_filt = {}
    sc_copy_count = 0
    for cocitpids, contexts in cocitpids2contexts.items():
        if len(cocitpids) > 3:
            continue
        else:
            # Sometimes the contexts are exact copies but from diff papers.
            # Get rid of these.
            con2pids = collections.defaultdict(list)
            for sc in contexts:
                # Sometimes they differ only by the inline citation numbers, replace those.
                sc_no_nums = re.sub(r'\d', '', sc[1])
                con2pids[sc_no_nums].append(sc)
            if len(con2pids) < len(contexts):
                sc_copy_count += 1
            uniq_scons = []
            for norm_con, contextt in con2pids.items():
                uniq_scons.append(contextt[0])
            fcons = []
            citing_pids = set()
            for sc in uniq_scons:
                # If the same paper is making the co-citation multiple times
                # only use the first of the co-citations. Multiple by the same citing
                # paper count as a single co-citation.
                if sc[0] in citing_pids:
                    continue
                # Filter context by length.
                if len(sc[1].split()) > 60 or len(sc[1].split()) < 5:
                    continue
                # Filter noisey citation contexts.
                elif ("(" not in sc[1] and ")" not in sc[1]) and ("[" not in sc[1] and "]" not in sc[1]):
                    continue
                else:
                    fcons.append(sc)
                # Update pids only if the sentence was used.
                citing_pids.add(sc[0])
            if len(fcons) > 0:
                cocitedpids2contexts_filt[cocitpids] = fcons

    # Write out filtered co-citations and their stats.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'wb') as fp:
        pickle.dump(cocitedpids2contexts_filt, fp)
        print(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(run_path, f'cocitpids2contexts-{area}-absnoisefilt.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(cocitedpids2contexts_filt.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        print(f'Wrote: {fp.name}')
    num_citcons = []
    example_count = 0  # The approximate number of triples which will be generated as training data.
    for cocitpids, citcontexts in cocitedpids2contexts_filt.items():
        num_citcons.append(len(citcontexts))
        if len(cocitpids) == 2:
            example_count += 1
        elif len(cocitpids) == 3:
            example_count += 3
    all_summ = pd.DataFrame(num_citcons).describe()
    print('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))
    print(f'Copies of co-citation context: {sc_copy_count}')
    print(f'Approximate number of possible triple examples: {example_count}')


def get_all_contexts_and_pids(train_copids, dev_copids, cocitedpids2contexts, train_size, dev_size):
    all_contexts = []
    all_pids = set()
    counter = 0
    lst = []
    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        logger.debug(f"Processing: {split_str}")
        split_examples = 0
        for cocitedpids in tqdm(split_copids):
            contexts = cocitedpids2contexts[cocitedpids] #
            if len(contexts) > 1:
                counter += 1
                lst.append(contexts)
            # Sample at most 10 context sentences at random to use for supervision.
            out_contexts = random.sample(contexts, min(10, len(contexts)))
            all_contexts.append(out_contexts)
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            all_pids.update(cocitedpids)
            split_examples += len(list(cidxs))
            if split_str == 'train' and split_examples > train_size:
                # logger.debug(f"####### Break #######\n split_examples: {split_examples}, train_size: {train_size}")
                break
            elif split_str == 'dev' and split_examples > dev_size:
                # logger.debug(f"####### Break #######\n split_examples: {split_examples}, dev: {dev_size}")
                break
    # logger.debug(f"Number of context list of length greater then 1: {counter}")
    # logger.debug(f"Example: {lst[0]}")
    '''
    DEBUG:__main__:Number of context list of length greater then 2: 3998
    DEBUG:__main__:Number of context list of length greater then 1: 24056
    '''
    all_pids = list(all_pids)
    print(f'Number of contexts: {len(all_contexts)}; Number of unique abstracts: {len(all_pids)}')
    return all_contexts, all_pids

def encode_stream(model, stream, save_path, type=""):
    # DEBUG:__main__:Context reps shape: (933360, 768); Stream sentences: 933360
    pool = model.start_multi_process_pool()
    start = time.time()
    logger.debug("Encoding abstract sents..........")
    reps = model.encode_multi_process(stream, pool)
    model.stop_multi_process_pool(pool)
    save_representations(save_path, reps)
    logger.debug('Encoding took: {:.4f}s'.format(time.time() - start))
    logger.debug(f"{type} shape: {reps.shape}; stream length: {len(stream)}")
    return reps

def generate_examples_aligned_cocitabs_rand(in_path, out_path, dataset, alignment_model, trained_model_path=None):
    """
    Assumes random (in-batch) negatives are used and only generates pair
    examples of query/anchor and positive for co-cited abstracts.
    - Also generate a alignment for the positive and negative based
    - Generate negatives for the dev set so its a frozen dev set.
    """

    train_size, dev_size = 1276820, 10000
    random.seed(69306)
    dataset2area = {
        's2orccompsci': 'compsci',
        's2orcbiomed': 'biomed'
    }
    area = dataset2area[dataset]
    logger.info(f'Reading: cocitpids2contexts-{area}-absnoisefilt.pickle')
    with codecs.open(os.path.join(in_path, f'cocitpids2contexts-{area}-absnoisefilt.pickle'), 'rb') as fp:
        cocitedpids2contexts = pickle.load(fp)
        logger.info(f'Read: {fp.name}')

    all_cocits = list(cocitedpids2contexts.keys())
    random.shuffle(all_cocits)
    total_copids = len(all_cocits)
    n = int(0.8 * total_copids)
    train_copids, dev_copids = all_cocits[:n], all_cocits[n:]
    logger.info(f'cocited pid sets; train: {len(train_copids)}; dev: {len(dev_copids)}')
    with codecs.open(os.path.join(in_path, f'pid2abstract-s2orc{area}.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
        all_abs_pids = list(pid2abstract.keys())
        logger.info(f'Read: {fp.name}')

    # all_cocits[0] -> ('30889122', '453426'), <class 'tuple'>
    # all_abs_pids[0] -> 18981358, <class 'numpy.int64'>
    logger.info(f"{all_cocits[0]}, {type(all_cocits[0])}\n {all_abs_pids[0]}, {type(all_abs_pids[0])}")
    if alignment_model in {'cosentbert'}:
        outfname_suffix = 'cocitabsalign'
        word_embedding_model = models.Transformer('allenai/aspire-sentence-embedder', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        sent_alignment_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        # logger.debug(os.system("nvidia-smi"))
        # logger.info(sent_alignment_model)
        # word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
        #                                           max_seq_length=512)
        # trained_model_fname = os.path.join(trained_model_path, 'sent_encoder_cur_best.pt')
        # word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        # sent_alignment_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    all_contexts, all_pids = get_all_contexts_and_pids(train_copids, dev_copids, cocitedpids2contexts,train_size, dev_size)
    context_stream = ContextSentenceStream(listofcontexts=all_contexts)
    abstract_stream = AbsSentenceStream(in_pids=all_pids, pid2abstract=pid2abstract)
    all_context_reps = load_representations(context_vectors_path)
    all_abs_sent_reps = load_representations(abstract_vectors_path)
    if all_context_reps is None:
        all_context_reps = encode_stream(model=sent_alignment_model,stream=context_stream,save_path=context_vectors_path, type="Context reps")
    if all_abs_sent_reps is None:
        # DEBUG:__main__:Encoding took: 5731.4139s
        all_abs_sent_reps = encode_stream(model=sent_alignment_model,stream=abstract_stream,save_path=context_vectors_path, type="Abstract sents reps")

    # Go over the abstract reps and put them into a dict
    logger.debug("Going over the abstract reps and put them into a dict.")
    abs_reps_start_idx = 0
    pid2abs_reps = {}
    for pid in all_pids:
        num_sents = len(pid2abstract[int(pid)]['abstract'])
        abs_reps = all_abs_sent_reps[abs_reps_start_idx:abs_reps_start_idx + num_sents, :]
        abs_reps_start_idx += num_sents
        pid2abs_reps[pid] = abs_reps
    logger.debug("Done!")
    # Now form examples.
    logger.debug("Now form training examples (triples)")
    contextsi = 0
    context_reps_start_idx = 0
    for split_str, split_copids in [('train', train_copids), ('dev', dev_copids)]:
        out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-{outfname_suffix}.jsonl'), 'w', 'utf-8')
        out_examples = 0
        num_context_sents = []
        for cocitedpids in tqdm(split_copids):
            out_contexts = all_contexts[contextsi]
            context_sents = [cc[1] for cc in out_contexts]
            citing_pids = [cc[0] for cc in out_contexts]
            context_reps = all_context_reps[context_reps_start_idx: context_reps_start_idx + len(context_sents), :]
            context_reps_start_idx += len(context_sents)
            contextsi += 1
            # Generate all combinations of length 2 given the contexts.
            cidxs = itertools.combinations(range(len(cocitedpids)), 2)
            for idxs in cidxs:
                anchor_pid = cocitedpids[idxs[0]]
                pos_pid = cocitedpids[idxs[1]]
                qabs_reps = pid2abs_reps[anchor_pid]
                posabs_reps = pid2abs_reps[pos_pid]
                cc2query_abs_sims = np.matmul(qabs_reps, context_reps.T)
                cc2query_idxs = np.unravel_index(cc2query_abs_sims.argmax(), cc2query_abs_sims.shape)
                cc2pos_abs_sims = np.matmul(posabs_reps, context_reps.T)
                cc2pos_idxs = np.unravel_index(cc2pos_abs_sims.argmax(), cc2pos_abs_sims.shape)
                abs2cc2abs_idx = (int(cc2query_idxs[0]), int(cc2pos_idxs[0]))
                q2pos_abs_sims = np.matmul(qabs_reps, posabs_reps.T)
                q2pos_idxs = np.unravel_index(q2pos_abs_sims.argmax(), q2pos_abs_sims.shape)
                abs2abs_idx = (int(q2pos_idxs[0]), int(q2pos_idxs[1]))
                anchor_abs = {'TITLE': pid2abstract[int(anchor_pid)]['title'],
                              'ABSTRACT': pid2abstract[int(anchor_pid)]['abstract']}
                pos_abs = {'TITLE': pid2abstract[int(pos_pid)]['title'],
                           'ABSTRACT': pid2abstract[int(pos_pid)]['abstract'],
                           'cc_align': abs2cc2abs_idx,
                           'abs_align': abs2abs_idx}
                out_ex = {
                    'citing_pids': citing_pids,
                    'cited_pids': cocitedpids,
                    'query': anchor_abs,
                    'pos_context': pos_abs,
                    'citing_contexts': context_sents
                }
                num_context_sents.append(len(citing_pids))
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_pid = random.choice(all_abs_pids)
                    rand_anch_idx, rand_neg_idx = random.choice(range(len(pid2abstract[int(anchor_pid)]['abstract']))), \
                        random.choice(range(len(pid2abstract[neg_pid]['abstract'])))
                    neg_cc_align = (rand_anch_idx, rand_neg_idx)
                    rand_anch_idx, rand_neg_idx = random.choice(range(len(pid2abstract[int(anchor_pid)]['abstract']))), \
                        random.choice(range(len(pid2abstract[neg_pid]['abstract'])))
                    neg_abs_align = (rand_anch_idx, rand_neg_idx)
                    # metadata_df[metadata_df['paper_id'] == int(neg_pid)]['title'].iloc[0]
                    neg_abs = {'TITLE': pid2abstract[int(neg_pid)]['title'],
                               'ABSTRACT': pid2abstract[int(neg_pid)]['abstract'],
                               'cc_align': neg_cc_align, 'abs_align': neg_abs_align}
                    out_ex['neg_context'] = neg_abs
                out_ex_file.write(json.dumps(out_ex) + '\n')
                out_examples += 1
                if out_examples % 1000 == 0:
                    logger.info(f'{split_str}; {out_examples}')
            # if out_examples > 1000:
            #     break
            # Do this only for 1.2m triples, then exit.
            if split_str == 'train' and out_examples > train_size:
                break
            elif split_str == 'dev' and out_examples > dev_size:
                break
        logger.info(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(num_context_sents).describe()
        logger.info('Number of cit contexts per triple: {:}'.format(all_summ))
        logger.info(f'Number of examples: {out_examples}')



def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Filter for abstract level models.
    filter_cocit_papers = subparsers.add_parser('filt_cocit_papers')
    filter_cocit_papers.add_argument('--run_path', required=True,
                                     help='Directory with absfilt cocitation pickle file. '
                                          'Also where outputs are written.')
    filter_cocit_papers.add_argument('--dataset', required=True,
                                     choices=['s2orccompsci', 's2orcbiomed'],
                                     help='Files of area to process.')
    # Write examples for sentence level models.
    write_example_sents = subparsers.add_parser('write_examples')
    write_example_sents.add_argument('--in_path', required=True,
                                     help='Directory with absfilt cocitation pickle file.')
    write_example_sents.add_argument('--out_path', required=True,
                                     help='Directory where outputs are written.')
    write_example_sents.add_argument('--model_path',
                                     help='Directory where trained sentence bert model is.')
    write_example_sents.add_argument('--model_name', choices=['cosentbert', 'specter', 'sbmpnet1B'],
                                     help='Model to use for getting alignments between abstracts.')
    write_example_sents.add_argument('--dataset', required=True,
                                     choices=['s2orccompsci', 's2orcbiomed'],
                                     help='Files of area to process.')
    write_example_sents.add_argument('--experiment', required=True,
                                     choices=['cosentbert', 'ictsentbert', 'cospecter',
                                              'sbalisentbienc'],
                                     help='Model writing examples for.')
    cl_args = parser.parse_args()
# python3 -m src.pre_process.pre_proc_cocits filt_cocit_papers --run_path /cs/labs/tomhope/idopinto12/aspire/datasets/train --dataset s2orcbiomed
    if cl_args.subcommand == 'filt_cocit_papers':
        filter_cocitation_papers(run_path=cl_args.run_path, dataset=cl_args.dataset)
# python3 -m src.pre_process.pp_cocits write_examples --in_path /cs/labs/tomhope/idopinto12/aspire/datasets/train --out_path /cs/labs/tomhope/idopinto12/aspire/datasets/train --dataset s2orcbiomed --model_name cosentbert --experiment sbalisentbienc
    elif cl_args.subcommand == 'write_examples':
        # pass
        if cl_args.experiment in {'sbalisentbienc'}:
            generate_examples_aligned_cocitabs_rand(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                                    dataset=cl_args.dataset, trained_model_path=cl_args.model_path,
                                                    alignment_model=cl_args.model_name)


if __name__ == '__main__':
    main()

'''

[('2340344', '7] [8] [9] [10] [11] Our findings are independent of sociodemographic factors and select learning and enrichment variables.')]

'''

# if all_context_reps is None:
#     # Encode documents.
#     pool = sent_alignment_model.start_multi_process_pool()
#     # Compute the embeddings using the multi-process pool
#     start = time.time()
#     logger.debug("Encoding contexts..........")
#     all_context_reps = sent_alignment_model.encode_multi_process(context_stream, pool)
#     sent_alignment_model.stop_multi_process_pool(pool)
#     save_representations(context_vectors_path, all_context_reps)
#     logger.debug('Encoding took: {:.4f}s'.format(time.time() - start))
#
# logger.debug(f"Context reps shape: {all_context_reps.shape}; Stream sentences: {len(context_stream)}")
# all_abs_sent_reps = load_representations(abstract_vectors_path)
# if all_abs_sent_reps is None:
#     # DEBUG:__main__:Context reps shape: (933360, 768); Stream sentences: 933360
#     pool = sent_alignment_model.start_multi_process_pool()
#     start = time.time()
#     logger.debug("Encoding abstract sents..........")
#     all_abs_sent_reps = sent_alignment_model.encode_multi_process(abstract_stream, pool)
#     sent_alignment_model.stop_multi_process_pool(pool)
#     save_representations(abstract_vectors_path, all_abs_sent_reps)
#     logger.debug('Encoding took: {:.4f}s'.format(time.time() - start))
# logger.debug(f"Abstract sentences representations shape: {all_abs_sent_reps.shape}; # of abstract-stream sentences: {len(abstract_stream)}")
