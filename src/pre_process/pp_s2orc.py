import argparse
import ast
import codecs
import collections
import gzip
import json
import csv
import os
import time
import pandas as pd
import spacy
import pp_settings as pps
import data_utils as du
import multiprocessing as mp
import pprint
import pickle
from tqdm import tqdm
import logging

# Configure logging to write to both console and file
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler(log_file, mode='w', encoding='utf-8')  # Logs to file
        ]
    )
    logger = logging.getLogger()
    return logger

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')

def filter_for_fulltext(args):
    """
    Filter metadata for papers that have full-text PDFs available and belong to the specified field of study.

    This function reads a JSON metadata file, filters rows where 'has_pdf_parse' is True,
    and further narrows down the results to papers within the 'Medicine' field of study
    if the 'mag_field_of_study' column contains a list with the value 'Medicine'.

    Args:
        args (tuple): A tuple containing:
            - in_fname (str): The input file path to the JSON metadata file.
            - area (str): The research area being filtered (unused directly, assumed 'Medicine').
            - _ (placeholder): An unused parameter, kept for compatibility with calling functions.

    Returns:
        tuple: A tuple containing:
            - total_row_count (int): The total number of rows in the original DataFrame.
            - filtered_df (pd.DataFrame): A DataFrame containing rows that passed the filtering criteria.
              If an error occurs, this will be an empty DataFrame.

    Logs:
        - Info log on reading the file and number of rows filtered.
        - Error log if processing fails for the given file.

    Notes:
        - The function extracts the batch number from the file name (assumes naming convention).
        - Adds a 'batch_num' column to the filtered DataFrame for tracking the batch source.

    Example:
        >>> filter_for_fulltext(("metadata_01.json.gz", "biomed", None))
        Reading metadata_01.json.gz
        Filtered metadata_01.json.gz: 150 rows
        (1000, filtered_df)
    """
    in_fname, area,logger, filtered_meta_path,_ = args
    try:
        logger.info(f"Reading {in_fname}")
        df = pd.read_json(in_fname, compression='gzip', lines=True)
        logger.info(f"Read {in_fname}: {df.shape[0]} rows")
        if area =='biomed':
            area_name = 'Medicine'
        elif area == 'compsci':
            area_name = 'Computer Science'
        filtered_df = df[
            (df['has_pdf_parse']) &
            (df['mag_field_of_study'].apply(lambda x: area_name in x if isinstance(x, list) else False))
            ]
        batch_num = int(in_fname.split('.')[0].split('_')[-1])
        filtered_df['batch_num'] = batch_num
        logger.info(f"Filtered {in_fname}: {filtered_df.shape[0]} rows")
        if not filtered_df.empty:
            intermediate_file = os.path.join(filtered_meta_path, f"intermediate_{batch_num}.tsv")
            filtered_df.to_csv(intermediate_file, sep='\t', index=False)
            logger.info(f"Wrote {intermediate_file}")
        return df.shape[0]
    except Exception as e:
        logger.error(f"Error processing {in_fname}: {e}")
        return 0


def filter_metadata(
    raw_meta_path,
    filtered_meta_path,
    filter_nan_cols=None,
    filter_method=None,
    area='',
    log_file=None
):
    """
    Filter metadata files and save the filtered results to a TSV file using memory-efficient concatenation.

    Args:
        raw_meta_path (str): Path to the directory containing raw metadata JSON files.
        filtered_meta_path (str): Path to the directory for saving filtered metadata results.
        filter_nan_cols (list, optional): Column names based on which rows with NaN values should be excluded.
        filter_method (str): Filtering method; only 'full text' is supported.
        area (str): Area of research for filtering (e.g., 'biomed', 'compsci').

    Returns:
        None: Writes the filtered results to a TSV file.

    Raises:
        ValueError: If an unsupported filter method is provided.

    Notes:
        - Uses multiprocessing and writes intermediate results to disk to handle large DataFrames.
        - Skips already processed batches by checking for existing intermediate files.
    """
    if log_file:
        logger = setup_logging(log_file)
    else:
        logger = logging.getLogger()

    if filter_method != 'full text':
        logger.error("Unsupported filter method. Only 'full text' is allowed.")
        raise ValueError("Unsupported filter method. Only 'full text' is allowed.")

    filt_function = filter_for_fulltext
    filt_file = os.path.join(filtered_meta_path, f'metadata-s2orcfulltext-{area}.tsv')
    logger.info(f"Final output file will be: {filt_file}...")

    # Get and sort metadata files
    raw_metadata_files = sorted(
        file for file in os.listdir(raw_meta_path) if file.endswith(".jsonl.gz")
    )

    processed_files = set(
        int(file.split('_')[-1].split('.')[0])
        for file in os.listdir(filtered_meta_path)
        if file.startswith("intermediate_") and file.endswith(".tsv")
    )

    logger.info(f"Found {len(processed_files)} already processed batches.")
    to_process = [file for file in raw_metadata_files if int(file.split('_')[-1].split('.')[0]) not in processed_files][:20]

    di = du.DirIterator(
        root_path=raw_meta_path,
        yield_list=to_process,
        args=(area, logger, filtered_meta_path, filter_nan_cols),
    )
    logger.info(f"Filtering {len(to_process)} batches...")

    # Prepare multiprocessing
    process_pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=10000)
    start_time = time.time()
    total_rows = 0
    try:
        # Process files using multiprocessing
        for i, total_row_count in enumerate(process_pool.imap_unordered(filt_function, di, chunksize=mp.cpu_count())):
            total_rows += total_row_count
            logger.info(f"Processed {total_rows} rows so far.")

    finally:
        process_pool.close()
        process_pool.join()

    # Combine intermediate results efficiently
    processed_files = [file for  file in os.listdir(filtered_meta_path) if file.startswith("intermediate_")]
    if len(processed_files) == len(raw_metadata_files):
        logger.info("Combining filtered intermediate files.")
        combined_df = pd.read_csv(os.path.join(filtered_meta_path,processed_files[0]), sep='\t')
        # Loop through the remaining files and skip their headers
        for i, file in enumerate(processed_files[1:]):
            df_temp = pd.read_csv(os.path.join(filtered_meta_path,file), sep='\t', header=0)  # `header=0` skips the first row
            combined_df = pd.concat([combined_df, df_temp], ignore_index=True)
            logger.info(f"Concatenated {2 + i} files.")

        logger.info(f"Final filtered rows: {combined_df.shape[0]}")
        combined_df.to_csv(filt_file, sep='\t', index=False)
        logger.info(f"Wrote filtered metadata to: {filt_file}")
    else:
        logger.info(f"Needs more {len(raw_metadata_files) - len(processed_files)} intermediate files to combine.")

    elapsed_time = time.time() - start_time
    logger.info(f"Filtering complete. Total time: {elapsed_time:.2f}s.")

def write_batch_papers(args):
    """
    Given a batch file, read the papers from it mentioned in the metadata-df
    and write it to disk as a jsonl file.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param pids: pids of the papers we want from the current batch file.
    :return: wrote_count: int; how many jsonl rows were written to the batch output.
    """
    jsonl_fname, pids, filtered_data_path, area, logger = args
    batch_num = int(os.path.basename(jsonl_fname).split('.')[0].split('_')[-1])
    logger.info(f"Processing raw batch:{batch_num} with pids:{len(pids)}")
    if len(pids) > 0:
        data_file = gzip.open(jsonl_fname)
        out_path = os.path.join(filtered_data_path, f'{batch_num}.jsonl')
        out_file = codecs.open(out_path, 'w', encoding='utf-8')
        # out_file = codecs.open(os.path.join(filtered_data_path, '{:d}.jsonl'.format(batch_num), 'w', 'utf-8')
        logger.info(f"Creating {out_path} ")
        for line in data_file:
            data_json = json.loads(line.strip())
            if int(data_json['paper_id']) in pids:
                out_file.write(json.dumps(data_json) + '\n')
        out_file.close()
        return len(pids)
    else:
        return 0

def gather_papers(meta_fname, raw_data_path,area, log_file):
    """
    Read metadata for (filtered) files and gather the filtered files from the full
    collection.
    :return:
    """
    if log_file:
        logger = setup_logging(log_file)
    else:
        logger = logging.getLogger()

    # Construct output dir path by removing "meta" and ".tsv" from end.
    filtered_data_path = os.path.join(os.path.dirname(meta_fname), os.path.basename(meta_fname)[4:-4])
    logger.info(f"Gathering {area} papers to dir: {filtered_data_path}")
    du.create_dir(filtered_data_path)
    metadata_df = pd.read_csv(meta_fname, delimiter='\t', error_bad_lines=False,
                              engine='python', quoting=csv.QUOTE_NONE)
    unique_batch_fnames = ['pdf_parses_{:d}.jsonl.gz'.format(bid) for bid in metadata_df['batch_num'].unique()]
    # unique_batch_fnames = ['pdf_parses_{:d}.jsonl.gz'.format(bid) for bid in range(0, 100)]

    processed_files = set(int(filename.split('.')[0]) for filename in os.listdir(filtered_data_path))
    logger.info(f"Found {len(processed_files)} already processed batches.")

    to_process = [file for file in unique_batch_fnames if int(file.split('_')[-1].split('.')[0]) not in processed_files]
    logger.info(f"Gathering {len(to_process)} batches...")

    di = du.DirMetaIterator(root_path=raw_data_path, yield_list=to_process, metadata_df=metadata_df, args=(filtered_data_path,area,logger,))
    # Start a pool of worker processes.
    process_pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=10000)
    start = time.time()
    gathered_total = 0
    logger.info('Gathering data from: {:s}; Shape: {:}'.format(meta_fname, metadata_df.shape))
    try:
        for wrote_count in process_pool.imap_unordered(write_batch_papers, di,
                                                       chunksize=mp.cpu_count()):
            gathered_total += wrote_count
            logger.info('Wrote rows: {:d}'.format(wrote_count))
    finally:
        # Close the pool.
        process_pool.close()
        process_pool.join()

    logger.info('Wrote papers to: {:s}'.format(filtered_data_path))
    logger.info('Wrote papers: {:d}'.format(gathered_total))
    logger.info('Took: {:.4f}s'.format(time.time() - start))

####################### CREATION OF PID2BATCH $ BATCH2PID, DONE ONCE###################################################
def gather_paper_batches(in_path, out_path, log_file=None):
    """
    For the entire S2ORC corpus build a map of batch to paper id.
    :return:
    """
    if log_file:
        logger = setup_logging(log_file)
    else:
        logger = logging.getLogger()

    batch_fnames = sorted(
        file for file in os.listdir(in_path) if file.endswith(".jsonl.gz")
    )
    batch2pid = {}
    pid2batch = []
    total_papers = 0
    start = time.time()
    logger.info("Starting to process batch files.")
    for bi, bfname in enumerate(tqdm(batch_fnames, total=len(batch_fnames))):
        try:
            logger.info(f"Processing {bfname}")
            meta_df = pd.read_json(
                os.path.join(in_path, bfname),
                compression='gzip',  # Specify the compression type
                lines=True  # Enable JSON lines (each line is a separate JSON object)
            )
            pids = meta_df['paper_id'].tolist()
            batch_num = int(bfname.split('.')[0].split('_')[-1])
            batch2pid[batch_num] = pids
            total_papers += len(pids)
            pid2batch.extend([(pid, batch_num) for pid in pids])

            if bi % 10 == 0:
                logger.info(f"Batch: {bi}; Total papers processed so far: {total_papers}")
        except Exception as e:
            logger.error(f"Error processing {bfname}: {e}")

    logger.info(f"Total papers processed: {total_papers}")

    try:
        pid2batch_path = os.path.join(out_path, 'pid2batch.json')
        with codecs.open(pid2batch_path, 'w', 'utf-8') as fp:
            pid2batch_dict = dict(pid2batch)
            json.dump(pid2batch_dict, fp)
            logger.info(f"pid2batch: {len(pid2batch_dict)} entries written to {fp.name}")
    except Exception as e:
        logger.error(f"Error writing pid2batch: {e}")

    try:
        batch2pid_path = os.path.join(out_path, 'batch2pids.json')
        with codecs.open(batch2pid_path, 'w', 'utf-8') as fp:
            json.dump(batch2pid, fp)
            logger.info(f"batch2pid: {len(batch2pid)} entries written to {fp.name}")
    except Exception as e:
        logger.error(f"Error writing batch2pid: {e}")

    elapsed_time = time.time() - start
    logger.info(f"Processing complete. Total time: {elapsed_time:.4f}s.")

def get_citation_count_large(query_meta_row, data_json,logger):
    """
    Given the metadata row for the paper making the citations and the
    full text json data, return the outgoing citation contexts counts.
    :param query_meta_row: dict(); Generated from a pd.Series.
    :param data_json: dict(); full paper dict from batch jsonl.
    :return:
    """
    # Sometimes the citations are NaN
    try:
        outbound_cits = ast.literal_eval(query_meta_row['outbound_citations'])
    except ValueError:
        # logger.error("Error parsing citations from metadata row.")
        return {}, {}
    # Sometimes its an empty list.
    if not outbound_cits:
        # logger.error("No citations found.")
        return {}, {}
    # Get the mapping from bib-id to the paper-id in the dataset.
    # logger.info('Mapping bib-id to paper-id')
    linked_bibid2pid = {}
    for bibid, bibmetadata in data_json['bib_entries'].items():
        if bibmetadata['link']:
            linked_bibid2pid[bibid] = bibmetadata['link']

    # Go over the citations and count up how often they occur in the text.
    # Only the linked citations will be counted up I think.
    pid2citcount = collections.defaultdict(int)
    # Each list element here will be (par_number, sentence_number, sentence_context)
    pid2citcontext = collections.defaultdict(list)
    for par_i, par_dict in enumerate(data_json['body_text']):
        par_text = par_dict['text']
        par_sentences = scispacy_model(par_text,
                                       disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                'lemmatizer', 'parser', 'ner'])
        par_sentences = [sent.text for sent in par_sentences.sents]
        for cit_span in par_dict['cite_spans']:
            # Check for the ref_id being in the linked bib2pids.
            if cit_span['ref_id'] and cit_span['ref_id'] in linked_bibid2pid:
                cit_span_text = par_text[cit_span['start']:cit_span['end']]
                pid = linked_bibid2pid[cit_span['ref_id']]
                pid2citcount[pid] += 1
                for sent_i, sent in enumerate(par_sentences):
                    if cit_span_text in sent:
                        context_tuple = (par_i, sent_i, sent)
                        # print(context_tuple)
                        pid2citcontext[pid].append(context_tuple)
    # logger.info(f"Processed batch.")

    return dict(pid2citcount), dict(pid2citcontext)

def write_batch_citation_contexts(args):
    """
    Given a batch file, read the papers from it mentioned in the metadata-df
    and write sentence contexts of outgoing citations.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param pids: pids of the papers we want from the current batch file.
    :return: wrote_count: int; how many jsonl rows were written to the batch output.
    """
    jsonl_fname, pids, batch_metadata_df, filtered_data_path, area, logger = args
    batch_num = int(os.path.basename(jsonl_fname)[:-6])  # Its 'batch_num.jsonl'
    logger.info(f"Processing batch {batch_num}")
    if len(pids) > 0:
        data_file = codecs.open(jsonl_fname, 'r', 'utf-8')
        citcontextf = codecs.open(os.path.join(filtered_data_path, f'pid2citcontext-{batch_num}-{area}.jsonl'), 'w', 'utf-8')
        citcountf = codecs.open(os.path.join(filtered_data_path, f'pid2citcount-{batch_num}-{area}.jsonl'), 'w', 'utf-8')
        pid2jsonl_idx = {}
        total_papers, valid_papers = 0, 0
        for line in tqdm(data_file):
            # load single paper json
            data_json = json.loads(line.strip())
            if int(data_json['paper_id']) in pids:
                row = batch_metadata_df[batch_metadata_df['paper_id'] == int(data_json['paper_id'])]
                assert(row.empty == False)
                row = row.to_dict('records')
                assert(len(row) == 1)
                row = row[0]
                total_papers += 1
                citation_counts, citation_contexts = get_citation_count_large(
                    query_meta_row=row, data_json=data_json,logger=logger)
                if len(citation_counts) == 0:
                    continue
                pid2jsonl_idx[row['paper_id']] = valid_papers
                valid_papers += 1
                citcontextf.write(json.dumps({row['paper_id']: citation_contexts})+'\n')
                citcountf.write(json.dumps({row['paper_id']: citation_counts})+'\n')
            # if valid_papers > 20:
            #     break
        with codecs.open(os.path.join(filtered_data_path, f'pid2jsonlidx-{batch_num}-{area}.json'),'w', 'utf-8') as fp:
            json.dump(pid2jsonl_idx, fp)
        citcontextf.close()
        citcountf.close()
        return total_papers, valid_papers
    else:
        return 0, 0

def merge_citation_files(filt_data_path, area, output_file_suffix='pid2citcontext', logger=None):
    """
    Merge multiple batch citation context files into a single file.
    :param filt_data_path: Directory containing the batch files.
    :param area: Area of research/topic identifier.
    :param output_file_suffix: File suffix for citation files ('pid2citcontext' or 'pid2citcount').
    """
    # List all files in the directory
    all_files = os.listdir(filt_data_path)

    # Filter files matching the pattern
    input_files = [
        os.path.join(filt_data_path, fname)
        for fname in all_files
        if fname.startswith(f"{output_file_suffix}-") and fname.endswith(f"-{area}.jsonl")
    ]

    # Output file
    output_file = os.path.join(filt_data_path, f'{output_file_suffix}-{area}.jsonl')

    # Initialize output data structure
    merged_data = {}

    # Loop through matching files
    for file_name in input_files:
        with open(file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    # Each line is a JSON object with {paper_id: data}
                    data = json.loads(line.strip())
                    merged_data.update(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON in file {file_name}: {e}")

    # Write the merged data to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for paper_id, data in merged_data.items():
            outfile.write(json.dumps({paper_id: data}) + '\n')

    logger.info(f"Merged files saved to {output_file}")


def gather_from_citationnw_large(filt_data_path, meta_fname, area, log_file):
    """
    Open up a metadata file of a fulltext-filtered subset of the s2orc dataset and
    check if the cited file is part of the s2orc data and count the number of times
    a cited paper is cited in the query paper and the citation contexts it is in and
    write out these counts for a set of query papers.
    Write out citation contexts and counts as per line jsons for a huge dataset
    and per batch which can them be merged with bash and python scripts (for the pid2idx file).
    :param filt_data_path:
    :param meta_fname: metadata file to gather cited papers for.
    :return:
    """
    if log_file:
        logger = setup_logging(log_file)
    else:
        logger = logging.getLogger()
    citations_dir = os.path.join(filt_data_path, 'cit_contexts_counts')

    # Check if the directory exists and create it if not
    if not os.path.exists(citations_dir):
        os.makedirs(citations_dir)
        logger.info(f"Directory '{citations_dir}' created successfully.")
    else:
        logger.info(f"Directory '{citations_dir}' already exists.")
    processed_files = set(int(filename.split('-')[1]) for filename in os.listdir(citations_dir))
    logger.info(f"Found {3*len(processed_files)} already processed batches.")
    query_meta = pd.read_csv(meta_fname, delimiter='\t', error_bad_lines=False, engine='python', quoting=csv.QUOTE_NONE)
    unique_batch_fnames = [f'{bid}.jsonl' for bid in query_meta['batch_num'].unique()]
    to_process = [filename for filename in unique_batch_fnames if int(filename.split('.')[0]) not in processed_files][:9]
    logger.info(f"Gathering {len(to_process)} batches...")
    di = du.DirMetaIterator(root_path=filt_data_path, yield_list=to_process, metadata_df=query_meta,
                            args=(citations_dir, area, logger), yield_meta=True)

    process_pool = mp.Pool(processes=mp.cpu_count()//2, maxtasksperchild=10000)
    start = time.time()
    total_papers, valid_papers = 0, 0
    try:
        for batch_processed_papers, batch_valid_papers in process_pool.imap_unordered(write_batch_citation_contexts, di, chunksize=max(1, mp.cpu_count() // 4)):
            total_papers += batch_processed_papers
            valid_papers += batch_valid_papers
            logger.info('Wrote rows: {:d}'.format(valid_papers))
    finally:
        process_pool.close()
        process_pool.join()

    processed_files = set(int(filename.split('-')[1]) for filename in os.listdir(citations_dir))
    if len(processed_files) == len(unique_batch_fnames):
        logger.info('Examined papers: {:d}; Valid query papers: {:d}'.format(total_papers, valid_papers))
        logger.info(f"Merges pid2citcontext-bid ")
        merge_citation_files(filt_data_path=citations_dir, area='biomed', output_file_suffix='pid2citcontext',logger=logger)
        logger.info(f"Merges pid2citcount-bid ")
        merge_citation_files(filt_data_path=citations_dir, area='biomed', output_file_suffix='pid2citcount',logger=logger)
        logger.info('Took: {:.4f}s'.format(time.time() - start))
    else:
        logger.info(f"Needs more {len(unique_batch_fnames) - len(processed_files)} intermediate files before merge.")


# TODO merge 'pid2citcontext-{bid}-{area}.jsonl to 'pid2citcontext-{area}.jsonl



def gather_cocitations(root_path, area, log_file):
    """
    - Read in citation contexts.
    - Go over the citation contexts and group them into co-citations.
    - Compute stats.
    - Save co-citations to disk.
    """
    logger = setup_logging(log_file) if log_file else logging.getLogger()
    logger.info('Reading citation contexts...')
    citation_contexts = codecs.open(os.path.join(root_path, f'pid2citcontext-{area}.jsonl'), 'r', 'utf-8')
    logger.info(f'pid2citcontext-{area}.jsonl loaded.')
    all_co_cited_pids2contexts = collections.defaultdict(list)
    single_cited2contexts = collections.defaultdict(list)
    examined_papers = 0
    for citation_context_line in citation_contexts:
        if examined_papers % 1000 == 0:
            logger.info(f'Examined papers: {examined_papers}')
        citcond = json.loads(citation_context_line.strip())
        citing_pid, cited2contexts = list(citcond.keys())[0], list(citcond.values())[0]
        paper_co_citations = collections.defaultdict(list)
        # Go over all the cited papers and get the co-citations by sentence position.
        for cited_pid, context_tuples in cited2contexts.items():
            # Cited papers can have multiple instances in the citing paper.
            for ct in context_tuples:  # ct is (par_i, sent_i, sent)
                par_i, sent_i, con_sent = ct[0], ct[1], ct[2]
                # Papers in the same sentence are co-cited.
                paper_co_citations[(par_i, sent_i)].append((cited_pid, con_sent))
        # Gather the co-cited papers by pid.
        paper_cocitpids2contexts = collections.defaultdict(list)
        for co_cited_tuple in paper_co_citations.values():
            # There has to be one element atleast and all of the sents will be the same.
            cit_sent = co_cited_tuple[0][1]
            # There can be repeated citations of the same thing in the same sentence
            # or somehow multiple instances of the same pid occur in the parsed spans.
            co_cited_pids = list(set([t[0] for t in co_cited_tuple]))
            co_cited_pids.sort()
            # The same co-cited set of pids in a paper may have mulitiple diff
            # cit contexts. Gather those here.
            paper_cocitpids2contexts[tuple(co_cited_pids)].append((citing_pid, cit_sent))
        # Merge the co-citations across the corpus.
        for cocitpids, citcontexts in paper_cocitpids2contexts.items():
            # Use this if writing to a json file instead of pickle.
            # cocitpids_key = '-'.join(list(cocitpids))
            if len(cocitpids) == 1:
                single_cited2contexts[cocitpids].extend(citcontexts)
            else:
                all_co_cited_pids2contexts[cocitpids].extend(citcontexts)
        examined_papers += 1
        # if examined_papers > 1000:
        #     break
    # Write out single citations and their stats.
    with codecs.open(os.path.join(root_path, f'single_cited2contexts-{area}.pickle'), 'wb') as fp:
        pickle.dump(single_cited2contexts, fp)
        logger.info(f'Wrote: {fp.name}')
    num_sincited_pids = []
    num_sincitcons = []
    for cocitpids, citcontexts in single_cited2contexts.items():
        num_sincited_pids.append(len(cocitpids))
        num_sincitcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_sincitcons).describe()
    logger.info('Single papers cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_sincitcons)))

    # Write out co-citations and their stats.
    with codecs.open(os.path.join(root_path, f'cocitpids2contexts-{area}.pickle'), 'wb') as fp:
        pickle.dump(all_co_cited_pids2contexts, fp)
        logger.info(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(root_path, f'cocitpids2contexts-{area}.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(all_co_cited_pids2contexts.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        logger.info(f'Wrote: {fp.name}')
    num_cocited_pids = []
    num_citcons = []
    for cocitpids, citcontexts in all_co_cited_pids2contexts.items():
        num_cocited_pids.append(len(cocitpids))
        num_citcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_cocited_pids).describe()
    logger.info('Papers co-cited together:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_cocited_pids)))
    all_summ = pd.DataFrame(num_citcons).describe()
    logger.info('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))

##################################################################################################################
def exclude_abstract(abstract_sents):
    """
    Given a json string check if it has everything an example should and return filtered dict.
    :param abstract_sents: list(string)
    :return: bool;
        True if the abstract looks noisy (too many sentences or too many tokens in a sentence)
        False if things look fine.
    """
    abs_sent_count = len(abstract_sents)
    if abs_sent_count < pps.MIN_ABS_LEN or abs_sent_count > pps.MAX_ABS_LEN:
        return True
    # Keep count of how many sentences in an abstract and how many tokens in a sentence.
    all_small_sents = True
    for sent in abstract_sents:
        num_toks = len(sent.split())
        if num_toks > pps.MIN_NUM_TOKS:
            all_small_sents = False
        if num_toks > pps.MAX_NUM_TOKS:
            return True
    # If all the sentences are smaller than a threshold then exclude the abstract.
    if all_small_sents:
        return True
    return False


def write_batch_absmeta(args):
    """
    Given a batch file, read the papers from it mentioned in the pids,
    filter out obviously noisy papers and write out the title and abstract
    and limited metadata to disk.
    :param jsonl_fname: string; filename for current batch.
    :param filtered_data_path: directory to which outputs should be written.
    :param to_write_pids: pids of the papers we want from the current batch file.
    :return:
        to_write_pids: list(int); to write pids.
        pids_written: list(string); actually written pids.
    """
    jsonl_fname, to_write_pids, filtered_data_path,logger = args
    batch_num = int(jsonl_fname.split('.')[0].split('/')[-1])
    pids_written = set()
    if len(to_write_pids) < 0:
        return 0, pids_written
    # data_file = gzip.open(jsonl_fname)
    # data_file = json.load(open(jsonl_fname))
    out_file = codecs.open(os.path.join(filtered_data_path, f'{batch_num}.jsonl'), 'w', 'utf-8')
    with open(jsonl_fname, 'r') as data_file:
        for line in tqdm(data_file):
            data_json = json.loads(line.strip())
            paper_id = data_json['paper_id']
            # The pids comes from metadata which saves it as an integer.
            if int(paper_id) not in to_write_pids:
                continue
            # Get title and abstract.
            # title_sent = data_json['metadata'].pop('title', None)
            # row_meta = batch_metadata_df[batch_metadata_df['paper_id'] == int(data_json['paper_id'])]
            # row_meta = row_meta.to_dict('records')
            #
            # title_sent = row_meta['title']
            # # Assuming this is present in the metadata; Suspect this is only if its gold and provided.
            # abstract_sents = []
            # try:
            #     # abstrast_str = data_json['metadata'].pop('abstract', None)
            #     abstract_str = row_meta['abstract']
            #     abstract_sents = scispacy_model(abstract_str,
            #                                     disable=['tok2vec', 'tagger', 'attribute_ruler',
            #                                              'lemmatizer', 'parser', 'ner'])
            #     abstract_sents = [sent.text for sent in abstract_sents.sents]
            # # Sometimes abstract is missing (is None) in the metadata.
            # except TypeError:
            abstract_sents = []
            try:
                for abs_par_dict in data_json['abstract']:
                    par_sents = scispacy_model(abs_par_dict['text'],
                                               disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                        'lemmatizer', 'parser', 'ner'])
                    par_sents = [sent.text for sent in par_sents.sents]
                    abstract_sents.extend(par_sents)
            except TypeError:
                pass
            if not abstract_sents:
                continue
            # Filter out abstrasts which are noisey.
            if exclude_abstract(abstract_sents):
                continue
            pids_written.add(paper_id)
            out_dict = {
                'paper_id': paper_id,
                # 'title': title_sent,
                'abstract': abstract_sents
            }
            out_file.write(json.dumps(out_dict) + '\n')
            # logger.info(f"Batch {batch_num} written to {out_file}")
            # if len(pids_written) > 20:
            #     break
    out_file.close()
    return to_write_pids, pids_written

def cocit_corpus_to_jsonl(meta_path, batch_data_path, root_path, out_path, area, log_file):
    """
    Given the co-citation information (which sets of papers are co-cited), write out a jsonl
    file with the abstracts and the metadata based on which training data for model will
    be formed (this will still need subsampling and additional cocitation stats based filtering)
    Also filter out data which is obviously noisey in the process.
    In multiprocessing each thread will write one jsonl. In the end, using bash to merge
    all the jsonl files into one jsonl file.
    :param meta_path: strong; directory with pid2citcount files.
    :param batch_data_path: string; directory with batched jsonl files.
    :param root_path: string; top level directory with pid2batch file.
    :param out_path: string; directory to write batch jsonl files to. Also where filtered citations
        get written.
    :param area: string; {'compsci', 'biomed'}
    :return: writes to disk.
    """
    logger = setup_logging(log_file) if log_file else logging.getLogger()
    logger.info(mp.cpu_count())
    batch_out_path = os.path.join(out_path, 'batch_data')
    du.create_dir(batch_out_path)
    with codecs.open(os.path.join(root_path, 'pid2batch.json'), 'r', 'utf-8') as fp:
        pid2batch = json.load(fp)
        logger.info('Read: {:s}'.format(fp.name))
        logger.info(f'pid2batch: {len(pid2batch)}')
    with codecs.open(os.path.join(meta_path, f'cocitpids2contexts-{area}.pickle'), 'rb') as fp:
        cocitpids2contexts = pickle.load(fp)
        logger.info('Read: {:s}'.format(fp.name))
    # Get all co-cited papers.
    co_cited_pids = set()
    for cocited_tuple in cocitpids2contexts.keys():
        co_cited_pids.update(cocited_tuple)
    # Get the batch numbers for the pids.
    batch2pids = collections.defaultdict(list)
    missing = 0
    for pid in co_cited_pids:
        try:
            batch_num = pid2batch[pid]
            batch2pids[batch_num].append(pid)
        except KeyError:
            missing += 1
            continue
    batch2pids = dict(batch2pids)
    logger.info(f'Total unique co-cited docs: {len(co_cited_pids)}; Missing in map: {missing}')
    logger.info(f'Number of batches: {len(batch2pids)}')
    del pid2batch
    unique_batch_fnames = ['{:d}.jsonl'.format(bid) for bid in batch2pids.keys()]
    di = du.DirMetaIterator(root_path=batch_data_path, yield_list=unique_batch_fnames, metadata_df=batch2pids,
                            args=(batch_out_path,logger))
    # Start a pool of worker processes.
    process_pool = mp.Pool(processes=mp.cpu_count() // 2, maxtasksperchild=10000)

    start = time.time()
    processed_total = 0
    written_total = 0
    all_written_pids = set()
    for batch_to_writepids, batch_written_pids in process_pool.imap_unordered(write_batch_absmeta, di,
                                                                              chunksize=max(1, mp.cpu_count() // 2)):
        all_written_pids.update(batch_written_pids)
        processed_total += len(batch_to_writepids)
        written_total += len(batch_written_pids)
        logger.info('Processed: {:d} Written: {:d}'.format(len(batch_to_writepids), len(batch_written_pids)))
    # Close the pool.
    process_pool.close()
    process_pool.join()

    # Exclude pids which were excluded.
    cocitedpids2contexts_filt = {}
    for cocit_pids, citcontexts in cocitpids2contexts.items():
        filt_cocit_pids = []
        for ccpid in cocit_pids:
            if ccpid not in all_written_pids:
                continue
            else:
                filt_cocit_pids.append(ccpid)
        if len(filt_cocit_pids) > 1:
            cocitedpids2contexts_filt[tuple(filt_cocit_pids)] = citcontexts

    # Write out filtered co-citations and their stats.
    with codecs.open(os.path.join(out_path, f'cocitpids2contexts-{area}-absfilt.pickle'), 'wb') as fp:
        pickle.dump(cocitedpids2contexts_filt, fp)
        logger.info(f'Wrote: {fp.name}')
    # Writing this out solely for readability.
    with codecs.open(os.path.join(out_path, f'cocitpids2contexts-{area}-absfilt.json'), 'w', 'utf-8') as fp:
        sorted_cocits = collections.OrderedDict()
        for cocitpids, citcontexts in sorted(cocitedpids2contexts_filt.items(), key=lambda i: len(i[1])):
            cocit_key = '-'.join(cocitpids)
            sorted_cocits[cocit_key] = citcontexts
        json.dump(sorted_cocits, fp, indent=1)
        logger.info(f'Wrote: {fp.name}')
    num_cocited_pids = []
    num_citcons = []
    for cocitpids, citcontexts in cocitedpids2contexts_filt.items():
        num_cocited_pids.append(len(cocitpids))
        num_citcons.append(len(citcontexts))
    all_summ = pd.DataFrame(num_cocited_pids).describe()
    logger.info('Papers co-cited together:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_cocited_pids)))
    all_summ = pd.DataFrame(num_citcons).describe()
    logger.info('Papers co-cited frequency:\n {:}'.format(all_summ))
    pprint.pprint(dict(collections.Counter(num_citcons)))

    logger.info('Unfiltered: {:d} Filtered written papers: {:d}'.format(processed_total, written_total))
    logger.info('Unfiltered cocited sets: {:d}; Filtered cocited sets: {:d}'.
          format(len(cocitpids2contexts), len(cocitedpids2contexts_filt)))
    logger.info('Took: {:.4f}s'.format(time.time() - start))

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Filter the metadata to group them by hosting service.
    filter_metadata_by_ft_area = subparsers.add_parser('filter_metadata_by_fulltext_and_area')
    filter_metadata_by_ft_area.add_argument('-i', '--raw_meta_path', required=True,
                                 help='Directory with batchwise metadata.')
    filter_metadata_by_ft_area.add_argument('-o', '--filt_meta_path', required=True,
                                 help='Directory where filtered metadata files should get written.')
    filter_metadata_by_ft_area.add_argument('-d', '--dataset', required=True,
                                 choices=['s2orcfulltext'],
                                 help='Directory where split files should get written.')
    filter_metadata_by_ft_area.add_argument('-a', '--area', required=True,choices=['compsci','biomed'])
    filter_metadata_by_ft_area.add_argument('-l','--log_fname', help='Filename of log file')

    # Gather filtered papers.
    gather_filtered_papers = subparsers.add_parser('gather_papers_by_filtered_metadata')
    gather_filtered_papers.add_argument('-i', '--in_meta_path', required=True,
                                 help='Directory with a filtered metadata tsv file.')
    gather_filtered_papers.add_argument('-o', '--raw_data_path', required=True,
                                 help='Directory where batches of raw data.')
    gather_filtered_papers.add_argument('-d', '--dataset', required=True,
                                 choices=['s2orcfulltext'],
                                 help='Directory where split files should get written.')
    gather_filtered_papers.add_argument('-a', '--area', required=True,choices=['compsci','biomed'])

    gather_filtered_papers.add_argument('-l','--log_fname', help='Filename of log file')


    # Gather pids and batches.
    batch_pids = subparsers.add_parser('get_batch_pids')
    batch_pids.add_argument('-i', '--in_path', required=True,
                            help='Directory with a batched tsv files.')
    batch_pids.add_argument('-o', '--out_path', required=True,
                            help='Directory to write batch to pid maps.')
    batch_pids.add_argument('-l','--log_fname', help='Filename of log file')

    # Gather citations.
    gather_citnw = subparsers.add_parser('gather_from_citationnw')
    gather_citnw.add_argument('-r', '--root_path', required=True,
                              help='Directory metadata, paper data and where outputs should be written.')
    gather_citnw.add_argument('-d', '--dataset', required=True,
                              choices=['s2orcfulltext'])
    gather_citnw.add_argument('-a', '--area', required=True,choices=['compsci','biomed'])

    gather_citnw.add_argument('-l','--log_fname', help='Filename of log file')


    # Gather co-citation contexts.
    gather_cocit_cons = subparsers.add_parser('gather_area_cocits')
    gather_cocit_cons.add_argument('--root_path', required=True,
                                   help='Directory with metadata, paper data and where '
                                        'outputs should be written.')
    gather_cocit_cons.add_argument('-a','--area', required=True,
                                   choices=['compsci', 'biomed'])
    gather_cocit_cons.add_argument('-l','--log_fname', help='Filename of log file')


    gather_cocitjsonl = subparsers.add_parser('gather_filtcocit_corpus')
    gather_cocitjsonl.add_argument('--root_path', required=True,
                                   help='Directory with pid2batch.')
    gather_cocitjsonl.add_argument('--in_meta_path', required=True,
                                   help='Directory with a filtered metadata tsv file.')
    gather_cocitjsonl.add_argument('--raw_data_path', required=True,
                                   help='Directory where batches of raw data.')
    gather_cocitjsonl.add_argument('--out_path', required=True,
                                   help='Directory where batches of title/abstract jsonl files '
                                        'and filtered citation map should be written.')
    gather_cocitjsonl.add_argument('-a', '--area', required=True,choices=['compsci','biomed'])
    gather_cocitjsonl.add_argument('-l','--log_name', help='Filename of log file')


    cl_args = parser.parse_args()

    if cl_args.subcommand == 'filter_metadata_by_fulltext_and_area':
        if cl_args.dataset == 's2orcfulltext':
            filter_metadata(raw_meta_path=cl_args.raw_meta_path,
                            filtered_meta_path=cl_args.filt_meta_path,
                            filter_method='full text',
                            area=cl_args.area,
                            log_file=cl_args.log_fname)

    elif cl_args.subcommand == 'gather_papers_by_filtered_metadata':
        if cl_args.dataset in {'s2orcfulltext'}:
            meta_fname = os.path.join(cl_args.in_meta_path, f'metadata-{cl_args.dataset}-{cl_args.area}.tsv')
            # /cs/labs/tomhope/idopinto12/aspire/datasets_raw/s2orc/s2orcbiomed/metadata-s2orcfulltext-biomed.tsv
            gather_papers(meta_fname=meta_fname, raw_data_path=cl_args.raw_data_path, area=cl_args.area,log_file=cl_args.log_fname)

    elif cl_args.subcommand == 'get_batch_pids':
        # Run once for the entire s2orc corpus, no need to re-run over and over.
        gather_paper_batches(in_path=cl_args.in_path, out_path=cl_args.out_path,log_file=cl_args.log_fname)

    elif cl_args.subcommand == 'gather_from_citationnw':
        filt_root_path = os.path.join(cl_args.root_path, 's2orcbiomed/')
        if cl_args.dataset == 's2orcfulltext':
            meta_fname = os.path.join(filt_root_path, f'metadata-s2orcfulltext-{cl_args.area}.tsv')
            batch_data_path = os.path.join(filt_root_path, f'data-s2orcfulltext-{cl_args.area}')
            gather_from_citationnw_large(filt_data_path=batch_data_path, meta_fname=meta_fname,area=cl_args.area,log_file=cl_args.log_fname)

    elif cl_args.subcommand == 'gather_area_cocits':
        root_path = os.path.join(cl_args.root_path, f's2orc{cl_args.area}')
        batch_data_path = os.path.join(root_path, f'data-s2orcfulltext-{cl_args.area}')
        filt_root_path = os.path.join(batch_data_path, f'cit_contexts_counts')
        gather_cocitations(root_path=filt_root_path, area=cl_args.area, log_file=cl_args.log_fname)


    elif cl_args.subcommand == 'gather_filtcocit_corpus':
        cocit_corpus_to_jsonl(meta_path=cl_args.in_meta_path, batch_data_path=cl_args.raw_data_path,
                              out_path=cl_args.out_path, area=cl_args.area, root_path=cl_args.root_path, log_file=cl_args.log_name)

if __name__ == '__main__':
    main()