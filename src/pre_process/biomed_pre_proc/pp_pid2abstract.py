import pandas as pd
import pickle
import os
import csv
import glob
import gzip
from tqdm import tqdm


def merge_jsonls():
    """
    Merges multiple JSONL files into a single compressed JSONL file.

    This function searches for all files matching the specified pattern (in this case, JSONL files)
    and merges them into one single compressed output file (Gzip format).

    The input files are read sequentially, and the contents are written to the output file with
    a new line between each file's contents.

    Args:
        None: This function does not take any parameters. The file pattern and output file path
              are hardcoded within the function.

    Returns:
        None: The function does not return any values. It saves the merged data into the specified
              output file.

    Output:
        A compressed Gzip file containing the merged data from all matched JSONL files.
    """
    # Specify the pattern to match all jsonl files
    file_pattern = "/cs/labs/tomhope/idopinto12/aspire/datasets/train/batch_data/*.jsonl"

    # Output file (compressed)
    output_file = "/cs/labs/tomhope/idopinto12/aspire/datasets/train/batch_data/merged.jsonl.gz"

    # Open the compressed output file
    with gzip.open(output_file, "wt") as outfile:  # "wt" is for writing text in gzip
        for filename in tqdm(sorted(glob.glob(file_pattern))):  # Sort files for sequential merging
            with open(filename, "r") as infile:
                outfile.write(infile.read())  # Append contents
                outfile.write("\n")  # Ensure new lines between files


def get_pid2abstracts(abs_path, metadata_path, out_path, area):
    """
    Extracts the paper ID, title, and abstract from the provided dataset and metadata,
    and saves the results as a pickle file.

    This function reads abstract data from a JSONL (or compressed JSONL) file and matches each
    paper ID with its corresponding title and abstract from the metadata file. It then saves
    the data in a dictionary and serializes it as a pickle file.

    Args:
        abs_path (str): The path to the dataset file containing abstracts (in either '.jsonl' or '.jsonl.gz' format).
        metadata_path (str): The path to the metadata file containing paper titles (in TSV format).
        out_path (str): The directory where the resulting pickle file will be saved.
        area (str): A string identifier for the dataset area (used in the output filename).

    Returns:
        None: This function does not return any values. It saves the result as a pickle file.

    Output:
        A pickle file containing a dictionary that maps paper IDs to their respective titles and abstracts.
    """
    # Handle the `abs_df` depending on the file type
    if abs_path.endswith('.jsonl.gz'):
        # For compressed files
        abs_df = pd.read_json(abs_path, lines=True, compression='gzip')
    elif abs_path.endswith('.jsonl'):
        # For regular JSON lines file
        abs_df = pd.read_json(abs_path, lines=True)
    else:
        raise ValueError("Unsupported abs_path file format. Please use '.jsonl' or '.jsonl.gz'.")
    print(f"{abs_path} loaded..")

    # Load metadata as DataFrame
    metadata_df = pd.read_csv(metadata_path, delimiter='\t', error_bad_lines=False,
                              engine='python', quoting=csv.QUOTE_NONE)

    # Prepare the dictionary for paper_id -> title and abstract
    pid2abstract = {}
    pids = abs_df['paper_id'].unique()

    for pid in tqdm(pids):
        # Gather abstract sentences
        abs_sents = abs_df[abs_df['paper_id'] == pid]['abstract'].iloc[0]
        # Gather title from metadata
        title = metadata_df[metadata_df['paper_id'] == pid]['title'].iloc[0]
        # Construct dictionary
        pid2abstract[pid] = {'title': title, 'abstract': abs_sents}

    # Save the dictionary as a pickle file
    output_file = os.path.join(out_path, f'pid2abstract-s2orc{area}.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(pid2abstract, f)

    print(f"Data successfully saved to {output_file}")


def main():
    """
    Main function to execute the process of extracting paper IDs, titles, and abstracts,
    and saving the result as a pickle file.

    This function is intended to be run as a script and calls the `get_pid2abstracts` function
    with predefined parameters for the abstract and metadata file paths, as well as the output path.

    Returns:
        None
    """
    # Example usage of get_pid2abstracts
    abs_path = '/cs/labs/tomhope/idopinto12/aspire/datasets/train/batch_data/merged.jsonl.gz'
    metadata_path = '/cs/labs/tomhope/idopinto12/aspire/datasets_raw/s2orc/s2orcbiomed/metadata-s2orcfulltext-biomed.tsv'
    out_path = '/cs/labs/tomhope/idopinto12/aspire/datasets/train/'
    area = 'biomed'

    # Extract and save paper ID to abstract dictionary
    get_pid2abstracts(abs_path, metadata_path, out_path, area)


if __name__ == '__main__':
    main()