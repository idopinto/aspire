import json
import os
import random
from tqdm import tqdm


def shuffle_and_split(dataset_file, output_dir, num_gpus, num_epochs, suffix):
    """
    Shuffle the dataset and split it into chunks for parallel processing across multiple GPUs.

    Args:
        dataset_file (str): The path to the input dataset file in JSONL format.
        output_dir (str): The directory where the shuffled and split dataset will be saved.
        num_gpus (int): The number of GPUs to split the dataset across.
        num_epochs (int): The number of epochs to shuffle and split the dataset.
        suffix (str): A suffix to append to the output file names for easy identification.

    Returns:
        None: This function does not return anything. It saves the shuffled and split dataset
              files into the specified output directory.

    Steps:
        1. Loads the dataset from the provided file.
        2. Shuffles the data randomly for each epoch.
        3. Splits the shuffled data into chunks corresponding to the number of GPUs.
        4. Writes each chunk to a separate output file named with GPU index and epoch number.
    """
    random.seed(69306)

    # Load the dataset
    train_basename = 'train-{:s}'.format(suffix)
    print(f"Loading dataset from {dataset_file}")
    data = []
    with open(dataset_file, 'r') as f:
        for line in tqdm(f):
            data.append(line.strip())  # Data is read line by line
    print(f"Loaded {len(data)} lines from {dataset_file}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle and split for each epoch
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        random.shuffle(data)
        chunks = [data[i::num_gpus] for i in range(num_gpus)]  # Split data into chunks for GPUs
        print(f"Shuffled {len(chunks)} chunks from {dataset_file}")
        for gpu_idx, chunk in enumerate(chunks):
            output_file = os.path.join(output_dir, f'{train_basename}-{gpu_idx}-{epoch}.jsonl')
            print(f"Writing {output_file}")
            with open(output_file, 'w') as out_f:
                out_f.writelines(chunk)
            print(f"Finished writing {output_file}")
    print(f'Finished shuffling {dataset_file} into {output_dir}')


def main():
    """
    Example usage of the shuffle_and_split function to shuffle and split a dataset into chunks
    for parallel processing across multiple GPUs. This function demonstrates how to call
    `shuffle_and_split` with specified parameters.

    Returns:
        None
    """
    shuffle_and_split(
        dataset_file='/cs/labs/tomhope/idopinto12/aspire/datasets/train/train-cocitabsalign.jsonl',
        output_dir='/cs/labs/tomhope/idopinto12/aspire/datasets/train/shuffled_data/',
        num_gpus=8,
        num_epochs=1,  # Number of epochs to shuffle and split the data
        suffix="cocitabsalign"
    )


if __name__ == '__main__':
    main()