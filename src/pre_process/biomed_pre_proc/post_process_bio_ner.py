import argparse

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import Set, List, Dict, Optional
from dataclasses import dataclass
import json

from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from bs4 import BeautifulSoup

S2ORC_GENERIC_TERMS = {}
GENERIC_TRECOVID_TERMS ={
            "virus", "cell", "sars", "cov", "death", 'disease', 'human','stop','black'
            'person', "viral", "infection", "protein", "gene", "dna", "rna",
            "mrna", "covid", 'people', 'participant', 'patient', 'individual',
            'subject', 'children', 'position','lash','female','male','network','region','food','introduction','method'
        }
GENERIC_RELISH_TERMS = {

}
GENERIC_TERMS = GENERIC_TRECOVID_TERMS.union(GENERIC_RELISH_TERMS).union(S2ORC_GENERIC_TERMS)
@dataclass
class PostProcessConfig:
    """Configuration for entity post-processing"""
    generic_terms: Set[str]
    max_entities_per_paper: int = 5
    min_entity_length: int = 2
    min_definition_length: int = 10
    max_tokens_for_definition: int = 2000
    min_idf_threshold: float = 1.0  # Higher value means more specific/rare terms
    output_dir: Optional[Path] = None
    model_name: str = 'allenai/specter'


class EntityPostProcessor:
    """Post-process extracted entities from scientific papers"""
    html_counter = 0
    too_long_counter = 0

    def __init__(self, config: PostProcessConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.logger.info(self.config)
        # Initialize SPECTER tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def clean_html_and_urls(self,html_string):
        # Parse HTML to remove tags
        soup = BeautifulSoup(html_string, "html.parser")
        text = soup.get_text(separator=" ")

        # Remove URLs
        clean_text = re.sub(r'http\S+', '', text)

        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        if clean_text != text:
            # self.logger.info(f"Cleaning {html_string}")
            # self.logger.info(f"Cleaned text: {clean_text}")
            # self.logger.info("Cleaned text.")
            self.html_counter += 1
        return clean_text
    def is_generic(self, entity: str) -> bool:
        """Check if entity is generic"""
        e = entity.lower()
        return any(term in e for term in self.config.generic_terms)

    def is_junk(self, entity: str) -> bool:
        """Check if entity is junk based on various criteria"""
        e = entity.strip()

        # Check length
        if len(e) < self.config.min_entity_length:
            return True

        # Allow patterns like: COVID-19, H1N1, CD4, IL-6, etc.
        # But exclude pure numbers or entities that start with numbers
        if e[0].isdigit():  # Exclude if starts with number
            return True

        # Allow alphanumeric with hyphens but exclude weird characters
        if not re.match(r"^[A-Za-z][A-Za-z0-9-]*$", e):
            return True

        return False

    def compute_idf_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute IDF scores for entities"""
        # Document frequency
        doc_freq = df.groupby("entity")["paper_id"].nunique()
        # Total documents
        total_docs = df["paper_id"].nunique()
        # IDF computation
        idf_scores = np.log(total_docs / doc_freq)
        return idf_scores.reset_index(name="idf")

    def get_top_entities(self, paper_df: pd.DataFrame, k: int) -> List[Dict]:
        """Get top-k entities for a paper based on IDF scores"""
        # Filter valid definitions and IDF threshold
        valid = paper_df[
            (paper_df['definitions'].notnull()) &
            (paper_df['definitions'].str.len() >= self.config.min_definition_length) &
            (paper_df['idf'] >= self.config.min_idf_threshold)  # Apply IDF threshold
            ]

        # Sort by IDF and get top-k
        topk = valid.sort_values(by="idf", ascending=False).head(k)

        # Format entities with definitions
        entities = []
        for _, row in topk.iterrows():
            # Use long form if available and valid, otherwise use original entity
            display_entity = row['entity'].capitalize()
            if pd.notna(row.get('long_form')) and row['long_form'].strip() not in ['-', '', None]:
                display_entity = f"{display_entity} ({row['long_form'].strip()})"

            # Take first definition if multiple exist (filter for not too long definitions)
            definition = row['definitions'].split('|')[0].strip()
            definition = self.clean_html_and_urls(definition)
            definition_tokens = len(self.tokenizer.encode(definition))

            if definition_tokens<= self.config.max_tokens_for_definition:
                entities.append({
                    'entity': display_entity,
                    'original_entity': row['entity'],
                    'long_form': row.get('long_form', ''),
                    'definition': definition,
                    'idf_score': row['idf']
                })
            else:
                self.too_long_counter += 1
                # self.logger.warning(f"Definition too long for entity: {display_entity}, {definition_tokens}, {definition}")

        return entities

    def calculate_length_statistics(self, entity_data: Dict) -> Dict:
        """Calculate word and token length statistics for entity and definition"""
        # Combine entity with long form if available
        entity_text = entity_data['entity']

        # Get the full text (entity + definition)
        full_text = f"{entity_text}: {entity_data['definition']}"

        # Calculate word counts
        entity_words = len(entity_text.split())
        definition_words = len(entity_data['definition'].split())
        total_words = len(full_text.split())

        # Calculate token counts using SPECTER tokenizer
        entity_tokens = len(self.tokenizer.encode(entity_text))
        definition_tokens = len(self.tokenizer.encode(entity_data['definition']))
        total_tokens = len(self.tokenizer.encode(full_text))

        return {
            'entity_words': entity_words,
            'definition_words': definition_words,
            'total_words': total_words,
            'entity_tokens': entity_tokens,
            'definition_tokens': definition_tokens,
            'total_tokens': total_tokens
        }

    def process(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Main processing pipeline"""
        self.logger.info(f"Starting processing with {len(input_df)} rows")
        idf_df_in= self.compute_idf_scores(input_df)
        df_with_idf_in = input_df.merge(idf_df_in, on="entity", how="left")
        analyze_idf_distribution(df_with_idf_in)
        # Filter generic and junk entities
        df_filtered = input_df[~input_df['entity'].apply(self.is_generic)]
        self.logger.info(f"After generic filtering: {len(df_filtered)} rows")

        df_filtered = df_filtered[~df_filtered['entity'].apply(self.is_junk)]
        self.logger.info(f"After junk filtering: {len(df_filtered)} rows")

        # Compute IDF scores
        idf_df = self.compute_idf_scores(df_filtered)
        df_with_idf = df_filtered.merge(idf_df, on="entity", how="left")
        analyze_idf_distribution(df_with_idf)

        # Process each paper
        results = []
        for paper_id, paper_df in df_with_idf.groupby("paper_id"):
            top_entities = self.get_top_entities(
                paper_df,
                self.config.max_entities_per_paper
            )

            # Calculate length statistics for each entity
            for entity in top_entities:
                entity.update(self.calculate_length_statistics(entity))

            results.append({
                'paper_id': paper_id,
                'entities': top_entities
            })

        self.logger.info(f"After html and urls cleaning: {len(df_filtered)} rows")
        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        self.logger.info(f"Processed {len(result_df)} papers")

        return result_df

    def save_results(self, dataset_name: str, result_df: pd.DataFrame):
        """Save processed results in JSONL format compatible with EvalDataset"""
        if self.config.output_dir is None:
            self.logger.warning("No output directory specified, skipping save")
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / f'{dataset_name}-ner.jsonl'

        # Create a single dictionary to hold all papers' data
        all_papers_data = {}

        for _, row in result_df.iterrows():
            paper_id = str(row['paper_id'])  # Ensure paper_id is string

            # Extract entities and definitions
            entities = []
            entities_with_def = []

            for entity in row['entities']:
                entity_text = entity['entity']
                definition = entity['definition']

                entities.append(entity_text)
                entities_with_def.append(f"{entity_text}: {definition}")

            # Store metadata
            metadata = [{
                'entity': entity['entity'],
                'idf_score': float(entity['idf_score']),
                'original_entity': entity['original_entity'],
                'long_form': entity.get('long_form', '-'),  # Default to '-' if no long form
                'entity_tokens': entity.get('entity_tokens', None),
                'definition_tokens': entity.get('definition_tokens', None)
            } for entity in row['entities']]

            # Add this paper's data to the main dictionary
            all_papers_data[paper_id] = {
                'entities': [entities],
                'entities_with_def': [entities_with_def],
                'metadata': metadata
            }

        # Save all data at once
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_papers_data, f, ensure_ascii=False, indent=None)

        self.logger.info(f"Results saved to {output_path}")

def analyze_idf_distribution(df: pd.DataFrame) -> None:
    """Analyze IDF score distribution to help set threshold"""
    idf_scores = df['idf'].dropna()
    percentiles = [25, 50, 75, 90, 95]
    stats = {
        'min': idf_scores.min(),
        'max': idf_scores.max(),
        'mean': idf_scores.mean(),
        'std': idf_scores.std(),
        **{f'p{p}': np.percentile(idf_scores, p) for p in percentiles}
    }
    print("\nIDF Score Distribution:")
    for metric, value in stats.items():
        print(f"{metric}: {value:.2f}")

    plt.figure(figsize=(8, 5))  # Set figure size
    plt.hist(idf_scores, bins=10, edgecolor='black', alpha=0.7)  # Histogram settings
    plt.xlabel('IDF Score')  # X-axis label
    plt.ylabel('Frequency')  # Y-axis label
    plt.title('IDF Score Distribution')  # Title
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines
    plt.show()  # Show the plot



def show_stats(result_df: pd.DataFrame, processor: EntityPostProcessor):
    print(f"HTML cleaned: {processor.html_counter}")
    print(f"Too long definitions: {processor.too_long_counter}")
    k = 3
    # Print example output
    print(f"\nExample output for the first {k} paper:")
    for i in range(k):
        if len(result_df) > 0:
            paper = result_df.iloc[i]
            paper_id = paper['paper_id']
            entities = paper['entities']
            print(f"Paper ID: {paper_id}")
            for entity in entities:
                print(f"- {entity['entity']}: {entity['definition']}")

    # Calculate and print length statistics
    all_stats = {
        'entity_words': [],
        'definition_words': [],
        'total_words': [],
        'entity_tokens': [],
        'definition_tokens': [],
        'total_tokens': []
    }

    for _, row in result_df.iterrows():
        for entity in row['entities']:
            for stat in all_stats:
                all_stats[stat].append(entity[stat])

    print(f"\nLength Statistics:")
    print("\nWord Counts:")
    print(f"Entity: mean={np.mean(all_stats['entity_words']):.1f}, "
          f"std={np.std(all_stats['entity_words']):.1f}, "
          f"max={max(all_stats['entity_words'])}")
    print(f"Definition: mean={np.mean(all_stats['definition_words']):.1f}, "
          f"std={np.std(all_stats['definition_words']):.1f}, "
          f"max={max(all_stats['definition_words'])}")
    print(f"Total: mean={np.mean(all_stats['total_words']):.1f}, "
          f"std={np.std(all_stats['total_words']):.1f}, "
          f"max={max(all_stats['total_words'])}")

    print("\nToken Counts (SPECTER):")
    print(f"Entity: mean={np.mean(all_stats['entity_tokens']):.1f}, "
          f"std={np.std(all_stats['entity_tokens']):.1f}, "
          f"max={max(all_stats['entity_tokens'])}")
    print(f"Definition: mean={np.mean(all_stats['definition_tokens']):.1f}, "
          f"std={np.std(all_stats['definition_tokens']):.1f}, "
          f"max={max(all_stats['definition_tokens'])}")
    print(f"Total: mean={np.mean(all_stats['total_tokens']):.1f}, "
          f"std={np.std(all_stats['total_tokens']):.1f}, "
          f"max={max(all_stats['total_tokens'])}")

    # Calculate average entities per paper
    avg_entities = np.mean([len(row['entities']) for _, row in result_df.iterrows()])
    total_papers = len(result_df)

    print(f"\nProcessing Summary:")
    print(f"Total papers processed: {total_papers}")
    print(f"Average entities per paper: {avg_entities:.2f}")
    unique_entities = len({entity['entity']
                           for _, row in result_df.iterrows()
                           for entity in row['entities']})
    total_entities = sum(len(row['entities']) for _, row in result_df.iterrows())

    print(f"\nTotal number of entities: {total_entities}")
    print(f"Number of unique entities: {unique_entities}")

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                                      default='allenai/specter',
                                      help='The name of the model and tokenizer.',
                                      choices=['allenai/specter'])
    parser.add_argument('--dataset', default='relish', required=True, help='Dataset to post-process its entities',
                        choices=['cocitabs', 'csfcube', 'relish', 'treccovid'])
    parser.add_argument('--output_dir', default='ner_outputs', help='Output directory to save results')
    parser.add_argument('--stats', action='store_true', help='Whether to show length statistics')

    args = parser.parse_args()

    config = PostProcessConfig(
        generic_terms=GENERIC_TERMS,
        max_entities_per_paper=5,
        min_entity_length=2,
        min_definition_length=10,
        max_tokens_for_definition=400,  # Optional: Set maximum definition length
        min_idf_threshold=3.0,  # This means term appears in < ~13% of documents
        output_dir=Path(args.output_dir)
    )

    # Initialize processor
    processor = EntityPostProcessor(config)

    # Load data
    root = Path(__file__).parent.parent
    data_path = root / "datasets" / "eval" / args.dataset / f"{args.dataset}-ner.parquet"
    df = pd.read_parquet(data_path)

    # Process data
    result_df = processor.process(df)

    # Save results
    processor.save_results(dataset_name=args.dataset, result_df=result_df)
    if args.stats:
        show_stats(result_df, processor)


if __name__ == "__main__":
    main()