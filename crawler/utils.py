import pandas as pd
from pathlib import Path
import pickle
from .downloader import download_file, extract_zip, fix_extracted_structure
from typing import Iterator, Optional, Dict
import re
import csv
from collections import Counter
from SPARQLWrapper import SPARQLWrapper, JSON


def prepare_dataset_directory(url: str, data_root: str) -> Path:
    """
    Prepare the dataset directory based on the URL and data_root.
    Creates the directory if it does not exist.
    """
    dataset_name = Path(url).stem.split(".")[0]
    dataset_dir = Path(data_root) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def download_dataset(url: str, data_root: str) -> str:
    """
    Download and extract the dataset from the provided URL.
    If the 'original' subdirectory with content exists, assumes the data is already available.
    """
    dataset_dir = prepare_dataset_directory(url, data_root)
    original_dir = dataset_dir / "original"

    # Check for the 'original' directory's existence and content
    if not original_dir.exists() or not any(original_dir.iterdir()):
        print(f"Downloading dataset '{dataset_dir.name}'...")
        zip_path = dataset_dir / "temp.zip"
        download_file(url, zip_path)

        # Extract to a temporary directory before moving to 'original'
        temp_extract = dataset_dir / "temp_extract"
        original_dir.mkdir(parents=True, exist_ok=True)
        temp_extract.mkdir(parents=True, exist_ok=True)

        extract_zip(zip_path, temp_extract)
        fix_extracted_structure(temp_extract, original_dir)
        zip_path.unlink()  # Remove the temporary zip file
    else:
        print(f"Dataset already found in '{original_dir}'. Skipping download.")

    return str(dataset_dir)


def apply_value_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    for col, mapping_dict in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping_dict)
    return df


def clean_kg_entity(entity_string: str) -> str:
    """
    Takes a string that could be a URI, a malformed URI, or a literal,
    and returns a clean, normalized final name. This is the single
    authoritative function for KG entity normalization.
    """
    if not isinstance(entity_string, str):
        return str(entity_string)

    # if it's a URI or a malformed URI, extract the last part.
    if entity_string.strip().lower().startswith('http'):
        # Use regex to split by any sequence of slashes, getting the last non-empty part.
        parts = re.split(r'/+', entity_string)
        cleaned = parts[-1] if parts[-1] else (parts[-2] if len(parts) > 1 else '')
    else:
        # already a literal
        cleaned = entity_string

    # replace spaces/special chars with underscores
    normalized = re.sub(r'[\s\W]+', '_', cleaned).strip('_')
    return normalized


def store_tsv(data: pd.DataFrame, path: str) -> None:
    """
    Saves a DataFrame of triples to a TSV file after applying final cleaning and normalization.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df_to_save = data.copy()
    df_to_save.dropna(subset=['s', 'p', 'o'], inplace=True)
    df_to_save[['s', 'p', 'o']] = df_to_save[['s', 'p', 'o']].astype(str)

    # cleaning
    df_to_save['s'] = df_to_save['s'].apply(clean_kg_entity)
    df_to_save['o'] = df_to_save['o'].apply(clean_kg_entity)

    df_to_save['p'] = df_to_save['p'].apply(clean_kg_entity)

    df_to_save.to_csv(path, sep='\t', index=False, header=False, lineterminator='\n')


def extract_resource(uri: str) -> str:
    if isinstance(uri, str):
        if uri.strip().startswith('http'):
            parts = re.split(r'[/\\]+', uri)
            return parts[-1] if parts else uri
    return uri


def chunker(lst: list, chunk_size: int) -> list:
    """
    Splits a list into smaller chunks of a specified size.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def create_mappings(knowledge_graph_path: str, output_dir: str):
    """Generate numeric mappings for entities and predicates, and store mapping files in a subdirectory within output_dir."""
    df = pd.read_csv(knowledge_graph_path, sep='\t', header=None, names=['s', 'p', 'o'])

    # collect unique entities and predicates
    entities = pd.concat([df['s'], df['o']]).unique()
    predicates = df['p'].unique()

    # create mapping dictionaries
    entity_to_id = {entity: idx for idx, entity in enumerate(entities, 1)}  # Start IDs from 1
    predicate_to_id = {pred: idx for idx, pred in enumerate(predicates, 1)}

    # create reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_predicate = {v: k for k, v in predicate_to_id.items()}

    # create subdirectory inside output_dir to store output of this function
    base_output_dir = Path(output_dir)
    mappings_output_dir = base_output_dir / "numeric_mappings"
    mappings_output_dir.mkdir(parents=True, exist_ok=True)

    # save mapping files in the new subdirectory
    pd.Series(entity_to_id).to_csv(mappings_output_dir / 'entity_to_id.tsv', sep='\t', header=False)
    pd.Series(predicate_to_id).to_csv(mappings_output_dir / 'predicate_to_id.tsv', sep='\t', header=False)
    pd.Series(id_to_entity).to_csv(mappings_output_dir / 'id_to_entity.tsv', sep='\t', header=False)
    pd.Series(id_to_predicate).to_csv(mappings_output_dir / 'id_to_predicate.tsv', sep='\t', header=False)

    return entity_to_id, predicate_to_id


def convert_to_numeric(knowledge_graph_path: str, mapping_dir: str, output_path: str):
    """Convert original KG to numeric IDs"""
    df = pd.read_csv(knowledge_graph_path, sep='\t', header=None, names=['s', 'p', 'o'])

    entity_map, predicate_map = create_mappings(knowledge_graph_path, mapping_dir)

    df['s'] = df['s'].map(entity_map)
    df['p'] = df['p'].map(predicate_map)
    df['o'] = df['o'].map(entity_map)

    df.to_csv(output_path, sep='\t', header=False, index=False)


def create_entity_name_mapping(entities_df: pd.DataFrame, output_path: str, config):
    """
    Creates a consistent mapping from a normalized KG entity name to its original local ID.
    e.g., 'Clerks_II_2006_film' -> '858'
    """
    id_col = config.columns.entity_id
    name_col = config.columns.fallback_uri  # e.g title

    final_mapping = {}

    for _, row in entities_df.dropna(subset=[id_col, name_col]).iterrows():
        movie_id = row[id_col]

        # get the original name from the dataset (e.g., "1984 (Nineteen Eighty-Four)")
        name_from_dataset = row[name_col]

        # normalize this name
        normalized_name = clean_kg_entity(name_from_dataset)

        # map KG name to the local ID
        final_mapping[normalized_name] = movie_id

    print(f"Created entity name mapping pickle with {len(final_mapping)} entries.")
    with open(output_path, 'wb') as f:
        pickle.dump(final_mapping, f)


# SMART SEARCH LOGIC

def build_uri(name: str, config) -> Iterator[str]:
    """
    Generate candidate DBpedia resource URIs for a given item name, including variants
    with and without the year and a "film" keyword.

    Example for "Clerks (1994)":
    - http://dbpedia.org/resource/Clerks
    - http://dbpedia.org/resource/Clerks_(film)
    - http://dbpedia.org/resource/Clerks_(1994_film)
    """
    base_uri = config.base_uri.rstrip('/') + '/'

    # extract year (if any) using a robust regex
    m_year = re.search(r'\((\d{4})\)', name)
    year = m_year.group(1) if m_year else None

    # Get the title part, cleaning out any parenthetical year or other info (1994)
    name_no_year = re.sub(r'\s*\([^)]*\)$', '', name).strip()

    # handle "Title, The" format
    if name_no_year.endswith(", The"):
        name_no_year = "The " + name_no_year[:-5]

    # normalize the title for a URI (replace spaces with underscores, etc.)
    cleaned_name = re.sub(r'[,\'\"()]', '', name_no_year)
    cleaned_name = re.sub(r'\s+', '_', cleaned_name).strip('_')

    # yield progressively more specific candidates

    # base name only
    yield base_uri + cleaned_name

    # base name with (film), (novel) suffix
    yield f"{base_uri}{cleaned_name}_(film)"
    yield f"{base_uri}{cleaned_name}_(novel)"

    # base name with (year) suffix
    if year:
        yield f"{base_uri}{cleaned_name}_({year})"

    # base name with (year_film or novel) suffix
    if year:
        yield f"{base_uri}{cleaned_name}_({year}_film)"
        yield f"{base_uri}{cleaned_name}_({year}_novel)"

def extract_leading_tokens(tsv_path, top_n=30):
    cnt = Counter()
    with open(tsv_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for line_number, row in enumerate(reader, 1):
            try:
                # itemId, title, kb_uri
                _, _, uri = row
                name = uri.rsplit('/', 1)[-1]  # e.g. "The_Last_Supper_(1995_film)"
                token = name.split('_', 1)[0]  # take everything before first underscore
                if token.istitle() and len(token) <= 5:
                    cnt[token] += 1
            except ValueError:
                print(f"Warning: Skipping malformed line {line_number}: {row}")
            except Exception as e:
                print(f"Error on line {line_number}: {e} — content: {row}")
    return cnt.most_common(top_n)


def lookup_uri(uri: str, smart_query: str, endpoint: str) -> bool:
    """
    Issue a SPARQL ASK to test whether <uri> matches the provided smart_query triple pattern.
    """
    sparql = SPARQLWrapper(endpoint)
    ask_query = f"ASK WHERE {{ <{uri}> {smart_query} }}"
    sparql.setQuery(ask_query)
    sparql.setReturnFormat(JSON)
    try:
        result = sparql.query().convert()
        return bool(result.get('boolean'))
    except Exception:
        # on failure/timeouts, assume no match
        return False


def find_first_valid_uri(name: str, config, default_articles=['The', 'A', 'An']) -> Optional[str]:
    """
    Iterate through candidate URIs and return the first one that satisfies the ASK query.
    """
    service_config = config.smart_search_service
    for uri in build_uri(name, config, default_articles):
        if lookup_uri(uri, service_config.smart_query, service_config.endpoint):
            return uri
    return None