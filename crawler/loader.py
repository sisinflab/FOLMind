import os
import pandas as pd
from pathlib import Path
import chardet
from tqdm import tqdm
from typing import Optional, Set, Any

def read_dat(file_path, encoding):
    """
    Reads a .dat file using a double colon separator.
    """
    return pd.read_csv(file_path, sep='::', engine='python', header=None, encoding=encoding, on_bad_lines='warn')

def read_tsv(file_path, encoding):
    """
    Reads a TSV file.
    """
    return pd.read_csv(file_path, sep='\t', header=None, encoding=encoding, on_bad_lines='warn')

def read_json(file_path, encoding):
    """
    Reads a JSON file, trying both lines and standard format.
    """
    try:
        # try to read as a JSON Lines file
        return pd.read_json(file_path, encoding=encoding, lines=True)
    except ValueError:
        # if fails, fall back to reading as a standard JSON file
        try:
            return pd.read_json(file_path, encoding=encoding, lines=False)
        except Exception as e:
            print(f"Warning: Could not parse JSON file {file_path}. Error: {e}")
            return pd.DataFrame() # Return empty DataFrame on failure

# Mapping file extensions to reader functions
READERS = {
    'dat': read_dat,
    'tsv': read_tsv,
    'json': read_json
}

def convert_file(file_path: Path, encoding: str):
    """
    Convert a file to a DataFrame, handling different delimiters and potential headers.
    """
    ext = file_path.suffix[1:].lower()
    if ext == 'dat':
        for sep in ['\t', '::']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, encoding=encoding)
                # Basic validation: If it has at least 2 columns, assume it's correctly parsed
                if df.shape[1] >= 2:
                    # Check if the first row looks like a header (string values)
                    if all(isinstance(col, str) for col in df.iloc[0]):
                        return pd.read_csv(file_path, sep=sep, header=0, encoding=encoding)
                    else:
                        return df
                # If parsing fails, try the other separator
            except UnicodeDecodeError:
                continue  # Try the next separator or encoding fallback

        # If all delimiters fail, try ISO-8859-1 fallback
        for sep in ['\t', '::']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, encoding='ISO-8859-1', on_bad_lines='warn')
                if df.shape[1] >= 2:
                    # Check for header in fallback encoding
                    if all(isinstance(col, str) for col in df.iloc[0]):
                        return pd.read_csv(file_path, sep=sep, header=0, encoding='ISO-8859-1', on_bad_lines='warn')
                    else:
                        return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Failed to decode {file_path} with both delimiters and encodings")

    elif ext == 'tsv':
        return pd.read_csv(file_path, sep='\t', header=None, encoding=encoding, on_bad_lines='warn')
    elif ext == 'json':
        return read_json(file_path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_mapping(mapping_file: str, mapping_columns: list, base_path: Path) -> dict:
    """Loads a TSV mapping file and returns a dictionary mapping entity IDs to KB URIs."""
    mapping_path = base_path / mapping_file
    if not mapping_path.exists():
        print(f"Mapping file not found at {mapping_path}. It will be created.")
        return {}
    try:
        mapping_df = pd.read_csv(mapping_path, sep='\t', header=None, names=mapping_columns)
        # Ensure the ID column is of a consistent type, trying integer first.
        try:
            id_type = int
            mapping_df[mapping_columns[0]] = mapping_df[mapping_columns[0]].astype(id_type)
        except ValueError:
            id_type = str  # Fallback to string if IDs are not purely numeric
            mapping_df[mapping_columns[0]] = mapping_df[mapping_columns[0]].astype(id_type)

        return dict(zip(mapping_df[mapping_columns[0]], mapping_df[mapping_columns[2]]))
    except pd.errors.EmptyDataError:
        print(f"Warning: Mapping file {mapping_path} is empty.")
        return {}

def find_csv_files(root_dir: str, keywords: list) -> list:
    """
    Searches the directory for CSV files whose names contain any of the specified keywords.
    """
    matches = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filename_without_ext = Path(filename).stem.lower() #Extracted filename
            if filename.endswith('.csv') and any(kw == filename_without_ext for kw in keywords): # exact match
                matches.append(os.path.join(root, filename))
    return matches

class DatasetConverter:
    """
    Converts raw dataset files (.dat, .tsv, .json) into CSV format.
    Handles encoding detection and assigns column names based on file keywords.
    """
    def __init__(self, data_dir: str, entity_kws: list, interaction_kws: list,
                 user_kws: list, entity_cols: list, interaction_cols: list, user_cols: list):
        self.data_dir = data_dir
        self.entity_kws = entity_kws
        self.interaction_kws = interaction_kws
        self.user_kws = user_kws
        self.entity_cols = entity_cols
        self.interaction_cols = interaction_cols
        self.user_cols = user_cols

    def convert_all(self) -> dict:
        """
        Iterates over all files in the dataset directory, converts supported files to CSV,
        assigns columns based on keywords, and deletes the original file.
        Returns a mapping of original filenames to new CSV file paths.
        """
        converted = {}
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.lower().endswith(('.dat', '.tsv', '.json')):
                    continue
                file_path = Path(root) / file
                with open(file_path, 'rb') as f:
                    rawdata = f.read(10000)
                    encoding = chardet.detect(rawdata)['encoding']
                df = convert_file(file_path, encoding)
                # Assign column names based on keywords in the filename
                filename_without_ext = Path(file).stem.lower()

                if any(kw == filename_without_ext for kw in self.entity_kws):
                    df.columns = self.entity_cols
                elif any(kw == filename_without_ext for kw in self.interaction_kws):
                    df.columns = self.interaction_cols
                elif any(kw == filename_without_ext for kw in self.user_kws):
                    df.columns = self.user_cols
                csv_path = file_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                file_path.unlink()  # Delete the original file
                converted[file] = str(csv_path)
        return converted


class Loader:
    """
    Manages dataset loading:
      - Converts raw files to CSV.
      - Loads entity-to-KB URI external_mappings.
      - Searches for CSV files corresponding to entities, interactions and users.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_dir = config.dataset.data_dir

        # Convert raw dataset files to CSV format
        self.converter = DatasetConverter(
            self.dataset_dir,
            config.processing.entity_keywords,
            config.processing.interaction_keywords,
            config.processing.user_keywords,
            config.processing.entity_columns,
            config.processing.interaction_columns,
            config.processing.user_columns
        )
        self.converted_files = self.converter.convert_all()

        # load KB external_mappings using the helper function
        base_path = Path(self.config.dataset.data_root).parent
        self.mapping = load_mapping(config.mapping.file, config.mapping.columns, base_path)

        # find CSV files for entities, interactions and users using helper function
        self.entity_files = find_csv_files(self.dataset_dir, config.processing.entity_keywords)
        self.interaction_files = None
        self.user_files = find_csv_files(self.dataset_dir, config.processing.user_keywords)

    def load_entities(self, queryer, valid_item_ids: Optional[Set[Any]] = None) -> pd.DataFrame:
        if not self.entity_files:
            raise FileNotFoundError(
                f"No entity files found with keywords {self.config.processing.entity_keywords} in {self.dataset_dir}")

        dfs = [pd.read_csv(file) for file in self.entity_files]
        entities_df = pd.concat(dfs, ignore_index=True)

        entity_id_col = self.detect_column(entities_df, 'entity_id')

        if valid_item_ids is not None:
            initial_count = len(entities_df)
            # Filter the entities DataFrame to only include items that are in the final train/val/test splits.
            # A .copy() is used to prevent potential SettingWithCopyWarning issues later.
            entities_df = entities_df[entities_df[entity_id_col].isin(valid_item_ids)].copy()
            print(f"Filtered entities to only include valid items (from train/val/test splits). "
                  f"Kept {len(entities_df)} of {initial_count} initial entities.")

        try:
            entities_df[entity_id_col] = entities_df[entity_id_col].astype(int)
        except (ValueError, TypeError):
            entities_df[entity_id_col] = entities_df[entity_id_col].astype(str)

        entities_df[self.config.columns.kb_uri] = entities_df[entity_id_col].map(self.mapping)

        mask = entities_df[self.config.columns.kb_uri].isna()
        missing_names = entities_df.loc[
            mask, [entity_id_col, self.config.columns.fallback_uri]].drop_duplicates().values.tolist()

        should_run_lookup = (
            # always run for wikidata to create the initial file
                self.config.knowledge_base == 'wikidata' or
                # For other KBs, check if the smart search service is configured and enabled
                (
                        hasattr(self.config, 'smart_search_service') and
                        getattr(self.config.smart_search_service, 'use_smart_search', False)
                )
        )

        if missing_names and should_run_lookup:
            print(f"Looking up {len(missing_names)} missing URIs for {self.config.knowledge_base}...")
            base_path = Path(self.config.dataset.data_root).parent
            mapping_path = base_path / self.config.mapping.file
            self.mapping = queryer.populate_mapping(missing_names, self.mapping, mapping_path)
            entities_df[self.config.columns.kb_uri] = entities_df[entity_id_col].map(self.mapping)

        entities_df[self.config.columns.kb_uri].fillna(entities_df[self.config.columns.fallback_uri], inplace=True)

        entities_df[self.config.columns.kb_uri].fillna('', inplace=True)
        return entities_df

    def detect_column(self, df: pd.DataFrame, key: str) -> str:
        """
        Detect actual column name in df using config.processing.column_aliases[key] plus canonical name.
        """
        primary = getattr(self.config.columns, key)
        aliases = self.config.processing.column_aliases
        fallbacks = aliases.get(key, [])

        candidates = [primary] + fallbacks
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        raise ValueError(
            f"Expected column for '{key}' not found. Tried {aliases} in {list(df.columns)}"
        )

    def load_interactions(self, entities_df) -> pd.DataFrame:
        """
        Loads interaction CSV files, maps entity IDs to KB URIs, and applies any user ID modifications.
        Validates the presence of required columns.
        """
        fallback_map = dict(zip(entities_df[self.config.columns.entity_id], entities_df[self.config.columns.fallback_uri]))
        self.interaction_files = find_csv_files(str(os.path.join(self.dataset_dir, 'splitting', 'train')), self.config.processing.interaction_keywords)
        # build mapping from entities df
        uri_map = dict(zip(entities_df[self.config.columns.entity_id], entities_df[self.config.columns.kb_uri]))

        dfs = []
        for file in self.interaction_files:
            df = pd.read_csv(file)
            entity_col = "itemId" #self._detect_column(df, 'entity_id')
            # Map entity id to kb_uri
            df[self.config.columns.kb_uri] = df[entity_col].map(uri_map)

            if 'userId' in df.columns:
                df['userId'] = 'user_' + df['userId'].astype(str)

            # no dropping but fill with "title"
            df[self.config.columns.kb_uri] = df[self.config.columns.kb_uri].fillna(df[entity_col].map(fallback_map))
            dfs.append(df)
        return pd.concat(dfs)

    def load_users(self) -> pd.DataFrame:
        """Load user data and standardize columns."""
        if not self.user_files:  # no files matching user_keywords found
            return pd.DataFrame()
        dfs = []
        for file in self.user_files:
            df = pd.read_csv(file)
            # if required columns exist
            if self.config.columns.user_id not in df.columns:
                raise ValueError(f"Missing user_id column in {file}")
            # add a prefix to user IDs if the column exists
            if self.config.columns.user_id in df.columns:
                df[self.config.columns.user_id] = 'user_' + df[self.config.columns.user_id].astype(str)
            dfs.append(df)
        return pd.concat(dfs)