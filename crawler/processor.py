import pandas as pd
import operator
import re

# mapping of supported operators for safe condition evaluation
OPERATORS = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}

def parse_condition(condition: str):
    """
    Parses a condition string (e.g., '>=3.0') and returns a tuple (operator_function, threshold).
    Raises a ValueError if the condition format is invalid.
    """
    pattern = r'(>=|<=|>|<|==|!=)\s*(\d+(\.\d+)?)'
    match = re.match(pattern, condition.strip())
    if not match:
        raise ValueError(f"Invalid condition format: {condition}")
    op_str, threshold_str, _ = match.groups()
    return OPERATORS[op_str], float(threshold_str)

class TripleProcessor:
    """
    Transforms raw data into knowledge graph triples based on configurable rules.
    Supports 'direct', 'conditional', and 'multi_value' triple creation.
    """
    def __init__(self, relation_config: list, predicate_mapping: dict):
        self.relation_config = relation_config
        self.predicate_mapping = predicate_mapping
        self.handlers = {
            'direct': self._handle_direct,
            'conditional': self._handle_conditional,
            'multi_value': self._handle_multi_value
        }

    def create_relations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all relation configurations to generate triples.
        Validates that the required columns exist in the DataFrame.
        """
        if df.empty:
            return pd.DataFrame()
        triples = []
        for config in self.relation_config:
            for key in ['object_col']:
                if key not in config or config[key] not in df.columns:
                    raise ValueError(f"Missing column '{config.get(key)}' in DataFrame")
            handler = self.handlers.get(config['type'])
            if not handler:
                continue
            # map the predicate using KB-specific mapping if available
            predicate = self.predicate_mapping.get(config['predicate'], config['predicate'])
            # generate triples and filter out s == o
            triple_df = handler(df, config, predicate)
            triple_df = triple_df[triple_df['s'] != triple_df['o']]
            triples.append(triple_df)
        return pd.concat(triples) if triples else pd.DataFrame()


    def _handle_direct(self, df: pd.DataFrame, config: dict, predicate: str) -> pd.DataFrame:
        """
        Creates triples by directly mapping columns.
        """
        return pd.DataFrame({
            's': df[config['subject_col']],
            'p': predicate,
            'o': df[config['object_col']]
        })

    def _handle_conditional(self, df: pd.DataFrame, config: dict, predicate: str) -> pd.DataFrame:
        """
        Creates triples only when the numeric value in 'value_col' meets the specified condition.
        Uses safe evaluation by parsing the condition and applying the corresponding operator.
        """
        df[config['value_col']] = pd.to_numeric(df[config['value_col']], errors='coerce')
        op_func, threshold = parse_condition(config['condition'])
        mask = df[config['value_col']].apply(lambda x: op_func(x, threshold))
        return pd.DataFrame({
            's': df['userId'],
            'p': predicate,
            'o': df[config['object_col']]
        })[mask]

    def _handle_multi_value(self, df: pd.DataFrame, config: dict, predicate: str) -> pd.DataFrame:
        """
        Handles columns containing multiple values separated by a delimiter.
        Splits the values and creates individual triples.
        """
        exploded = df[config['object_col']].str.split(config['separator']).explode()
        return pd.DataFrame({
            's': df[config['subject_col']].repeat(exploded.groupby(level=0).size()),
            'p': predicate,
            'o': exploded.str.strip()
        })
