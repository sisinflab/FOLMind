"""
Module description:
    AMIE processor class
"""

__version__ = '0.0'
__email__ = ''

import os.path
import subprocess
from typing import List
from pathlib import Path
import logging
import pandas as pd
import re
from tqdm import tqdm
from enum import Enum
import csv
from .AMIEConfig import AMIEConfig
from .AMIEGrounder import AMIEGrounder

logger = logging.getLogger(__name__)


class AMIEProcessor:
    def __init__(self, config: AMIEConfig, train_path, data_path, jar: str = "amie.jar"):
        self.config = config
        self.train_dir = train_path
        self.dataset_path = data_path
        jar_path = Path.cwd().joinpath("folm_core", "amie", jar)
        self.jar_path = Path(jar_path)
        self.grounder = AMIEGrounder()

    def run(self, input_tsv: str, output_file: str = None) -> None:
        """Execute AMIE-3.5 with the configured parameters"""
        # - dataset info
        print("###############-DATASET INFO-#################")
        self._dataset_info(input_tsv=input_tsv)
        print("###############--------------#################")
        # run command
        args = self._build_args()
        cmd = self._build_command(args, input_tsv)
        self._execute_command(cmd, output_file)

    def ground(self, train_input_tsv: str, rules_file: str, dataset_dir: str):
        """Execute AMIE grounding from AMIEGrounder.py"""
        self.grounder.ground_rules(train_path=train_input_tsv, rules_path=rules_file,
                                   out_path=os.path.join(self.dataset_path, 'amie_mined', 'groundings.txt'),
                                   dataset_path=dataset_dir)

    def _build_args(self) -> List[str]:
        """Convert AMIEConfig to command-line arguments"""
        args = []
        config_data = self.config.model_dump(exclude_unset=True)

        param_mapping = {
            # core parameters
            'mins': '-mins',
            'minis': '-minis',
            'minhc': '-minhc',
            'pm': '-pm',
            'maxad': '-maxad',
            'minpca': '-minpca',
            'minc': '-minc',

            # rule constraints
            'bexr': '-bexr',
            'hexr': '-hexr',
            'iexr': '-iexr',
            'htr': '-htr',
            'btr': '-btr',
            'itr': '-itr',

            # constants handling
            'const': '-const',
            'fconst': '-fconst',

            # execution parameters
            'nc': '-nc',
            'bias': '-bias',
            'caos': '-caos',
            'verbose': '-verbose',
            'rl': '-rl',

            # optimization flags
            'optimcb': '-optimcb',
            'optimfh': '-optimfh',
            'optimai': '-optimai',

            # output control
            'oute': '-oute',
            'datalog': '-datalog',
            'oout': '-oout',
            'full': '-full',
            'ost': '-ostd',

            # advanced parameters
            'auta': '-auta',
            'deml': '-deml',
            'dqrw': '-dqrw',
            'dpr': '-dpr',
            'noHeuristics': '-noHeuristics',
            'noKbRewrite': '-noKbRewrite',
            'noKbExistsDetection': '-noKbExistsDetection',
            'noSkyline': '-noSkyline',
            'vo': '-vo',
            'mlg': '-mlg',
            'd': '-d',

            # file parameters
            'ef': '-ef'
        }

        for field, value in config_data.items():
            flag = param_mapping[field]

            if isinstance(value, bool):
                if value:
                    args.append(flag)
            elif isinstance(value, list):
                args.extend([flag, ",".join(value)])
            elif isinstance(value, Enum):
                args.extend([flag, str(value.value)])
            else:
                args.extend([flag, str(value)])

        return args

    def _build_command(self, args: List[str], input_tsv: str) -> List[str]:
        """Construct the full command-line invocation"""
        cmd = [
                  "java",
                  "-jar",
                  str(self.jar_path.resolve()),
                  input_tsv
              ] + args

        return cmd

    def _dataset_info(self, input_tsv):
        df = pd.read_csv(input_tsv, sep='\t', header=None, names=['subject', 'predicate', 'object'], dtype=str)
        print(df.head())
        print("Unique relations:", df['predicate'].nunique())
        print("Top relations:\n", df['predicate'].value_counts().head(10))

    def _execute_command(self, cmd: list, output_file: str) -> None:
        """Execute AMIE-3.5 command"""
        try:
            logger.info(f"Executing AMIE command: {' '.join(cmd)}\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            rule_lines = []  # store mined rule lines
            capture_table = False  # flag to detect the rules table
            headers = None

            # read AMIE output line by line
            for line in process.stdout:
                print(line, end="")  # print AMIE logs in real-time

                # detect the start of the rules table
                if "Rule\tHead Coverage" in line:
                    capture_table = True
                    headers = line.strip().split("\t")  # extract column names
                    continue  # skip this line (headers already extracted)

                # capture only rule lines
                if capture_table:
                    if line.strip() == "" or "Mining done" in line:
                        break  # stop capturing when mining ends
                    # convert commas to dots in numeric values
                    formatted_line = re.sub(r'(\d),(\d)', r'\1.\2', line.strip())
                    rule_lines.append(formatted_line.split("\t"))  # properly split columns

            process.wait()

            # capture any errors
            stderr_output = process.stderr.read()
            if stderr_output:
                logger.error(f"AMIE STDERR:\n{stderr_output}")

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            # save only the rules as a structured CSV file
            if rule_lines and headers:
                rules_df = pd.DataFrame(rule_lines, columns=headers)
                csv_numeric_file = str(output_file)
                rules_df.to_csv(csv_numeric_file, index=False, encoding="utf-8", sep=",")
                logger.info(f"Rules successfully extracted to {csv_numeric_file}")

                # reverse mapping -> from numeric relations to string entity
                mapping_tsv = os.path.join(str(self.train_dir), 'numeric_mappings', "id_to_predicate.tsv")
                csv_entity_file = csv_numeric_file.replace('mined_rules.csv', 'mined_rules_string.csv')
                self.convert_numeric_to_string(csv_numeric_file, mapping_tsv, csv_entity_file)
                logger.info(f"String rules successfully extracted to {csv_entity_file}")

        except subprocess.CalledProcessError as e:
            logger.error(f"AMIE execution failed: {e}")
            raise

    # REVERSE MAPPING FUNCTIONS
    def load_mapping(self, mapping_file):
        """
        Reads the mapping TSV file where each row contains:
          id <tab> relation
        and returns a dictionary: { id: relation }
        """
        df = pd.read_csv(mapping_file, sep='\t', header=None, names=['id', 'relation'])
        mapping = dict(zip(df['id'].astype(str), df['relation']))
        return mapping

    def process_triples(self, triple_str, id_to_relation):
        """
        Processes a string containing one or more triples.

        The triple string is expected to contain tokens in multiples of three:
          subject predicateID object [subject predicateID object ...]

        This function converts each predicateID into its corresponding relation label
        using the id_to_relation mapping and then joins the processed triples back into a string.
        """
        tokens = triple_str.strip().split()
        triple_list = []

        # process tokens in groups of three
        for i in range(0, len(tokens), 3):
            if i + 2 < len(tokens):
                subj = tokens[i]
                pred = tokens[i + 1]
                obj = tokens[i + 2]
                pred_label = id_to_relation.get(pred, pred)
                triple_list.append(f"{subj}  {pred_label}  {obj}")
        return " ".join(triple_list)

    def convert_numeric_to_string(self, input_path, mapping_file, output_path):
        """
        Processes an input TSV file where each line is formatted as:
          left_triple(s) => right_triple(s)

        Each side may contain one or more triples (i.e., tokens grouped in threes).
        The predicate IDs are converted to their string labels using the mapping file.

        Example:
          Input:  ?a 1 ?b ?a 3 ?b=> ?a 2 ?b
          Output: ?a likes ?b ?a dislikes ?b => ?a dislikes ?b
        (assuming mapping file defines 1->likes, 2->dislikes, 3->dislikes, or as provided)
        """
        id_to_relation = self.load_mapping(mapping_file)

        with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', newline='',
                                                                  encoding='utf-8') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)

            _ = next(reader)  # skip header row
            for row in tqdm(reader, desc="Reverse mapping rules.."):
                if not row or not row[0]:
                    continue

                rule = row[0]  # the "Rule" column
                if '=>' not in rule:
                    writer.writerow([rule])
                    continue

                left_str, right_str = rule.split('=>')
                left_processed = self.process_triples(left_str, id_to_relation)
                right_processed = self.process_triples(right_str, id_to_relation)
                writer.writerow([f"{left_processed} => {right_processed}"])




