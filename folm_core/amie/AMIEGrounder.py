import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import psutil


class AMIEGrounder:

    def _parse_rule(self, rule_str):
        # splits a rule of the form "ATOM1 ATOM2 ... => HEAD" into head and body atoms
        parts = rule_str.split("=>")
        body_str, head_str = parts[0].strip(), parts[1].strip()
        tokens = body_str.split()
        # group tokens in triples for each body atom
        body_atoms = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(0, len(tokens), 3)]
        head_tokens = head_str.split()
        head_atom = (head_tokens[0], head_tokens[1], head_tokens[2])
        return head_atom, body_atoms

    def _get_bindings(self, atom, df):
        # atom is a tuple (subject_var, relation, object_var)
        subj, rel, obj = atom
        df_bind = df[df['predicate'] == rel].copy()
        df_bind = df_bind.rename(columns={'subject': subj, 'object': obj})
        return df_bind[[subj, obj]]

    def safe_merge(self, left_df, right_df, join_keys, usable_memory, rule_str):
        """
        Estimate the memory cost and if the estimated merge size exceeds the memory budget,
        process the merge in batches.
        """
        # estimate average memory per row for left and right
        row_mem_left = left_df.memory_usage(deep=True).sum() / max(len(left_df), 1)
        row_mem_right = right_df.memory_usage(deep=True).sum() / max(len(right_df), 1)
        row_mem_product = row_mem_left + row_mem_right
        max_product_rows = usable_memory / row_mem_product
        estimated_size = len(left_df) * len(right_df)

        if estimated_size > max_product_rows:
            # batch processing
            chunk_size = int(max_product_rows // len(right_df))
            if chunk_size < 1:
                chunk_size = 1
            product_chunks = []
            for start in range(0, len(left_df), chunk_size):
                chunk = left_df.iloc[start: start + chunk_size].copy()
                merged_chunk = pd.merge(chunk, right_df, on=join_keys, how='inner')
                product_chunks.append(merged_chunk)
            if product_chunks:
                return pd.concat(product_chunks, ignore_index=True)
            else:
                return pd.DataFrame(
                    columns=list(left_df.columns) + [col for col in right_df.columns if col not in join_keys])
        else:
            return pd.merge(left_df, right_df, on=join_keys, how='inner')

    def reverse_map_groundings(self, input_path: str, output_path: str, dataset_dir: str):
        """Convert numeric groundings back to original values"""
        # load reverse mappings
        id_to_entity = pd.read_csv(
            'data' / Path(dataset_dir) / 'splitting' / 'train' / 'numeric_mappings/id_to_entity.tsv',
            sep='\t', header=None, names=['id', 'entity']
        ).set_index('id')['entity'].to_dict()

        id_to_predicate = pd.read_csv(
            'data' / Path(dataset_dir) / 'splitting' / 'train' / 'numeric_mappings/id_to_predicate.tsv',
            sep='\t', header=None, names=['id', 'predicate']
        ).set_index('id')['predicate'].to_dict()

        # regular expression to capture triples within parentheses.
        triple_pattern = re.compile(r'\((\d+)[\t ]+(\d+)[\t ]+(\d+)\)')

        with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, desc='Processing groundings'):
                matches = triple_pattern.findall(line)
                reversed_triples = []
                if not matches:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                for s, p, o in matches:
                    reversed_s = id_to_entity.get(int(s), s)
                    reversed_p = id_to_predicate.get(int(p), p)
                    reversed_o = id_to_entity.get(int(o), o)
                    reversed_triples.append(f"({reversed_s}\t{reversed_p}\t{reversed_o})")

                f_out.write('\t'.join(reversed_triples) + '\n')
        print(f"Reverse mapping complete. Output written to {output_path}")

    def ground_rules(self, train_path: str, rules_path: str, out_path, dataset_path: str):
        # load training facts and rules.
        df = pd.read_csv(train_path, sep='\t', header=None, names=['subject', 'predicate', 'object'], dtype=str)
        df['predicate'] = df['predicate'].str.strip()
        rules_df = pd.read_csv(rules_path)

        with open(out_path, 'w', encoding='utf-8') as file_out:
            # use 50% of available memory as our budget.
            usable_memory = psutil.virtual_memory().available * 0.5

            for rule_str in tqdm(rules_df['Rule'], desc='Grounding rules'):
                head_atom, body_atoms = self._parse_rule(rule_str)
                bindings = [self._get_bindings(atom, df) for atom in body_atoms]
                merged_df = bindings[0]

                for bdf in bindings[1:]:
                    common_vars = list(set(merged_df.columns) & set(bdf.columns))
                    if common_vars:
                        merged_df = self.safe_merge(merged_df, bdf, join_keys=common_vars,
                                                    usable_memory=usable_memory, rule_str=rule_str)
                    else:
                        # for Cartesian product, add a key column.
                        merged_df['key'] = 1
                        bdf['key'] = 1
                        merged_df = self.safe_merge(merged_df, bdf, join_keys=['key'],
                                                    usable_memory=usable_memory, rule_str=rule_str)
                        merged_df = merged_df.drop('key', axis=1)

                head_subj, head_rel, head_obj = head_atom
                for idx, row in merged_df.iterrows():
                    grounded_body = []
                    for (var1, rel, var2) in body_atoms:
                        grounded_body.append(f"({row[var1]}\t{rel}\t{row[var2]})")
                    grounded_head = f"({row[head_subj]}\t{head_rel}\t{row[head_obj]})"
                    file_out.write("\t".join(grounded_body + [grounded_head]) + "\n")
        print(f"Grounding done. New facts in numeric form saved to {out_path}")
        string_out_path = out_path.parent / (out_path.stem + '_string' + out_path.suffix)
        self.reverse_map_groundings(out_path, string_out_path, dataset_path)
        print(f"Grounding done. New facts in string form saved to {out_path}")





