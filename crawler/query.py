from .utils import chunker, build_uri
from SPARQLWrapper import SPARQLWrapper, JSON
import tempfile
import multiprocessing
import pandas as pd
import requests
from tqdm import tqdm
import os
import re
from typing import Optional, List

# a global session and config for each worker process.
session = None
worker_config = None


def initialize_worker(config_obj):
    """Initializes a requests.Session and config for each worker in the pool."""
    global session, worker_config
    if session is None:
        session = requests.Session()
        session.headers.update({
            'User-Agent': "FOLM-Crawler/1.0 (https://github.com/FOLMIND)"
        })
    if worker_config is None:
        worker_config = config_obj


def _get_clean_search_term(raw_keyword: str) -> str:
    """A robust function to extract the most likely English search title from a messy string."""
    term = raw_keyword
    while re.search(r'\s*\([^)]*\)$', term):
        term = re.sub(r'\s*\([^)]*\)$', '', term).strip()
    if term.endswith(", The"):
        term = "The " + term[:-5].strip()
    elif term.endswith(", A"):
        term = "A " + term[:-3].strip()
    elif term.endswith(", An"):
        term = "An " + term[:-4].strip()
    if ':' in term:
        term = term.split(':', 1)[0].strip()
    return term.strip()


def _search_wikidata_candidates_sparql(raw_keyword: str) -> tuple[str, list[str]]:
    """
    Worker function: Finds candidate QIDs using a powerful and reliable SPARQL text search.
    """
    global session, worker_config
    search_term = _get_clean_search_term(raw_keyword)
    if not search_term:
        return raw_keyword, []

    svc_config = worker_config.lookup_services.get(worker_config.knowledge_base, {})
    class_filter = getattr(svc_config, 'search_class_filter', 'Q11424')

    query = f"""
        SELECT ?item WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:api "EntitySearch";
                            wikibase:endpoint "www.wikidata.org";
                            mwapi:search "{search_term.replace('"', '')}";
                            mwapi:language "en".
            ?item wikibase:apiOutput mwapi:item.
          }}
          ?item wdt:P31/wdt:P279* wd:{class_filter}.
        }}
        LIMIT 15
    """

    try:
        endpoint_url = worker_config.sparql.endpoint
        headers = {
            'Accept': 'application/sparql-results+json',
            'User-Agent': "FOLM-Crawler/1.0 (Python/requests)"
        }
        response = session.get(endpoint_url, headers=headers, params={'query': query, 'format': 'json'}, timeout=20)
        response.raise_for_status()
        data = response.json()
        qids = [res['item']['value'].split('/')[-1] for res in data.get("results", {}).get("bindings", [])]
        return raw_keyword, qids
    except Exception:
        return raw_keyword, []


class SparqlQueryer:
    """Executes multi-hop SPARQL queries and handles entity lookups for configurable KBs."""

    def __init__(self, config):
        self.config = config
        self.sparql = SPARQLWrapper(config.sparql.endpoint)
        self.sparql.agent = "FOLM-Crawler/1.0 (https://github.com/FOLMIND) python-SPARQLWrapper"
        self.sparql.setReturnFormat(JSON)
        self.predicate_map = config.predicate_mapping.get(config.knowledge_base, {})
        initialize_worker(config)
        self.session = session

    def _generate_queries(self, uris: list, predicate: str) -> list:
        template = self.config.sparql.templates[predicate]
        valid_uris = [uri for uri in uris if uri]
        if not valid_uris: return []
        chunks = chunker(valid_uris, self.config.sparql.chunk_size)
        if self.config.knowledge_base == 'wikidata':
            uri_formatter = lambda chunk: " ".join(f"wd:{uri.split('/')[-1]}" for uri in chunk)
        else:
            uri_formatter = lambda chunk: " ".join(f"<{uri}>" for uri in chunk)
        return [template % {"values": uri_formatter(chunk)} for chunk in chunks if chunk]

    def multi_hop_query(self, initial_uris: list) -> pd.DataFrame:
        all_triples = []
        current_uris = initial_uris
        visited = set(initial_uris)
        hop_categories = getattr(self.config.sparql, "hop_categories", None)
        for hop in range(self.config.sparql.max_hops):
            new_uris = []
            category = hop_categories[hop] if hop_categories and hop < len(hop_categories) else None
            print(f"Hop {hop}: using category '{category}'")
            predicates_for_hop = self.predicate_map.get(category, {})
            predicates = [p for p in self.config.sparql.templates if p in predicates_for_hop]
            for predicate in predicates:
                queries = self._generate_queries(current_uris, predicate)
                if not queries: continue
                results = self._execute_parallel(queries)
                if not results: continue
                predicate_name = predicates_for_hop[predicate]
                for s, o in results:
                    clean_s = s.strip("<>")
                    clean_o = o.strip("<>")
                    all_triples.append((clean_s, predicate_name, clean_o))
                    if clean_o not in visited and clean_o.startswith(self.config.base_uri):
                        new_uris.append(clean_o)
                        visited.add(clean_o)
            current_uris = new_uris
            if not current_uris: break
        return pd.DataFrame(all_triples, columns=['s', 'p', 'o'])

    def _execute_parallel(self, queries: list) -> list:
        with multiprocessing.Pool(self.config.sparql.processes) as pool:
            results = pool.map(self._execute_single_query, queries)
        return [item for sublist in results for item in sublist]

    def _execute_single_query(self, query: str) -> list:
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            bindings = results['results']['bindings']
            return [(res['s']['value'], res['o']['value']) for res in bindings]
        except Exception:
            return []

    def _sanitize(self, keyword: str) -> str:
        cleaned = re.sub(r"\s*\(\d{4}\)$", "", keyword)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _lookup_dbpedia_uri(self, name: str) -> tuple[str, str]:
        """
        Finds the first valid DBpedia URI for a name by generating candidates
        and verifying them with a SPARQL ASK query. This is a self-identifying
        worker function for parallel processing.
        """
        # generate all potential URI candidates using the smart build_uri function
        uri_candidates = build_uri(name, self.config)

        # get the verification query pattern from the configuration
        smart_query_pattern = self.config.smart_search_service.smart_query

        # ASK each one until we find a valid one
        for uri in uri_candidates:
            query = f"ASK WHERE {{ <{uri}> {smart_query_pattern} }}"
            try:
                self.sparql.setQuery(query)
                result = self.sparql.query().convert()
                if result.get('boolean', False):
                    # return the original name and the found URI
                    return name, uri
            except Exception:
                continue

        # if no valid URI was found after checking all candidates
        return name, ""


    # dbpedia populator
    def _populate_mapping_default(self, missing_names, mapping):
        """
        High-performance mapping population for DBpedia.
        """
        names_to_lookup = [title for (_, title) in missing_names]

        # parallel
        with multiprocessing.Pool(processes=self.config.sparql.processes) as pool:
            imap_it = pool.imap_unordered(self._lookup_dbpedia_uri, names_to_lookup)

            # build the dictionary from the (name, uri) tuples returned by the workers
            title_to_uri = {
                name: uri for name, uri in tqdm(imap_it, total=len(names_to_lookup), desc="Finding valid DBpedia URIs")
            }

        newly_mapped_count = 0
        # reverse map from title to the original itemId
        title_to_mid = {title: str(mid) for mid, title in missing_names}

        for title, uri in title_to_uri.items():
            if uri:
                mid = title_to_mid.get(title)
                if mid and mid not in mapping:
                    mapping[mid] = uri
                    newly_mapped_count += 1

        return newly_mapped_count

    def lookup_uri(self, raw_keyword: str) -> str:
        if self.config.knowledge_base == 'dbpedia':
            _name, uri = self._lookup_dbpedia_uri(raw_keyword)
            return uri
        return ""

    def batch_lookup_labels(self, uris: List[str]) -> dict:
        if not uris or self.config.knowledge_base != 'wikidata':
            return {}
        label_map = {}
        uri_chunks = chunker(uris, 150)
        for chunk in tqdm(uri_chunks, desc="Batch-fetching labels"):
            values_clause = " ".join(f"wd:{uri.split('/')[-1]}" for uri in chunk if 'Q' in uri)
            if not values_clause: continue
            query = f"""
                SELECT ?item ?itemLabel WHERE {{
                  VALUES ?item {{ {values_clause} }}
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                }}
            """
            try:
                headers = {'Accept': 'application/sparql-results+json', 'User-Agent': self.sparql.agent}
                response = self.session.get(self.config.sparql.endpoint, params={'query': query, 'format': 'json'},
                                            headers=headers, timeout=45)
                response.raise_for_status()
                data = response.json()
                for res in data["results"]["bindings"]:
                    uri = res['item']['value']
                    label = res['itemLabel']['value']
                    label_map[uri] = label
            except Exception as e:
                print(f"\nWarning: Batch label lookup query failed for a chunk. {e}")
        return label_map

    def _disambiguate_qids_by_year(self, qids_to_check: list, year: int) -> Optional[str]:
        if not qids_to_check or not year:
            return None

        svc_config = self.config.lookup_services.get(self.config.knowledge_base, {})
        disamb_prop = getattr(svc_config, 'disambiguation_property', 'P577')
        values_clause = " ".join(f"wd:{qid}" for qid in qids_to_check)
        query = f"""
            SELECT ?item WHERE {{
              VALUES ?item {{ {values_clause} }}
              ?item wdt:{disamb_prop} ?date_value .
              FILTER(YEAR(?date_value) = {year})
            }}
            LIMIT 1
        """
        try:
            headers = {
                'Accept': 'application/sparql-results+json',
                'Content-Type': 'application/sparql-query',
                'User-Agent': self.sparql.agent
            }
            response = self.session.post(self.config.sparql.endpoint, data=query.encode('utf-8'), headers=headers,
                                         timeout=20)
            response.raise_for_status()
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0]['item']['value'].split('/')[-1]
        except Exception:
            return None
        return None

    def _populate_mapping_wikidata(self, missing_names, mapping):
        names_to_lookup = [title for (_, title) in missing_names]
        print(f"Stage 1: Searching for verified candidates for {len(names_to_lookup)} entities...")
        with multiprocessing.Pool(processes=self.config.sparql.processes, initializer=initialize_worker,
                                  initargs=(self.config,)) as pool:
            imap_it = pool.imap_unordered(_search_wikidata_candidates_sparql, names_to_lookup)
            title_to_candidates = {title: qids for title, qids in tqdm(imap_it, total=len(names_to_lookup))}

        print("Stage 2: Disambiguating and creating final mappings...")
        svc_config = self.config.lookup_services.get(self.config.knowledge_base, {})
        year_regex = getattr(svc_config, 'year_regex', None)
        title_to_mid = {title: str(mid) for mid, title in missing_names}
        base_uri = self.config.base_uri
        newly_mapped_count = 0

        for raw_title, candidates in tqdm(title_to_candidates.items(), desc="Disambiguating entities"):
            final_qid = None
            if not candidates: continue
            year = None
            if year_regex:
                match = re.search(year_regex, raw_title)
                if match:
                    try:
                        year = int(match.group(1))
                    except (ValueError, TypeError):
                        pass
            if year and len(candidates) > 1:
                final_qid = self._disambiguate_qids_by_year(candidates, year)
            if not final_qid:
                final_qid = candidates[0]
            if final_qid:
                mid = title_to_mid.get(raw_title)
                if mid is not None and mid not in mapping:
                    mapping[mid] = f"{base_uri}{final_qid}"
                    newly_mapped_count += 1
        return newly_mapped_count

    def populate_mapping(self, missing_names, mapping, mapping_file):
        if not missing_names: return mapping
        print(f"Populating mapping for {self.config.knowledge_base}...")
        if self.config.knowledge_base == 'wikidata':
            newly_mapped_count = self._populate_mapping_wikidata(missing_names, mapping)
        else:
            newly_mapped_count = self._populate_mapping_default(missing_names, mapping)
        print(f"Successfully linked {newly_mapped_count} new entities.")
        if not mapping:
            print("Warning: No entities were successfully mapped.")
            return mapping

        mapping_path = str(mapping_file)
        dirpath = os.path.dirname(mapping_path) or "."
        os.makedirs(dirpath, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".mapping_tmp_", suffix=".tsv", dir=dirpath)

        id_col, title_col, uri_col = self.config.mapping.columns
        all_titles_df = pd.DataFrame(missing_names, columns=[id_col, title_col]).drop_duplicates(subset=[id_col])
        try:
            existing_titles = pd.read_csv(
                mapping_path,
                sep='\t',
                header=None,
                names=self.config.mapping.columns,
                dtype={id_col: str}
            )[[id_col, title_col]]
            all_titles_df = pd.concat([all_titles_df, existing_titles]).drop_duplicates(subset=[id_col])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

        map_df = pd.DataFrame(list(mapping.items()), columns=[id_col, uri_col])
        map_df[id_col] = map_df[id_col].astype(str)
        all_titles_df[id_col] = all_titles_df[id_col].astype(str)

        final_df = pd.merge(map_df, all_titles_df, on=id_col, how='left')
        final_df[title_col] = final_df[title_col].fillna('')
        final_df = final_df[self.config.mapping.columns]

        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            final_df.to_csv(tmp_f, sep='\t', header=False, index=False, lineterminator='\n')
        os.replace(tmp_path, mapping_path)
        return mapping

    def find_first_valid_uri(self, name: str) -> tuple[str, str]:
        """
        Generates candidate URIs for a name, returns the first valid one,
        and returns the original name to allow for unordered processing.
        """
        uri_candidates = build_uri(name, self.config)
        for uri in uri_candidates:
            query = f"ASK WHERE {{ <{uri}> a dbo:Film . }}"
            try:
                self.sparql.setQuery(query)
                result = self.sparql.query().convert()
                if result.get('boolean', False):
                    # return the original name along with the found URI
                    return name, uri
            except Exception:
                continue

        # if nothing is found, still return the name with an empty URI
        return name, ""





