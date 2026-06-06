import argparse
import logging
import os

from folm_core import (
    ConfigManager,
    AMIEProcessor,
    LLMEmb,
    MetaEmb,
    UserEmbeddingAggregator
)
from crawler import Crawler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(args):
    """
    Main function to orchestrate the recommendation system pipeline.
    """
    # Construct the full path from the provided config filename
    config_path = os.path.join("configs", args.config)
    logging.info(f"Loading configuration from: {config_path}")
    config_manager = ConfigManager(config_path)

    # call initialize() to fully populate the ConfigManager object
    # this method sets up all the necessary paths and sub-configurations
    # like LLMEmb_config, amie_config, etc.
    config_manager.initialize()

    print(" ################################## ")
    print("Dataset:", config_manager.config_data["dataset"])
    print("Train Path:", config_manager.dataset_config.train_path)
    print(" ################################## ")

    if args.run_crawler:
        logging.info("--- Starting Crawler Stage ---")
        crawler_config_filename = config_manager.LLMEmb_config.crawler_config
        crawler_config_path = os.path.join("configs", crawler_config_filename)
        logging.info(f"Using crawler configuration: {crawler_config_path}")
        Crawler(config_manager, config_path=crawler_config_path)
        logging.info("--- Crawler Stage Finished ---")

    if args.run_amie:
        logging.info("--- Starting AMIE Rule Mining Stage ---")
        if os.path.exists(config_manager.amie_rules_path):
            logging.warning("AMIE mined rules already exist. Skipping.")
        else:
            amie_processor = AMIEProcessor(
                config=config_manager.amie_config,
                train_path=config_manager.train_dir,
                data_path=config_manager.dataset_dir
            )
            amie_processor.run(
                input_tsv=config_manager.dataset_config.train_path,
                output_file=config_manager.amie_rules_path
            )
        logging.info("--- AMIE Stage Finished ---")

    if args.run_llm_calls:
        logging.info("--- Starting LLM-Based Embedding API Calls ---")
        llm_embedder = LLMEmb(
            datadir=config_manager.config_data["dataset"],
            config=config_manager.LLMEmb_config
        )
        llm_embedder.run_llm_api_pipeline()
        logging.info("--- LLM-Based Embedding API Calls Finished ---")

    if args.run_llm_pipeline:
        logging.info("--- Starting LLM-Based Embedding Generarion ---")
        llm_embedder = LLMEmb(
            datadir=config_manager.config_data["dataset"],
            config=config_manager.LLMEmb_config
        )
        llm_embedder.generate_embeddings_from_api_responses()
        logging.info("--- LLM-Based Embedding Generation Finished ---")

    if args.run_meta_pipeline:
        logging.info("--- Starting Metadata-Based (MetaEmb) Pipeline ---")
        meta_embedder = MetaEmb(
            datadir=config_manager.config_data["dataset"],
            config=config_manager.LLMEmb_config
        )
        meta_embedder.generate_all_embeddings()
        user_aggregator = UserEmbeddingAggregator(
            datadir=config_manager.config_data["dataset"],
            predicates=['likes', 'dislikes']
        )
        user_aggregator.aggregate_for_all_item_embeddings()
        logging.info("--- Metadata-Based (MetaEmb) Pipeline Finished ---")

    logging.info("Pipeline execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FOLM recommendation pipeline.")
    parser.add_argument(
        '--config', type=str, default='example_base_settings.yml',
        help='Filename of the main YAML configuration file located in the ./configs directory.'
    )
    parser.add_argument('--run-crawler', action='store_true', help='Execute the data crawling stage.')
    parser.add_argument('--run-amie', action='store_true', help='Execute the AMIE rule mining stage.')
    parser.add_argument('--run_llm_calls', action='store_true', help='Run the LLM-based api calls.')
    parser.add_argument('--run-llm-pipeline', action='store_true', help='Run the full LLM-based embedding generation.')
    parser.add_argument('--run-meta-pipeline', action='store_true', help='Run the metadata-based embedding pipeline.')
    parser.add_argument('--run-all', action='store_true', help='A shortcut to run all pipeline stages.')

    args = parser.parse_args()

    if args.run_all:
        args.run_crawler = True
        args.run_amie = True
        args.run_llm_calls = True
        args.run_llm_pipeline = True
        args.run_meta_pipeline = True

    main(args)


# --- Command-Line Examples ---
#
# The script uses argparse to allow for flexible execution of the pipeline.
# Below are some common usage examples that can be run from your terminal.

# 1. Run the entire pipeline from start to finish using the ***DEFAULT*** config:
#    This is the most common command for a fresh run. It will execute the crawler,
#    AMIE, the LLM pipeline, and the MetaEmb pipeline in sequence.
#
#    python run.py --run-all

# 2. Run the entire pipeline with a ***CUSTOM** configuration file:
#    Useful if you have multiple datasets or experiment configurations.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : SET THE CRAWLER NAME CONFIG (eg: 'example_crawler_config_dbpedia.yml') in the
#    MAIN CONFIG (eg: 'example_base_settings.yml') --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#    HERE YOU MUST SET CORRECTLY THE NAME OF THE CRAWLER CONFIG AND PASS to "--config your_custom_config.yml"
#    THE BASE SETTING CONFIG LIKE: "--config example_base_settings.yml"
#
#    python run.py --config your_custom_config.yml --run-all

# 3. Run only the initial data setup stages (Crawler and AMIE):
#    Use this if you only want to prepare the dataset and mine the rules without
#    generating embeddings yet.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : HERE YOU MUST PASS THE "*_base_setting.yml" and SET CORRECTLY THE CRAWLER IN
#    --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#
#    python run.py --run-crawler --run-amie --config your_custom_config.yml



# 4. Run only the LLM-based embedding pipeline:
#    This is useful if the data is already crawled and rules are mined, and you have the api calls
#    (or some of them if you don't want to continue) and you want to run/re-run only embedding generation.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : SET THE CRAWLER NAME CONFIG (eg: 'example_crawler_config_dbpedia.yml') in the
#    MAIN CONFIG (eg: 'example_base_settings.yml') --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#
#    python run.py --run-llm-pipeline --config your_custom_config.yml

# 5. Run only the Metadata-based (MetaEmb) pipeline:
#    Use this to generate item embeddings from KG metadata and aggregate user
#    embeddings, assuming the KG and entity mappings already exist.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : SET THE CRAWLER NAME CONFIG (eg: 'example_crawler_config_dbpedia.yml') in the
#    MAIN CONFIG (eg: 'example_base_settings.yml') --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#
#    python run.py --run-meta-pipeline --config your_custom_config.yml

# 6. Generate all embeddings without re-running Crawler or AMIE:
#    A common scenario for experimentation with embedding models. This will
#    run both the LLM and the MetaEmb pipelines.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : SET THE CRAWLER NAME CONFIG (eg: 'example_crawler_config_dbpedia.yml') in the
#    MAIN CONFIG (eg: 'example_base_settings.yml') --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#
#    python run.py --run-llm-pipeline --run-meta-pipeline --config your_custom_config.yml
#
# 7. Run only the LLM-based API calls:
#    This is useful if the data is already crawled and rules are mined, and you
#    want to re-run only the API calls.
#
#    [!!!!] IMPORTANT NOTE [!!!!] : SET THE CRAWLER NAME CONFIG (eg: 'example_crawler_config_dbpedia.yml') in the
#    MAIN CONFIG (eg: 'example_base_settings.yml') --> (LLMEmb parameter) > crawler_config: "#crawler_config_name.yml"
#
#    python run.py --run_llm_calls --config your_custom_config.yml
