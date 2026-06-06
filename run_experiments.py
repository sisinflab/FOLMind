import argparse
import logging
import os.path

from folm_core import ConfigManager, Recommender


def main():
    """
    Only run experiments
    """
    # argparse setup
    parser = argparse.ArgumentParser(description="Run recommender model experiments from a configuration file.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file for the experiments."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"--- Starting Experiment Run using config: {args.config} ---")
    config_path = os.path.join(os.getcwd(), 'configs', args.config)
    # load the configuration using ConfigManager
    try:
        config_manager = ConfigManager(config_path)
    except Exception as e:
        logger.error(f"Failed to load or parse the configuration file at {config_path}. Error: {e}")
        return

    print("\n" + " " + "#" * 34 + " ")
    print(f"# Dataset: {config_manager.config_data['dataset']}")
    print(f"# Strategy: {config_manager.dataset_config.strategy}")
    print(" " + "#" * 34 + " \n")

    # initialize the configuration manager.
    logger.info("Initializing paths and sub-configurations...")
    config_manager.initialize()
    logger.info("Paths initialized successfully.")
    print(f"Train Path: {config_manager.dataset_config.train_path}")
    print(f"Recs/Results Dir: {config_manager.recs_dir}\n")

    # initialize and run the Recommender
    logger.info("Initializing the Recommender engine...")
    recommender = Recommender(
        datadir=config_manager.config_data["dataset"],
        recs_config=config_manager.rec_config,
        recs_dir=config_manager.recs_dir
    )

    logger.info("Starting the experiment execution loop...")
    recommender.run()

    logger.info("--- Experiment Run Finished ---")


if __name__ == "__main__":
    main()

    # NO FOL
    # nohup python run_experiments.py -c books_base_settings_qwen4BNOFOL.yml > books_base_settings_qwen4BNOFOL.log 2>&1 &
    # nohup python run_experiments.py -c ml1m_base_settings_qwen4BNOFOL.yml > ml1m_base_settings_qwen4BNOFOL.log 2>&1 &

    # FOL
    # nohup python run_experiments.py -c books_base_settings_qwen4BFOL.yml > books_base_settings_qwen4BFOL.log 2>&1 &
    # nohup python run_experiments.py -c ml1m_base_settings_qwen4BFOL.yml > ml1m_base_settings_qwen4BFOL.log 2>&1 &

    # baselines (results saved in the 'book-genome' and 'ml-1m' dir)
    # nohup python run_experiments.py -c books_base_settings_baseline.yml > books_base_settings_baseline.log 2>&1 &
    # nohup python run_experiments.py -c ml1m_base_settings_baseline.yml > ml1m_base_settings_baseline.log 2>&1 &
