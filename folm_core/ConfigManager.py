from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Any, List
import yaml
import logging
from pathlib import Path
import os

from .dataset.DatasetConfig import DatasetConfig
from .amie.AMIEConfig import AMIEConfig
from .Emb.LLMEmb import LLMEmbConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecConfig(BaseModel):
    """
    Config for experiments
    """
    models: Dict[str, Any]
    evaluation: Dict[str, List] = Field(default_factory=dict)
    top_k: int = 10
    gpu: int = 0
    seed: int = 42

    class Config:
        extra = "allow"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        try:
            return cls(**config_dict)
        except ValidationError as e:
            logger.error(f"Config validation error for RecConfig: {e}")
            raise


class BaseConfig(BaseModel):
    """Base configuration model for all components"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        try:
            return cls(**config_dict)
        except ValidationError as e:
            logger.error(f"Config validation error: {e}")
            raise


class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        self.dataset_config = self._init_dataset_config()
        # __init__ only defines the path, it doesn't create anything
        self.dataset_dir = os.path.join(os.getcwd(), 'data', self.config_data["dataset"])

        # all other properties will be set in initialize()

    def initialize(self):
        """
        Initializes all paths and sub-configurations. 
        This should be called AFTER the crawler ensures the dataset exists.
        """
        # define and create necessary directories
        self.train_dir = os.path.join(self.dataset_dir, 'splitting/train')
        self.recs_dir = os.path.join(self.dataset_dir, 'recs')

        # os.makedirs is safe to call even if the directory exists
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.recs_dir, exist_ok=True)

        # set the file path
        self.dataset_config.train_path = os.path.join(self.train_dir, 'knowledge_graph_numeric.tsv')

        # initialize all other components
        self.amie_config = self._init_amie_config()
        self.LLMEmb_config = self._init_LLMEmb()
        self.rec_config = self._init_rec()
        self.amie_output_path, self.amie_rules_path, self.amie_facts_path = self._init_amie_output()
        logger.info("ConfigManager fully initialized.")

    def _load_config(self) -> Dict[str, Any]:
        """Load entire configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _init_amie_config(self):
        """Initialize AMIE configuration"""
        return AMIEConfig.from_dict(self.config_data.get('amie', {}))

    def _init_dataset_config(self):
        """Initialize Dataset configuration"""
        return DatasetConfig.from_dict(self.config_data.get('data_config', {}))

    def _init_amie_output(self):
        """Initialize AMIE output folder"""
        dataset_name = self.config_data['dataset']
        output_path = Path.cwd().joinpath("data", dataset_name, "amie_mined")
        os.makedirs(output_path, exist_ok=True)
        rule_path = Path.joinpath(output_path, "mined_rules.csv")
        ground_path = Path.joinpath(output_path, "grounded_facts.tsv")
        return output_path, rule_path, ground_path

    def _init_LLMEmb(self):
        """Initialize Emb configuration"""
        return LLMEmbConfig.from_dict(self.config_data.get('LLMEmb', {}))

    def _init_rec(self):
        """Initialize the configuration for the recommendation phase."""
        logger.info("Initializing experiment configuration...")
        return RecConfig.from_dict(self.config_data.get('experiment', {}))
