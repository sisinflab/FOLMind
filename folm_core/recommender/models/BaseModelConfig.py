import re
from pydantic import BaseModel, Extra, Field, model_validator
from typing import Dict, Type, Literal, Optional, List, Any


class EmbeddingInitConfig(BaseModel):
    """
    Configuration for initializing model factors with pre-trained embeddings.
    This class acts as a control panel for the feature.
    """
    enabled: bool = Field(
        default=False,
        description="Enable initialization from pre-trained embeddings."
    )
    entities: List[Literal['user', 'item']] = Field(
        default_factory=lambda: ['user', 'item'],
        description="A list of entities ('user', 'item') for which to load embeddings."
    )
    type_emb: Literal['mean', 'last', 'step', 'bert', 'ST'] = Field(
        default='mean',
        description="The aggregation type of the embedding, used to build the filename."
    )
    info_emb: Literal['MetaEmb', 'LLMEmb'] = Field(
        default='LLMEmb',
        description="The source of the embeddings, used to build the directory path."
    )
    emb_source_model: str = Field(
        default='',
        description="The embeddings model source, used to build the filename. [ "
                    "user/item_*_embeddings_MODELNAME.safetensors]"
    )


class FolmConfig(BaseModel):
    refined: bool = False
    type_emb: Literal['mean', 'last', 'step', 'bert', 'ST'] = Field(default='last')
    top_k_similar: int = Field(default=-1, ge=-1,
                               description='Top K similar users to consider from embeddings. -1 means all.')
    mode: Optional[Literal['inject', 'hybrid_blend']] = Field(
        default=None, description='Refinement mode: "inject" for kNN, "hybrid_blend" for MF.')
    info_emb: Literal['MetaEmb', 'LLMEmb'] = Field(
        default='LLMEmb', description='Loads embedding from the chosen directory.')
    entity_emb: Literal['user', 'item'] = Field(
        default='user', description='Loads embeddings for the chosen entity.')
    emb_source_model: str = Field(
        default='',
        description="The embeddings model source, used to build the filename. [ "
                    "user/item_*_embeddings_MODELNAME.safetensors]"
    )
    alpha: Optional[float] = Field(
        default=0.5, ge=0, le=1,
        description='Blending coefficient. For "inject", it weights the embedding signal. For "hybrid_blend", it weights the original MF model signal.'
    )
    #  search_space for l'HPO
    search_space: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Spazio di ricerca per gli iperparametri di Folm."
    )

    @model_validator(mode='before')
    def _check_mode_fields(cls, values: dict) -> dict:
        if not values.get('refined'):
            return values
        m = values.get('mode')
        if m not in ['inject', 'hybrid_blend']:
            raise ValueError(f'Invalid mode: {m}. Must be "inject" or "hybrid_blend" when refined=true.')
        # Non richiediamo più alpha obbligatorio qui, perché potrebbe essere definito nello search_space
        return values


class HPOConfig(BaseModel):
    """Config Hyperparameter Optimization."""
    enabled: bool = Field(default=False, description="HPO enabled.")
    n_trials: int = Field(default=20, ge=1, description="N trial.")
    metric_to_optimize: str = Field(default="ndcg@10", description="Maximize metric on val set.")
    search_space: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameter Optimization space.")


class MetaConfig(BaseModel):
    save_recs: bool = False


class BaseModelConfig(BaseModel):
    meta: MetaConfig = Field(default_factory=MetaConfig)
    folm: FolmConfig = Field(default_factory=FolmConfig)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    embedding_init: EmbeddingInitConfig = Field(default_factory=EmbeddingInitConfig)

    class Config:
        extra = "allow"
