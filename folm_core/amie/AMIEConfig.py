"""
Module description:
    AMIE config classes
"""

__version__ = '0.0'
__email__ = ''

from typing import List, Optional
from pydantic import BaseModel, model_validator, field_validator
from enum import Enum


class AMIEBias(str, Enum):
    DEFAULT = "default"
    ONE_VAR = "oneVar"
    LAZY = "lazy"
    LAZIT = "lazit"
    CUSTOM = "custom"


class VariableOrder(str, Enum):
    APP = "app"
    FUN = "fun"
    IFUN = "ifun"


class AMIEConfig(BaseModel):
    # Core mining parameters
    mins: int = 100
    minis: int = 100
    minhc: float = 0.01
    pm: str = "headcoverage"
    maxad: int = 3
    minpca: float = 0.0
    minc: float = 0.0

    # Rule constraints
    bexr: Optional[List[str]] = None
    hexr: Optional[List[str]] = None
    iexr: Optional[List[str]] = None
    htr: Optional[List[str]] = None
    btr: Optional[List[str]] = None
    itr: Optional[List[str]] = None

    # Constants handling
    const: bool = False
    fconst: bool = False

    # Execution parameters
    nc: Optional[int] = None
    bias: AMIEBias = AMIEBias.DEFAULT
    caos: bool = False
    verbose: bool = False
    rl: Optional[int] = None

    # Optimization flags
    optimcb: bool = False
    optimfh: bool = False
    optimai: bool = False

    # Output control
    oute: bool = False
    datalog: bool = False
    oout: bool = False
    full: bool = False
    ost: bool = False

    # Advanced parameters
    auta: bool = False
    deml: bool = False
    dqrw: bool = False
    dpr: bool = False
    noHeuristics: bool = False
    noKbRewrite: bool = False
    noKbExistsDetection: bool = False
    noSkyline: bool = False
    vo: VariableOrder = VariableOrder.FUN
    mlg: bool = False
    d: str = "\t"

    # File parameters
    ef: Optional[str] = None

    # Validation and constraints
    @model_validator(mode='after')
    def check_mutually_exclusive(self) -> 'AMIEConfig':
        if self.htr and self.hexr:
            raise ValueError("htr and hexr are mutually exclusive")
        if self.btr and self.bexr:
            raise ValueError("btr and bexr are mutually exclusive")
        if self.itr and self.iexr:
            raise ValueError("itr and iexr are mutually exclusive")
        return self

    @field_validator('pm')
    def validate_pm(cls, v: str) -> str:  # <-- cls here
        if v not in {"support", "headcoverage"}:
            raise ValueError("Invalid pruning metric")
        return v

    @field_validator('maxad')
    def validate_maxad(cls, v):
        if v < 1:
            raise ValueError("maxad must be at least 1")
        return v

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

