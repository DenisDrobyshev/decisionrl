"""Core abstractions: spaces, environment API, agent API and shared types."""

from .agent import BaseAgent
from .env import Env, Wrapper
from .spaces import Box, Dict, Discrete, Space, flatdim, flatten, is_discrete
from .types import Transition

__all__ = [
    "BaseAgent",
    "Env",
    "Wrapper",
    "Space",
    "Box",
    "Discrete",
    "Dict",
    "is_discrete",
    "flatdim",
    "flatten",
    "Transition",
]
