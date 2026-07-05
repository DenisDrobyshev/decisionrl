"""Core abstractions: spaces, environment API, agent API and shared types."""

from .agent import BaseAgent
from .env import Env, Wrapper
from .spaces import Box, Discrete, Space, flatdim, is_discrete
from .types import Transition

__all__ = [
    "BaseAgent",
    "Env",
    "Wrapper",
    "Space",
    "Box",
    "Discrete",
    "is_discrete",
    "flatdim",
    "Transition",
]
