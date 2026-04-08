from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlAction(Action):
    optimized_query: str = Field(..., description="Optimized SQL query")


class SqlObservation(Observation):
    query: str = Field(default="", description="Current SQL query")
    task: str = Field(default="", description="Task difficulty")