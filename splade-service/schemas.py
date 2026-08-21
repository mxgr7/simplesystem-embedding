from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    inputs: str | list[str]


class AddBackendRequest(BaseModel):
    url: str
    weight: float = Field(default=1.0, gt=0)
    max_concurrency: int = Field(default=1, gt=0)
    max_client_batch: int = Field(default=8, gt=0)
    timeout_s: float = Field(default=120, gt=0)
    api_key: str = ""


class PatchBackendRequest(BaseModel):
    weight: float = Field(ge=0)
