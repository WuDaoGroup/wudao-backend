from pydantic import BaseModel


class CausalInfo(BaseModel):
    key: str
    causal: str
