from pydantic import BaseModel


class FeatureInfo(BaseModel):
    key: str
    value: str
    type: str
