from pydantic import BaseModel


class ExplanationInfo(BaseModel):
    dimmension: str
    type: str
    learningrate: str
    target: str


class FeatureCorrFeaturesInfo(BaseModel):
    object: str
    k_number: str
