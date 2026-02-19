from pydantic import BaseModel


class TrainConfig(BaseModel):
    seed: int = 42
    test_size: float = 0.2
