from dataclasses import dataclass, field


@dataclass
class Config:
    experiment: str
    column_encodings: str
    dataset: str
    model: str
    use_folktexts: bool
    task_prompt: str
    additional_params: dict = field(default_factory=dict)
    random_seed: int = 42
