from typing import List, Optional

from pydantic import BaseModel


class Finding(BaseModel):
    finding_name: str
    explanation: str


class Method(BaseModel):
    method_name: str
    explanation: str
    citation: Optional[str]


class Evaluation(BaseModel):
    metric: str
    benchmark: str
    value: float
    observation: str


class PaperInfo(BaseModel):
    main_findings: List[Finding]  # The main findings of the paper
    novel_methods: List[Method]  # The novel methods proposed in the paper
    existing_methods: List[Method]  # The existing methods used in the paper
    machine_learning_techniques: List[
        Method
    ]  # The machine learning techniques used in the paper
    metrics: List[Evaluation]  # The evaluation metrics used in the paper
    github_repository: (
        str  # The link to the GitHub repository of the paper (if there is any)
    )
    hardware: str  # The hardware or accelerator setup used in the paper
    further_research: List[
        str
    ]  # The further research directions suggested in the paper
