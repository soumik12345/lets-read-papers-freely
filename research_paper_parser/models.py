import requests
from enum import Enum
from io import BytesIO
from typing import Optional, Union

import instructor
import pymupdf
import pymupdf4llm
import weave
from instructor import Instructor
from openai import OpenAI

from .schema import PaperInfo


class ResearchPaperReadingMode(Enum):
    UNSTRUCTURED_RESPONSE = 0
    DIRECT_STRUCTURED_RESPONSE = 1
    NL2STRUCTURED_RESPONSE = 2


class ResearchPaperReaderModel(weave.Model):
    openai_model: str
    openai_model_for_extraction: Optional[str] = "gpt-4o"
    max_retries: int = 5
    seed: int = 42
    system_prompt: Optional[str] = None
    _lm_client: OpenAI = None
    _structured_lm_client: Instructor = None

    def __init__(
        self,
        openai_model: str,
        openai_model_for_extraction: Optional[str] = "gpt-4o",
        max_retries: int = 5,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(
            openai_model=openai_model,
            openai_model_for_extraction=openai_model_for_extraction,
            max_retries=max_retries,
            seed=seed,
            system_prompt=system_prompt,
        )
        if self.system_prompt is None:
            self.system_prompt = """
You are a helpful assistant to a machine learning researcher who is reading a paper from arXiv.
You are to extract the following information from the paper:

- a list of main findings in from the paper and their corresponding detailed explanations
- the list of names of the different novel methods proposed in the paper and their corresponding detailed explanations
- the list of names of the different existing methods used in the paper, their corresponding detailed explanations, and
    their citations
- the list of machine learning techniques used in the paper, such as architectures, optimizers, schedulers, etc., their
    corresponding detailed explanations, and their citations
- the list of evaluation metrics used in the paper, the benchmark datasets used, the values of the metrics, and their
    corresponding detailed observation in the paper
- the link to the GitHub repository of the paper if there is any
- the hardware or accelerators used to perform the experiments in the paper if any
- a list of possible further research directions that the paper suggests

Here are some rules to follow:
1. When looking for the main findings in the paper, you should look for the abstract.
2. When looking for the explanations for the main findings, you should look for the introduction and methods section of
    the paper.
3. When looking for the list of existing methods used in the paper, first look at the citations, and then try explaining
    how they were used in the paper.
4. When looking for the list of machine learning methods used in the paper, first look at the citations, and then try
    explaining how they were used in the paper.
5. When looking for the evaluation metrics used in the paper, first look at the results section of the paper, and then
    try explaining the observations made from the results. Pay special attention to the tables to find the metrics,
    their values, the corresponding benchmark and the observation association with the result.
6. If there are no github repositories associated with the paper, simply return "None".
7. When looking for hardware and accelerators, pay special attentions to the quantity of each type of hardware and
    accelerator. If there are no hardware or accelerators used in the paper, simply return "None".
8. When looking for further research directions, look for the conclusion section of the paper.
"""
        self._lm_client = OpenAI()
        self._structured_lm_client = instructor.from_openai(self._lm_client)

    @weave.op()
    def get_markdown_from_arxiv(self, url):
        response = requests.get(url)
        with pymupdf.open(stream=BytesIO(response.content), filetype="pdf") as doc:
            return pymupdf4llm.to_markdown(doc)

    @weave.op()
    def get_direct_structured_response(self, md_text: str) -> PaperInfo:
        return self._structured_lm_client.chat.completions.create(
            model=self.openai_model,
            response_model=PaperInfo,
            max_retries=self.max_retries,
            seed=self.seed,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": md_text},
            ],
        )

    @weave.op()
    def get_two_staged_structured_response(self, md_text: str) -> PaperInfo:
        natural_language_response = (
            self._lm_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": md_text},
                ],
                seed=self.seed,
            )
            .choices[0]
            .message.content
        )
        return self._structured_lm_client.chat.completions.create(
            model=self.openai_model_for_extraction,
            response_model=PaperInfo,
            max_retries=self.max_retries,
            seed=self.seed,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful assistant to a machine learning researcher.
You are responsible for extracting the following information from the summary of a research paper:
- - a list of main findings in from the paper and their corresponding detailed explanations
- the list of names of the different novel methods proposed in the paper and their corresponding detailed explanations
- the list of names of the different existing methods used in the paper, their corresponding detailed explanations, and
    their citations
- the list of machine learning techniques used in the paper, such as architectures, optimizers, schedulers, etc., their
    corresponding detailed explanations, and their citations
- the list of evaluation metrics used in the paper, the benchmark datasets used, the values of the metrics, and their
    corresponding detailed observation in the paper
- the link to the GitHub repository of the paper if there is any
- the hardware or accelerators used to perform the experiments in the paper if any
- a list of possible further research directions that the paper suggests
                """,
                },
                {"role": "user", "content": natural_language_response},
            ],
        )

    @weave.op()
    def get_unstructured_response(self, md_text: str) -> str:
        return (
            self._lm_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": md_text},
                ],
                seed=self.seed,
            )
            .choices[0]
            .message.content
        )

    @weave.op()
    def predict(
        self, url_pdf: str, reading_mode: ResearchPaperReadingMode
    ) -> Optional[Union[PaperInfo, str]]:
        md_text = self.get_markdown_from_arxiv(url_pdf)
        if reading_mode == ResearchPaperReadingMode.DIRECT_STRUCTURED_RESPONSE:
            return self.get_direct_structured_response(md_text=md_text)
        elif reading_mode == ResearchPaperReadingMode.NL2STRUCTURED_RESPONSE:
            return self.get_two_staged_structured_response(md_text=md_text)
        return self.get_unstructured_response(md_text=md_text)
