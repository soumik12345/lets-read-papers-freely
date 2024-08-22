import asyncio
import fire
import functools
import weave
from research_paper_parser import ResearchPaperReaderModel, ResearchPaperReadingMode
from research_paper_parser.metrics import (
    extraction_numeracy,
    method_prediction_accuracy_score,
)


def evaluate_direct_structured_outputs(
    project_name: str = "research_paper_parser",
    max_evaluation_samples: int = 50,
    openai_model="gpt-4o",
    max_retries: int = 5,
    seed: int = 42,
    evaluate_direct_structured_response: bool = True,
):
    weave.init(project_name=project_name)
    dataset = (
        weave.ref(
            "weave:///geekyrakshit/research_paper_parser/object/cv-papers:T4m4tp4XZ0N3d9bY8qDu0TpkX1Ruw34WhZFVwYHUHL8"
        )
        .get()
        .rows[:max_evaluation_samples]
    )
    model = ResearchPaperReaderModel(
        openai_model=openai_model, max_retries=max_retries, seed=seed
    )
    evaluation = weave.Evaluation(
        name="direct_structured_outputs",
        dataset=dataset,
        scorers=[extraction_numeracy, method_prediction_accuracy_score],
    )
    reading_mode = (
        ResearchPaperReadingMode.DIRECT_STRUCTURED_RESPONSE
        if evaluate_direct_structured_response
        else ResearchPaperReadingMode.NL2STRUCTURED_RESPONSE
    )
    asyncio.run(
        evaluation.evaluate(functools.partial(model.predict, reading_mode=reading_mode))
    )


if __name__ == "__main__":
    fire.Fire(evaluate_direct_structured_outputs)
