import asyncio
import fire
import weave
from research_paper_parser import ResearchPaperReaderModel, ResearchPaperReadingMode
from research_paper_parser.metrics import (
    extraction_numeracy,
    method_prediction_accuracy_score,
)


def evaluate_direct_structured_outputs(
    project_name: str = "research_paper_parser",
    max_evaluation_samples: int = 30,
    openai_model="gpt-4o",
    openai_model_for_extraction: str = "gpt-4o-mini",
    max_retries: int = 5,
    seed: int = 42,
    evaluation_name: str = "direct_structured_outputs",
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
    reading_mode = (
        ResearchPaperReadingMode.DIRECT_STRUCTURED_RESPONSE
        if evaluate_direct_structured_response
        else ResearchPaperReadingMode.NL2STRUCTURED_RESPONSE
    )
    model = ResearchPaperReaderModel(
        openai_model=openai_model,
        reading_mode=reading_mode,
        openai_model_for_extraction=openai_model_for_extraction,
        max_retries=max_retries,
        seed=seed,
    )
    evaluation = weave.Evaluation(
        name=evaluation_name,
        dataset=dataset,
        scorers=[extraction_numeracy, method_prediction_accuracy_score],
    )
    asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    fire.Fire(evaluate_direct_structured_outputs)
