from typing import Dict, List, Optional

import weave
from .schema import PaperInfo


@weave.op()
def method_prediction_accuracy_score(
    method: List[Dict], model_output: Optional[PaperInfo]
) -> Dict[str, float]:
    if model_output is None:
        return {"method_prediction_accuracy": 0.0}
    predicted_methods = (
        model_output.novel_methods
        + model_output.existing_methods
        + model_output.machine_learning_techniques
    )
    num_correct_methods = 0
    for gt_method in method:
        for predicted_method in predicted_methods:
            predicted_method = (
                f"{predicted_method.method_name}\n{predicted_method.explanation}"
            )
            if (
                gt_method["name"].lower() in predicted_method.lower()
                or gt_method["full_name"].lower() in predicted_method.lower()
            ):
                num_correct_methods += 1
    return {
        "method_prediction_accuracy_score": num_correct_methods / len(predicted_methods)
    }


@weave.op()
def extraction_numeracy(model_output: Optional[PaperInfo]) -> Dict[str, int]:
    if model_output is None:
        return {
            "num_main_main_findings": 0,
            "num_methods": 0,
            "num_further_research": 0,
            "num_metrics": 0,
        }
    predicted_methods = (
        model_output.novel_methods
        + model_output.existing_methods
        + model_output.machine_learning_techniques
    )
    return {
        "num_main_main_findings": len(model_output.main_findings),
        "num_methods": len(predicted_methods),
        "num_further_research": len(model_output.further_research),
        "num_metrics": len(model_output.metrics),
    }
