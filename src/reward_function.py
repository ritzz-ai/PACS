from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def compute_score(
    data_source, solution_str, ground_truth, extra_info=None, timeout_score=0
):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return {
        "score": ret_score,
        "data_source": data_source,
        "ground_truth": ground_truth,
    }
