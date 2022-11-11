from typing import List
import diskcache

from transformers.pipelines import pipeline
from kga2c.typings import AnalysisLabelScore, GameFeedback

cache_base_dir = ".cache"

zero_shot_classify_pipeline = pipeline(task="zero-shot-classification", device="cuda:0")
analyse_sentiment_pipeline = pipeline(task="text-classification", device="cuda:0")

sentiment_analyser_cache = diskcache.Cache(f"{cache_base_dir}/sentiment_analysis")


def sentiment_analyser(prompt) -> List[AnalysisLabelScore]:
    if prompt in sentiment_analyser_cache:
        return sentiment_analyser_cache[prompt]  # type: ignore
    else:
        result = analyse_sentiment_pipeline(  # type: ignore
            prompt,
            return_all_scores=True,
        )[0]
        sentiment_analyser_cache[prompt] = result
        return result  # type: ignore


success_failure_classifier_cache = diskcache.Cache(
    f"{cache_base_dir}/success_failure_classification"
)


def success_failure_classifier(prompt) -> List[AnalysisLabelScore]:
    if prompt in success_failure_classifier_cache:
        return success_failure_classifier_cache[prompt]  # type: ignore
    else:
        predictions = zero_shot_classify_pipeline(  # type: ignore
            prompt,
            ["succeeded", "failed"],
            return_all_scores=True,
        )
        result = [
            {"label": label, "score": score}
            for label, score in zip(predictions["labels"], predictions["scores"])  # type: ignore
        ]
        success_failure_classifier_cache[prompt] = result
        return result  # type: ignore
    
def action_feedback_prompt_format(action: str, obs: str) -> str:
    return f"Action: {action.strip()}. Effect: {obs.strip()}"

