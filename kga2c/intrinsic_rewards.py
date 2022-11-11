#%%
from typing import NamedTuple
from kga2c.text_classification import action_feedback_prompt_format, sentiment_analyser


class SentimentAnalysisResult(NamedTuple):
    negative: float
    positive: float


def get_intrinsic_reward(action: str, obs: str, game_max_reward: float):
    prompt = action_feedback_prompt_format(action, obs)
    analysis = sentiment_analyser(prompt)
    positive = [i["score"] for i in analysis if i["label"] == "POSITIVE"][0]
    negative = [i["score"] for i in analysis if i["label"] == "NEGATIVE"][0]
    return SentimentAnalysisResult(positive=positive, negative=negative)


# %%
