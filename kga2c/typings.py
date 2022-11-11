from typing import Callable, Dict, Iterator, List, Tuple, NamedTuple, Union
import jericho

GameFeedback = Union[Tuple[str, dict], Tuple[str, float, bool, dict]]
PromptFormatter = Callable[[str, GameFeedback], str]
Policy = Callable[[jericho.FrotzEnv, GameFeedback], str]
GameName = Union[str, str]


AnalysisLabelScore = Dict


PromptClassifier = Callable[[str], List[AnalysisLabelScore]]


TrajectoryAnalysis = Dict

class Trajectory(NamedTuple):
    previous_observation: str
    action: str
    reward: float
    observation: str
    analysis: TrajectoryAnalysis
    score: float


class Experiment(NamedTuple):
    name: str
    game_name: GameName
    prompt_formatter: PromptFormatter
    policy: Policy
    episodes: int
    max_episode_length: int
    classifier: PromptClassifier


class ExperimentResults(NamedTuple):
    avg_entropy: float
    avg_prediction_scores: Dict[str, float]
    trajectories: List[Trajectory]
