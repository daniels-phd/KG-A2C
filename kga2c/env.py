import collections
import random
from typing import Any, NamedTuple, Tuple, Union

import jericho
import numpy as np
import redis
from jericho import TemplateActionGenerator
from networkx import DiGraph
from sentencepiece import SentencePieceProcessor

from kga2c.lib import make_admissible_actions_cache
from representations import StateAction


class Action(NamedTuple):
    text: str
    id: int
    objs: list[str]
    obj_ids: list[int]


class GraphInfo(NamedTuple):
    objs: list[str]
    ob_rep: np.ndarray
    act_rep: list[int]
    graph_state: DiGraph
    graph_state_rep: Tuple[list[Any], np.ndarray]
    admissible_actions: list[Action]
    admissible_actions_rep: list[list[int]]


def generate_actions(
    act_gen: TemplateActionGenerator,
    objs: list[str],
    obj_ids: list[int],
):
    actions: list[Action] = []

    for id, template in enumerate(act_gen.templates):
        holes = template.count("OBJ")
        if holes <= 0:
            actions.append(Action(text=template, id=id, objs=[], obj_ids=[]))
        elif holes == 1:
            actions.extend(
                [
                    Action(
                        text=template.replace("OBJ", obj),
                        id=id,
                        objs=[obj],
                        obj_ids=[obj_ids[obj_i]],
                    )
                    for obj_i, obj in enumerate(objs)
                ]
            )
        elif holes == 2:
            for o1 in objs:
                for o2 in objs:
                    if o1 != o2:
                        actions.append(
                            Action(
                                text=template.replace("OBJ", o1, 1).replace(
                                    "OBJ", o2, 1
                                ),
                                objs=objs,
                                obj_ids=obj_ids,
                                id=id,
                            )
                        )
    return actions


def load_vocab(env: jericho.FrotzEnv):
    vocab = {i + 2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = " "
    vocab[1] = "<s>"
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev


def clean_obs(s: str):
    garbage_chars = ["*", "-", "!", "[", "]"]
    for c in garbage_chars:
        s = s.replace(c, " ")
    return s.strip()


class KGA2CEnv:
    """

    KGA2C environment performs additional graph-based processing.

    """

    def __init__(
        self,
        rom_path: str,
        seed: int,
        spm_model: SentencePieceProcessor,
        tsv_file: str,
        step_limit=None,
        stuck_steps=10,
        gat=True,
    ):
        random.seed(seed)
        np.random.seed(seed)
        self.rom_path = rom_path
        self.seed = seed
        self.episode_steps = 0
        self.stuck_steps = 0
        self.valid_steps = 0
        self.spm_model = spm_model
        self.tsv_file = tsv_file
        self.step_limit = step_limit
        self.max_stuck_steps = stuck_steps
        self.gat = gat
        self.env = jericho.FrotzEnv(self.rom_path, self.seed)
        self.bindings: dict[str, Any] = self.env.bindings  # type: ignore
        self.act_gen: TemplateActionGenerator = self.env.act_gen  # type: ignore
        self.max_word_len: int = self.bindings["max_word_length"]  # type: ignore
        self.vocab, self.vocab_rev = load_vocab(self.env)
        self.admissible_actions_cache = make_admissible_actions_cache(
            self.bindings["name"]  # type: ignore
        )

    def create(self):
        self.env = jericho.FrotzEnv(self.rom_path, self.seed)
        self.bindings: dict[str, Any] = self.env.bindings  # type: ignore
        self.act_gen: TemplateActionGenerator = self.env.act_gen  # type: ignore

    def _get_admissible_actions(self, objs):
        world_state_hash = self.env.get_world_state_hash()
        admissible: Union[list[Action], None] = (
            list(self.admissible_actions_cache[world_state_hash])  # type: ignore
            if world_state_hash in self.admissible_actions_cache
            else None
        )
        if admissible is None:
            obj_ids = [
                self.vocab_rev[o[: self.max_word_len]]
                for o in objs
                if o[: self.max_word_len] in self.vocab_rev
            ]
            objs_clean = [self.vocab[id] for id in obj_ids]
            admissible = generate_actions(self.act_gen, objs_clean, obj_ids)
            self.admissible_actions_cache[world_state_hash] = admissible

        return admissible

    def _build_graph_rep(self, action, ob_r):
        """Returns various graph-based representations of the current state."""
        objs = [o[0] for o in self.env._identify_interactive_objects(ob_r)]
        objs.append("all")
        admissible_actions = self._get_admissible_actions(objs)
        admissible_actions_rep = (
            [self.state_rep.get_action_rep_drqa(a) for a in admissible_actions]
            if admissible_actions
            else [[0] * 20]
        )
        try:  # Gather additional information about the new state
            saved_state = self.env.get_state()
            ob_l: str = self.env.step("look")[0]
            self.env.set_state(saved_state)
            ob_i: str = self.env.step("inventory")[0]
            self.env.set_state(saved_state)
        except RuntimeError:
            print(f"RuntimeError: {clean_obs(ob_r)}")
            ob_l = ob_i = ""
        ob_rep = self.state_rep.get_obs_rep(ob_l, ob_i, ob_r, action)
        cleaned_obs = clean_obs(ob_l + " " + ob_r)

        rules, tocache = self.state_rep.step(
            cleaned_obs, ob_i, objs, action, gat=self.gat
        )

        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)
        return GraphInfo(
            objs,
            ob_rep,
            action_rep,
            graph_state,
            graph_state_rep,
            admissible_actions,
            admissible_actions_rep,
        )

    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        info["valid"] = self.env._world_changed() or done
        info["steps"] = self.episode_steps
        if info["valid"]:
            self.valid_steps += 1
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
        if (
            self.step_limit and self.valid_steps >= self.step_limit
        ) or self.stuck_steps > self.max_stuck_steps:
            done = True
        if done:
            graph_info = GraphInfo(
                objs=["all"],
                ob_rep=self.state_rep.get_obs_rep(obs, obs, obs, action),
                act_rep=self.state_rep.get_action_rep_drqa(action),
                graph_state=self.state_rep.graph_state,
                graph_state_rep=self.state_rep.graph_state_rep,
                admissible_actions=[],
                admissible_actions_rep=[],
            )
        else:
            graph_info = self._build_graph_rep(action, obs)
        return obs, reward, done, info, graph_info

    def reset(self):
        self.state_rep = StateAction(
            self.spm_model, self.vocab, self.vocab_rev, self.tsv_file, self.max_word_len
        )
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0
        obs, info = self.env.reset()
        info["valid"] = False
        info["steps"] = 0
        graph_info = self._build_graph_rep("look", obs)
        return obs, info, graph_info

    def close(self):
        self.env.close()
