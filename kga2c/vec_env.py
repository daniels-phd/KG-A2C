from queue import Queue
from threading import Event, Thread

from kga2c.env import KGA2CEnv
import sentencepiece as spm


def worker(queue: Queue, event: Event, params):

    sp = spm.SentencePieceProcessor()
    sp.Load(params["spm_file"])
    env = KGA2CEnv(
        params["rom_file_path"],
        params["seed"],
        sp,
        params["tsv_file"],
        step_limit=params["reset_steps"],
        stuck_steps=params["stuck_steps"],
        gat=params["gat"],
    )
    try:
        done = False
        while True:
            event.wait()
            latest = queue.get()
            cmd = latest[0]
            data = latest[1]
            if cmd == "step":
                if done:
                    ob, info, graph_info = env.reset()
                    rew = 0
                    done = False
                else:
                    ob, rew, done, info, graph_info = env.step(data)
                queue.put((ob, rew, done, info, graph_info))
                event.set()
            elif cmd == "reset":
                ob, info, graph_info = env.reset()
                queue.put((ob, info, graph_info))
                event.set()
            elif cmd == "close":
                env.close()
                event.set()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print("SubprocVecEnv worker: got KeyboardInterrupt")
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, params):
        self.closed = False
        self.total_steps = 0
        self.num_envs = num_envs
        sp = spm.SentencePieceProcessor()
        sp.Load(params["spm_file"])
        self.remotes = [
            KGA2CEnv(
                params["rom_file_path"],
                params["seed"],
                sp,
                params["tsv_file"],
                step_limit=params["reset_steps"],
                stuck_steps=params["stuck_steps"],
                gat=params["gat"],
            )
            for _ in range(num_envs)
        ]
        """ self.ps = [
            Thread(target=worker, args=(remote, event, params))
            for (remote, event) in self.remotes
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for (remote, event) in self.remotes:
            event.clear()
            remote.empty() """

    def step(self, actions):
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."

        results = [env.step(action) for (env, action) in zip(self.remotes, actions)]
        self.waiting = False
        return zip(*results)

    def reset(self):
        self._assert_not_closed()

        results = [env.reset() for env in self.remotes]
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for env in self.remotes:
            env.close()

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on a SubprocVecEnv after calling close()"
