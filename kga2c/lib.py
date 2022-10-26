from diskcache import Cache


def make_admissible_actions_cache(game: str):
    return Cache(f"./.cache/{game}/admissible_actions")

