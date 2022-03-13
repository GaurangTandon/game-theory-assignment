from typing import List, Set

import numpy as np
import pygambit
import pytest

from q1 import Game


def helper_convert_gambit_psne_to_ours(
    gambit_g: pygambit.Game, gambit_psne: List[List[float]]
) -> Set[List[int]]:
    psne_strats: List[List[int]] = []
    processed_idx = 0

    for psne in gambit_psne:
        psne = list(psne)
        strat: List[int] = []

        for i in range(len(gambit_g.players)):
            strat.append(
                1
                + np.argmax(
                    psne[
                        processed_idx : processed_idx
                        + len(gambit_g.players[i].strategies)
                    ]
                )
            )
        psne_strats.append(strat)

    return set(psne_strats)

@pytest.mark.timeout(1)
def test_provided_games(nfg_str: str):
    nfg_str = nfg_str.strip()
    gambit_g = pygambit.Game.parse_game(nfg_str)

    n = len(gambit_g.players)
    utility_list = list(map(int, nfg_str.split("\n")[-1].split()))
    n_strategies = [len(gambit_g.players[i].strategies) for i in range(n)]

    if len(utility_list) != n * np.prod(n_strategies):
        return

    if len(utility_list) > int(1e6):
        return

    psne_gambit_g = pygambit.nash.enumpure_solve(gambit_g)
    g = Game(n, n_strategies, utility_list)
    psne_g = set(g.list_all_psne())
    assert psne_g == helper_convert_gambit_psne_to_ours(gambit_g, psne_gambit_g)
