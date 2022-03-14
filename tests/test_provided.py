from typing import List, Tuple

import numpy as np
import pygambit
import pytest

from q1 import Game


def helper_convert_gambit_psne_to_ours(
    gambit_g: pygambit.Game, gambit_psne: List[List[float]]
) -> List[List[int]]:
    psne_strats: List[List[int]] = []

    for psne in gambit_psne:
        psne = list(psne)
        strat: List[int] = []
        processed_idx = 0

        for player in gambit_g.players:
            strat.append(
                1
                + np.argmax(
                    psne[processed_idx : processed_idx + len(player.strategies)]
                )
            )
            processed_idx += len(player.strategies)

        psne_strats.append(strat)

    return psne_strats


@pytest.mark.timeout(1)
def test_provided_games(nfg_str: str):
    nfg_str = nfg_str.strip()

    if nfg_str.split(" ", 4)[2] != "R":
        pytest.skip("not correct format")

    gambit_g = pygambit.Game.parse_game(nfg_str)

    n = len(gambit_g.players)
    utility_list = list(map(eval, nfg_str.split("\n")[-1].split()))
    n_strategies = [len(gambit_g.players[i].strategies) for i in range(n)]

    if len(utility_list) != n * np.prod(n_strategies):
        pytest.skip("not correct format")

    if len(utility_list) > int(1e6):
        pytest.skip("too large test case")

    psne_gambit_g = pygambit.nash.enumpure_solve(gambit_g, external=True)

    g = Game(n, n_strategies, utility_list)
    psne_g = g.list_all_psne()

    assert sorted(psne_g) == sorted(
        helper_convert_gambit_psne_to_ours(gambit_g, psne_gambit_g)
    )


@pytest.mark.timeout(1)
def test_manual_games(game_args: Tuple, ans: List[List[int]]):
    g = Game(*game_args)
    psne_g = g.list_all_psne()

    assert sorted(psne_g) == sorted(ans)
