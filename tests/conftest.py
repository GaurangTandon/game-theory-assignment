import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def skip_test():
    pytest.skip("Skipping test due to cli args")


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")
    parser.addoption("--manual", action="store_true", help="run manual tests")
    parser.addoption("--gambit", action="store_true", help="run gambit tests")
    parser.addoption("--stressed_out", action="store_true", help="run stress tests")

def million_players():
    pc = int(30)
    dim_count = pc + 1
    payoffs_shape = np.full((dim_count,), 1, dtype=np.int64)
    payoffs_shape[-1] = pc
    payoffs = np.ones(payoffs_shape)
    vwdse_strategy = [[1] for _ in range(pc)] 
    psne_state = [[1 for _ in range(pc)]]

    return (
        (pc, [1 for _ in range(pc)], [], payoffs),
        psne_state, vwdse_strategy
    )

def get_large_tests():
    tests = []
    # 1e6 players
    tests.append(million_players())

    return tests

def pytest_generate_tests(metafunc):
    if "nfg_str" in metafunc.fixturenames:
        if metafunc.config.getoption("gambit") or metafunc.config.getoption("all"):
            gambit_nfgs = (REPO_ROOT / "gambit/contrib/games").glob("*.nfg")
            metafunc.parametrize("nfg_str", [nfg.read_text() for nfg in gambit_nfgs])
        else:
            metafunc.fixturenames.insert(0, "skip_test")
    elif "game_args" in metafunc.fixturenames:
        if metafunc.config.getoption("manual") or metafunc.config.getoption("all"):
            metafunc.parametrize(
                "game_args,psne_strats,vwdse_strats",
                [
                    # 0
                    (
                        (
                            2,
                            [2, 2],
                            [],
                            np.array(
                                [
                                    [[2, 2], [6, 4]],
                                    [[4, 6], [2, 2]],
                                ]
                            ),
                        ),
                        [[1, 2], [2, 1]],
                        [[], []],
                    ),
                    # 1
                    (
                        (
                            2,
                            [2, 2],
                            [],
                            np.array(
                                [
                                    [[3, 2], [2, 4]],
                                    [[2, 4], [3, 2]],
                                ]
                            ),
                        ),
                        [],
                        [[], []],
                    ),
                    # 2
                    (
                        (
                            2,
                            [3, 3],
                            [],
                            np.array(
                                [
                                    [[1, 6], [5, 7], [3, 1]],
                                    [[1, 5], [2, 3], [5, 4]],
                                    [[1, 1], [3, 2], [5, 5]],
                                ]
                            ),
                        ),
                        [[1, 2], [3, 3], [2, 1]],
                        [[], []],
                    ),
                    # 3
                    (
                        (
                            3,
                            [2, 2, 2],
                            None,
                            np.array(
                                [
                                    [
                                        [[0, 0, 0], [1, -1, 1]],
                                        [[-1, 1, 1], [0, 0, 2]],
                                    ],
                                    [
                                        [[1, 1, -1], [2, 0, 0]],
                                        [[0, 2, 0], [1, 1, 1]],
                                    ],
                                ]
                            ),
                        ),
                        [[2, 2, 2]],
                        [[2], [2], [2]],
                    ),
                    # 4
                    (
                        (
                            2,
                            [2, 2],
                            [],
                            np.array(
                                [
                                    [[1, 1], [0, 3]],
                                    [[1, 2], [2, 2]],
                                ]
                            ),
                        ),
                        None,
                        [[2], [2]],
                    ),
                    # 5
                    (
                        (
                            2,
                            [3, 3],
                            [],
                            np.array(
                                [
                                    [[0, 5], [1, 5], [3, 1]],
                                    [[1, 3], [2, 3], [3, 2]],
                                    [[1, 5], [2, 3], [3, 3]],
                                ]
                            ),
                        ),
                        None,
                        [[2, 3], [1]],
                    ),
                    # 6
                    (
                        (
                            2,
                            [3, 3],
                            [],
                            np.array(
                                [
                                    [[0, 5], [1, 5], [3, 1]],
                                    [[1, 3], [2, 3], [3, 2]],
                                    [[1, 5], [2, 5], [3, 3]],
                                ]
                            ),
                        ),
                        None,
                        [[2, 3], [1, 2]],
                    ),
                    # 7
                    (
                        (
                            2,
                            [2, 3],
                            [],
                            np.array(
                                [
                                    [[1, 5], [3, 4], [1, 1]],
                                    [[2, 5], [2, 3], [3, 2]],
                                ]
                            ),
                        ),
                        None,
                        [[], [1]],
                    ),
                    # 8
                    (
                        (
                            3,
                            [2, 2, 2],
                            [],
                            np.array(
                                [
                                    [
                                        [[0, 0, 0], [1, 1, -1]],
                                        [[1, -1, 1], [2, 0, 0]],
                                    ],
                                    [
                                        [[-1, 1, 1], [0, 2, 0]],
                                        [[0, 0, 2], [1, 1, 1]],
                                    ],
                                ]
                            ),
                        ),
                        [[1, 1, 1]],
                        [[1], [1], [1]],
                    ),
                    # 9
                    (
                        (
                            2,
                            [4, 1],
                            [],
                            np.array(
                                [
                                    [[0, 1]],
                                    [[1, 0]],
                                    [[3, 2]],
                                    [[-1, 5]],
                                ]
                            ),
                        ),
                        None,
                        [[3], [1]],
                    ),
                    # 10
                    (
                        (5, [1, 1, 1, 1, 1], [], np.array([[[[[[1, 2, 2, 10, 1]]]]]])),
                        [[1, 1, 1, 1, 1]],
                        [[1], [1], [1], [1], [1]],
                    ),
                    # 11
                    (
                        (
                            2,
                            [4, 1],
                            [],
                            np.array(
                                [
                                    [[0, 1]],
                                    [[0, 0]],
                                    [[0, 2]],
                                    [[0, 5]],
                                ]
                            ),
                        ),
                        None,
                        [[1, 2, 3, 4], [1]],
                    ),
                    # 12
                    (
                        (
                            3,
                            [2, 2, 2],
                            [],
                            np.array(
                                [
                                    [
                                        [[5, 0, 0], [1, 1, -1]],
                                        [[1, -1, 1], [2, 0, 0]],
                                    ],
                                    [
                                        [[-1, 1, 1], [0, 2, 0]],
                                        [[0, 0, 2], [1, 1, 1]],
                                    ],
                                ]
                            ),
                        ),
                        None,
                        [[1], [1], [1]],
                    ),
                    # 13
                    (
                        (
                            3,
                            [2, 2, 2],
                            [],
                            np.array(
                                [
                                    [
                                        [[0, 0, 0], [1, 1, -1]],
                                        [[1, -1, 1], [2, 0, 0]],
                                    ],
                                    [
                                        [[5, 1, 1], [0, 2, 0]],
                                        [[0, 0, 2], [1, 1, 1]],
                                    ],
                                ]
                            ),
                        ),
                        None,
                        [[], [1], [1]],
                    ),
                ],
            )
        elif metafunc.config.getoption("stressed_out") or metafunc.config.getoption("all"):
            metafunc.parametrize(
                "game_args,psne_strats,vwdse_strats", get_large_tests()
            )
        else:
            metafunc.fixturenames.insert(0, "skip_test")
