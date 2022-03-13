import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")
    parser.addoption("--manual", action="store_true", help="run manual tests")
    parser.addoption("--gambit", action="store_true", help="run gambit tests")


def pytest_generate_tests(metafunc):
    gambit_nfgs = (REPO_ROOT / "gambit/contrib/games").glob("*.nfg")
    if "nfg_str" in metafunc.fixturenames:
        if metafunc.config.getoption("gambit") or metafunc.config.getoption("all"):
            metafunc.parametrize("nfg_str", [nfg.read_text() for nfg in gambit_nfgs])
    elif "game_args" in metafunc.fixturenames:
        if metafunc.config.getoption("manual") or metafunc.config.getoption("all"):
            metafunc.parametrize(
                "game_args,ans",
                [
                    (
                        (2, [2, 2], [], np.array([[[2, 2], [6, 4]], [[4, 6], [2, 2]]])),
                        [[1, 2], [2, 1]],
                    ),
                    (
                        (2, [2, 2], [], np.array([[[3, 2], [2, 4]], [[2, 4], [3, 2]]])),
                        [],
                    ),
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
                    ),
                    (
                        (
                            3,
                            [2, 2, 2],
                            [],
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
                    ),
                ],
            )
