import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))


def pytest_generate_tests(metafunc):
    gambit_nfgs = (REPO_ROOT / "gambit/contrib/games").glob("*.nfg")
    if "nfg_str" in metafunc.fixturenames:
        metafunc.parametrize("nfg_str", [nfg.read_text() for nfg in gambit_nfgs])
