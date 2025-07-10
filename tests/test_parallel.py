import os
import random
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from battleship import BattleshipPlayer


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(0)


def test_joblib_backend_generates_correct_number():
    player = BattleshipPlayer(dim=4, ships=[2], boards=20,
                              parallel_backend="joblib", n_jobs=2)
    player.generate_random_boards()
    assert len(player.random_boards) == 20


def test_multiprocessing_backend_generates_correct_number():
    player = BattleshipPlayer(dim=4, ships=[2], boards=20,
                              parallel_backend="multiprocessing", n_jobs=2)
    player.generate_random_boards()
    assert len(player.random_boards) == 20
