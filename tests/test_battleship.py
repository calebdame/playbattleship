import os
import random
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from battleship import BattleshipBoard, BattleshipPlayer


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(0)


def test_generate_component_layouts():
    board = BattleshipBoard(dim=4, ships=[2])
    assert board.poss_ships_num['a'] == 24


def test_coords_to_bit():
    board = BattleshipBoard(dim=4, ships=[2])
    bits = board.coords_to_bit({(0, 0), (1, 0)})
    expected = (1 << 0) | (1 << 4)
    assert bits == expected


def test_bit_to_coords():
    board = BattleshipBoard(dim=4, ships=[2])
    coords = board.bit_to_coords((1 << 0) | (1 << 4))
    assert coords == {(0, 0), (1, 0)}


def test_random_board_no_overlap():
    board = BattleshipBoard(dim=4, ships=[2, 3])
    boards = board.random_board(batch_size=5)
    for b in boards:
        coords = board.bit_to_coords(b)
        assert len(coords) == 5
        assert len(coords) == len(set(coords))


def test_generate_random_boards_respects_hits_misses():
    player = BattleshipPlayer(dim=4, ships=[2])
    hit = next(iter(player.enemy_ships[0]))
    player.hits.add(hit)
    miss = (3, 3)
    player.misses.add(miss)
    player.last_move = 2
    player.generate_random_boards()
    miss_bit = 1 << (miss[0] * player.board.dim + miss[1])
    hit_bit = 1 << (hit[0] * player.board.dim + hit[1])
    boards_without_miss = [b for b in player.random_boards if not (b & miss_bit)]
    assert boards_without_miss
    assert any(b & hit_bit for b in boards_without_miss)


def test_take_turn_returns_valid_coord():
    player = BattleshipPlayer(dim=4, ships=[2])
    coord = player.take_turn()
    assert coord not in player.hits
    assert coord not in player.misses
    assert 0 <= coord[0] < 4
    assert 0 <= coord[1] < 4


def test_update_game_state_and_check_all_sunk():
    player = BattleshipPlayer(dim=4, ships=[1])
    ship = player.enemy_ships[0]
    for x, y in ship:
        player.update_game_state(x, y)
    assert player.check_all_sunk()


def test_reset_resets_state():
    player = BattleshipPlayer(dim=4, ships=[1])
    player.hits.add((0, 0))
    player.misses.add((1, 1))
    player.reset()
    assert not player.hits and not player.misses
