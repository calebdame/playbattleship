"""Battleship Monte-Carlo sampling engine.

Typical usage::

    from battleship import BattleshipBoard, BattleshipPlayer

    player = BattleshipPlayer(dim=10, ships=[5, 4, 3, 3, 2], boards=10000)
    coord = player.take_turn()
    player.update_game_state(*coord)
"""

from .board import BattleshipBoard
from .player import BattleshipPlayer

__all__ = ["BattleshipBoard", "BattleshipPlayer"]
