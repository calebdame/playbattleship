import random
import string
from collections import Counter
from itertools import chain

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class BattleshipBoard:
    """
    Main Battleship class where rules of the game are defined such as
    dimensions, ships, etc., and the initial board is built.
    
    Attributes:
        dim (int): Size of the square board (dim x dim).
        names (tuple): Names of the ships derived from the English alphabet.
        ship_lengths (dict): Dictionary mapping ship names to their lengths.
        poss_ships_num (dict): Dictionary storing the number of possible
            positions for each ship.
        poss_ships (dict): Dictionary storing possible positions for each ship.
        poss_ship_bits (dict): Dictionary storing bitboard integers for each
            possible ship placement. Used to speed up board sampling.
        
    Args:
        dim (int, optional): Dimension of the square board. Defaults to 10.
        ships (list, optional): List of ship lengths that fit on the board. 
            Defaults to [5,4,3,3,2].
    """
    
    def __init__(
        self,
        dim: int = 10,
        ships: Optional[Sequence[int]] = None,
    ) -> None:
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.dim = dim
        self.names = tuple(string.ascii_lowercase[:len(ships)])
        self.ship_lengths = dict(zip(self.names, ships))
        self.generate_component_layouts()

    def generate_component_layouts(self) -> None:
        """
        Generate all legal moves for each ship to define possible moves based on
        the ship's length and board dimensions. The possible moves are stored
        in dictionaries.
        """
        self.poss_ships_num, self.poss_ships = {}, {}
        self.poss_ship_bits = {}
        dim_range = range(self.dim)
        
        for name in self.names:
            ship_length = range(self.ship_lengths[name])
            dim_limit = range(self.dim - self.ship_lengths[name] + 1)
            
            placements = [
                {(i + temp, j) for temp in ship_length}
                for i in dim_limit for j in dim_range
            ] + [
                {(j, i + temp) for temp in ship_length}
                for i in dim_limit for j in dim_range
            ]

            self.poss_ships[name] = placements
            self.poss_ship_bits[name] = [
                self.coords_to_bit(p) for p in placements
            ]
            self.poss_ships_num[name] = len(placements)

    def coords_to_bit(self, coords: Iterable[Tuple[int, int]]) -> int:
        """Convert a placement set to a bitboard integer."""
        bit = 0
        for x, y in coords:
            bit |= 1 << (x * self.dim + y)
        return bit

    def random_board(
        self,
        initial: bool = False,
        names: Optional[Sequence[str]] = None,
        batch_size: int = 1,
    ) -> List[Set[Tuple[int, int]]]:
        """Generate random boards using bitboards for overlap checks.

        This method mirrors the original ``random_board`` API but internally
        relies on bitboards for faster collision detection.  It still returns
        sets of coordinate tuples so the rest of the code base remains
        unchanged.

        Args:
            initial (bool, optional): Whether to draw all locations jointly.
                Defaults to False.
            names (list, optional): List of ship names to consider while
                building the random board. If not provided, considers all ships.
                Defaults to None.
            batch_size (int, optional): Number of random boards to generate.
                Defaults to 1.

        Returns:
            list: Random boards represented as sets of coordinates (or lists of
            sets when ``initial`` is True).
        """
        if names is None:
            names = []

        results = []

        if initial or not names:
            first_name, *rest_names = self.names
            names = self.names
        else:
            first_name, *rest_names = names

        bits = self.poss_ship_bits
        sets = self.poss_ships

        if self.poss_ships_num[first_name] == 1:
            start_idx = [0 for name in names if self.poss_ships_num[name] == 1]
            start_sets = [
                sets[name][0]
                for name in names
                if self.poss_ships_num[name] == 1
            ]
            start_bits = [
                bits[name][0]
                for name in names
                if self.poss_ships_num[name] == 1
            ]
            rest_names = [
                name for name in names if self.poss_ships_num[name] != 1
            ]
            start_t = 0
            for b in start_bits:
                start_t |= b

            while len(results) < batch_size:
                while True:
                    end = True
                    t_bit = start_t
                    cur_sets = list(start_sets)
                    for name in rest_names:
                        idx = int(self.poss_ships_num[name] * random.random())
                        k_bit = bits[name][idx]
                        if k_bit & t_bit:
                            end = False
                            break
                        t_bit |= k_bit
                        cur_sets.append(sets[name][idx])
                    if end:
                        board_set = set.union(*cur_sets)
                        results.append(cur_sets if initial else board_set)
                        break
        else:
            while len(results) < batch_size:
                while True:
                    end = True
                    idx = int(self.poss_ships_num[first_name] * random.random())
                    t_bit = bits[first_name][idx]
                    cur_sets = [sets[first_name][idx]]
                    for name in rest_names:
                        idx = int(self.poss_ships_num[name] * random.random())
                        k_bit = bits[name][idx]
                        if k_bit & t_bit:
                            end = False
                            break
                        t_bit |= k_bit
                        cur_sets.append(sets[name][idx])
                    if end:
                        board_set = set.union(*cur_sets)
                        results.append(cur_sets if initial else board_set)
                        break

        return results


class BattleshipPlayer:
    """A Monte‑Carlo driven battleship opponent.

    Attributes:
        board (BattleshipBoard): Player's board configuration.
        enemy_ships (list): Sets representing the enemy ship positions.
        enemy_board (set): Tiles occupied by all enemy ships.
        enemy_ship_statuses (dict): Status of each enemy ship.
        n_boards (int): Number of random boards to generate for simulations.
        hits (set): Set of tuples representing the tiles that have been hit.
        misses (set): Tiles that have been guessed but were misses.
        random_boards (list): List of random boards generated for simulations.
        name_order (list): List of ship names ordered for strategic gameplay.
        target_hits (int): Total number of hits needed to sink all ships.
        last_move (int): 1 if the previous shot missed, 2 if it hit.
        num_hits (int): Number of successful hits made so far.

    Args:
        dim (int, optional): Dimension of the square board. Defaults to 10.
        ships (list, optional): Ship lengths. Defaults to [5, 4, 3, 3, 2].
        boards (int, optional): Number of random boards for simulations.
            Defaults to 10000.
    """
    def __init__(
        self,
        dim: int = 10,
        ships: Optional[Sequence[int]] = None,
        boards: int = 10000,
    ) -> None:
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.board = BattleshipBoard(dim=dim, ships=ships)
        self.enemy_ships = self.board.random_board(initial=True)[0]
        self.enemy_board = set.union(*self.enemy_ships)
        self.enemy_ship_statuses = {
            name: {
                'sunk': False,
                'confirmed_pos': set(),
                'hit_count': 0,
            }
            for name in self.board.names
        }
        self.n_boards = boards
        self.hits, self.misses, self.random_boards = set(), set(), []
        self.name_order = [name for name in self.board.names]
        self.target_hits = sum(self.board.ship_lengths.values())
        self.last_move = 0
        self.num_hits = 0

    def generate_random_boards(self) -> None:
        """
        Generate random boards based on the current game state, considering
        hits and misses. Random boards are used to make educated guesses in
        future turns.
        """
        if self.last_move == 1:
            self.random_boards = [
                b for b in self.random_boards if not bool(b & self.misses)
            ]
        else:
            self.random_boards = [
                b for b in self.random_boards if not bool(self.hits - b)
            ]
            
        self.name_order = sorted(
            self.name_order,
            key=lambda x: self.board.poss_ships_num.get(x, float("inf")),
        )
        num_b = len(self.random_boards)
        self.random_boards += self.board.random_board(
            names=self.name_order,
            batch_size=self.n_boards - num_b,
        )
    
    def update_possible_ship_positions(self) -> None:
        """
        Update the possible positions of each ship based on the current game
        state. Confirmed positions, hits and misses narrow down the potential
        placements.
        """
        for name in self.board.names:
            if len(self.board.poss_ships[name]) == 1:
                continue
            necessary = self.enemy_ship_statuses[name]['confirmed_pos']
            hits, misses = self.hits, self.misses
            if necessary:
                len_necessity = self.enemy_ship_statuses[name]['hit_count']
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not bool(conf & misses)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if (
                            len(necessary & conf)
                            == len_necessity
                            == len(hits & conf)
                        )
                    ]
            else:
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not bool(conf & misses)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not bool(hits & conf)
                    ]
            self.board.poss_ships_num[name] = len(self.board.poss_ships[name])

    def take_turn(self) -> Optional[Tuple[int, int]]:
        """
        Take a turn by choosing the most common tile among the sampled
        boards. This aims to make an educated guess based on the current game
        state.

        Returns:
            tuple: The chosen tile to target in the current turn.
        """
        self.generate_random_boards()
        board_tiles = Counter(chain.from_iterable(self.random_boards))
        for tile, _ in board_tiles.most_common(len(self.hits) + 1)[::-1]:
            if tile not in self.hits and tile not in self.misses:
                return tile

        # Fallback in case all of the suggested tiles were already tried
        remaining = [
            (x, y)
            for x in range(self.board.dim)
            for y in range(self.board.dim)
            if (x, y) not in self.hits and (x, y) not in self.misses
        ]
        if remaining:
            return random.choice(remaining)
        return None
    
    def update_game_state(self, x: int, y: int) -> None:
        """
        Update the game state based on the outcome of the current turn.
        Hits and misses update the sets and enemy ship status.

        Args:
            x (int): The x-coordinate of the targeted tile.
            y (int): The y-coordinate of the targeted tile.
        """

        coords = (x,y)
        if coords in self.enemy_board:
            if (x,y) not in self.hits:
                self.hits.add(coords)
                self.num_hits += 1
                self.last_move = 2
                for name, ship in zip(self.board.names, self.enemy_ships):
                    if (x,y) in ship:
                        self.enemy_ship_statuses[name]['hit_count'] += 1
                        self.enemy_ship_statuses[name][
                            'confirmed_pos'
                        ].add(coords)
                        if (
                            self.enemy_ship_statuses[name]['hit_count']
                            == len(ship)
                        ):
                            self.enemy_ship_statuses[name]['sunk'] = True
                        break
                self.update_possible_ship_positions()
        elif coords not in self.misses:
            self.last_move = 1
            self.misses.add(coords)
            self.update_possible_ship_positions()

    def check_all_sunk(self) -> bool:
        """
        Check if all enemy ships have been sunk based on the number of
        successful hits.

        Returns:
            bool: True if all enemy ships have been sunk, False otherwise.
        """
        return self.num_hits == self.target_hits
    
    def reset(self) -> None:
        """Reset the player and enemy boards to start a new game."""
        self.board.generate_component_layouts()
        self.enemy_ships = self.board.random_board(initial=True)[0]
        self.enemy_board = set.union(*self.enemy_ships)        self.enemy_ship_statuses = {
            name: {
                'sunk': False,
                'confirmed_pos': set(),
                'hit_count': 0,
            }
            for name in self.board.names
        }
        self.last_move = 0
        self.num_hits = 0
        self.hits, self.misses, self.random_boards = set(), set(), []
        self.name_order = list(self.board.names)
