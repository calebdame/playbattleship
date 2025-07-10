import random
import string
from collections import Counter
from itertools import chain

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _single_random_board(args: Tuple["BattleshipBoard", Sequence[str]]) -> int:
    """Helper for multiprocessing joblib backends to generate one board."""
    board, names = args
    return board.random_board(names=names, batch_size=1)[0]


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
        poss_ships (dict): Dictionary storing possible positions for each ship
            as bitboard integers.
        poss_ship_bits (dict): Alias for ``poss_ships`` kept for backward
            compatibility.
        
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
        self.dim2 = dim * dim
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
        # Ship placements are stored primarily as bitboards to speed up
        # intersection checks. ``poss_ship_bits`` is kept for backwards
        # compatibility, but mirrors ``poss_ships``.
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

            bits = [self.coords_to_bit(p) for p in placements]
            # Store only the bitboard representations to avoid the overhead
            # of Python ``set`` operations during filtering.
            self.poss_ships[name] = bits
            self.poss_ship_bits[name] = bits
            self.poss_ships_num[name] = len(bits)

    def coords_to_bit(self, coords: Iterable[Tuple[int, int]]) -> int:
        """Convert a placement set to a bitboard integer.

        Using an integer avoids Python set intersections when checking ship
        collisions. Each bit corresponds to a tile, so overlapping ships can be
        detected via bitwise AND operations.
        """
        bit = 0
        for x, y in coords:
            bit |= 1 << (x * self.dim + y)
        return bit

    def bit_to_coords(self, bit: int) -> Set[Tuple[int, int]]:
        """Convert an integer bitboard back into a set of coordinates."""
        coords = set()
        for idx in range(self.dim * self.dim):
            if bit & (1 << idx):
                coords.add(divmod(idx, self.dim))
        return coords

    def random_board(
        self,
        initial: bool = False,
        names: Optional[Sequence[str]] = None,
        batch_size: int = 1,
    ) -> List[int]:
        """Generate random boards using bitboards for overlap checks.

        All returned boards are integers representing occupied tiles. When
        ``initial`` is True, each result is a list of bitboards corresponding to
        individual ships.

        Args:
            initial (bool, optional): Whether to draw all locations jointly.
                Defaults to False.
            names (list, optional): Ship names to consider while building the
                random board. If omitted, considers all ships. Defaults to None.
            batch_size (int, optional): Number of random boards to generate.
                Defaults to 1.

        Returns:
            list: Random boards represented as bitboard integers, or lists of
            bitboards when ``initial`` is True.
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

        if self.poss_ships_num[first_name] == 1:
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
                    cur_bits = list(start_bits)
                    for name in rest_names:
                        idx = int(self.poss_ships_num[name] * random.random())
                        k_bit = bits[name][idx]
                        if k_bit & t_bit:
                            end = False
                            break
                        t_bit |= k_bit
                        cur_bits.append(k_bit)
                    if end:
                        results.append(cur_bits if initial else t_bit)
                        break
        else:
            while len(results) < batch_size:
                while True:
                    end = True
                    idx = int(self.poss_ships_num[first_name] * random.random())
                    t_bit = bits[first_name][idx]
                    cur_bits = [bits[first_name][idx]]
                    for name in rest_names:
                        idx = int(self.poss_ships_num[name] * random.random())
                        k_bit = bits[name][idx]
                        if k_bit & t_bit:
                            end = False
                            break
                        t_bit |= k_bit
                        cur_bits.append(k_bit)
                    if end:
                        results.append(cur_bits if initial else t_bit)
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
        hits (set): Coordinates of tiles that have been hit. ``hits_bit`` stores
            the same information as a bitboard for fast operations.
        misses (set): Tiles that have been guessed but were misses.
            ``misses_bit`` mirrors these as a bitboard.
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
        parallel_backend (str, optional): ``'joblib'`` or ``'multiprocessing'``
            to generate boards in parallel. Any other value disables
            parallelism. Defaults to ``'sequential'``.
        n_jobs (int, optional): Number of worker processes when using a
            parallel backend. Defaults to 1.
    """
    def __init__(
        self,
        dim: int = 10,
        ships: Optional[Sequence[int]] = None,
        boards: int = 10000,
        parallel_backend: str = "sequential",
        n_jobs: int = 1,
    ) -> None:
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.board = BattleshipBoard(dim=dim, ships=ships)
        enemy_bits = self.board.random_board(initial=True)[0]
        self.enemy_ships = [
            self.board.bit_to_coords(b) for b in enemy_bits
        ]
        self.enemy_board = set.union(*self.enemy_ships)
        self.enemy_ship_statuses = {
            name: {
                'sunk': False,
                'confirmed_pos': 0,
                'hit_count': 0,
            }
            for name in self.board.names
        }
        self.n_boards = boards
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        # Maintain sets for external callers but operate primarily on bitboards.
        self.hits, self.misses, self.random_boards = set(), set(), []
        self.hits_bit = 0
        self.misses_bit = 0
        self.name_order = [name for name in self.board.names]
        self.target_hits = sum(self.board.ship_lengths.values())
        self.last_move = 0
        self.num_hits = 0

    def generate_random_boards(self) -> None:
        """Refresh the Monte-Carlo pool based on hits and misses.

        Instead of recomputing the entire sample each turn, previously valid
        boards are filtered and only the remaining count is generated.  This
        incremental approach keeps the simulation responsive when ``n_boards``
        is large.
        """
        if self.last_move == 1:
            self.random_boards = [
                b for b in self.random_boards if not (b & self.misses_bit)
            ]
        else:
            self.random_boards = [
                b for b in self.random_boards if (b & self.hits_bit) == self.hits_bit
            ]
            
        self.name_order = sorted(
            self.name_order,
            key=lambda x: self.board.poss_ships_num.get(x, float("inf")),
        )
        num_b = len(self.random_boards)
        needed = self.n_boards - num_b
        if needed <= 0:
            return

        if self.parallel_backend == "joblib":
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self.board.random_board)(names=self.name_order, batch_size=1)
                for _ in range(needed)
            )
            self.random_boards += [r[0] for r in results]
        elif self.parallel_backend == "multiprocessing":
            import multiprocessing as mp

            with mp.Pool(self.n_jobs) as pool:
                results = pool.map(
                    _single_random_board,
                    [(self.board, self.name_order)] * needed,
                )
            self.random_boards += results
        else:
            self.random_boards += self.board.random_board(
                names=self.name_order,
                batch_size=needed,
            )
    
    def update_possible_ship_positions(self) -> None:
        """
        Update the possible positions of each ship based on the current game
        state. Confirmed positions, hits and misses narrow down the potential
        placements.
        """
        hits_bit = self.hits_bit
        misses_bit = self.misses_bit

        for name in self.board.names:
            if len(self.board.poss_ships[name]) == 1:
                continue
            necessary = self.enemy_ship_statuses[name]['confirmed_pos']
            if necessary:
                need_len = self.enemy_ship_statuses[name]['hit_count']
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not (conf & misses_bit)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if (
                            (conf & necessary).bit_count() == need_len
                            and (conf & hits_bit).bit_count() == need_len
                        )
                    ]
            else:
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not (conf & misses_bit)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf for conf in self.board.poss_ships[name]
                        if not (conf & hits_bit)
                    ]
            self.board.poss_ships_num[name] = len(self.board.poss_ships[name])

    def take_turn(self) -> Optional[Tuple[int, int]]:
        """Return the most likely coordinate to contain a ship.

        The function tallies tile frequencies across the current Monte-Carlo
        sample and fires at the most common unseen tile.  This simple heuristic
        performs well for Battleship without requiring an exhaustive search.

        Returns:
            tuple: The chosen tile to target in the current turn.
        """
        self.generate_random_boards()
        dim2 = self.board.dim2
        counts = [0] * dim2
        for b in self.random_boards:
            bit = b
            while bit:
                lsb = bit & -bit
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                bit &= bit - 1
        unseen_mask = ~(self.hits_bit | self.misses_bit)
        best_idx = None
        best_count = -1
        for idx, c in enumerate(counts):
            if unseen_mask & (1 << idx) and c > best_count:
                best_idx = idx
                best_count = c
        if best_idx is not None:
            x, y = divmod(best_idx, self.board.dim)
            return (x, y)

        # If every candidate has already been fired at, fall back to a random
        # unseen position to guarantee progress toward game completion.
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
        Hits and misses are tracked as both coordinate sets (for
        compatibility) and bitboards for faster filtering.

        Args:
            x (int): The x-coordinate of the targeted tile.
            y (int): The y-coordinate of the targeted tile.
        """

        coords = (x, y)
        bit = 1 << (x * self.board.dim + y)
        if coords in self.enemy_board:
            if coords not in self.hits:
                self.hits.add(coords)
                self.hits_bit |= bit
                self.num_hits += 1
                self.last_move = 2
                for name, ship in zip(self.board.names, self.enemy_ships):
                    if coords in ship:
                        self.enemy_ship_statuses[name]['hit_count'] += 1
                        self.enemy_ship_statuses[name]['confirmed_pos'] |= bit
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
            self.misses_bit |= bit
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
        """Reset the player and enemy boards to start a new game.

        Regenerating component layouts ensures that subsequent games do not
        reuse any state from previous rounds, keeping the sampling unbiased.
        """
        self.board.generate_component_layouts()
        enemy_bits = self.board.random_board(initial=True)[0]
        self.enemy_ships = [
            self.board.bit_to_coords(b) for b in enemy_bits
        ]
        self.enemy_board = set.union(*self.enemy_ships)
        self.enemy_ship_statuses = {
            name: {
                'sunk': False,
                'confirmed_pos': 0,
                'hit_count': 0,
            }
            for name in self.board.names
        }
        self.last_move = 0
        self.num_hits = 0
        self.hits, self.misses, self.random_boards = set(), set(), []
        self.hits_bit = 0
        self.misses_bit = 0
        self.name_order = list(self.board.names)
