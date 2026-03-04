import string
import time as _time
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import random

try:
    from ._fast_board import fast_random_boards as _fast_gen
except ImportError:
    _fast_gen = None


class BattleshipBoard:
    """Board geometry and random-board generation for Battleship.

    This class encodes the rules of standard Battleship — board dimensions,
    ship lengths, and legal placements — and provides fast random-board
    generation backed by bitboard arithmetic.

    Attributes:
        dim: Size of the square board (dim x dim).
        dim2: Total number of tiles (dim * dim).
        names: Ship identifiers derived from the English alphabet.
        ship_lengths: Mapping from ship name to its length.
        poss_ships_num: Number of legal placements per ship.
        poss_ships: Legal placements per ship as bitboard integers.
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
        self.names = tuple(string.ascii_lowercase[: len(ships)])
        self.ship_lengths = dict(zip(self.names, ships))
        self.generate_component_layouts()

    # ------------------------------------------------------------------
    # Layout pre-computation
    # ------------------------------------------------------------------

    def generate_component_layouts(self) -> None:
        """Pre-compute every legal placement for each ship as bitboards."""
        self.poss_ships_num, self.poss_ships = {}, {}
        dim_range = range(self.dim)

        for name in self.names:
            ship_length = range(self.ship_lengths[name])
            dim_limit = range(self.dim - self.ship_lengths[name] + 1)

            placements = [
                {(i + temp, j) for temp in ship_length}
                for i in dim_limit
                for j in dim_range
            ] + [
                {(j, i + temp) for temp in ship_length}
                for i in dim_limit
                for j in dim_range
            ]

            bits = [self.coords_to_bit(p) for p in placements]
            self.poss_ships[name] = bits
            self.poss_ships_num[name] = len(bits)

    # ------------------------------------------------------------------
    # Bitboard helpers
    # ------------------------------------------------------------------

    def coords_to_bit(self, coords: Iterable[Tuple[int, int]]) -> int:
        """Convert a set of (row, col) coordinates to a bitboard integer."""
        bit = 0
        for x, y in coords:
            bit |= 1 << (x * self.dim + y)
        return bit

    def bit_to_coords(self, bit: int) -> Set[Tuple[int, int]]:
        """Convert a bitboard integer back into a set of coordinates."""
        coords = set()
        for idx in range(self.dim2):
            if bit & (1 << idx):
                coords.add(divmod(idx, self.dim))
        return coords

    # ------------------------------------------------------------------
    # Random board generation
    # ------------------------------------------------------------------

    def random_board(
        self,
        initial: bool = False,
        names: Optional[Sequence[str]] = None,
        batch_size: int = 1,
        time_limit: Optional[float] = None,
        backend: str = "auto",
    ) -> List[int]:
        """Generate one or more random non-overlapping ship layouts.

        Args:
            initial: When True each result is a *list* of per-ship bitboards
                rather than a single combined bitboard.
            names: Subset of ship names to place.  Defaults to all ships.
            batch_size: How many boards to generate (ignored when
                *time_limit* is set).
            time_limit: When set, generate boards until this many seconds
                have elapsed instead of producing exactly *batch_size*.
            backend: Which generation backend to use.  ``'auto'`` (default)
                uses Cython when available, otherwise Python.  ``'cython'``
                forces the Cython extension (raises ``RuntimeError`` if it is
                not installed).  ``'python'`` forces the pure-Python path.

        Returns:
            A list of bitboard integers (or lists of bitboards when *initial*
            is True).
        """
        if names is None:
            names = []

        # Resolve active ship list once for both paths.
        if initial or not names:
            active = self.names
        else:
            active = names

        # ── Cython fast path (128-bit BB → up to 11x11) ──────────
        if backend == "cython":
            if _fast_gen is None:
                raise RuntimeError(
                    "Cython backend requested but '_fast_board' extension is not"
                    " available.  Build it with: python setup.py build_ext --inplace"
                )
            if self.dim2 > 128:
                raise RuntimeError(
                    f"Cython backend only supports dim2 <= 128 (got {self.dim2})."
                )
        use_cython = (
            backend == "cython"
            or (backend == "auto" and _fast_gen is not None and self.dim2 <= 128)
        )
        if use_cython:
            tl = -1.0 if time_limit is None else time_limit
            return _fast_gen(
                [self.poss_ships[n] for n in active],
                [self.poss_ships_num[n] for n in active],
                batch_size,
                initial,
                tl,
            )

        # ── Python fallback ───────────────────────────────────────
        results: list = []
        first_name, *rest_names = active

        bits = self.poss_ships

        timed = time_limit is not None
        deadline = _time.monotonic() + time_limit if timed else 0.0

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

            while True:
                if timed:
                    if _time.monotonic() >= deadline:
                        break
                elif len(results) >= batch_size:
                    break
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
            while True:
                if timed:
                    if _time.monotonic() >= deadline:
                        break
                elif len(results) >= batch_size:
                    break
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
