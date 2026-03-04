# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Fast rejection-sampling board generator for Battleship.

Replaces the hot inner loop of BattleshipBoard.random_board with typed C,
using 128-bit bitboards (2 x uint64) and xorshift64* PRNG.

Supports boards up to 11x11 (121 tiles <= 128 bits).
"""

from libc.stdlib cimport malloc, free, realloc
from libc.stdint cimport uint64_t, uint32_t

import random as _py_random

# ── monotonic clock (for time-limited generation) ─────────────────

cdef extern from "<time.h>" nogil:
    cdef struct timespec:
        long tv_sec
        long tv_nsec
    int CLOCK_MONOTONIC
    int clock_gettime(int, timespec *)

cdef inline double _monotonic() noexcept nogil:
    cdef timespec ts
    clock_gettime(CLOCK_MONOTONIC, &ts)
    return <double>ts.tv_sec + <double>ts.tv_nsec * 1e-9

# ── 128-bit bitboard ──────────────────────────────────────────────

ctypedef struct BB:
    uint64_t lo
    uint64_t hi

cdef inline BB bb_or(BB a, BB b) noexcept nogil:
    cdef BB r
    r.lo = a.lo | b.lo
    r.hi = a.hi | b.hi
    return r

cdef inline bint bb_overlap(BB a, BB b) noexcept nogil:
    # Explicit != 0 avoids uint64_t → int truncation losing upper bits.
    return (a.lo & b.lo) != 0 or (a.hi & b.hi) != 0

# ── xorshift64* PRNG ─────────────────────────────────────────────

cdef uint64_t _rng = 88172645463325252ULL

cdef inline void _seed(uint64_t s) noexcept nogil:
    global _rng
    _rng = s if s else 1

cdef inline uint64_t _next() noexcept nogil:
    global _rng
    cdef uint64_t x = _rng
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    _rng = x
    return x * <uint64_t>0x2545F4914F6CDD1D

cdef inline uint32_t _rand_below(uint32_t n) noexcept nogil:
    return <uint32_t>(_next() % <uint64_t>n)

# ── Python int <-> BB ─────────────────────────────────────────────

cdef inline BB _to_bb(object pyint):
    cdef BB b
    cdef object v = int(pyint)
    b.lo = <uint64_t>(v & 0xFFFFFFFFFFFFFFFF)
    b.hi = <uint64_t>((v >> 64) & 0xFFFFFFFFFFFFFFFF)
    return b

cdef inline object _from_bb(BB b):
    return int(b.lo) | (int(b.hi) << 64)

# ── Public API ────────────────────────────────────────────────────

def fast_random_boards(
    list ship_placements,
    list ship_counts,
    int batch_size,
    bint initial,
    double time_limit=-1.0,
):
    """Generate random non-overlapping ship layouts.

    Parameters
    ----------
    ship_placements : list[list[int]]
        Legal placements per ship, as Python-int bitboards.
    ship_counts : list[int]
        Number of placements available per ship.
    batch_size : int
        How many boards to produce (ignored when *time_limit* > 0).
    initial : bool
        True  -> return list of lists (per-ship bitboards).
        False -> return list of combined bitboards.
    time_limit : float, optional
        When positive, generate boards until this many seconds have
        elapsed instead of producing exactly *batch_size* boards.

    Returns
    -------
    list[int] or list[list[int]]
    """
    cdef int n_ships = len(ship_placements)
    cdef bint timed = time_limit > 0.0

    if n_ships == 0 or (not timed and batch_size <= 0):
        return []

    # Seed from Python's random state so callers that seed random get
    # deterministic behaviour.
    _seed(<uint64_t>_py_random.getrandbits(64))

    # ── Convert placement lists into C arrays ─────────────────────
    cdef BB **c_pl = <BB **>malloc(n_ships * sizeof(BB *))
    cdef uint32_t *c_n = <uint32_t *>malloc(n_ships * sizeof(uint32_t))
    if c_pl == NULL or c_n == NULL:
        raise MemoryError()

    cdef int i, j
    for i in range(n_ships):
        c_n[i] = <uint32_t>ship_counts[i]
        c_pl[i] = <BB *>malloc(c_n[i] * sizeof(BB))
        if c_pl[i] == NULL:
            raise MemoryError()
        plist = ship_placements[i]
        for j in range(<int>c_n[i]):
            c_pl[i][j] = _to_bb(plist[j])

    # ── Working storage ───────────────────────────────────────────
    cdef int capacity
    if timed:
        capacity = 4096
    else:
        capacity = batch_size

    cdef BB *cur_bits = <BB *>malloc(n_ships * sizeof(BB))
    cdef BB *combined = <BB *>malloc(capacity * sizeof(BB))
    cdef BB *all_ships = NULL
    if initial:
        all_ships = <BB *>malloc(capacity * n_ships * sizeof(BB))
        if all_ships == NULL:
            raise MemoryError()
    if cur_bits == NULL or combined == NULL:
        raise MemoryError()

    # ── Rejection-sampling loop (no GIL) ──────────────────────────
    cdef BB t_bit, k_bit
    cdef uint32_t idx
    cdef bint collision
    cdef int produced
    cdef double deadline
    cdef BB *new_combined
    cdef BB *new_all_ships

    with nogil:
        if timed:
            deadline = _monotonic() + time_limit

        produced = 0
        while True:
            # ── stop condition ────────────────────────────────────
            if timed:
                if _monotonic() >= deadline:
                    break
            else:
                if produced >= batch_size:
                    break

            # ── rejection-sample one valid board ──────────────────
            while True:
                collision = False
                idx = _rand_below(c_n[0])
                t_bit = c_pl[0][idx]
                cur_bits[0] = c_pl[0][idx]

                for i in range(1, n_ships):
                    idx = _rand_below(c_n[i])
                    k_bit = c_pl[i][idx]
                    if bb_overlap(k_bit, t_bit):
                        collision = True
                        break
                    t_bit = bb_or(t_bit, k_bit)
                    cur_bits[i] = k_bit

                if not collision:
                    break

            # ── grow buffer if needed (timed mode only) ───────────
            if timed and produced >= capacity:
                capacity = capacity * 2
                new_combined = <BB *>realloc(combined, capacity * sizeof(BB))
                if new_combined == NULL:
                    break
                combined = new_combined
                if all_ships != NULL:
                    new_all_ships = <BB *>realloc(all_ships, capacity * n_ships * sizeof(BB))
                    if new_all_ships == NULL:
                        break
                    all_ships = new_all_ships

            combined[produced] = t_bit
            if all_ships != NULL:
                for i in range(n_ships):
                    all_ships[produced * n_ships + i] = cur_bits[i]
            produced += 1

    # ── Convert results back to Python ────────────────────────────
    cdef list results = []
    if initial:
        for i in range(produced):
            results.append([
                _from_bb(all_ships[i * n_ships + j])
                for j in range(n_ships)
            ])
    else:
        for i in range(produced):
            results.append(_from_bb(combined[i]))

    # ── Cleanup ───────────────────────────────────────────────────
    if all_ships != NULL:
        free(all_ships)
    free(combined)
    free(cur_bits)
    for i in range(n_ships):
        free(c_pl[i])
    free(c_pl)
    free(c_n)

    return results
