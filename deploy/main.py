import uuid
import random
from typing import List, Tuple, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from battleship import BattleshipPlayer

app = FastAPI()
app.mount("/", StaticFiles(directory="deploy/static", html=True), name="static")

BOARD_DIM = 10
SHIP_LENGTHS = [5, 4, 3, 3, 2]

class ShipPlacement(BaseModel):
    ships: List[List[Tuple[int, int]]]

class Guess(BaseModel):
    x: int
    y: int

class Game:
    def __init__(self) -> None:
        # player_ai knows computer board and offers suggestions
        self.player_ai = BattleshipPlayer(dim=BOARD_DIM, ships=SHIP_LENGTHS)
        self.player_ai.generate_random_boards()
        self.computer_board = self.player_ai.enemy_ships
        self.player_ships: List[List[Tuple[int,int]]] = []
        self.computer_ai: BattleshipPlayer | None = None
        self.turn = random.choice(["player", "computer"])
        self.winner: str | None = None

games: Dict[str, Game] = {}

@app.post("/api/new_game")
def new_game():
    game_id = str(uuid.uuid4())
    games[game_id] = Game()
    return {"game_id": game_id, "turn": games[game_id].turn}

@app.post("/api/{game_id}/set_ships")
def set_ships(game_id: str, placement: ShipPlacement):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="game not found")
    game = games[game_id]
    if game.player_ships:
        raise HTTPException(status_code=400, detail="ships already set")
    if len(placement.ships) != len(SHIP_LENGTHS):
        raise HTTPException(status_code=400, detail="wrong number of ships")
    player_ships = [set(tuple(c) for c in ship) for ship in placement.ships]
    # simple validation for overlap
    seen = set()
    for ship,length in zip(player_ships, SHIP_LENGTHS):
        if len(ship) != length:
            raise HTTPException(status_code=400, detail="ship length mismatch")
        if seen & ship:
            raise HTTPException(status_code=400, detail="overlapping ships")
        seen |= ship
    game.player_ships = [list(ship) for ship in player_ships]

    # computer_ai uses player's ships as enemy board
    comp = BattleshipPlayer(dim=BOARD_DIM, ships=SHIP_LENGTHS)
    comp.enemy_ships = [set(ship) for ship in player_ships]
    comp.enemy_board = set.union(*comp.enemy_ships)
    comp.enemy_ship_statuses = {
        name: {"sunk": False, "confirmed_pos": set(), "hit_count": 0}
        for name in comp.board.names
    }
    comp.hits.clear(); comp.misses.clear();
    comp.hits_bit = 0; comp.misses_bit = 0
    comp.name_order = list(comp.board.names)
    comp.target_hits = sum(comp.board.ship_lengths.values())
    comp.last_move = 0
    comp.generate_random_boards()
    game.computer_ai = comp
    return {"status": "ok"}

@app.get("/api/{game_id}/state")
def state(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="game not found")
    game = games[game_id]
    best = None
    if not game.winner and game.turn == "player":
        guess = game.player_ai.take_turn()
        if guess:
            best = list(guess)
    return {
        "turn": game.turn,
        "winner": game.winner,
        "player_hits": list(game.player_ai.hits),
        "player_misses": list(game.player_ai.misses),
        "computer_hits": list(game.computer_ai.hits if game.computer_ai else []),
        "computer_misses": list(game.computer_ai.misses if game.computer_ai else []),
        "player_ships": game.player_ships,
        "best_guess": best,
    }

@app.post("/api/{game_id}/player_guess")
def player_guess(game_id: str, g: Guess):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="game not found")
    game = games[game_id]
    if game.winner:
        return {"winner": game.winner}
    if game.turn != "player":
        raise HTTPException(status_code=400, detail="not player's turn")
    game.player_ai.update_game_state(g.x, g.y)
    if game.player_ai.check_all_sunk():
        game.winner = "player"
    else:
        game.turn = "computer"
    return {"turn": game.turn, "winner": game.winner}

@app.post("/api/{game_id}/computer_turn")
def computer_turn(game_id: str):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="game not found")
    game = games[game_id]
    if not game.computer_ai:
        raise HTTPException(status_code=400, detail="ships not set")
    if game.winner:
        return {"winner": game.winner}
    if game.turn != "computer":
        raise HTTPException(status_code=400, detail="not computer's turn")
    coord = game.computer_ai.take_turn()
    if coord:
        x,y = coord
        game.computer_ai.update_game_state(x,y)
    if game.computer_ai.check_all_sunk():
        game.winner = "computer"
    else:
        game.turn = "player"
    return {"turn": game.turn, "winner": game.winner, "coord": coord}
