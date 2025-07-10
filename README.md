# Battleship Monte-Carlo Solver

This project is a small Battleship game driven by a Monte‑Carlo search. A Streamlit interface lets you play locally or inside Codespaces.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game

Execute the Streamlit app:

```bash
streamlit run dev/streamlit_app.py
```

## How It Works

`BattleshipPlayer` in `battleship.py` repeatedly generates random boards that respect the hits and misses so far. The method [`generate_random_boards`](battleship.py#L192-L205) filters previously generated boards and adds new samples. The most common tile across these boards is picked in [`take_turn`](battleship.py#L224-L238). This Monte‑Carlo sampling uses the number defined by `boards_sim` in [`streamlit_app.py`](dev/streamlit_app.py#L4-L12).

## Codespaces

If using GitHub Codespaces, see [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) for environment setup and automatic Streamlit startup.

## Railway Deployment

When you connect this repository to [Railway](https://railway.app/), the
configuration in `railway.json` will install the dependencies and start the
FastAPI service using Uvicorn. No additional setup is required.
