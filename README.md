# March-Madness-Predictor-and-Bracket-Simulator
This repository contains my final project for the Data Science and Advanced Programming class. This project aims to explore why predicting March Madness is almost impossible by building a Python-based model analysing past tournament data as well as simulating outcomes of games.

## Project Structure (Course Requirement)

This repository follows the required ML project layout:

- `main.py` is the runnable entry point (pipeline + evaluation + plots).
- `src/` contains modules that document the intended separation of concerns
  (data loading, models, evaluation). Core logic currently remains in `main.py`.
- `data/raw/` is where Kaggle CSV inputs should be placed locally (not committed).
- `results/` stores generated figures/metrics when running the pipeline.
- `notebooks/` is for exploration and prototyping.

## Running the Project

The primary grading criterion for this project is that:

**`python3 main.py` runs successfully on a fresh machine.**

All commands below must be run from the **repository root**
(the folder containing `main.py`).

### End-to-end commands (copy & paste)

```bash
git clone https://github.com/jackmoloney679-cloud/March-Madness-Predictor-and-Bracket-Simulator.git
cd March-Madness-Predictor-and-Bracket-Simulator

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/raw
ls data/raw
cat data/raw/README.md

python3 main.py
```

If `python3 main.py` completes without errors, the project has run successfully.












