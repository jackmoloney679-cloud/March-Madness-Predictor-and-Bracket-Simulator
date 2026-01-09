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

Step 1: Clone the repository

Open a terminal and run:

git clone https://github.com/jackmoloney679-cloud/March-Madness-Predictor-and-Bracket-Simulator.git
cd March-Madness-Predictor-and-Bracket-Simulator

Step 2: Create a Python virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Step 3: Download and place the dataset

This project uses Kaggle’s March Machine Learning Mania dataset.

Download the dataset from Kaggle and place the required CSV files into:

data/raw/


The exact list of required CSV files is documented in:

data/raw/README.md


File names must match exactly (case-sensitive).

Step 4: Run the project

From the repository root, run:

python3 main.py
