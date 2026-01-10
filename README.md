# March Madness Predictor and Bracket Simulator

This repository predicts NCAA tournament game outcomes using machine learning and generates evaluation metrics and visualizations.

## Data Source

This project uses the Kaggle **March Machine Learning Mania 2025** dataset.  
The raw Kaggle CSV files are **not committed** to this repository. You must download them and place them locally (instructions below).

---

## Running the Project (TA / Grader Instructions)

The primary grading criterion for this project is that:

**`python3 main.py` runs successfully on a fresh machine.**

All commands below must be run from the **repository root** (the folder containing `main.py`).

```bash
git clone https://github.com/jackmoloney679-cloud/March-Madness-Predictor-and-Bracket-Simulator.git
cd March-Madness-Predictor-and-Bracket-Simulator

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/raw
cat data/raw/README.md
ls data/raw

python3 main.py















