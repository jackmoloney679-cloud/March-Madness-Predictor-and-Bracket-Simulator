# March Madness Predictor and Bracket Simulator

This repository predicts NCAA tournament game outcomes using machine learning and generates evaluation metrics and visualizations.

## Data Source

This project uses the Kaggle **March Machine Learning Mania 2025** dataset.  
The raw Kaggle CSV files are **not committed** to this repository. You must download them and place them locally (instructions below).

---

## Running the Project (TA / Grader Instructions)

The primary grading criterion for this project is that:

**`python3 main.py` runs successfully on a fresh machine.**

After cloning the git repository (for example on VS Code), all commands below must be run from the **New Terminal**

**Note** : the spaces between the different code segments indicates that each segment needs to be ran before running the next one. Instructions on how to put the CSV files in the correct place are shown when running the second block.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/raw
cat data/raw/README.md

python3 main.py

















