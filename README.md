# Road Accident Hotspot & Safe Route Finder

This project is a simple Streamlit app for accident analysis. It compares the **fastest** and **safest** routes, shows accident hotspot areas, and includes a basic severity prediction feature.

## Files In This Repo

- `app.py` - Streamlit app
- `pipeline.py` - data loading, graph building, and clustering
- `Final_Project_Module3.ipynb` - notebook version of the project
- `Peshawar_Accidents_Synthetic.csv` - sample dataset used by default
- `models/rf_model.pkl` - Random Forest model
- `models/ann_model.pkl` - ANN model
- `LINK_DATASET` - source dataset reference

## Dataset

Large files are **not** pushed to GitHub.

Download the full dataset here:

- [US Accidents Dataset (Kaggle)](https://kaggle.com/datasets/sobhanmoosavi/us-accidents)

If you want to use the Miami data, download `US_Accidents_March23.csv` and place it in the project folder next to `app.py`.

## Run The Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What The App Does

- Finds the safest route
- Finds the fastest route
- Shows accident hotspot clusters
- Predicts accident severity
- Works with Peshawar sample data by default

## Notes

- `dataset.zip`, `archive/`, and `__pycache__/` are not pushed
- The repo on GitHub contains only the tracked project files
