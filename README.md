# Road Accident Hotspot & Safe Route Finder

An interactive civic AI project that analyzes accident-prone streets, highlights hotspot clusters, and compares the **safest** and **fastest** routes between two locations.

This repository contains:

- A **Streamlit web app** for exploring routes, hotspot maps, and severity predictions.
- A **data pipeline** that builds a street-level graph from accident records.
- A **Jupyter notebook** documenting the end-to-end analysis and machine learning experiments.
- Pretrained **Random Forest** and **ANN** model artifacts used during the project.

## What Problem This Project Solves

Road navigation tools usually optimize for speed or distance, but they do not explain whether a route passes through accident-heavy areas. This project adds a safety layer by:

- Aggregating accidents at the street level
- Turning streets into a connected graph
- Assigning each street and road segment a risk score
- Finding both the fastest and the safest route with graph algorithms
- Highlighting dangerous accident clusters on a map
- Predicting accident severity from location-based features

## Main Features

- **Safest vs fastest routing** using Dijkstra's algorithm on different edge weights
- **Accident hotspot detection** through K-Means clustering
- **Interactive map views** with Folium inside Streamlit
- **Severity prediction** using a pretrained Random Forest model
- **Two data modes**
  - `Peshawar`: uses the included synthetic dataset and works out of the box
  - `Miami`: uses a subset of the Kaggle US Accidents dataset if you provide it locally

## How It Works

### 1. Data loading

The app loads accident data for the selected city:

- `Peshawar` reads `Peshawar_Accidents_Synthetic.csv`
- `Miami` tries to read `US_Accidents_March23.csv`

If the Miami file is not available, the app automatically falls back to the Peshawar dataset so the interface still works.

### 2. Street-level aggregation

Accidents are grouped by `Street`, and the pipeline computes:

- accident count per street
- average latitude per street
- average longitude per street

This creates one summarized record for each street.

### 3. Graph construction

Each street becomes a **node** in a NetworkX graph. Streets that are geographically close are connected by an **edge**.

Each edge stores:

- `distance`: approximate physical distance between streets
- `risk`: average accident count of the two connected streets

This graph is the foundation for route selection.

### 4. Route optimization

The app calculates two route types:

- **Fastest route**: minimizes total distance
- **Safest route**: minimizes cumulative risk

Both routes are computed with Dijkstra's algorithm and displayed side by side so users can compare the safety tradeoff.

### 5. Clustering and hotspot analysis

The pipeline clusters streets with K-Means using:

- average latitude
- average longitude
- accident count

Clusters are ranked by their mean accident count, and the highest-risk cluster is labeled as the **danger cluster**.

If `scikit-learn` is unavailable, the code falls back to quantile-based grouping so the app can still run.

### 6. Machine learning prediction

The app includes a **Random Forest classifier** that predicts accident severity from:

- latitude
- longitude
- distance
- cluster indicator

The repository also contains an ANN model artifact from the notebook experimentation phase.

## Project Structure

```text
.
+-- app.py                           # Streamlit interface
+-- pipeline.py                      # Data loading, graph building, clustering logic
+-- Final_Project_Module3.ipynb      # Full notebook analysis and experiments
+-- Peshawar_Accidents_Synthetic.csv # Included demo dataset
+-- LINK_DATASET                     # Reference link to the Kaggle source dataset
+-- models/
|   +-- rf_model.pkl                 # Random Forest severity classifier
|   +-- ann_model.pkl                # ANN route danger model artifact
+-- requirements.txt                 # Python dependencies
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dataset Notes

### Included in this repository

- `Peshawar_Accidents_Synthetic.csv`

This file is small and intended for quick testing, demos, and project review.

### Not included in this repository

The full US accidents dataset is intentionally **not committed** because it is too large for a normal GitHub repository.

Source:

- Kaggle: [US Accidents Dataset](https://kaggle.com/datasets/sobhanmoosavi/us-accidents)

If you want to enable Miami mode properly:

1. Download the dataset from Kaggle.
2. Extract the file named `US_Accidents_March23.csv`.
3. Place it in the project root directory next to `app.py`.

## How To Use the App

1. Launch the Streamlit app.
2. Select a city from the sidebar.
3. Choose a source street and a target street.
4. Review the generated safest and fastest routes.
5. Open the hotspot map to inspect cluster risk.
6. Use the prediction tab to estimate severity for a custom location.
7. Check the summary tab for dataset totals and high-risk streets.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- NetworkX
- Scikit-learn
- Folium
- Plotly
- Matplotlib

## Model and Analysis Notes

- The Streamlit app currently exposes the **Random Forest** prediction workflow in the UI.
- The notebook contains the broader experimentation flow, including:
  - graph construction
  - alternate route generation
  - BFS hotspot detection
  - K-Means clustering
  - Random Forest severity classification
  - ANN-based route danger scoring
- The model files were previously trained and serialized. If your local `scikit-learn` version differs, you may see compatibility warnings while loading them.

## Current Limitations

- The Miami workflow depends on a large local CSV that is not bundled in the repo.
- Distances are estimated from latitude/longitude differences, so they are approximate rather than road-network exact.
- The ANN model is included as a project artifact, but it is not wired into the Streamlit interface yet.
- The route graph is generated from street centroids, not from a true GIS road network.

## Recommended Next Improvements

- Replace centroid distance with real road-network travel data
- Add model retraining scripts to reproduce the `.pkl` files
- Add automated tests for the pipeline and graph logic
- Add a downloadable report or route export feature
- Deploy the Streamlit app online

## Repository Hygiene

Large local files such as raw datasets and archives are excluded with `.gitignore` so the repository stays lightweight and GitHub-compatible.

## License

Add a license file if you plan to publish or reuse this project publicly.
