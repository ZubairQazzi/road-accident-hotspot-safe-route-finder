import pandas as pd
import numpy as np
import networkx as nx
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Try sklearn, if broken pyarrow DLL just skip clustering
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@st.cache_data
def load_and_build(city='Peshawar'):
    if city == 'Miami':
        try:
            df = pd.read_csv('US_Accidents_March23.csv', nrows=100000, engine='python',
                             usecols=['ID','Severity','Start_Lat','Start_Lng','Street','City','Distance(mi)'])
            df = df[df['City'] == 'Miami'].copy()
            df = df.dropna(subset=['Start_Lat','Start_Lng','Street'])
            df = df[(df['Start_Lat'].between(24,27)) & (df['Start_Lng'].between(-81,-79))]
        except Exception:
            st.error("Miami CSV not found or too large. Switching to Peshawar.")
            city = 'Peshawar'
            df = pd.read_csv('Peshawar_Accidents_Synthetic.csv')
    else:
        df = pd.read_csv('Peshawar_Accidents_Synthetic.csv')

    sd = df.groupby('Street').agg(
        accident_count=('ID','count'),
        avg_lat=('Start_Lat','mean'),
        avg_lng=('Start_Lng','mean')
    ).reset_index()

    if city == 'Miami':
        sd = sd[sd['accident_count'] >= 1].head(200).reset_index(drop=True)
        radius = 8.0
    else:
        sd = sd.reset_index(drop=True)
        radius = 5.0

    G = nx.Graph()
    for _, row in sd.iterrows():
        G.add_node(row['Street'], pos=(row['avg_lng'], row['avg_lat']), risk=row['accident_count'])

    for i, r1 in sd.iterrows():
        for j, r2 in sd.iterrows():
            if i < j:
                d = np.sqrt((r1['avg_lat']-r2['avg_lat'])**2 + (r1['avg_lng']-r2['avg_lng'])**2) * 111
                if d <= radius:
                    rw = round((r1['accident_count']+r2['accident_count'])/2, 2)
                    G.add_edge(r1['Street'], r2['Street'], distance=round(d,2), risk=rw)

    n_clusters = 5 if city == 'Miami' else 4
    danger_cluster = 0

    if HAS_SKLEARN:
        features = sd[['avg_lat','avg_lng','accident_count']].values
        X_scaled = StandardScaler().fit_transform(features)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        sd['cluster'] = km.fit_predict(X_scaled)
    else:
        # Simple fallback: assign clusters by accident count quantiles
        sd['cluster'] = pd.qcut(sd['accident_count'], q=n_clusters, labels=False, duplicates='drop')

    cluster_risk = sd.groupby('cluster')['accident_count'].mean().sort_values(ascending=False)
    danger_cluster = cluster_risk.index[0]

    cluster_map = dict(zip(sd['Street'], sd['cluster']))
    for node in G.nodes():
        G.nodes[node]['cluster'] = cluster_map.get(node, -1)

    return df, sd, G, n_clusters, danger_cluster, cluster_risk
