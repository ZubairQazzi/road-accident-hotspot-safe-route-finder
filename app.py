import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
except ImportError:
    st.error("networkx not installed. Run: pip install networkx")
    st.stop()

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from pipeline import load_and_build

st.set_page_config(page_title="Road Accident Hotspot & Safe Route Finder", layout="wide")
st.title("Road Accident Hotspot & Safe Route Finder")


st.sidebar.markdown("---")
city = st.sidebar.selectbox("Select City", ["Peshawar", "Miami"])

try:
    df, sd, G, n_clusters, danger_cluster, cluster_risk = load_and_build(city)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

nodes_list = sorted(G.nodes())
default_src = nodes_list.index('GT Road') if 'GT Road' in nodes_list else 0
default_tgt = nodes_list.index('Hayatabad Phase 5') if 'Hayatabad Phase 5' in nodes_list else min(5, len(nodes_list)-1)
source = st.sidebar.selectbox("Source", nodes_list, index=default_src)
target = st.sidebar.selectbox("Target", nodes_list, index=default_tgt)

tab1, tab2, tab3, tab4 = st.tabs(["Route Finder", "Hotspot Map", "ML Predictions", "Summary"])

with tab1:
    st.subheader("Safest & Fastest Route")
    try:
        safest_path = nx.dijkstra_path(G, source, target, weight='risk')
        fastest_path = nx.dijkstra_path(G, source, target, weight='distance')
        s_risk = sum(G[safest_path[i]][safest_path[i+1]]['risk'] for i in range(len(safest_path)-1))
        f_risk = sum(G[fastest_path[i]][fastest_path[i+1]]['risk'] for i in range(len(fastest_path)-1))
        s_dist = sum(G[safest_path[i]][safest_path[i+1]]['distance'] for i in range(len(safest_path)-1))
        f_dist = sum(G[fastest_path[i]][fastest_path[i+1]]['distance'] for i in range(len(fastest_path)-1))

        col1, col2, col3 = st.columns(3)
        col1.metric("Safest Risk", f"{s_risk:.1f}")
        col2.metric("Fastest Risk", f"{f_risk:.1f}")
        reduction = ((f_risk - s_risk)/f_risk*100) if f_risk > 0 else 0
        col3.metric("Risk Reduction", f"{reduction:.1f}%")

        st.markdown(f"**Safest:** {' -> '.join(safest_path)} ({s_dist:.2f} km)")
        st.markdown(f"**Fastest:** {' -> '.join(fastest_path)} ({f_dist:.2f} km)")

        if HAS_FOLIUM:
            pos = nx.get_node_attributes(G, 'pos')
            center_lat = np.mean([v[1] for v in pos.values()])
            center_lng = np.mean([v[0] for v in pos.values()])
            m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='CartoDB positron')
            for i in range(len(safest_path)-1):
                p1, p2 = pos[safest_path[i]], pos[safest_path[i+1]]
                folium.PolyLine([(p1[1],p1[0]),(p2[1],p2[0])], color='blue', weight=5).add_to(m)
            for i in range(len(fastest_path)-1):
                p1, p2 = pos[fastest_path[i]], pos[fastest_path[i+1]]
                folium.PolyLine([(p1[1],p1[0]),(p2[1],p2[0])], color='red', weight=3, dash_array='10').add_to(m)
            folium.Marker([pos[source][1], pos[source][0]], popup=source, icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([pos[target][1], pos[target][0]], popup=target, icon=folium.Icon(color='red')).add_to(m)
            st_folium(m, height=500, use_container_width=True)
        else:
            st.info("Install folium and streamlit-folium for interactive maps.")
    except nx.NetworkXNoPath:
        st.error("No path exists between selected nodes.")
    except Exception as e:
        st.error(f"Routing error: {e}")

with tab2:
    st.subheader("K-Means Cluster Map")
    try:
        if HAS_FOLIUM:
            pos = nx.get_node_attributes(G, 'pos')
            center_lat = np.mean([v[1] for v in pos.values()])
            center_lng = np.mean([v[0] for v in pos.values()])
            hm = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles='CartoDB positron')
            colours = ['red','blue','green','purple','orange','darkred','lightred','beige']
            for _, row in sd.iterrows():
                cid = int(row['cluster'])
                is_danger = (cid == danger_cluster)
                folium.CircleMarker(
                    location=[row['avg_lat'], row['avg_lng']], radius=8 if is_danger else 5,
                    color=colours[cid % len(colours)], fill=True, fill_opacity=0.8,
                    popup=f"{row['Street']}<br>Cluster {cid}<br>Accidents: {row['accident_count']}"
                ).add_to(hm)
            st_folium(hm, height=500, use_container_width=True)

        if go:
            st.subheader("Cluster Risk Ranking")
            fig = go.Figure(go.Bar(
                x=[f"Cluster {i}" for i in cluster_risk.index], y=cluster_risk.values,
                marker_color=['#B71C1C' if i == danger_cluster else '#1565C0' for i in cluster_risk.index],
                text=[f"{v:.0f}" for v in cluster_risk.values], textposition='auto'))
            fig.update_layout(title='Mean Accident Count per Cluster', height=350)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Cluster map error: {e}")

with tab3:
    st.subheader("ML Predictions")
    st.markdown("**Random Forest Severity Prediction**")
    col1, col2, col3 = st.columns(3)
    test_lat = col1.number_input("Latitude", value=float(sd['avg_lat'].mean()), format="%.4f")
    test_lng = col2.number_input("Longitude", value=float(sd['avg_lng'].mean()), format="%.4f")
    test_dist = col3.number_input("Distance (mi)", value=0.3, format="%.2f")
    if st.button("Predict Severity"):
        try:
            import pickle
            with open('models/rf_model.pkl', 'rb') as f:
                rf = pickle.load(f)
            pred = rf.predict(np.array([[test_lat, test_lng, test_dist, 0]]))[0]
            sev_labels = {1:'Minor (1)', 2:'Moderate (2)', 3:'Serious (3)', 4:'Severe (4)'}
            st.metric("Predicted Severity", sev_labels.get(pred, str(pred)))
        except Exception as e:
            st.warning(f"Model not available: {e}")

with tab4:
    st.subheader("Project Summary")
    st.markdown(f"**City:** {city} | **Streets:** {G.number_of_nodes()} | **Connections:** {G.number_of_edges()}")
    st.markdown(f"**Total records:** {len(df)} | **Clusters:** {n_clusters} | **Danger cluster:** {danger_cluster}")
    st.dataframe(sd[['Street','accident_count','cluster']].sort_values('accident_count', ascending=False).head(20))
