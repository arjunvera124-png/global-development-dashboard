import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go


# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="Global Development Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Did Globalization Reduce Global Inequality?")
st.markdown(
    """
    A Multi-Indicator, Multi-Decade Visual Analysis of Global Development Trajectories
    """
)

# -----------------------------------
# Load Data
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/global_development.csv")

df = load_data()

# -----------------------------------
# Sidebar Filters
# -----------------------------------

st.sidebar.header("Filters")

year = st.sidebar.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    2020
)

selected = df[df["year"] == year].copy()

# -----------------------------------
# Tabs
# -----------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Bubble Chart",
    "🌍 World Map",
    "🔍 PCA + Clustering",
    "🔄 Sankey",
    "📊 Parallel Coordinates",
    "🌲 Feature Importance"
])

# ===================================
# TAB 1: Bubble Chart
# ===================================

with tab1:

    st.header("Global Development Overview")

    fig = px.scatter(
        selected,
        x="gdp_per_capita",
        y="life_expectancy",
        size="population",
        color="gini",
        hover_name="country",
        size_max=50,
        color_continuous_scale="Viridis",
        log_x=True,
        title=f"GDP vs Life Expectancy ({year})"
    )

    fig.update_layout(height=700)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ===================================
# TAB 2: World Map
# ===================================

# ===================================
# TAB 2: World Map
# ===================================

with tab2:

    st.header("Global Geographic Distribution")

    metric = st.selectbox(
        "Select Indicator",
        [
            "gdp_per_capita",
            "life_expectancy",
            "gini",
            "internet_users",
            "child_mortality",
            "co2_per_capita"
        ],
        key="map_metric"
    )

    map_df = selected.copy()

    # Convert ISO-3 codes to uppercase
    map_df["country_code"] = (
        map_df["country_code"]
        .astype(str)
        .str.upper()
    )

    fig_map = px.choropleth(
        map_df,
        locations="country_code",
        locationmode="ISO-3",
        color=metric,
        hover_name="country",
        hover_data={
            metric: ":,.2f"
        },
        color_continuous_scale="Viridis",
        projection="natural earth",
        title=f"{metric.replace('_', ' ').title()} ({year})"
    )

    fig_map.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# ===================================
# TAB 3: PCA + KMeans
# ===================================

with tab3:

    st.header("Development Archetypes")

    cluster_df = selected[
        [
            "country",
            "gdp_per_capita",
            "life_expectancy",
            "child_mortality",
            "co2_per_capita",
            "internet_users",
            "gini"
        ]
    ].dropna()

    features = [
        "gdp_per_capita",
        "life_expectancy",
        "child_mortality",
        "co2_per_capita",
        "internet_users",
        "gini"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(
        cluster_df[features]
    )

    n_clusters = st.slider(
        "Number of Clusters",
        2,
        8,
        4
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cluster_df["cluster"] = kmeans.fit_predict(
        X_scaled
    )

    pca = PCA(n_components=2)

    components = pca.fit_transform(
        X_scaled
    )

    cluster_df["PC1"] = components[:, 0]
    cluster_df["PC2"] = components[:, 1]

    fig_pca = px.scatter(
    cluster_df,
    x="PC1",
    y="PC2",
    color=cluster_df["cluster"].astype(str),
    hover_name="country",
    title=f"Development Clusters and PCA Biplot ({year})"
    )

    # Feature Loadings
    loadings = pca.components_.T

    scale_factor = 4

    for i, feature in enumerate(features):

        fig_pca.add_trace(
            go.Scatter(
                x=[0, loadings[i, 0] * scale_factor],
                y=[0, loadings[i, 1] * scale_factor],
                mode="lines+text",
                line=dict(
                    color="red",
                    width=2
                ),
                text=["", feature],
                textposition="top center",
                showlegend=False,
                hoverinfo="text",
                name=feature
            )
        )

    fig_pca.update_layout(
        height=700,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    st.write(
        f"Variance Explained: "
        f"PC1 = {pca.explained_variance_ratio_[0]:.2%}, "
        f"PC2 = {pca.explained_variance_ratio_[1]:.2%}"
    )
    st.markdown("""
### Interpreting the Biplot

- Countries located near each other have similar development profiles.
- Red arrows represent the contribution of indicators to the principal components.
- Longer arrows indicate stronger influence on the PCA dimensions.
- Countries aligned with an arrow tend to exhibit higher values of that indicator.
""")

# ===================================
# TAB 4: Sankey Diagram
# ===================================

with tab4:

    st.header("Economic Development and Inequality Flows")

    sankey_df = selected[
        [
            "country",
            "gdp_per_capita",
            "gini"
        ]
    ].dropna()

    # Categorize GDP
    sankey_df["GDP Category"] = pd.qcut(
        sankey_df["gdp_per_capita"],
        4,
        labels=[
            "Low GDP",
            "Lower-Middle GDP",
            "Upper-Middle GDP",
            "High GDP"
        ]
    )

    # Categorize Gini
    sankey_df["Gini Category"] = pd.qcut(
        sankey_df["gini"],
        4,
        labels=[
            "Low Inequality",
            "Moderate Inequality",
            "High Inequality",
            "Very High Inequality"
        ]
    )

    # Count flows
    flow_df = (
        sankey_df
        .groupby(
            ["GDP Category", "Gini Category"],
            observed=False
        )
        .size()
        .reset_index(name="count")
    )

    # Define nodes
    labels = [
        "Low GDP",
        "Lower-Middle GDP",
        "Upper-Middle GDP",
        "High GDP",
        "Low Inequality",
        "Moderate Inequality",
        "High Inequality",
        "Very High Inequality"
    ]

    label_to_index = {
        label: idx
        for idx, label in enumerate(labels)
    }

    source = flow_df["GDP Category"].map(
        label_to_index
    )

    target = flow_df["Gini Category"].map(
        label_to_index
    )

    value = flow_df["count"]

    fig_sankey = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(
                        color="black",
                        width=0.5
                    ),
                    label=labels
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )
        ]
    )

    fig_sankey.update_layout(
        title=f"GDP Categories vs Inequality Categories ({year})",
        font_size=12,
        height=700
    )

    st.plotly_chart(
        fig_sankey,
        use_container_width=True
    )

    st.markdown("""
    **Interpretation:**
    
    - Countries are grouped into GDP quartiles.
    - Countries are also grouped into inequality quartiles.
    - The Sankey diagram shows how economic development relates to inequality levels.
    """)

# ===================================
# TAB 5: Parallel Coordinates
# ===================================

with tab5:

    st.header("Multi-Dimensional Country Comparison")

    parallel_df = selected[
        [
            "country",
            "gdp_per_capita",
            "life_expectancy",
            "child_mortality",
            "co2_per_capita",
            "internet_users",
            "gini"
        ]
    ].dropna()

    if len(parallel_df) > 20:

        sample_size = st.slider(
            "Number of Countries",
            20,
            min(100, len(parallel_df)),
            50,
            key="parallel_slider"
        )

        parallel_df = parallel_df.sample(
            sample_size,
            random_state=42
        )

        fig_parallel = px.parallel_coordinates(
            parallel_df,
            dimensions=[
                "gdp_per_capita",
                "life_expectancy",
                "child_mortality",
                "co2_per_capita",
                "internet_users",
                "gini"
            ],
            color="gini",
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Comparing Countries Across Multiple Development Indicators"
        )

        st.plotly_chart(
            fig_parallel,
            use_container_width=True
        )

    else:

        st.warning(
            "Not enough data available."
        )

# ===================================
# TAB 6: Feature Importance
# ===================================

# ===================================
# TAB 6: Feature Importance
# ===================================

with tab6:

    st.header("Drivers of Life Expectancy")

    rf_df = selected[
        [
            "gdp_per_capita",
            "child_mortality",
            "co2_per_capita",
            "internet_users",
            "gini",
            "life_expectancy"
        ]
    ].dropna()

    X = rf_df[
        [
            "gdp_per_capita",
            "child_mortality",
            "co2_per_capita",
            "internet_users",
            "gini"
        ]
    ]

    y = rf_df["life_expectancy"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=True
    )

    fig_rf = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance for Predicting Life Expectancy"
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

    st.dataframe(
        importance.sort_values(
            "Importance",
            ascending=False
        )
    )

    st.markdown("---")

st.header("Key Insights")

st.markdown("""
### Economic Development and Health

- Countries with higher GDP per capita generally exhibit higher life expectancy.

### Inequality Persists

- High-income countries do not necessarily have low Gini coefficients.
- Globalization appears to improve living standards while inequality remains unevenly distributed.

### Development Archetypes

- PCA and clustering reveal distinct groups of countries sharing similar development characteristics.

### Digital Divide

- Internet adoption strongly correlates with overall development indicators.

### Environmental Trade-offs

- Higher development levels are often associated with increased CO₂ emissions.
""")