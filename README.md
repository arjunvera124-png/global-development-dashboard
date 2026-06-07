# 🌍 Did Globalization Reduce Global Inequality?

A Multi-Indicator, Multi-Decade Visual Analysis of Global Development Trajectories

## Project Overview

This project investigates the relationship between globalization, economic development, and inequality using multiple socio-economic indicators across countries and decades. The interactive dashboard enables users to explore patterns in global development through various visualization techniques and machine learning approaches.

### Research Question

**Did globalization reduce global inequality, or did it merely redistribute it?**

---

## Features

### 📈 Interactive Bubble Chart

* GDP per capita vs Life Expectancy
* Bubble size represents Population
* Bubble color represents Gini Index (inequality)
* Year selection for temporal analysis

### 🌍 Choropleth World Map

* Geographic distribution of indicators
* Supports multiple metrics:

  * GDP per capita
  * Life Expectancy
  * Gini Index
  * Internet Usage
  * Child Mortality
  * CO₂ Emissions per capita

### 🔍 PCA and Clustering Analysis

* Principal Component Analysis (PCA)
* K-Means clustering for identifying development archetypes
* PCA Biplot showing feature contributions

### 🔄 Sankey Diagram

* Visualizes relationships between GDP categories and inequality levels.

### 📊 Parallel Coordinates Plot

* Multi-dimensional comparison of countries across indicators.

### 🌲 Feature Importance Analysis

* Random Forest Regression used to identify the most influential indicators related to development outcomes.

---

## Dataset

Data were obtained from **Gapminder**, incorporating indicators from sources such as:

* World Bank
* United Nations
* Gapminder Foundation

Indicators used:

* GDP per capita
* Life Expectancy
* Population
* Child Mortality
* CO₂ Emissions per capita
* Internet Users (%)
* Gini Index

---

## Technologies Used

* Python 3
* Streamlit
* Pandas
* Plotly
* Scikit-learn
* NumPy

---

## Project Structure

```
global-development-dashboard/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   └── process_data.py
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd global-development-dashboard
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## Authors

**Arjun Verma**

Information Visualization Project
TU Wien

---

## Future Improvements

* Additional predictive modeling techniques
* Enhanced cross-filtering between visualizations
* Deployment to Streamlit Cloud
* More advanced temporal transition analyses
