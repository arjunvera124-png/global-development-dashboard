import pandas as pd

df = pd.read_csv("data/processed/global_development.csv")

print(df["country_code"].head(20))
