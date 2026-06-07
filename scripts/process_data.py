import pandas as pd
from functools import reduce
import os

# Define datasets and their output column names
DATASETS = {
    "gdp_pcap.csv": "gdp_per_capita",
    "lex.csv": "life_expectancy",
    "pop.csv": "population",
    "child_mortality_0_5_year_olds_dying_per_1000_born.csv": "child_mortality",
    "co2_pcap_cons.csv": "co2_per_capita",
    "internet_users.csv": "internet_users",
    "gini.csv": "gini"
}

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/global_development.csv"


def load_and_transform(filename, value_name):
    """Convert wide-format Gapminder data to long format."""

    filepath = os.path.join(RAW_DIR, filename)

    print(f"Processing {filename}...")

    df = pd.read_csv(filepath)

    # Keep geo and name columns
    id_vars = ["geo", "name"]

    # Identify year columns
    year_cols = [col for col in df.columns if col.isdigit()]

    # Convert wide -> long
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year",
        value_name=value_name
    )

    df_long.rename(
        columns={
            "geo": "country_code",
            "name": "country"
        },
        inplace=True
    )

    df_long["year"] = df_long["year"].astype(int)

    return df_long


def main():

    dfs = []

    for filename, value_name in DATASETS.items():
        df = load_and_transform(filename, value_name)
        dfs.append(df)

    print("Merging datasets...")

    merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["country_code", "country", "year"],
            how="outer"
        ),
        dfs
    )

    print("Handling missing values...")

    merged = merged.sort_values(
        by=["country_code", "year"]
    )

    numeric_cols = [
        "gdp_per_capita",
        "life_expectancy",
        "population",
        "child_mortality",
        "co2_per_capita",
        "internet_users",
        "gini"
    ]

    # Interpolate country-wise
    merged[numeric_cols] = (
        merged
        .groupby("country_code")[numeric_cols]
        .transform(
            lambda x: x.interpolate(
                method="linear",
                limit_direction="both"
            )
        )
    )

    # Drop rows with too much missing data
    merged = merged.dropna(
        subset=[
            "gdp_per_capita",
            "life_expectancy"
        ]
    )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDone!")
    print(f"Saved to {OUTPUT_FILE}")

    print("\nDataset shape:")
    print(merged.shape)

    print("\nMissing values:")
    print(
        merged[numeric_cols]
        .isna()
        .sum()
    )


if __name__ == "__main__":
    main()