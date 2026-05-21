"""
Loads data from .csv & parses dates, returns cleaned data frames
Each method loads data from 1 csv
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent


def load_country_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "ebola_country_yearly.csv")
    df["year"] = df["year"].astype(int)
    return df


def load_monthly_trends() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "ebola_monthly_trends.csv")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.sort_values(["country", "date"]).reset_index(drop=True)


def load_outbreaks() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "ebola_outbreaks.csv",
        parse_dates=["start_date", "end_date"],
    )
    df["who_emergency"] = df["who_emergency_status"].str.strip().eq("Yes")
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    return df


def load_outbreak_timeline() -> pd.DataFrame:
    # `cases`, `deaths`, `fatality_rate` contain literal strings like "ongoing"
    # and "emergency" for the active 2026 cluster - coerce so numerics survive.
    df = pd.read_csv(DATA_DIR / "ebola_outbreak_timeline.csv")
    for col in ("cases", "deaths", "fatality_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "ebola_clinical.csv")
    df["symptom_list"] = (
        df["symptoms"].str.split(";").apply(lambda xs: [s.strip() for s in xs])
    )
    df["deceased"] = df["outcome"].eq("Deceased").astype(int)
    return df


def load_transmission_factors() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "ebola_transmission_factors.csv")
    impact_order = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    df["impact_rank"] = df["impact_level"].map(impact_order)
    return df


def load_virus_species() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ebola_virus_species.csv")


def load_virus_facts() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ebola_virus_facts.csv")


def load_master() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ebola_master.csv")


def load_data_dictionary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "data_dictionary.csv")


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "country_yearly": load_country_yearly(),
        "monthly_trends": load_monthly_trends(),
        "outbreaks": load_outbreaks(),
        "outbreak_timeline": load_outbreak_timeline(),
        "clinical": load_clinical(),
        "transmission_factors": load_transmission_factors(),
        "virus_species": load_virus_species(),
        "virus_facts": load_virus_facts(),
        "master": load_master(),
        "data_dictionary": load_data_dictionary(),
    }


if __name__ == "__main__":
    for name, df in load_all().items():
        print(f"{name:22s} shape={df.shape}")
