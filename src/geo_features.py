import pandas as pd
from src.team_names import TEAM_RENAME

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def _split_hosts(x: str) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    # Normalise séparateurs possibles
    s = s.replace(" and ", "/").replace("&", "/")
    # Exemple: "Korea/Japan"
    parts = [p.strip() for p in s.split("/") if p.strip()]
    return parts if parts else [s]

def add_host_features(df: pd.DataFrame, worldcups: pd.DataFrame) -> pd.DataFrame:
    wc = _clean_cols(worldcups)

    # Ton fichier a "year" et "country" après normalisation
    if "year" not in wc.columns or "country" not in wc.columns:
        raise ValueError(f"WorldCups.csv doit contenir Year et Country. Colonnes trouvées: {list(worldcups.columns)}")

    host_map = wc[["year", "country"]].drop_duplicates().copy()
    host_map["host_team"] = host_map["country"].apply(_split_hosts)
    host_map = host_map.explode("host_team")
    host_map["host_team"] = host_map["host_team"].replace(TEAM_RENAME)

    out = df.copy()
    out["team"] = out["team"].replace(TEAM_RENAME)

    out = out.merge(host_map[["year", "host_team"]], on="year", how="left")

    # 1 si l’équipe = pays hôte cette année-là
    out["is_host"] = (out["team"] == out["host_team"]).fillna(False).astype(int)

    # (option) si tu veux éviter une colonne texte en plus:
    # out = out.drop(columns=["host_team"])

    return out