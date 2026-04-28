import pandas as pd
import unicodedata

def clean_player_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name
#Remove duplicates in dataset
def clean_df(df):
    # If minutes column exists, keep the row with the highest MP for duplicate players
    if "MP" in df.columns:
        df = df.sort_values(by="MP", ascending=False)

    # Keep only one row per player
    if "Player" in df.columns:
        df = df.drop_duplicates(subset="Player", keep="first")

    return df


def load_and_prepare(per_game_path, advanced_path):
    # Load both CSV files
    per_game = pd.read_csv(per_game_path)
    advanced = pd.read_csv(advanced_path)

    per_game["Player"] = per_game["Player"].apply(clean_player_name)
    advanced["Player"] = advanced["Player"].apply(clean_player_name)
    # Clean each dataframe
    per_game = clean_df(per_game)
    advanced = clean_df(advanced)

    # Merge datasets for each player
    df = pd.merge(per_game, advanced, on="Player", suffixes=("_pg", "_adv"))
    
    # Only use selected states
    selected_columns = [
        "Player",
        "PTS",
        "AST",
        "TRB",
        "FG%",
        "3P%",
        "TS%",
        "WS",
        "PER",
        "VORP",
        "BPM"
        
    ]

    # Keep only needed columns
    df = df[selected_columns]
    # Drop rows with missing values in selected columns
    df = df.dropna(subset=["PTS", "AST", "TRB", "WS"])
    # Keep top 100 players by PPG
    df = df.sort_values(by="PTS", ascending=False).head(100)
    # Split into names and feature matrix
    names = df["Player"]
    X = df.drop(columns=["Player"])
    return df, X, names


if __name__ == "__main__":
    df, X, names = load_and_prepare("perGame.csv", "Advanced.csv")

    print("Final cleaned dataset:")
    print(df)

    print("Selected features:")
    print(X)

    print("Player names:")
    print(names)