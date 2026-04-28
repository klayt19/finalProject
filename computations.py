import numpy as np
import pandas as pd


# Score Functions

def compute_total_score(X_clean, weights_dict):
    
    #Compute optional total player score using:
    #Score = w1*x1 + w2*x2 + ... + wn*xn
    return X_clean.mul(pd.Series(weights_dict)).sum(axis=1)


def get_score_rankings(names, X_clean, weights_dict, top_n=10):
    
    #Return top_n players ranked by optional score.
    
    scores = compute_total_score(X_clean, weights_dict)

    results = list(zip(names.tolist(), scores.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_n]


# Euclidean Distance Function
def euclidean_distance(x, y):
    #Compute Euclidean distance between two vectors.
    
    return np.linalg.norm(x - y)


def cosine_similarity(x, y):
    #Compute cosine similarity between two vectors.
    
    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    if denominator == 0:
        return 0.0

    return np.dot(x, y) / denominator



def get_player_index(names, player_name):
    
    #Return the dataframe index of a player by name.
    matches = names[names.str.lower() == player_name.lower()]

    if matches.empty:
        return None

    return matches.index[0]


def get_player_profile(df, player_name):
    
    #Return a player's cleaned dataframe row.
    matches = df[df["Player"].str.lower() == player_name.lower()]

    if matches.empty:
        return None

    return matches.iloc[0]



# Similiarity Functions
def find_similar_players_euclidean(player_index, X_weighted, names, top_n=5):
    
    #Find most similar players using Euclidean distance.
    target_vector = X_weighted.loc[player_index].values
    results = []

    for i in X_weighted.index:
        if i == player_index:
            continue

        other_vector = X_weighted.loc[i].values
        dist = euclidean_distance(target_vector, other_vector)
        results.append((names.loc[i], dist))

    results.sort(key=lambda x: x[1])
    return results[:top_n]


def find_similar_players_cosine(player_index, X_weighted, names, top_n=5):
    
    #Find most similar players using cosine similarity.
    
    target_vector = X_weighted.loc[player_index].values
    results = []

    for i in X_weighted.index:
        if i == player_index:
            continue

        other_vector = X_weighted.loc[i].values
        sim = cosine_similarity(target_vector, other_vector)
        results.append((names.loc[i], sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


def get_similar_players(player_name, X_weighted, names, method="euclidean", top_n=5):
    
    #Wrapper function to find similar players by name.
    player_index = get_player_index(names, player_name)

    if player_index is None:
        return None

    if method == "euclidean":
        return find_similar_players_euclidean(player_index, X_weighted, names, top_n)

    if method == "cosine":
        return find_similar_players_cosine(player_index, X_weighted, names, top_n)

    raise ValueError("method must be 'euclidean' or 'cosine'")