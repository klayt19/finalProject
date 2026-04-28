import numpy as np
from scipy.optimize import minimize


def objective(weights, target_vector, average_vector):
    #Get Score for selected player and average score
    target_score = np.dot(weights, target_vector)
    average_score = np.dot(weights, average_vector)
    #Return the negative value
    return -(target_score - average_score)


def optimize_weights_for_player(X_norm, player_index):
    # Get target player's normalized vector
    target_vector = X_norm.loc[player_index].values

    # Get average player vector
    average_vector = X_norm.mean().values

    num_features = len(target_vector)

    # Start with equal weights
    initial_weights = np.ones(num_features) / num_features

    # Weights must sum to 1(Used online sources)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    ]

    #Don't allow all weight to go to highest stat
    bounds = [(0.05, 0.30) for _ in range(num_features)]
    #Used online source on how to utilize minimize function
    result = minimize(
        objective,
        initial_weights,
        args=(target_vector, average_vector),
        bounds=bounds,
        constraints=constraints
    )

    return result.x


def optimized_weights_to_dict(X_norm, optimized_weights):
    # Convert optimized weight array to dictionary
    return dict(zip(X_norm.columns, optimized_weights))


def get_optimized_weights_for_player(X_norm, names, player_name):
    # Find player index by name
    matches = names[names.str.lower() == player_name.lower()]

    if matches.empty:
        return None

    player_index = matches.index[0]
    optimized_weights = optimize_weights_for_player(X_norm, player_index)

    return optimized_weights_to_dict(X_norm, optimized_weights)