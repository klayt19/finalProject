import pandas as pd


def normalize(X):
    
    #Apply z-score normalization to each column.
    
    return (X - X.mean()) / X.std()


def get_default_weights():
    
    #Return default weights for each feature.
    
    return {
        "PTS": 1.0,
        "AST": 1.0,
        "TRB": 1.0,
        "FG%": 1.0,
        "3P%": 1.0,
        "TS%": 1.0,
        "WS": 1.0,
        "PER": 1.0,
        "VORP": 1.0,
        "BPM": 1.0
    }


def normalize_weights(weights_dict):
    
    #Normalize weights so that they sum to 1.
    
    total = sum(weights_dict.values())

    if total == 0:
        raise ValueError("At least one weight must be greater than 0.")

    normalized = {}
    for key in weights_dict:
        normalized[key] = weights_dict[key] / total

    return normalized


def get_user_weights(features):
    
    #Ask the user to enter weights for each feature.
    weights = {}

    print("\nEnter a weight for each stat.")
    print("Use 0 to ignore a stat, or larger numbers to emphasize it more.")
    print("Example: 0 = ignore, 1 = low, 5 = very important\n")

    for feature in features:
        while True:
            try:
                value = float(input(f"Weight for {feature}: "))
                if value < 0:
                    print("Please enter a nonnegative number.")
                    continue
                weights[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    return weights  


def choose_weights(features):

    #Prompt user to choose default weights or custom weights.
    
    default_weights = get_default_weights()

    print("\nWould you like to use default weights or choose your own?")
    print("1. Use default weights")
    print("2. Enter custom weights")

    while True:
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return default_weights  

        elif choice == "2":
            return get_user_weights(features)

        else:
            print("Invalid choice. Please enter 1 or 2.")


def weights_to_array(X, weights_dict):
    
    #Convert weight dictionary into a list/array matching X column order.
    
    return [weights_dict[col] for col in X.columns]


def apply_weights(X, weights):
    
    #Apply weights to the normalized dataframe.
    
    return X * weights


def prepare_weighted_data(X, weights_dict):
   
    # Normalize stats
    X_norm = normalize(X)

    # Normalize weights for vector space
    normalized_weights = normalize_weights(weights_dict)

    # Convert weights into correct order
    weights_array = weights_to_array(X_norm, normalized_weights)

    # Apply weights
    X_weighted = apply_weights(X_norm, weights_array)

    return X_norm, normalized_weights, weights_array, X_weighted