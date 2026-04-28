from read_data import load_and_prepare
from weighted_values import choose_weights, prepare_weighted_data
from computations import get_score_rankings, get_similar_players, get_player_profile
from visualize import build_radar_dataframe, visualize_player_radar, plot_player_space_3d
from optimize import get_optimized_weights_for_player


def print_menu():
    print("\nNBA Player Analysis Menu")
    print("1. Show top 10 players by optional total score")
    print("2. Show a player's profile")
    print("3. Show similar players")
    print("4. Show radar chart")
    print("5. Show 3d Vector")
    print("6. Get optimized weights")
    print("7. Exit")


def main():
    # Load and prepare the cleaned dataset
    df, X, names = load_and_prepare("perGame.csv", "Advanced.csv")

    # Get raw weights from the user
    weights_dict = choose_weights(list(X.columns))

    # Prepare normalized and weighted data
    X_norm, normalized_weights, weights_array, X_weighted = prepare_weighted_data(X, weights_dict)
    # Build radar dataframe (uses normalized data)
    radar_df = build_radar_dataframe(X_weighted, names)
    print("\nData loaded successfully.")
    print(f"Number of players: {len(names)}")

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            rankings = get_score_rankings(names, X_norm, weights_dict, top_n=10)

            print("\nTop 10 Players by Optional Total Score:")
            for i, (name, score) in enumerate(rankings, start=1):
                print(f"{i}. {name}: {score:.4f}")

        elif choice == "2":
            player_name = input("Enter player name: ").strip()
            profile = get_player_profile(df, player_name)

            if profile is None:
                print("Player not found.")
            else:
                print("\nPlayer Profile:")
                print(profile)

        elif choice == "3":
            player_name = input("Enter player name: ").strip()
            method = input("Choose method (euclidean/cosine): ").strip().lower()

            try:
                similar_players = get_similar_players(
                    player_name,
                    X_weighted,
                    names,
                    method=method,
                    top_n=5
                )

                if similar_players is None:
                    print("Player not found.")
                else:
                    print(f"\nMost similar players to {player_name}:")
                    for name, value in similar_players:
                        print(f"{name}: {value:.4f}")

            except ValueError as e:
                print(e)
        elif choice == "4":
            player_name = input("Enter player name for radar chart: ").strip()

            success = visualize_player_radar(radar_df, player_name)

            if not success:
                print("Player not found.")
        elif choice == "5":
            player_name = input("Highlight a player (or press Enter to skip): ").strip()
            plot_player_space_3d(X_weighted, names, highlight_player=player_name or None)
        elif choice == "6":
            player_name = input("Enter player name for optimized weights: ").strip()

            optimized_weights = get_optimized_weights_for_player(X_norm, names, player_name)

            if optimized_weights is None:
                print("Player not found.")
            else:
                print(f"\nOptimized weights for {player_name}:")
                for stat, weight in optimized_weights.items():
                    print(f"{stat}: {weight:.4f}")
        elif choice == "7":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()