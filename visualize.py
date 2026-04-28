import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

def plot_player_space_3d(X_weighted, names, highlight_player=None):
    clean = X_weighted.copy()
    clean["Player"] = names.values
    clean = clean.dropna()
    names_clean = clean["Player"]
    clean = clean.drop(columns=["Player"])

    pca = PCA(n_components=3)
    coords = pca.fit_transform(clean.values)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_.T  # shape: (n_features, 3)

    plot_df = pd.DataFrame({
        'PC1': coords[:, 0],
        'PC2': coords[:, 1],
        'PC3': coords[:, 2],
        'Player': names_clean.values,
        'Color': ['Highlighted' if highlight_player and n.lower() == highlight_player.lower()
                  else 'Player' for n in names_clean]
    })

    fig = px.scatter_3d(
        plot_df,
        x='PC1', y='PC2', z='PC3',
        hover_name='Player',
        color='Color',
        color_discrete_map={'Player': 'steelblue', 'Highlighted': 'red'},
        title='NBA Players in 3D Feature Space (PCA)',
        labels={
            'PC1': f'PC1 ({explained[0]*100:.1f}%)',
            'PC2': f'PC2 ({explained[1]*100:.1f}%)',
            'PC3': f'PC3 ({explained[2]*100:.1f}%)'
        }
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))

    # Scale arrows to fit the player scatter
    scale = max(abs(coords).max(axis=0)) * 0.6

    for i, stat in enumerate(clean.columns):
        x_end = loadings[i, 0] * scale
        y_end = loadings[i, 1] * scale
        z_end = loadings[i, 2] * scale

        # Draw the arrow as a line from origin to loading
        fig.add_trace(go.Scatter3d(
            x=[0, x_end],
            y=[0, y_end],
            z=[0, z_end],
            mode='lines+text',
            line=dict(color='orange', width=4),
            text=['', stat],
            textposition='top center',
            textfont=dict(size=10, color='orange'),
            name=stat,
            showlegend=False
        ))

    fig.show()


def get_player_radar_data(radar_df, player_name):
    
    #Return the selected player's categories and values for radar plotting.
    matches = radar_df[radar_df["Player"].str.lower() == player_name.lower()]

    if matches.empty:
        return None

    player_row = matches.iloc[0]
    categories = [col for col in radar_df.columns if col != "Player"]
    values = [player_row[col] for col in categories]

    return categories, values


def plot_radar_chart(player_name, categories, values):
    
    #Plot a radar chart for one player.
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values = values + [values[0]]
    angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"{player_name} Profile Radar Chart", pad=20)

    plt.tight_layout()
    plt.show()


def visualize_player_radar(radar_df, player_name):
    
    #Find a player's radar data and plot it.
 
    radar_data = get_player_radar_data(radar_df, player_name)

    if radar_data is None:
        return False

    categories, values = radar_data
    plot_radar_chart(player_name, categories, values)
    return True


def build_radar_dataframe(X_norm, names):
    
    #Build a dataframe for radar chart plotting using normalized stats.

    radar_df = X_norm.copy()
    radar_df["Player"] = names
    radar_df = radar_df[["Player"] + list(X_norm.columns)]

    return radar_df