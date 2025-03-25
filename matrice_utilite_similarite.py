import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement des données
print("Chargement des données MovieLens...")
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

print(f"Nombre d'utilisateurs: {ratings['userId'].nunique()}")
print(f"Nombre de films: {ratings['movieId'].nunique()}")
print(f"Nombre d'évaluations: {len(ratings)}")

# 2. Création de la matrice d'utilité
print("\nCréation de la matrice d'utilité...")
# Pivot pour créer la matrice utilisateurs (lignes) x films (colonnes)
utility_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Afficher les dimensions de la matrice
print(f"Dimensions de la matrice d'utilité: {utility_matrix.shape}")

# Afficher un aperçu de la matrice d'utilité
print("\nAperçu de la matrice d'utilité:")
print(utility_matrix.iloc[:5, :5])  # 5 premiers utilisateurs et 5 premiers films

# Calculer le pourcentage de valeurs manquantes
missing_values = utility_matrix.isna().sum().sum()
total_values = utility_matrix.size
print(f"Pourcentage de valeurs manquantes: {missing_values / total_values * 100:.2f}%")

# Définition de la fonction améliorée pour trouver les paires d'utilisateurs
def find_users_with_most_common_ratings(utility_matrix, min_common=10):
    user_pairs = []

    # Sélectionner les utilisateurs les plus actifs
    active_users = utility_matrix.count(axis=1).sort_values(ascending=False)[:100].index

    # Comparer tous les utilisateurs actifs entre eux
    for i, user1 in enumerate(active_users):
        user1_ratings = utility_matrix.loc[user1]

        # Comparer avec tous les autres utilisateurs actifs
        for j, user2 in enumerate(active_users[i+1:]):
            user2_ratings = utility_matrix.loc[user2]

            # Films notés par les deux utilisateurs
            common_movies = user1_ratings.notna() & user2_ratings.notna()
            common_count = common_movies.sum()

            if common_count >= min_common:
                user_pairs.append((user1, user2, common_count))

    # Trier par nombre de films en commun (décroissant)
    user_pairs.sort(key=lambda x: x[2], reverse=True)
    return user_pairs

# Rechercher des utilisateurs avec plus de films en commun
print("\nRecherche d'utilisateurs avec plus de films en commun...")
user_pairs = find_users_with_most_common_ratings(utility_matrix, min_common=20)

if user_pairs:
    # Trier par nombre de films en commun
    best_pair = user_pairs[0]
    print(f"Meilleure paire: utilisateurs {best_pair[0]} et {best_pair[1]} avec {best_pair[2]} films en commun")
    print(f"Nombre total de paires trouvées: {len(user_pairs)}")

    # Utiliser cette paire pour l'analyse
    user1_id = best_pair[0]
    user2_id = best_pair[1]
else:
    print("Aucune paire d'utilisateurs n'a suffisamment de films en commun. Utilisation des utilisateurs par défaut.")
    # En cas d'échec, utiliser les utilisateurs par défaut
    user1_id = 1
    # Trouver un autre utilisateur qui a noté plusieurs des mêmes films
    common_movies = utility_matrix.loc[user1_id].dropna()
    other_users = utility_matrix.loc[:, common_movies.index].dropna(how='all')
    other_users = other_users.index.tolist()
    other_users.remove(user1_id) if user1_id in other_users else None
    user2_id = other_users[0] if other_users else 2  # En cas de problème, on utilise l'ID 2

print(f"\nAnalyse de similarité entre les utilisateurs {user1_id} et {user2_id}...")

# Extraire les évaluations des deux utilisateurs
user1_ratings = utility_matrix.loc[user1_id]
user2_ratings = utility_matrix.loc[user2_id]

# Trouver les films évalués par les deux utilisateurs
common_movies = user1_ratings.notna() & user2_ratings.notna()
user1_common = user1_ratings[common_movies]
user2_common = user2_ratings[common_movies]

# Calcul de la similarité cosinus
def cosine_sim(x, y):
    # Gestion des valeurs NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    return 1 - cosine(x[mask], y[mask])

# Calcul de la corrélation de Pearson amélioré
def pearson_sim(x, y):
    # Gestion des valeurs NaN
    mask = ~np.isnan(x) & ~np.isnan(y)

    # Vérifier qu'il y a suffisamment de points pour une corrélation significative
    if np.sum(mask) < 3:  # Au moins 3 points communs
        return np.nan

    x_valid = x[mask]
    y_valid = y[mask]

    # Vérifier si l'écart-type est suffisant pour éviter la division par zéro
    if np.std(x_valid) < 1e-8 or np.std(y_valid) < 1e-8:
        return np.nan

    return np.corrcoef(x_valid, y_valid)[0, 1]

# Calcul des similarités
cosine_similarity_users = cosine_sim(user1_common, user2_common)
pearson_similarity_users = pearson_sim(user1_common, user2_common)

print(f"Nombre de films notés en commun: {len(user1_common)}")
print(f"Similarité cosinus entre les utilisateurs: {cosine_similarity_users:.4f}")
print(f"Corrélation de Pearson entre les utilisateurs: {pearson_similarity_users:.4f}")

# Visualisation des évaluations communes
plt.figure(figsize=(10, 6))
plt.scatter(user1_common, user2_common, alpha=0.7)
plt.plot([0, 5], [0, 5], 'r--')  # Ligne de référence
plt.xlabel(f'Évaluations de l\'utilisateur {user1_id}')
plt.ylabel(f'Évaluations de l\'utilisateur {user2_id}')
plt.title(f'Comparaison des évaluations entre utilisateurs {user1_id} et {user2_id}')
plt.grid(True)
plt.savefig('matrice/user_comparison.png')

# 4. Calcul de la similarité entre deux films
# Choisir des films populaires avec beaucoup d'évaluations
movie_ratings_count = ratings['movieId'].value_counts()
popular_movies = movie_ratings_count[movie_ratings_count > 100].index[:50]  # 50 films populaires

# Obtenir les titres de ces films
popular_movie_titles = movies[movies['movieId'].isin(popular_movies)][['movieId', 'title']]

# Sélectionner deux films populaires qui existent dans notre matrice d'utilité
movie_cols = utility_matrix.columns.intersection(popular_movies)
if len(movie_cols) >= 2:
    movie1_id = movie_cols[0]
    movie2_id = movie_cols[1]
else:
    # En cas d'absence, utiliser les deux premiers films
    movie1_id = utility_matrix.columns[0]
    movie2_id = utility_matrix.columns[1]

# Obtenir les titres des films sélectionnés
movie1_title = movies[movies['movieId'] == movie1_id]['title'].values[0]
movie2_title = movies[movies['movieId'] == movie2_id]['title'].values[0]

print(f"\nAnalyse de similarité entre les films:")
print(f"Film 1 (ID: {movie1_id}): {movie1_title}")
print(f"Film 2 (ID: {movie2_id}): {movie2_title}")

# Extraire les évaluations des deux films
movie1_ratings = utility_matrix[movie1_id]
movie2_ratings = utility_matrix[movie2_id]

# Trouver les utilisateurs qui ont évalué les deux films
common_users = movie1_ratings.notna() & movie2_ratings.notna()
movie1_common = movie1_ratings[common_users]
movie2_common = movie2_ratings[common_users]

# Calcul des similarités
cosine_similarity_movies = cosine_sim(movie1_common, movie2_common)
pearson_similarity_movies = pearson_sim(movie1_common, movie2_common)

print(f"Nombre d'utilisateurs ayant évalué les deux films: {len(movie1_common)}")
print(f"Similarité cosinus entre les films: {cosine_similarity_movies:.4f}")
print(f"Corrélation de Pearson entre les films: {pearson_similarity_movies:.4f}")

# Visualisation des évaluations communes
plt.figure(figsize=(10, 6))
plt.scatter(movie1_common, movie2_common, alpha=0.7)
plt.plot([0, 5], [0, 5], 'r--')  # Ligne de référence
plt.xlabel(f'Évaluations de {movie1_title}')
plt.ylabel(f'Évaluations de {movie2_title}')
plt.title(f'Comparaison des évaluations entre films')
plt.grid(True)
plt.savefig('matrice/movie_comparison.png')

# 5. Interprétation des résultats (version corrigée)
print("\nInterprétation des résultats:")
print("Similarité entre utilisateurs:")
if cosine_similarity_users > 0.8:
    print(f"- Les utilisateurs {user1_id} et {user2_id} ont des goûts très similaires (cosinus > 0.8)")
elif cosine_similarity_users > 0.5:
    print(f"- Les utilisateurs {user1_id} et {user2_id} ont des goûts modérément similaires (cosinus > 0.5)")
else:
    print(f"- Les utilisateurs {user1_id} et {user2_id} ont des goûts différents (cosinus < 0.5)")

# Vérifier si la corrélation de Pearson est NaN
if np.isnan(pearson_similarity_users):
    print(f"- La corrélation de Pearson n'a pas pu être calculée (données insuffisantes)")
elif pearson_similarity_users > 0.7:
    print(f"- Forte corrélation positive dans leurs évaluations (Pearson > 0.7)")
elif pearson_similarity_users > 0.3:
    print(f"- Corrélation positive modérée dans leurs évaluations (Pearson > 0.3)")
elif pearson_similarity_users > -0.3:
    print(f"- Faible corrélation dans leurs évaluations (Pearson entre -0.3 et 0.3)")
else:
    print(f"- Corrélation négative dans leurs évaluations (Pearson < -0.3)")

print("\nSimilarité entre films:")
if cosine_similarity_movies > 0.8:
    print(f"- Les films ont des profils d'évaluation très similaires (cosinus > 0.8)")
elif cosine_similarity_movies > 0.5:
    print(f"- Les films ont des profils d'évaluation modérément similaires (cosinus > 0.5)")
else:
    print(f"- Les films ont des profils d'évaluation différents (cosinus < 0.5)")

if np.isnan(pearson_similarity_movies):
    print(f"- La corrélation de Pearson n'a pas pu être calculée pour les films (données insuffisantes)")
elif pearson_similarity_movies > 0.7:
    print(f"- Les films sont fortement appréciés par les mêmes utilisateurs (Pearson > 0.7)")
elif pearson_similarity_movies > 0.3:
    print(f"- Les films sont modérément appréciés par les mêmes utilisateurs (Pearson > 0.3)")
elif pearson_similarity_movies > -0.3:
    print(f"- Faible corrélation entre les évaluations des films (Pearson entre -0.3 et 0.3)")
else:
    print(f"- Les utilisateurs qui aiment un film ont tendance à ne pas aimer l'autre (Pearson < -0.3)")