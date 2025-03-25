import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement des données
print("Chargement des données MovieLens...")
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')
tags = pd.read_csv('data/tags.csv')  # Utilisation des tags pour enrichir le contenu

print(f"Nombre de films: {len(movies)}")
print(f"Nombre d'évaluations: {len(ratings)}")
print(f"Nombre de tags: {len(tags)}")

print("\nAperçu des données de films:")
print(movies.head())

# 2. Prétraitement pour TF-IDF
print("\nPréparation des données pour TF-IDF...")

# Traitement des genres pour TF-IDF
movies['genres_text'] = movies['genres'].str.replace('|', ' ')

# Enrichissement avec les tags (nouveauté)
print("\nEnrichissement du contenu avec les tags...")
# Regrouper tous les tags d'un même film
film_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.str.lower())).reset_index()
film_tags.columns = ['movieId', 'tags_text']

# Fusionner les tags avec les films
movies_enriched = pd.merge(movies, film_tags, on='movieId', how='left')
movies_enriched['tags_text'] = movies_enriched['tags_text'].fillna('')

# Combinaison des genres et des tags pour une représentation plus riche
movies_enriched['content'] = movies_enriched['genres_text'] + ' ' + movies_enriched['tags_text']

# Calculer un score de popularité basé sur le nombre d'évaluations
movie_popularity = ratings['movieId'].value_counts().reset_index()
movie_popularity.columns = ['movieId', 'rating_count']

# Normaliser pour avoir un score entre 0 et 1
max_ratings = movie_popularity['rating_count'].max()
movie_popularity['popularity_score'] = movie_popularity['rating_count'] / max_ratings

# Ajouter à notre DataFrame de films
movies_enriched = pd.merge(movies_enriched, movie_popularity, on='movieId', how='left')
movies_enriched['popularity_score'] = movies_enriched['popularity_score'].fillna(0)

# Afficher quelques exemples de contenu
print("\nExemples de contenu enrichi (genres + tags):")
sample_indices = np.random.randint(0, len(movies_enriched), 3)
for idx in sample_indices:
    movie = movies_enriched.iloc[idx]
    print(f"\nTitre: {movie['title']}")
    print(f"Genres: {movie['genres']}")
    print(f"Tags: {movie['tags_text'][:100] + '...' if len(movie['tags_text']) > 100 else movie['tags_text']}")
    print(f"Contenu combiné: {movie['content'][:100] + '...' if len(movie['content']) > 100 else movie['content']}")
    print(f"Popularité: {movie['popularity_score']:.4f}")

# 3. Application de TF-IDF sur le contenu enrichi
print("\nApplication de TF-IDF sur le contenu...")


# Fonction simple de normalisation sans dépendre de NLTK
def simple_normalize(text):
    """Normalisation simple: minuscules et filtre des caractères spéciaux"""
    return ' '.join([word.lower() for word in text.split() if word.isalpha()])


# Appliquer la normalisation simple
print("Application d'une normalisation simple (minuscules, filtrage des caractères spéciaux)")
movies_enriched['content_normalized'] = movies_enriched['content'].apply(simple_normalize)

# Utiliser cette version normalisée pour TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies_enriched['content_normalized'])

print(f"Dimensions de la matrice TF-IDF: {tfidf_matrix.shape}")
print(f"Vocabulaire TF-IDF (termes uniques): {len(tfidf.vocabulary_)}")
print("Premiers termes du vocabulaire:", list(tfidf.vocabulary_.keys())[:10], "...")

# 4. Calcul de la similarité entre tous les films
print("\nCalcul de la similarité cosinus entre les films...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Dimensions de la matrice de similarité: {cosine_sim.shape}")

# 5. Création du système de recommandation
indices = pd.Series(movies_enriched.index, index=movies_enriched['title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim, movies=movies_enriched, indices=indices, n=5):
    """
    Fonction pour recommander des films similaires
    """
    try:
        # Obtenir l'index du film correspondant au titre
        idx = indices[title]
    except KeyError:
        print(f"Film '{title}' non trouvé. Voici quelques titres disponibles:")
        print(movies['title'].sample(5).tolist())
        return None

    # Obtenir les scores de similarité avec tous les autres films
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Trier les films par similarité décroissante
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Récupérer les indices des films les plus similaires (exclure le film lui-même)
    sim_scores = sim_scores[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]

    # Retourner les films recommandés avec leur score de similarité
    recommendations = movies.iloc[movie_indices].copy()
    recommendations['similarity_score'] = similarities
    return recommendations[['title', 'genres', 'similarity_score']]


def get_recommendations_with_popularity(title, cosine_sim=cosine_sim, movies=movies_enriched, indices=indices, n=5,
                                        popularity_weight=0.2):
    """
    Fonction pour recommander des films en tenant compte de la popularité
    """
    try:
        # Obtenir l'index du film correspondant au titre
        idx = indices[title]
    except KeyError:
        print(f"Film '{title}' non trouvé. Voici quelques titres disponibles:")
        print(movies['title'].sample(5).tolist())
        return None

    # Obtenir les scores de similarité avec tous les autres films
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Combiner similarité et popularité
    weighted_scores = []
    for i, score in sim_scores:
        if i == idx:  # Sauter le film lui-même
            continue
        popularity = movies.iloc[i]['popularity_score']
        # Score combiné: (1-w) * similarité + w * popularité
        combined_score = (1 - popularity_weight) * score + popularity_weight * popularity
        weighted_scores.append((i, score, popularity, combined_score))

    # Trier par score combiné
    weighted_scores.sort(key=lambda x: x[3], reverse=True)
    weighted_scores = weighted_scores[:n]

    # Récupérer les infos
    result = []
    for i, sim, pop, combined in weighted_scores:
        movie = movies.iloc[i]
        result.append({
            'title': movie['title'],
            'genres': movie['genres'],
            'similarity': sim,
            'popularity': pop,
            'combined_score': combined
        })

    return pd.DataFrame(result)


def get_diverse_recommendations(title, cosine_sim=cosine_sim, movies=movies_enriched, indices=indices, n=5):
    """
    Fonction pour recommander des films similaires mais diversifiés
    """
    # Obtenir d'abord plus de recommandations que nécessaire
    try:
        idx = indices[title]
    except KeyError:
        print(f"Film '{title}' non trouvé. Voici quelques titres disponibles:")
        print(movies['title'].sample(5).tolist())
        return None

    # Obtenir les scores de similarité
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n * 3]  # Prendre plus de recommandations initiales

    # Extraire les infos des films
    movie_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]
    candidates = movies.iloc[movie_indices].copy()
    candidates['similarity_score'] = similarities

    # Diversification: sélectionner des films de genres différents
    selected = []
    selected_genres = set()

    for _, movie in candidates.iterrows():
        movie_genres = set(movie['genres'].split('|'))
        # Si le film ajoute au moins un nouveau genre, le sélectionner
        if len(movie_genres.difference(selected_genres)) > 0:
            selected.append(movie)
            selected_genres.update(movie_genres)

        # S'arrêter quand on a suffisamment de recommandations
        if len(selected) >= n:
            break

    # S'il n'y a pas assez de films diversifiés, compléter avec les meilleurs scores
    if len(selected) < n:
        remaining = n - len(selected)
        # Exclure les films déjà sélectionnés
        selected_indices = [movie.name for movie in selected]
        candidates = candidates[~candidates.index.isin(selected_indices)]
        for _, movie in candidates.head(remaining).iterrows():
            selected.append(movie)

    # Convertir en DataFrame
    result = pd.DataFrame(selected[:n])
    return result[['title', 'genres', 'similarity_score']]


# 6. Démonstration du système de recommandation
# Sélectionner quelques films populaires pour la démonstration
popular_movies_ids = ratings['movieId'].value_counts().head(100).index
popular_movie_titles = movies[movies['movieId'].isin(popular_movies_ids)]['title'].sample(5).tolist()

for title in popular_movie_titles[:1]:  # Prendre juste le premier pour l'exemple
    print(f"\n{'=' * 50}")
    print(f"Recommandations pour : {title}")
    print(f"{'=' * 50}")

    recs = get_recommendations(title)
    if recs is not None:
        for i, (_, row) in enumerate(recs.iterrows(), 1):
            print(f"{i}. {row['title']}")
            print(f"   Genres: {row['genres']}")
            print(f"   Score de similarité: {row['similarity_score']:.4f}")

# 7. Comparaison des recommandations avec et sans tags
# A. Calculer la similarité avec seulement les genres
print("\n\nComparaison des recommandations avec et sans tags:")

# Appliquer TF-IDF sur les genres seulement
tfidf_genres = TfidfVectorizer(stop_words='english')
tfidf_matrix_genres = tfidf_genres.fit_transform(movies['genres_text'])
cosine_sim_genres = cosine_similarity(tfidf_matrix_genres, tfidf_matrix_genres)
indices_genres = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# Fonction pour obtenir les recommandations basées uniquement sur les genres
def get_genre_recommendations(title, n=5):
    try:
        idx = indices_genres[title]
    except KeyError:
        print(f"Film '{title}' non trouvé.")
        return None

    sim_scores = list(enumerate(cosine_sim_genres[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]


# Choisir un film pour la comparaison
demo_film = popular_movie_titles[0]
print(f"\nComparaison pour le film: {demo_film}")

# Recommandations avec contenu enrichi (genres + tags)
print("\nRecommandations avec genres + tags:")
recs_enriched = get_recommendations(demo_film)
if recs_enriched is not None:
    for i, (_, row) in enumerate(recs_enriched.iterrows(), 1):
        print(f"{i}. {row['title']} - Score: {row['similarity_score']:.4f}")

# Recommandations avec genres seulement
print("\nRecommandations avec genres seulement:")
recs_genres = get_genre_recommendations(demo_film)
if recs_genres is not None:
    for i, (_, row) in enumerate(recs_genres.iterrows(), 1):
        print(f"{i}. {row['title']}")

# B. Recommandations avec popularité
print("\nRecommandations en tenant compte de la popularité:")
recs_pop = get_recommendations_with_popularity(demo_film, popularity_weight=0.3)
if recs_pop is not None:
    for i, (_, row) in enumerate(recs_pop.iterrows(), 1):
        print(f"{i}. {row['title']}")
        print(f"   Similarité: {row['similarity']:.4f}, Popularité: {row['popularity']:.4f}")
        print(f"   Score combiné: {row['combined_score']:.4f}")

# C. Recommandations diversifiées
print("\nRecommandations diversifiées:")
recs_div = get_diverse_recommendations(demo_film)
if recs_div is not None:
    for i, (_, row) in enumerate(recs_div.iterrows(), 1):
        print(f"{i}. {row['title']} - Genres: {row['genres']}")

# 8. Visualisation de la similarité
# Créer une visualisation pour comparer les films recommandés
plt.figure(figsize=(12, 8))

if recs_enriched is not None:
    titles = [title[:30] + '...' if len(title) > 30 else title for title in recs_enriched['title']]
    scores = recs_enriched['similarity_score']

    plt.barh(titles, scores, color='skyblue')
    plt.xlabel('Score de similarité')
    plt.title(f'Films similaires à "{demo_film[:30] + "..." if len(demo_film) > 30 else demo_film}"')
    plt.tight_layout()
    plt.savefig('recommandation/content_based_recommendations.png')
    print("\nVisualisations des similarités sauvegardées dans 'content_based_recommendations.png'")


# 9. Évaluation de la pertinence des recommandations
def evaluate_recommendations(recommendations, ground_truth_genres):
    """
    Évalue si les recommandations partagent des genres avec le film original
    """
    if recommendations is None:
        return 0

    common_genres = 0
    total_comparisons = 0

    for _, rec in recommendations.iterrows():
        rec_genres = set(rec['genres'].split('|'))
        common = len(rec_genres.intersection(ground_truth_genres))
        common_genres += common
        total_comparisons += len(ground_truth_genres)

    return common_genres / total_comparisons if total_comparisons > 0 else 0


# Évaluation pour quelques films
print("\nÉvaluation de la pertinence des recommandations:")
test_movies = ['Interstellar (2014)', 'Shutter Island (2010)', 'Armageddon (1998)', 'Fight Club (1999)']

for movie in test_movies:
    try:
        original_genres = set(movies[movies['title'] == movie]['genres'].iloc[0].split('|'))

        # Évaluer les différentes méthodes
        recs_basic = get_recommendations(movie)
        precision_basic = evaluate_recommendations(recs_basic, original_genres)

        recs_pop = get_recommendations_with_popularity(movie)
        precision_pop = evaluate_recommendations(recs_pop, original_genres)

        recs_div = get_diverse_recommendations(movie)
        precision_div = evaluate_recommendations(recs_div, original_genres)

        print(f"\nÉvaluation pour {movie}:")
        print(f"Genres originaux: {original_genres}")
        print(f"Précision des genres - Recommandation standard: {precision_basic:.2f}")
        print(f"Précision des genres - Avec popularité: {precision_pop:.2f}")
        print(f"Précision des genres - Diversifiée: {precision_div:.2f}")

    except IndexError:
        print(f"Film '{movie}' non trouvé dans la base de données")

# 10. Avantages et limitations de l'approche basée sur le contenu
print("\n" + "=" * 80)
print("AVANTAGES ET LIMITATIONS DE L'APPROCHE BASÉE SUR LE CONTENU")
print("=" * 80)

print("\nAvantages:")
print("1. Ne nécessite pas de données d'autres utilisateurs (cold start réduit)")
print("2. Peut recommander des éléments nouveaux ou peu populaires")
print("3. Transparent : les recommandations sont explicables (basées sur le contenu similaire)")
print("4. Personnalisé : recommandations adaptées aux préférences individuelles")
print("5. Utile quand peu de données d'évaluation sont disponibles")

print("\nLimitations:")
print("1. Diversité limitée : tendance à recommander des films très similaires (sur-spécialisation)")
print("2. Ne découvre pas les intérêts latents ou nouveaux des utilisateurs")
print("3. Dépend fortement de la qualité et la richesse des descriptions")
print("4. Ne prend pas en compte la qualité ou la popularité des films")
print("5. Ne capture pas l'évolution des goûts de l'utilisateur dans le temps")
print("6. L'enrichissement avec les tags améliore les recommandations, mais reste limité")
print("   par rapport à des descriptions textuelles complètes ou des analyses d'images/audio.")

print("\nAméliorations possibles:")
print("1. Combiner avec un filtrage collaboratif pour une approche hybride")
print("2. Utiliser le traitement du langage naturel plus avancé (embedding de phrases)")
print("3. Intégrer des métadonnées supplémentaires (acteurs, réalisateurs, date)")
print("4. Ajouter la diversification des recommandations (comme implémenté)")
print("5. Incorporer la popularité dans le score final (comme implémenté)")
print("6. Ajout de feedback utilisateur pour affiner les recommandations")
