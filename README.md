
### groupe:
- DA SILVA Hugo
- DHIVERT Maxime
- ANGO Shalom
- CAPOLUNGHI Romain

# Filtrage Collaboratif et Basé sur le Contenu

## Question 1 : Matrice d’utilité et similarité
1. Téléchargez le dataset MovieLens Latest Small (Lien Kaggle).
2. Créez une matrice d’utilité à partir des données, avec les utilisateurs en lignes et les films en colonnes. Remplissez les valeurs avec les notes données par les utilisateurs.
3. Calculez la similarité entre deux utilisateurs et deux films de votre choix en utilisant :
   • Cosine Similarity
   • Pearson Corrélation Interprétez les résultats obtenus.

## output pour la question 1 : (matrice_utilite_similarite.py)

```
M2PEX_ML_TD/matrice_utilite_similarite.py 
Chargement des données MovieLens...
Nombre d'utilisateurs: 610
Nombre de films: 9724
Nombre d'évaluations: 100836

Création de la matrice d'utilité...
Dimensions de la matrice d'utilité: (610, 9724)

Aperçu de la matrice d'utilité:
movieId    1   2    3   4   5
userId                       
1        4.0 NaN  4.0 NaN NaN
2        NaN NaN  NaN NaN NaN
3        NaN NaN  NaN NaN NaN
4        NaN NaN  NaN NaN NaN
5        4.0 NaN  NaN NaN NaN
Pourcentage de valeurs manquantes: 98.30%

Recherche d'utilisateurs avec plus de films en commun...
Meilleure paire: utilisateurs 414 et 599 avec 1338 films en commun
Nombre total de paires trouvées: 4899

Analyse de similarité entre les utilisateurs 414 et 599...
Nombre de films notés en commun: 1338
Similarité cosinus entre les utilisateurs: 0.9657
Corrélation de Pearson entre les utilisateurs: 0.5108

Analyse de similarité entre les films:
Film 1 (ID: 1): Toy Story (1995)
Film 2 (ID: 32): Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
Nombre d'utilisateurs ayant évalué les deux films: 104
Similarité cosinus entre les films: 0.9637
Corrélation de Pearson entre les films: 0.1201

Interprétation des résultats:
Similarité entre utilisateurs:
- Les utilisateurs 414 et 599 ont des goûts très similaires (cosinus > 0.8)
- Corrélation positive modérée dans leurs évaluations (Pearson > 0.3)

Similarité entre films:
- Les films ont des profils d'évaluation très similaires (cosinus > 0.8)
- Faible corrélation entre les évaluations des films (Pearson entre -0.3 et 0.3)

Process finished with exit code 0
```


Question 2 : Recommandation Bas´ee sur le Contenu
1. Utilisez la méthode TF-IDF pour représenter les descriptions des films disponibles dans le dataset.
2. Construisez un système de recommandation basé sur le contenu qui recommande 5 films similaires `a un film donné.
3. Expliquez les avantages et limitations d’une approche basée sur le contenu.

## output pour la question 2 : (recommandation_basee_contenue.py)

```
M2PEX_ML_TD/recommandation_basee_contenue.py
Chargement des données MovieLens...
Nombre de films: 9742
Nombre d'évaluations: 100836
Nombre de tags: 3683

Aperçu des données de films:
movieId  ...                                       genres
0        1  ...  Adventure|Animation|Children|Comedy|Fantasy
1        2  ...                   Adventure|Children|Fantasy
2        3  ...                               Comedy|Romance
3        4  ...                         Comedy|Drama|Romance
4        5  ...                                       Comedy

[5 rows x 3 columns]

Préparation des données pour TF-IDF...

Enrichissement du contenu avec les tags...

Exemples de contenu enrichi (genres + tags):

Titre: Stanley & Iris (1990)
Genres: Drama|Romance
Tags:
Contenu combiné: Drama Romance
Popularité: 0.0061

Titre: Begin Again (2013)
Genres: Comedy|Romance
Tags:
Contenu combiné: Comedy Romance
Popularité: 0.0091

Titre: Kaspar Hauser (1993)
Genres: Drama|Mystery
Tags:
Contenu combiné: Drama Mystery
Popularité: 0.0030

Application de TF-IDF sur le contenu...
Application d'une normalisation simple (minuscules, filtrage des caractères spéciaux)
Dimensions de la matrice TF-IDF: (9742, 1621)
Vocabulaire TF-IDF (termes uniques): 1621
Premiers termes du vocabulaire: ['adventure', 'animation', 'children', 'comedy', 'fantasy', 'pixar', 'fun', 'magic', 'board', 'game'] ...

Calcul de la similarité cosinus entre les films...
Dimensions de la matrice de similarité: (9742, 9742)

==================================================
Recommandations pour : Independence Day (a.k.a. ID4) (1996)
==================================================
1. Arrival, The (1996)
   Genres: Action|Sci-Fi|Thriller
   Score de similarité: 0.9398
2. Day the Earth Stood Still, The (1951)
   Genres: Drama|Sci-Fi|Thriller
   Score de similarité: 0.8684
3. Men in Black (a.k.a. MIB) (1997)
   Genres: Action|Comedy|Sci-Fi
   Score de similarité: 0.8662
4. Astronaut's Wife, The (1999)
   Genres: Horror|Sci-Fi|Thriller
   Score de similarité: 0.8224
5. Signs (2002)
   Genres: Horror|Sci-Fi|Thriller
   Score de similarité: 0.8224


Comparaison des recommandations avec et sans tags:

Comparaison pour le film: Independence Day (a.k.a. ID4) (1996)

Recommandations avec genres + tags:
1. Arrival, The (1996) - Score: 0.9398
2. Day the Earth Stood Still, The (1951) - Score: 0.8684
3. Men in Black (a.k.a. MIB) (1997) - Score: 0.8662
4. Astronaut's Wife, The (1999) - Score: 0.8224
5. Signs (2002) - Score: 0.8224

Recommandations avec genres seulement:
1. Independence Day (a.k.a. ID4) (1996)
2. Escape from L.A. (1996)
3. Abyss, The (1989)
4. Escape from New York (1981)
5. Star Trek: First Contact (1996)

Recommandations en tenant compte de la popularité:
1. Men in Black (a.k.a. MIB) (1997)
   Similarité: 0.8662, Popularité: 0.5015
   Score combiné: 0.7568
2. Arrival, The (1996)
   Similarité: 0.9398, Popularité: 0.0669
   Score combiné: 0.6780
3. Alien (1979)
   Similarité: 0.7686, Popularité: 0.4438
   Score combiné: 0.6711
4. Signs (2002)
   Similarité: 0.8224, Popularité: 0.1915
   Score combiné: 0.6331
5. Day the Earth Stood Still, The (1951)
   Similarité: 0.8684, Popularité: 0.0760
   Score combiné: 0.6307

Recommandations diversifiées:
1. Arrival, The (1996) - Genres: Action|Sci-Fi|Thriller
2. Day the Earth Stood Still, The (1951) - Genres: Drama|Sci-Fi|Thriller
3. Men in Black (a.k.a. MIB) (1997) - Genres: Action|Comedy|Sci-Fi
4. Astronaut's Wife, The (1999) - Genres: Horror|Sci-Fi|Thriller
5. My Stepmother Is an Alien (1988) - Genres: Comedy|Romance|Sci-Fi

Visualisations des similarités sauvegardées dans 'content_based_recommendations.png'

Évaluation de la pertinence des recommandations:

Évaluation pour Interstellar (2014):
Genres originaux: {'Sci-Fi', 'IMAX'}
Précision des genres - Recommandation standard: 0.20
Précision des genres - Avec popularité: 0.10
Précision des genres - Diversifiée: 0.10

Évaluation pour Shutter Island (2010):
Genres originaux: {'Thriller', 'Mystery', 'Drama'}
Précision des genres - Recommandation standard: 0.67
Précision des genres - Avec popularité: 0.67
Précision des genres - Diversifiée: 0.73

Évaluation pour Armageddon (1998):
Genres originaux: {'Sci-Fi', 'Thriller', 'Action', 'Romance'}
Précision des genres - Recommandation standard: 0.55
Précision des genres - Avec popularité: 0.60
Précision des genres - Diversifiée: 0.55

Évaluation pour Fight Club (1999):
Genres originaux: {'Thriller', 'Action', 'Drama', 'Crime'}
Précision des genres - Recommandation standard: 0.50
Précision des genres - Avec popularité: 0.50
Précision des genres - Diversifiée: 0.45

================================================================================
AVANTAGES ET LIMITATIONS DE L'APPROCHE BASÉE SUR LE CONTENU
================================================================================

Avantages:
1. Ne nécessite pas de données d'autres utilisateurs (cold start réduit)
2. Peut recommander des éléments nouveaux ou peu populaires
3. Transparent : les recommandations sont explicables (basées sur le contenu similaire)
4. Personnalisé : recommandations adaptées aux préférences individuelles
5. Utile quand peu de données d'évaluation sont disponibles

Limitations:
1. Diversité limitée : tendance à recommander des films très similaires (sur-spécialisation)
2. Ne découvre pas les intérêts latents ou nouveaux des utilisateurs
3. Dépend fortement de la qualité et la richesse des descriptions
4. Ne prend pas en compte la qualité ou la popularité des films
5. Ne capture pas l'évolution des goûts de l'utilisateur dans le temps
6. L'enrichissement avec les tags améliore les recommandations, mais reste limité
   par rapport à des descriptions textuelles complètes ou des analyses d'images/audio.

Améliorations possibles:
1. Combiner avec un filtrage collaboratif pour une approche hybride
2. Utiliser le traitement du langage naturel plus avancé (embedding de phrases)
3. Intégrer des métadonnées supplémentaires (acteurs, réalisateurs, date)
4. Ajouter la diversification des recommandations (comme implémenté)
5. Incorporer la popularité dans le score final (comme implémenté)
6. Ajout de feedback utilisateur pour affiner les recommandations

Process finished with exit code 0
```