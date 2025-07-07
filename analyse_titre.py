from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re




# Téléchargement des stopwords français à la première exécution
import nltk
nltk.data.path.append("nltk_data")


# ------------------------------------------------------------------------------
#  1. Nettoyage des titres
# ------------------------------------------------------------------------------
def nettoyer_titres(serie_titre):
    """
    Nettoie une série pandas contenant des titres :
    - Remplace les valeurs manquantes par des chaînes vides.
    - Convertit tous les caractères en minuscule.

    Exemple :
        Input : "Problème PC - Accès réseau"
        Output : "problème pc - accès réseau"
    """
    titres = serie_titre.fillna("").str.lower()
    return titres

# ------------------------------------------------------------------------------
#  2. Vectorisation TF-IDF (bag of words pondéré)
# ------------------------------------------------------------------------------
def vectoriser_titres(titres_nettoyes):
    """
    Transforme une liste de titres textuels en une matrice numérique (TF-IDF).
    Supprime les mots vides, les chiffres, mois, années.

    Retourne :
        - X : matrice TF-IDF sparse (n_samples, n_features)
        - vectoriseur : l'objet TfidfVectorizer (contenant le vocabulaire)

    Exemple :
        "vpn accès refusé réseau" → vecteur [0.2, 0.3, 0.1, 0.0, ...]
    """
    stop_words_fr = stopwords.words("french")

    # Liste personnalisée
    mots_perso = {
        "dr", "cs", "mr", "mme", "rdv","mvt","re","vers","non","oui",
        "janvier", "fevrier", "mars", "avril", "mai", "juin",
        "juillet", "aout", "septembre", "octobre", "novembre", "décembre"
    }

    chiffres = [str(i).zfill(2) for i in range(1, 32)]
    annees = [str(y) for y in range(2020, 2035)]

    mots_a_ignorer = set(stop_words_fr) | mots_perso | set(chiffres) | set(annees)

    vectoriseur = TfidfVectorizer(stop_words=list(mots_a_ignorer))
    X = vectoriseur.fit_transform(titres_nettoyes)
    return X, vectoriseur

# ------------------------------------------------------------------------------
#  3. Clustering KMeans
# ------------------------------------------------------------------------------
def creer_clusters(X, n_clusters=6):
    """
    Regroupe automatiquement les titres similaires avec l'algorithme KMeans.
    Chaque cluster contiendra des titres proches dans l’espace sémantique.

    Paramètres :
        - X : matrice TF-IDF
        - n_clusters : nombre de groupes à créer

    Retourne :
        - clusters : tableau numpy indiquant l'appartenance de chaque titre
        - modele : objet KMeans entraîné
    """
    modele = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = modele.fit_predict(X)
    return clusters, modele

# ------------------------------------------------------------------------------
#  4. Mots clés par cluster
# ------------------------------------------------------------------------------
def mots_cles_par_cluster(modele, vectoriseur, nb_mots=5):
    """
    Pour chaque cluster généré par KMeans, extrait les mots les plus représentatifs
    (ayant les poids IDF les plus élevés dans le centre du cluster).

    Exemple :
        Groupe 1 : ['vpn', 'réseau', 'accès', 'refusé', 'serveur']
    """
    mots = vectoriseur.get_feature_names_out()
    indices = modele.cluster_centers_.argsort()[:, ::-1]

    resultats = []
    for i in range(modele.n_clusters):
        top_mots = [mots[indice] for indice in indices[i, :nb_mots]]
        resultats.append(top_mots)
    return resultats

# ------------------------------------------------------------------------------
#  5. Découverte de thèmes latents (LDA)
# ------------------------------------------------------------------------------
def appliquer_lda(X, vectorizer, n_topics=6, n_mots=5):
    """
    Applique le modèle Latent Dirichlet Allocation pour découvrir les thèmes
    dominants dans les titres.

    Retourne une liste de listes de mots-clés pour chaque "topic".

    Exemple :
        Thème 1 : ['vpn', 'accès', 'réseau']
        Thème 2 : ['mot', 'passe', 'oublié']
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    termes = vectorizer.get_feature_names_out()
    themes = []
    for topic in lda.components_:
        mots_cles = [termes[j] for j in topic.argsort()[-n_mots:][::-1]]
        themes.append(mots_cles)
    return themes

# ------------------------------------------------------------------------------
#  6. Fréquence des mots (filtrés)
# ------------------------------------------------------------------------------
def plot_mots_frequents(titres, nb_mots=20):
    """
    Affiche un graphique des mots les plus fréquents dans les titres,
    après avoir filtré les stopwords, chiffres et mots courts.

    Retourne une figure matplotlib (bar chart).
    """
    mots = " ".join(titres).split()

    stop_words = set(stopwords.words("french"))
    mots_perso = {"dr","mvt"}
    chiffres = {str(i).zfill(2) for i in range(1, 32)}
    annees = {str(y) for y in range(2007, 2099)}

    a_exclure = stop_words | mots_perso | chiffres | annees

    mots_filtres = [m for m in mots if m.lower() not in a_exclure and len(m) > 2 and not m.isdigit()]

    freq = pd.Series(mots_filtres).value_counts().head(nb_mots)
    fig, ax = plt.subplots(figsize=(10, 5))
    freq.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Mots les plus fréquents (filtrés)")
    ax.set_ylabel("Fréquence")
    return fig

# ------------------------------------------------------------------------------
#  7. Calcul de l'entropie (diversité thématique)
# ------------------------------------------------------------------------------
def calculer_entropie(df, colonne="Groupe de titres"):
    """
    Calcule l'entropie (diversité) de répartition des titres dans les groupes.
    Une entropie élevée → les groupes sont équilibrés.
    Une entropie faible → certains groupes dominent.
        Afin de calculer l'entropoe d'un gorupe de clusters : H(p) = - (∑ (i) p(i) log(p(i)))

    Retourne un float ∈ [0, log(n_clusters)]
    """
    distribution = df[colonne].value_counts(normalize=True)
    return entropy(distribution)

# ------------------------------------------------------------------------------
#  8. Entraînement d’un modèle prédictif (Logistic Regression)
# ------------------------------------------------------------------------------
def entrainer_model(X, y):
    """
    Entraîne un modèle de classification supervisée pour prédire
    à quel groupe un titre appartient (à partir de ses mots).

    Retourne :
        - acc : précision
        - cm : matrice de confusion

    Exemples d’usage :
        - évaluer la cohérence des clusters
        - prédire automatiquement le thème d’un nouveau ticket
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = LogisticRegression(max_iter=1000)
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# ------------------------------------------------------------------------------
# 9.Extrait les mots utiles d’un texte en supprimant les mots vides, courts ou contenant des chiffres.
# ------------------------------------------------------------------------------
def nettoyer(texte):
    stop_words = set(stopwords.words('french'))
    mots_perso = {"mvt"}

    mots = re.findall(r'\b\w+\b', texte.lower())
    return [
        mot for mot in mots
        if mot not in stop_words | mots_perso
        and len(mot) > 2
        and not any(c.isdigit() for c in mot) 
    ]



