import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from datetime import datetime
from collections import Counter
import networkx as nx
from analyse_titre import (
    nettoyer_titres,
    vectoriser_titres,
    creer_clusters,
    mots_cles_par_cluster,
    appliquer_lda,
    calculer_entropie,
    entrainer_model,
    plot_mots_frequents,
    nettoyer
)



# Configuration de la page
st.set_page_config(
    page_title="Statistiques GLPI",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style> 
        .main { background-color: #f8f9fa; }
        .stMetric { border-radius: 10px; padding: 15px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stMetric label { font-size: 1rem !important; font-weight: 600 !important; color: #4a4a4a !important; }
        .stMetric div { font-size: 1.5rem !important; font-weight: 700 !important; color: #2c3e50 !important; }
        .stPlotlyChart { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: white; padding: 15px; }
        .stDataFrame { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stExpander { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: white; padding: 15px; }
        #.stAlert{ background-color: #f9f0a3;}
        .css-1aumxhk { background-color: #2c3e50; color: white; }
        h1 { color: #2c3e50; border-bottom: 2px solid #066c5d; padding-bottom: 10px; }
        h2 { color: #2c3e50; }
        .stMarkdown { background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.title("📊 Tableau de Bord GLPI")
st.markdown("""
    <div style="background-color:#066c5d;padding:15px;border-radius:10px;color:white;margin-bottom:20px;">
        <h3 style="margin:0;color:white;">Un aperçu interactif et visuel des tickets générés dans GLPI</h3>
    </div>
""", unsafe_allow_html=True)

# Import du fichier de données
with st.expander("📤 Importer des données", expanded=True):
    fichier_importe = st.file_uploader(
        "Sélectionnez un fichier CSV ou Excel exporté depuis GLPI",
        type=["csv", "xlsx"],
        help="Le fichier doit contenir les colonnes standards de GLPI"
    )

# Barre latérale : Logo, navigation, infos, options, pied de page
with st.sidebar:

    # Infos du fichier importé
    st.markdown("## ℹ️ Informations du fichier")
    if fichier_importe:
        st.markdown(f"**Nom du fichier :** `{fichier_importe.name}`")
        st.markdown(f"**Taille :** {round(fichier_importe.size / 1024, 2)} ko")
        st.markdown(f"**Type :** {fichier_importe.type}")
    else:
        st.markdown("Aucun fichier chargé.")

    # Acces rapide
    st.markdown("""
    <div style='padding: 5px; border-radius: 10px;'>
    <h4 style='margin-bottom: 10px;'>📍 Accès rapide</h4>
    <ul style='list-style: none; padding-left: 0; font-size: 15px;'>
    <li><a href="#kpi"> KPIs</a></li>
    <li><a href="#visualisation"> Visualisation</a></li>
    <li><a href="#analysesdetaillees"> Analyses détaillées</a></li>
    <li><a href="#ticketsinactifs"> Tickets inactifs</a></li>
    <li><a href="#graphedeconcurrence"> Graphe de cooccurrence</a></li>
    <li><a href="#analyseintelligente"> Analyse intelligente des titres</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)




if fichier_importe:
    try:
        # Lecture du fichier selon son type
        if fichier_importe.name.endswith('.csv'):
            donnees = pd.read_csv( fichier_importe, encoding='utf-8-sig', sep=';', dayfirst=True)
        else:
            donnees = pd.read_excel(fichier_importe)
        
        # Nettoyage
        donnees = donnees.dropna(how='all')
        st.success("✅ Données chargées avec succès !")
        with st.sidebar: 
            donnees["Dernière modification"] = pd.to_datetime(donnees["Dernière modification"], errors='coerce',dayfirst=True)
            derniere_date = donnees["Dernière modification"].max()
            st.markdown(f"**Ce fichier traite les tickets jusqu'au :** {derniere_date.strftime('%Y-%m-%d %H:%M:%S')}")

            # Pied de page
            st.markdown("""
            <small>Développé par Kenza</small><br>
            <small>Version 1.0.0</small>
            """, unsafe_allow_html=True)

        # KPIs
        st.markdown('<a name="kpi"></a>', unsafe_allow_html=True)
        st.subheader("📈 Indicateurs clés")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="🎟️ Nombre total de tickets",
                value=len(donnees),
                help="Total des tickets dans le fichier importé"
            )
        with col2:
            if "Plugins - Champs Sup - Etablissement" in donnees.columns:
                services_count = donnees["Plugins - Champs Sup - Etablissement"].nunique()
            else:
                services_count = "N/A"
            st.metric(
                label="👥 Nombre de services",
                value=services_count,
                help="Nombre de services générant des tickets"
            )
        with col3:
            if "Priorité" in donnees.columns:
                priorites_count = donnees["Priorité"].nunique()
            else:
                priorites_count = "N/A"
            st.metric(
                label="🔥 Types de priorité",
                value=priorites_count,
                help="Nombre de niveaux de priorité différents"  
            )
        with col4:
            tickets_ouverts = donnees[donnees["Statut"] == "Nouveau"]
            st.metric(
                label="📬 Nouveaux tickets",
                value=len(tickets_ouverts),
                help="Nombre de tickets non ouverts"  
            )

        st.markdown("---")
        st.subheader("🔍 Exploration des données")
        with st.expander("📋 Aperçu des données brutes", expanded=False):
            st.dataframe(donnees.style.background_gradient(cmap='Blues'), use_container_width=True)
            csv_export = donnees.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les données",
                data=csv_export,
                file_name='glpi_donnees_filtrees.csv',
                mime='text/csv',
                help="Téléchargez les données visibles"
            )
        
        # ------------------------------
        # 📊 Visualisations interactives
        # ------------------------------
        # Cette section propose 3 visualisations majeures à partir des données :
        # 1. Par service (répartition des tickets)
        # 2. Par priorité (niveaux d'urgence)
        # 3. Par chronologie (tendance temporelle + activité horaire)

        st.subheader("📊 Visualisations")
        st.markdown('<a name="visualisation"></a>', unsafe_allow_html=True)
        # Onglets de navigation thématique
        tab_service, tab_prio = st.tabs(["📂 Par service", "🎯 Par priorité"])

        # -------------------------------------------
        # 🏢 Onglet 1 : Visualisation par service
        # -------------------------------------------
        with tab_service:
            st.markdown("""
            ### 🏢 Analyse par service
            Ce graphique permet de visualiser quels services génèrent le plus de tickets.
            Cela aide à identifier les unités les plus sollicitées.
            """)
            if "Plugins - Champs Sup - Etablissement" in donnees.columns:
                df_service = donnees["Plugins - Champs Sup - Etablissement"].value_counts().reset_index()
                df_service.columns = ["Service", "Tickets"]

                # Deux colonnes : graphique + statistiques clés
                c1, c2 = st.columns([3, 1])

                with c1:
                    fig_service = px.bar(
                        df_service.head(10),
                        x="Tickets", y="Service", orientation='h',
                        title="Top 10 des services par nombre de tickets",
                        color="Service", template="plotly_white", height=600
                    )
                    st.plotly_chart(fig_service, use_container_width=True)

                with c2:
                    st.metric(
                        label="🏆 Service le plus actif",
                        value=df_service.iloc[0]["Service"],
                        delta=f"{df_service.iloc[0]['Tickets']} tickets"
                    )
                    st.metric(
                        label="📊 % du total",
                        value=f"{(df_service.iloc[0]['Tickets']/len(donnees)*100):.1f} %"
                    )
            else:
                st.warning("La colonne 'Plugins - Champs Sup - Etablissement' est absente.")

        # -------------------------------------------
        # 🎯 Onglet 2 : Visualisation par priorité
        # -------------------------------------------
        with tab_prio:
            st.markdown("""
            ### 🎯 Analyse par priorité
            Ici, on visualise la répartition des tickets selon leur degré d'urgence ou de criticité (basse, moyenne, haute).
            Cela permet de comprendre si l'organisation traite majoritairement des incidents critiques ou des demandes mineures.
            """)
            if "Priorité" in donnees.columns:
                df_prio = donnees["Priorité"].value_counts().reset_index()
                df_prio.columns = ["Priorité", "Tickets"]

                c1, c2 = st.columns([3, 1])

                with c1:
                    # Diagramme circulaire des priorités
                    fig_prio = px.pie(
                        df_prio, names="Priorité", values="Tickets",
                        title="Répartition des tickets par priorité",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig_prio, use_container_width=True)

                with c2:
                    st.metric(
                        label="⚠️ Priorité la plus fréquente",
                        value=df_prio.iloc[0]["Priorité"],
                        delta=f"{df_prio.iloc[0]['Tickets']} tickets"
                    )
                    # Petit graphique complémentaire en barres
                    fig_prio_bar = px.bar(
                        df_prio, x="Priorité", y="Tickets",
                        color="Priorité", height=220
                    )
                    st.plotly_chart(fig_prio_bar, use_container_width=True)
            else:
                st.warning("La colonne 'Priorité' est absente.")
    
        st.markdown("---")

        # -----------------------------
        # 📌 ANALYSES DÉTAILLÉES
        # -----------------------------
        # Cette section approfondit l'analyse des tickets via plusieurs perspectives :
        # 🔹 Statuts et priorités (tableau croisé dynamique)
        # 📆 Activité temporelle : mois, jour, heure
        # 👤 Analyse des acteurs : demandeurs, techniciens
        # 🗂️ Répartition par catégories

        st.subheader("📌 Analyses détaillées")
        st.markdown('<a name="analysesdetaillees"></a>', unsafe_allow_html=True)
 
        # Onglets principaux

        tab_statut,tab_mois, tab_jour, tab_heure = st.tabs([
            "🔹 Statuts par priorité",
            "📆 Activité par mois",
            "📅 Activité par jour",
            "🕒 Activité par heure"
        ])

        # -------------------------------
        # 🔹 Statuts par priorité
        # -------------------------------
        with tab_statut:
            st.markdown("""
            Ce tableau croisé dynamique montre comment les statuts des tickets (nouveau, résolu, en attente...) se répartissent selon leur priorité (haute, basse...).
            Cela permet de détecter, par exemple, si les tickets urgents restent souvent non traités.
            """)
            if "Statut" in donnees.columns and "Priorité" in donnees.columns:
                tab_stat = pd.crosstab(donnees["Priorité"], donnees["Statut"])
                st.dataframe(tab_stat)
                fig_statut = px.bar(
                    tab_stat,
                    barmode='group',
                    title="📊 Répartition des statuts par priorité"
                )
                st.plotly_chart(fig_statut, use_container_width=True)
            else:
                st.warning("Colonnes 'Statut' ou 'Priorité' manquantes.")

        # -------------------------------
        # 📆 Activité par mois
        # -------------------------------
        with tab_mois:
            st.markdown("""
            Cette double visualisation met en évidence des périodes de forte ou faible activité (vacances, incidents, etc). Elle permet de voir :
            - l'évolution mensuelle des tickets (barre + courbe)
            - la moyenne mobile sur 7 jours (tendance lissée)
            """)
            if "Dernière modification" in donnees.columns:
                try:
                    donnees["Mois"] = pd.to_datetime(donnees["Dernière modification"]).dt.to_period("M").astype(str)
                    mensuels = donnees["Mois"].value_counts().sort_index()
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_mensuel_line = px.line(
                            mensuels, title="Évolution mensuelle des tickets",
                            labels={'value': 'Tickets', 'index': 'Mois'}, markers=True
                        )
                        st.plotly_chart(fig_mensuel_line, use_container_width=True)
                    with c2:
                        fig_mensuel_bar = px.bar(
                            mensuels, title="Répartition mensuelle des tickets",
                            labels={'value': 'Tickets', 'index': 'Mois'},
                            color=mensuels.index
                        )
                        st.plotly_chart(fig_mensuel_bar, use_container_width=True)                  
                except:
                    st.warning("Analyse mensuelle impossible avec les dates.")
            else:
                st.warning("La colonne 'Dernière modification' est absente.")

        # -------------------------------
        # 📅 Activité par jour + heure
        # -------------------------------
        with tab_jour:
            st.markdown("""
            Cette heatmap représente l'activité des tickets selon l'heure et la date.
            Elle est utile pour repérer les horaires les plus intenses ou les jours à forte charge.
            """)

            if "Dernière modification" in donnees.columns:
                # Conversion de la colonne en datetime
                donnees["Dernière modification"] = pd.to_datetime(donnees["Dernière modification"], dayfirst=True)

                # Création des colonnes "Date" et "Heure"
                donnees["Date"] = donnees["Dernière modification"].dt.date.astype(str)
                donnees["Heure"] = donnees["Dernière modification"].dt.hour

                # Tickets par date (compte)
                tickets_pour_date = donnees["Date"].value_counts().sort_index()

                # 📊 Affichage côte à côte : tableau + graphique
                col1, col2 = st.columns([1, 3])  # 1/4 pour le tableau, 3/4 pour le graphique

                with col1:
                    st.subheader("Résumé quotidien")
                    st.dataframe(tickets_pour_date.rename("Nombre de tickets"))

                with col2:
                    fig2 = px.line(
                        tickets_pour_date,
                        title="Évolution du nombre de tickets par jour",
                        labels={'value': 'Tickets', 'index': 'Date'},
                        template="plotly_white"
                    )
                    fig2.update_traces(line_color='#3498db', line_width=3)
                    st.plotly_chart(fig2, use_container_width=True, key="lineplot1")

                # 🔥 Heatmap : activité par heure et jour
                heatmap_data = donnees.pivot_table(index="Heure", columns="Date", aggfunc='size', fill_value=0)

                fig1 = px.imshow(
                    heatmap_data,
                    labels=dict(x="Date", y="Heure", color="Nombre de tickets"),
                    aspect="auto",
                    title="Heatmap du nombre de tickets par jour et heure"
                )
                st.plotly_chart(fig1, use_container_width=True, key="heatmap1")

            else: 
                st.warning("La colonne 'Dernière modification' n'existe pas dans ce fichier.")


        # -------------------------------
        # 🕒 Activité par heure uniquement
        # -------------------------------
        with tab_heure:
            st.markdown("""
            Ce graphique montre les heures où le plus de tickets sont modifiés.
            """)
            if "Dernière modification" in donnees.columns:
                donnees["Heure"] = pd.to_datetime(donnees["Dernière modification"], errors='coerce',dayfirst=True).dt.hour
                heure_count = donnees["Heure"].value_counts().sort_index()
                st.bar_chart(heure_count)
            else:
                st.warning("Colonne 'Dernière modification' absente.")



        tab_tech,tab_demandeur, tab_categorie = st.tabs(["👨‍💻 Par technicien", "👤 Tickets par demandeur", "🗂️ Par catégorie"])

        # -------------------------------
        # 👨‍💻 Par technicien
        # -------------------------------
        with tab_tech:
            st.markdown("""
            Ce graphique affiche les techniciens les plus sollicités selon le nombre de tickets qui leur sont assignés.
            Cela permet d'évaluer la charge de travail de chaque intervenant technique.
            """)
            if "Attribué à - Technicien" in donnees.columns:
                donnees["Attribué à - Technicien"] = donnees["Attribué à - Technicien"].fillna("Non assigné")
                df_tech = donnees["Attribué à - Technicien"].value_counts().reset_index()
                df_tech.columns = ["Technicien", "Tickets"]
                fig_tech = px.bar(
                    df_tech.head(20),
                    x="Tickets", y="Technicien", orientation='h',
                    title="Top 20 des techniciens par nombre de tickets assignés",
                    color="Technicien", height=600
                )
                st.plotly_chart(fig_tech, use_container_width=True)
            else:
                st.warning("La colonne 'Attribué à - Technicien' est absente.")

        # -------------------------------
        # 📅 Par demandeur
        # -------------------------------
        with tab_demandeur:
            st.markdown("""
            Cette vue classe les demandeurs (utilisateurs finaux) selon le nombre de tickets créés.
            On peut y identifier les services ou personnes les plus actives.
            """)
            if "Demandeur - Demandeur" in donnees.columns:
                donnees["Demandeur - Demandeur"] = donnees["Demandeur - Demandeur"].fillna("Non assigné").replace("", "Non assigné")
                df_demandeur = donnees["Demandeur - Demandeur"].value_counts().reset_index()
                df_demandeur.columns = ["Demandeur", "Tickets"]
                fig_demandeur = px.bar(
                    df_demandeur.head(20),
                    x="Tickets", y="Demandeur", orientation='h',
                    title="Top 20 des demandeurs de tickets",
                    color="Demandeur", height=600
                )
                st.plotly_chart(fig_demandeur, use_container_width=True)
            else:
                st.warning("La colonne 'Demandeur - Demandeur' est absente.")

        # -------------------------------
        # 🗂️ Par catégorie principale
        # -------------------------------
        with tab_categorie:
            st.markdown("""
            Ici, les tickets sont regroupés selon les grandes catégories (ex : Logiciels, Réseau...).
            Cela facilite la priorisation des interventions et la compréhension des domaines les plus sensibles.
            """)
            if "Catégorie" in donnees.columns:
                donnees[["Catégorie principale", "Sous‑catégorie"]] = donnees["Catégorie"].str.split(">", n=1, expand=True)
                donnees["Catégorie principale"] = donnees["Catégorie principale"].str.strip()
                donnees["Sous‑catégorie"] = donnees["Sous‑catégorie"].str.strip()

                df_cat = donnees["Catégorie principale"].value_counts().reset_index()
                df_cat.columns = ["Catégorie", "Tickets"]
                fig_treemap = px.treemap(
                    df_cat,
                    path=['Catégorie'],
                    values='Tickets',
                    title="Répartition par catégorie principale"
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

                categorie_sel = st.selectbox(
                    "Sélectionnez une catégorie principale pour voir ses sous‑catégories :",
                    donnees["Catégorie principale"].dropna().unique()
                )
                if categorie_sel:
                    df_sub = donnees[donnees["Catégorie principale"]==categorie_sel]["Sous‑catégorie"].value_counts().reset_index()
                    df_sub.columns = ["Sous‑catégorie", "Tickets"]
                    fig_sub = px.bar(
                        df_sub, x="Tickets", y="Sous‑catégorie",
                        orientation='h',
                        title=f"Sous‑catégories dans : {categorie_sel}",
                        height=500
                    )
                    st.plotly_chart(fig_sub, use_container_width=True)
            else:
                st.warning("La colonne 'Catégorie' est absente.")
                
        
        st.markdown("---")
        # -----------------------
        # 🔢 Tickets inactifs depuis X jours
        # -----------------------
        st.subheader("⏳ Tickets inactifs")
        st.markdown('<a name="ticketsinactifs"></a>', unsafe_allow_html=True)

        jours_seuil = st.slider("Inactivité depuis (jours)", min_value=1, max_value=60, value=7)
        if "Dernière modification" in donnees.columns:
            donnees["Dernière modification"] = pd.to_datetime(donnees["Dernière modification"], errors="coerce",dayfirst=True)
            inactifs = donnees[donnees["Dernière modification"] < (pd.Timestamp.now() - pd.Timedelta(days=jours_seuil))]
            
            st.info(f"🎯 {len(inactifs)} tickets sont inactifs depuis plus de **{jours_seuil} jours**.")
            st.dataframe(inactifs[["ID", "Titre", "Statut", "Dernière modification"]].sort_values(by="Dernière modification"))
        else:
            st.warning("Colonne 'Dernière modification' absente.")
        st.markdown("---")


        # -----------------------
        # 🔢 Menu déroulant pour choisir la visualisation
        # -----------------------

        choix_visu = st.selectbox(
            "Choisissez la visualisation à afficher :",
            ["🧠 Mots fréquents (camembert)", "🌐 Graphe de cooccurrence"]
        )

        # -----------------------
        # 🧠 Mots-clés fréquents dans les tickets non résolus (Camembert)
        # -----------------------

        if choix_visu == "🧠 Mots fréquents (camembert)":
            st.subheader("🧠 Mots fréquents dans les tickets non résolus")

            if "Titre" in donnees.columns and "Statut" in donnees.columns:
                non_resolus = donnees[~donnees["Statut"].isin(["Résolu", "Clos", "Fermé", "Terminé"])]
                titres = non_resolus["Titre"].dropna().astype(str).apply(nettoyer)
                flat = [mot for liste in titres for mot in liste]
                freqs = Counter(flat).most_common(15)

                mots, valeurs = zip(*freqs)

                fig_donut = px.pie(
                    names=mots,
                    values=valeurs,
                    hole=0.4,
                    title="Distribution des mots les plus fréquents (tickets non résolus)",
                    color_discrete_sequence=px.colors.sequential.Plasma_r
                )
                st.plotly_chart(fig_donut, use_container_width=True)

                top_mots = ", ".join([f"'{m}'" for m in mots[:10]])

                st.success(
                    f"""
                    Les 10 mots les plus fréquents dans les titres des tickets non résolus sont : {top_mots}.  
                    Leur fréquence élevée suggère qu’ils représentent des thématiques récurrentes dans les problèmes non traités.  
                    """
                )

            else:
                st.warning("Colonnes 'Titre' ou 'Statut' manquantes.")

        # -----------------------
        # 🌐 Graphe de cooccurrence des mots-clés
        # -----------------------

        elif choix_visu == "🌐 Graphe de cooccurrence":
            st.subheader("🌐 Graphe de cooccurrence des mots dans les titres")
            st.markdown('<a name="graphedeconcurrence"></a>', unsafe_allow_html=True)

            if "Titre" in donnees.columns:
                titres_clean = donnees["Titre"].dropna().astype(str).apply(nettoyer)

                # Calcul cooccurrences
                cooccurrence = Counter()
                for mots_titre in titres_clean:
                    mots_uniques = list(set(mots_titre))
                    for i in range(len(mots_uniques)):
                        for j in range(i + 1, len(mots_uniques)):
                            paire = tuple(sorted((mots_uniques[i], mots_uniques[j])))
                            cooccurrence[paire] += 1

                seuil = 3
                edges = [(a, b, {"weight": w}) for (a, b), w in cooccurrence.items() if w >= seuil]

                G = nx.Graph()
                G.add_edges_from(edges)

                if len(G.nodes) == 0:
                    st.info(f"Aucun lien de cooccurrence trouvé avec un seuil >= {seuil}. Essayez de baisser le seuil.")
                else:
                    nb_mots = len(G.nodes)
                    nb_liens = len(G.edges)
                    centralite = nx.degree_centrality(G)
                    mot_central = max(centralite, key=centralite.get)
                    centralite_max = centralite[mot_central]
                    densite = nx.density(G)

                    pos = nx.spring_layout(G, k=0.5, seed=42)
                    plt.figure(figsize=(10, 8))
                    nx.draw(
                        G, pos,
                        with_labels=True,
                        node_size=[v * 1000 for v in centralite.values()],
                        node_color='#407fb2',
                        font_size=7,
                        edge_color='#0877d1',
                        width=[G[u][v]['weight'] * 0.2 for u, v in G.edges()]
                    )
                    st.pyplot(plt)
                    plt.clf()

                    # Résumé chiffré
                    st.markdown(f"""
                    <table style="width:100%; border-collapse: collapse; border: 1px solid #ddd;">
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">🧩 <b>Nombre total de mots-clés distincts analysés</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align:center;"><b>{nb_mots}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">🔗 <b>Nombre de relations fréquentes entre ces mots</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align:center;"><b>{nb_liens}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">📊 <b>Densité du réseau de cooccurrence</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align:center;"><b>{densite:.3f}</b></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">⭐ <b>Mot-clé le plus central</b></td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align:center;"><b>'{mot_central}' (centralité = {centralite_max:.2f})</b></td>
                    </tr>
                    </table>
                    """, unsafe_allow_html=True)

                    # 🟢 Conclusion explicative
                    mots_connectes = sorted(list(G.nodes))[:5]
                    mots_connectes_str = ", ".join([f"'{m}'" for m in mots_connectes])

                    st.success(
                        f"""
                        Le réseau de cooccurrence met en évidence des liens fréquents entre des mots comme {mots_connectes_str}.  
                        Cela montre que ces termes apparaissent ensemble dans de nombreux titres, traduisant des problématiques liées ou des tickets traitant de thèmes communs.  
                        
                        Le mot **'{mot_central}'**, central dans ce graphe, semble jouer un rôle de pivot thématique. Il pourrait désigner une source fréquente de blocage ou un sujet transversal.

                        Une densité de **{densite:.3f}** suggère un niveau de connectivité {'élevé' if densite > 0.15 else 'modéré' if densite > 0.05 else 'faible'} : les sujets traités dans les tickets sont donc {'fortement reliés entre eux' if densite > 0.15 else 'parfois liés' if densite > 0.05 else 'plutôt isolés'}.

                        👉 Ces informations permettent d’identifier rapidement les thématiques dominantes et leur interconnexion, pour mieux structurer les résolutions ou organiser les priorités.
                        """
                    )

            else:
                st.warning("Colonne 'Titre' absente.")

        st.markdown("---")

        # --------------------------
        # 🧠 Analyse NLP des titres améliorée
        # --------------------------
        st.subheader("🧠 Analyse intelligente des titres")
        st.markdown('<a name="analyseintelligente"></a>', unsafe_allow_html=True)

        if "Titre" in donnees.columns:
            titres_nettoyes = nettoyer_titres(donnees["Titre"])
            donnees_valides = donnees.loc[titres_nettoyes.index]
            X, vectoriseur = vectoriser_titres(titres_nettoyes)
            clusters, modele = creer_clusters(X)
            donnees_valides["Groupe de titres"] = clusters

            

            # 📈 Mots fréquents
            st.markdown("### 📈 Mots les plus fréquents")
            with st.expander("ℹ️ Explications détaillées sur le regroupement en groupes"):
                st.markdown("""
                Cette section génère un graphique en barres (matplotlib) représentant les mots les plus fréquents dans les titres.  
                Avant l’analyse, les titres ont été nettoyés à l’aide d’une fonction qui :  
                - supprime les stopwords (mots très courants comme "le", "la", "de"),  
                - élimine les chiffres, la ponctuation et les mots trop courts,  
                - homogénéise les mots pour éviter les doublons inutiles.  

                Ce traitement permet de se concentrer sur les termes les plus significatifs et d’éviter que des mots sans réelle valeur sémantique ne biaisent l’analyse.
                """)

            fig_freq = plot_mots_frequents(titres_nettoyes)
            st.pyplot(fig_freq)

            # 🔑 Mots-clés 
            st.markdown("### 🔑 Mots-clés par groupe")
            with st.expander("ℹ️ Explications détaillées sur le regroupement en groupes"):
                st.markdown("""
                Dans cette section, on regroupe les titres dans des clusters et cela en se basant sur deux alogrithmes : 

                2. **Vectorisation TF-IDF**  
                Chaque titre est transformé en un vecteur numérique représentant l'importance des mots présents dans ce titre par rapport à tous les titres.  
                Cela permet de comparer facilement les titres entre eux.

                3. **Clustering (regroupement en groupes)**  
                Un algorithme (K-means) regroupe les titres en 6 clusters (groupes) selon leur similarité sémantique.  
                Par exemple, un cluster peut regrouper tous les titres parlant de problèmes d'imprimante, un autre ceux liés à la connexion internet, etc.

                4. **Mots-clés par groupe**  
                Pour chaque groupe, on extrait les 5 mots les plus caractéristiques, permettant de comprendre rapidement le thème de chaque cluster.
                """)
            mots_par_groupe = mots_cles_par_cluster(modele, vectoriseur)
            for i, mots in enumerate(mots_par_groupe):
                st.markdown(f"**Groupe {i}** : {' - '.join(mots)}")
            st.markdown("""
            _Ces mots sont les plus représentatifs de chaque groupe. Par exemple, si le groupe 0 affiche « connexion - réseau - wifi », cela signifie que les titres de ce groupe parlent principalement de problèmes de connexion réseau._
            """)

            # 🔍 Exemples
            st.markdown("### 🔍 Aperçu dynamique des titres par groupe")
            groupe_choisi = st.selectbox("Choisir un groupe :", sorted(donnees_valides["Groupe de titres"].unique()))
            exemples = donnees_valides[donnees_valides["Groupe de titres"] == groupe_choisi]["Titre"].head(10)
            st.table(exemples.to_frame())
            st.markdown("""
            _Vous pouvez sélectionner un groupe pour voir quelques exemples de titres qui en font partie, afin de mieux comprendre le thème du groupe._
            """)

            # 📊 Répartition (bar + pie)
            distribution = donnees_valides["Groupe de titres"].value_counts().sort_index()
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Nombre de titres par groupe")
                st.markdown("> Ce graphique montre combien de titres ont été automatiquement regroupés dans chaque groupe.")
                fig_bar, ax_bar = plt.subplots()
                sns.barplot(x=distribution.index, y=distribution.values, ax=ax_bar, palette="Set2")
                ax_bar.set_xlabel("Groupe")
                ax_bar.set_ylabel("Nombre de titres")
                st.pyplot(fig_bar)

            with col2:
                st.markdown("#### 🥧 Répartition en pourcentage")
                st.markdown("> Le camembert illustre la proportion relative de chaque groupe dans l’ensemble des tickets.")
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(distribution, labels=[f"Groupe {i}" for i in distribution.index],
                        autopct="%1.1f%%", startangle=90)
                ax_pie.axis("equal")
                st.pyplot(fig_pie)

            # 🧮 Entropie
            st.markdown("### 🧮 Diversité sémantique")
            st.markdown("""
            > L'entropie est un indicateur qui permet d'évaluer à quel point les titres sont **répartis de manière équilibrée entre les groupes**.  
            > Plus l'entropie est élevée, plus la diversité thématique est grande.
            """)

            # Calculs
            entropie_val = calculer_entropie(donnees_valides)
            max_entropie = np.log(6)  # log du nombre de groupes
            pourcentage = (entropie_val / max_entropie) * 100

            # Affichage
            st.info(f"Entropie mesurée : **{entropie_val:.2f}** / **{max_entropie:.2f}** (soit **{pourcentage:.0f}%** de diversité possible)")

            # Analyse automatique
            if pourcentage >= 70:
                st.success("✔️ Les titres sont bien répartis entre les groupes. Cela indique une **bonne diversité thématique**.")
            elif 40 <= pourcentage < 70:
                st.warning("⚠️ Les titres sont partiellement concentrés. Certains groupes dominent légèrement.")
            else:
                st.error("❗ Faible diversité : la majorité des titres semblent concerner **un sujet principal**.")

                # Trouver le groupe dominant
                groupe_majoritaire = donnees_valides["Groupe de titres"].value_counts().idxmax()
                mots_dominants = mots_par_groupe[groupe_majoritaire][:5]
                st.markdown(f"""
                _Dans notre cas, le **groupe {groupe_majoritaire}** est largement dominant._
                
                Ce groupe est associé aux mots-clés suivants : **{'**, **'.join(mots_dominants)}**  
                👉 Cela suggère que les titres parlent principalement de : **{' '.join(mots_dominants)}**
                """)

            # Info pédagogique
            st.markdown(f"""
            _À titre de référence :_
            - Une entropie proche de **{max_entropie:.2f}** signifie une excellente répartition (100% diversité)
            - Une entropie proche de **0** signifie une très forte concentration sur un seul thème
            """)


            # 🤖 Modèle
            st.markdown("### 🤖 Modèle de prédiction")
            with st.expander("🧠 En savoir plus : comment fonctionne ce modèle ?"):
                st.markdown("""
                    > Un modèle de machine learning (ici une **régression logistique**) a été entraîné pour apprendre à **reconnaître automatiquement à quel groupe appartient un titre**, uniquement à partir des mots qu'il contient.
                    >
                    > Ce modèle est utile si tu veux **classer automatiquement de nouveaux tickets** dans les bons thèmes sans intervention humaine.

                    La **régression logistique** est un algorithme simple mais puissant qui apprend à **associer des mots à des classes**.

                    Voici les étapes :
                    1. Chaque titre est converti en vecteur de mots (grâce à la vectorisation)
                    2. Le modèle apprend sur ces vecteurs et leurs groupes réels
                    3. Ensuite, pour un nouveau titre, il **calcule la probabilité** qu'il appartienne à chaque groupe
                    4. Il choisit celui avec la plus forte probabilité

                    C’est un algorithme largement utilisé pour les tâches de **classification automatique de texte**, car :
                    - il est rapide à entraîner
                    - il fonctionne bien sur des données peu complexes
                    - il peut donner une **probabilité pour chaque classe**, ce qui est utile pour mesurer la confiance du modèle
                """)
          

            # Entraînement + score + matrice
            score, matrice = entrainer_model(X, donnees_valides["Groupe de titres"])
            st.success(f"🎯 Précision du modèle : **{score:.2%}**")

            # Explication en langage simple
            st.markdown(f"""
            _La précision représente le **pourcentage de titres pour lesquels le modèle a correctement prédit le groupe**.  
            Dans notre cas, cela signifie que le modèle trouve le bon groupe dans environ **{score:.0%} des cas**._

            > Exemple : si un nouveau ticket parle de "wifi qui se déconnecte souvent", le modèle va analyser les mots clés du titre  
            et le **classer automatiquement dans le groupe associé aux problèmes de réseau ou de connexion**.
            
            """)

            # Matrice de confusion avec taille réduite
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))  # plus petit que défaut
            sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Matrice de confusion")
            st.pyplot(fig_cm)

            st.markdown("""
            _👉 La matrice de confusion montre comment les prédictions du modèle sont réparties :_
            - Les **cases sur la diagonale** montrent les bonnes prédictions (où le modèle a vu juste)
            - Les **autres cases** montrent les erreurs (ex. : titres du groupe 1 classés à tort comme groupe 2)

            Cela permet de savoir **quels groupes sont parfois confondus**, ce qui peut aider à améliorer le modèle ou à ajuster les groupes.
            """)


            st.markdown("### 🧠 Thèmes latents détectés (LDA)")
            st.markdown("""
            LDA (Latent Dirichlet Allocation) est un modèle qui **détecte automatiquement les grands sujets présents dans les titres**, en regroupant des mots qui apparaissent souvent ensemble.

            C’est une façon de voir quels sont les **sujets implicites** qui reviennent le plus dans les tickets, sans avoir besoin de les définir à l'avance.
            """)

            themes = appliquer_lda(X, vectoriseur)
            for i, mots in enumerate(themes):
                st.markdown(f"**Thème {i+1}** : {', '.join(mots)}")
                #st.markdown(f"_Ce thème semble concerner : {' et '.join(mots[:2])}..._")

            st.info("""
            💡 LDA est particulièrement utile quand on a un grand volume de titres,  
            et que qu'on souhaite **identifier automatiquement les sujets récurrents**,  
            même sans classer chaque ticket individuellement.
            """)

           # 📌 Résumé automatique enrichi
            st.markdown("### 🧾 Résumé automatique des observations")

            # Stats complémentaires
            nb_groupes = donnees_valides["Groupe de titres"].nunique()
            groupe_majoritaire = donnees_valides["Groupe de titres"].value_counts().idxmax()
            nb_total = len(donnees_valides)
            nb_dominant = donnees_valides["Groupe de titres"].value_counts().max()
            part_dominant = (nb_dominant / nb_total) * 100
            top_themes = [", ".join(t[:3]) for t in themes[:3]]

            # Mots fréquents globaux
            mots_cles_principaux = [mot for mots in mots_par_groupe for mot in mots]
            mots_resumables = pd.Series(mots_cles_principaux).value_counts().head(5).index.tolist()

            resume_intro = f"""
            Parmi les **{nb_total} titres analysés**, le système a identifié **{nb_groupes} groupes thématiques distincts**.
            - Les mots les plus fréquents sont : **{'**, **'.join(mots_resumables)}**
            - Le **groupe dominant est le groupe {groupe_majoritaire}**, représentant **{part_dominant:.1f}%** des titres
            """

            # Diversité
            if pourcentage >= 70:
                resume_diversite = "🟢 Les titres sont bien répartis entre les groupes, indiquant une **bonne diversité thématique**."
            elif 40 <= pourcentage < 70:
                resume_diversite = "🟠 Les titres sont partiellement concentrés : **certains thèmes dominent légèrement**."
            else:
                mots_dominants = mots_par_groupe[groupe_majoritaire][:5]
                resume_diversite = f"🔴 La majorité des titres se concentrent sur un **seul thème**, associé à : **{'**, **'.join(mots_dominants)}**."

            # Prédiction automatique
            resume_model = f"🤖 Le modèle de machine learning atteint une **précision de {score:.0%}**, ce qui le rend fiable pour classer automatiquement de nouveaux tickets."

            # LDA
            resume_lda = f"""
            📚 L’analyse LDA a mis en évidence les thèmes latents suivants :
            - **Thème 1** : {top_themes[0]}
            - **Thème 2** : {top_themes[1]}
            - **Thème 3** : {top_themes[2]}
            """

            # Affichage global
            st.success(f"{resume_intro}\n\n{resume_diversite}\n\n{resume_model}\n\n{resume_lda}")



        else:
            st.warning("Colonne 'Titre' absente du fichier.")

        
        st.markdown("---")
        st.markdown(f"""
            <div style="text-align:center;padding:20px;background-color:#2c3e50;color:white;border-radius:10px;">
                <p>© 2025 – Tableau de Bord GLPI – Développé avec Streamlit</p>
                <p style="font-size:0.8em;">Dernière mise à jour : {datetime.now().strftime("%d/%m/%Y %H:%M")}</p>
            </div>
        """, unsafe_allow_html=True)
    
    except Exception as erreur:
        st.error(f"❌ Erreur lors du chargement des données : {erreur}")
else:
    st.info("ℹ️ Veuillez importer un fichier CSV ou Excel depuis GLPI pour démarrer.")
    #⚠️
    #st.markdown("""
    #<div style="padding:20px;background-color:#f4f6ff;border-radius:10px;border-left:5px solid #c9ba2e;">
    #    <h4 style="margin-top:0;">Comment utiliser ce tableau de bord :</h4>
    #    <ol>
    #       <li>Exportez vos données depuis GLPI (format CSV ou Excel).</li>
    #        <li>Choisissez ou glissez-déposez votre fichier ci‑dessous.</li>
    #        <li>Explorez les visualisations et analyses proposées.</li>
    #        <li>Affinez votre sélection via les filtres latéraux.</li>
    #    </ol>
    #</div>
    #""", unsafe_allow_html=True)


