import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Config de la page
st.set_page_config(page_title="Dashboard TARDIS", layout="wide")
st.title("🚆 TARDIS : Prédiction des Retards SNCF")

# Chargement des donnees et du modele
@st.cache_data
def load_data():
    df = pd.read_csv("cleared_dataset.csv")
    
    # On met toutes les gares en majuscules pour regrouper les doublons
    df["Gare de départ"] = df["Gare de départ"].astype(str).str.upper()
    df["Gare d'arrivée"] = df["Gare d'arrivée"].astype(str).str.upper()
    
    # On force les colonnes à etre de vrais nombre pour éviter l'erreur de calcul
    cols_num = [
        "Retard moyen des trains en retard au départ", 
        "Nombre de circulations prévues", 
        "Nombre de trains annulés"
    ]
    for col in cols_num:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# filtre
st.sidebar.header("🔍 Filtres d'analyse")

# On récupère les services et on cache le faux service "0.0"
liste_services = df["Service"].dropna().unique().tolist()
if 0.0 in liste_services: liste_services.remove(0.0)
if "0.0" in liste_services: liste_services.remove("0.0")

choix_service = st.sidebar.selectbox("Choix du service", ["Tous"] + liste_services)

# On trie les donnees
if choix_service != "Tous":
    df_filtre = df[df["Service"] == choix_service]
else:
    df_filtre = df

# stats en general
st.header("📊 Statistiques Globales")
col1, col2, col3 = st.columns(3)

retard_moyen = df_filtre["Retard moyen des trains en retard au départ"].mean()
total_trajets = df_filtre["Nombre de circulations prévues"].sum()
total_annules = df_filtre["Nombre de trains annulés"].sum()

col1.metric("Retard Moyen (minutes)", round(retard_moyen, 2))
# On affiche bien en int
col2.metric("Total Trajets Prévus", int(total_trajets) if pd.notna(total_trajets) else 0)
col3.metric("Total Trains Annulés", int(total_annules) if pd.notna(total_annules) else 0)

st.divider()

# rendu
st.header("📈 Visualisation des Retards")
st.write("Top 10 des gares de départ avec le plus de retard moyen.")

top_gares = df_filtre.groupby("Gare de départ")["Retard moyen des trains en retard au départ"].mean().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 5))
top_gares.plot(kind="barh", ax=ax, color='#1f77b4')
ax.set_xlabel("Retard moyen (minutes)")
ax.set_ylabel("Gare de départ")
ax.invert_yaxis() 
st.pyplot(fig)

st.divider()

# modele qui prédit
st.header("🔮 Prédire un Retard")

with st.form("form_prediction"):
    c1, c2 = st.columns(2)
    
    with c1:
        gare_dep = st.selectbox("Gare de départ", df["Gare de départ"].unique())
        service_input = st.selectbox("Service", liste_services) # On utilise la liste propre
        circulations = st.number_input("Nombre de circulations prévues", min_value=0, value=100)
        
    with c2:
        gare_arr = st.selectbox("Gare d'arrivée", df["Gare d'arrivée"].unique())
        annules = st.number_input("Nombre de trains annulés", min_value=0, value=0)
        
    bouton = st.form_submit_button("Lancer la prédiction")

if bouton:
    # On recupere les infos (on les remet en texte pour tromper le modèle qui s'attend à du texte)
    input_user = pd.DataFrame({
        "Gare de départ": [gare_dep],
        "Gare d'arrivée": [gare_arr],
        "Service": [service_input],
        "Nombre de circulations prévues": [str(circulations)],
        "Nombre de trains annulés": [str(annules)]
    })
    
    features = ["Gare de départ", "Gare d'arrivée", "Service", "Nombre de circulations prévues", "Nombre de trains annulés"]
    
    # On recharge le fichier brut juste pour aligner les colonnes avec le get_dummies
    df_brut = pd.read_csv("cleared_dataset.csv")[features]
    
    df_combine = pd.concat([df_brut, input_user], ignore_index=True)
    df_dummies = pd.get_dummies(df_combine, drop_first=True)
    
    ligne_utilisateur = df_dummies.tail(1)
    
    # on force les colonnes à correspondre exactement à ce que le modele a appris
    ligne_utilisateur = ligne_utilisateur.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # On lance la prediction
    prediction = model.predict(ligne_utilisateur)[0]
    
    st.success(f"⏳ Le retard estimé pour ce trajet est de : **{prediction:.2f} minutes**")