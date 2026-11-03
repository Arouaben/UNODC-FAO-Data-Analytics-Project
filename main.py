import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

print("=======================================================================")
# TRILOGIE DE PROJETS DE DATA SCIENCE (SIMULATION UNODC & FAO)
print("=======================================================================\n")

# =====================================================================
# THÈME 1 : TRAFIC DE MARCHANDISES & CONTREBANDE (UNODC)
# =====================================================================
print("--- 📑 Exécution du Thème 1 : Trafic & Contrebande ---")
np.random.seed(42)
n_records = 1000

df_trafic = pd.DataFrame({
    'Incident_ID': [f"INC-{i:04d}" for i in range(1, n_records + 1)],
    'Produit': np.random.choice(['Contrefaçons Électroniques', 'Tabac de contrebande', 'Produits Chimiques Régulés', 'Œuvres d\'art'], n_records),
    'Quantite_Saisie_KG': np.random.exponential(scale=150, size=n_records).round(2),
    'Pays_Origine': np.random.choice(['Tunisia', 'Turkey', 'China', 'Libya'], n_records),
    'Mode_Transport': np.random.choice(['Maritime', 'Aérien', 'Routier'], n_records),
    'Valeur_Estimee_USD': np.random.uniform(5000, 500000, n_records).round(2)
})

scores_cpi = {'Tunisia': 40, 'Turkey': 34, 'China': 45, 'Libya': 18}
df_trafic['CPI_Score_2025'] = df_trafic['Pays_Origine'].map(scores_cpi)
df_trafic['Valeur_par_KG'] = (df_trafic['Valeur_Estimee_USD'] / df_trafic['Quantite_Saisie_KG']).round(2)

X_t = StandardScaler().fit_transform(df_trafic[['Quantite_Saisie_KG', 'CPI_Score_2025']])
df_trafic['Niveau_Menace'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_t)
df_trafic['Niveau_Menace'] = df_trafic['Niveau_Menace'].map({0: 'Risque Modéré', 1: 'Risque Critique', 2: 'Risque Faible'})

X_reg_t = pd.concat([df_trafic[['Quantite_Saisie_KG', 'CPI_Score_2025']], pd.get_dummies(df_trafic[['Produit', 'Mode_Transport']], drop_first=True)], axis=1)
model_t = LinearRegression().fit(X_reg_t, df_trafic['Valeur_Estimee_USD'])
df_trafic['Valeur_Predite_USD'] = model_t.predict(X_reg_t).round(2)


# =====================================================================
# THÈME 2 : CRIMINALITÉ FINANCIÈRE & BLANCHIMENT (UNODC)
# =====================================================================
print("--- 💸 Exécution du Thème 2 : Criminalité Financière ---")
np.random.seed(101)
n_tx = 1200

df_finance = pd.DataFrame({
    'Transaction_ID': [f"TX-{i:05d}" for i in range(1, n_tx + 1)],
    'Type_Canal': np.random.choice(['Virement Bancaire', 'Crypto-actifs', 'Société Écran', 'Espèces'], n_tx),
    'Montant_USD': np.random.exponential(scale=50000, size=n_tx).round(2),
    'Pays_Emetteur': np.random.choice(['Tunisia', 'Cayman Islands', 'Switzerland', 'Panama', 'Luxembourg'], n_tx)
})

risques_fsi = {'Tunisia': 45, 'Cayman Islands': 82, 'Switzerland': 65, 'Panama': 78, 'Luxembourg': 60}
df_finance['FSI_Score_2026'] = df_finance['Pays_Emetteur'].map(risques_fsi).fillna(55)

X_f = StandardScaler().fit_transform(df_finance[['Montant_USD', 'FSI_Score_2026']])
df_finance['Statut_Transaction'] = KMeans(n_clusters=3, random_state=101, n_init=10).fit_predict(X_f)
df_finance['Statut_Transaction'] = df_finance['Statut_Transaction'].map({0: 'Flux Standard', 1: 'Flux Opaque', 2: 'Flux Hautement Suspect'})

X_reg_f = pd.concat([df_finance[['FSI_Score_2026']], pd.get_dummies(df_finance['Type_Canal'], drop_first=True)], axis=1)
model_f = LinearRegression().fit(X_reg_f, df_finance['Montant_USD'])
df_finance['Montant_Predit_USD'] = model_f.predict(X_reg_f).round(2)


# =====================================================================
# THÈME 3 : IMPACT DE LA CRIMINALITÉ SUR L'AGRICULTURE (FAO)
# =====================================================================
print("--- 🌾 Exécution du Thème 3 : Impact Agricole ---")
np.random.seed(42)
n_parcelles = 800

df_agri = pd.DataFrame({
    'Parcelle_ID': [f"PRC-{i:04d}" for i in range(1, n_parcelles + 1)],
    'Filiere_Fruit': np.random.choice(['Dattes Deglet Nour', 'Agrumes', 'Olives', 'Grenades'], n_parcelles),
    'Effectif_Recolte_Manuelle': np.random.randint(5, 40, size=n_parcelles)
})
df_agri['Production_Attendue_Tonnes'] = (df_agri['Effectif_Recolte_Manuelle'] * np.random.uniform(1.2, 3.5, n_parcelles)).round(2)
df_agri['Taux_Perte_Criminalite_Pct'] = np.random.uniform(2, 28, n_parcelles).round(2)
df_agri['Volume_Perdu_Tonnes'] = ((df_agri['Production_Attendue_Tonnes'] * df_agri['Taux_Perte_Criminalite_Pct']) / 100).round(2)

X_a = StandardScaler().fit_transform(df_agri[['Production_Attendue_Tonnes', 'Taux_Perte_Criminalite_Pct']])
df_agri['Cluster_Agri'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_a)
df_agri['Profil_Vulnerabilite'] = df_agri['Cluster_Agri'].map({0: 'Exploitation Résiliente', 1: 'Exploitation Sous Menace Élevée', 2: 'Petite Exploitation Ciblée'})

X_reg_a = pd.concat([df_agri[['Effectif_Recolte_Manuelle', 'Production_Attendue_Tonnes']], pd.get_dummies(df_agri['Filiere_Fruit'], drop_first=True)], axis=1)
model_a = LinearRegression().fit(X_reg_a, df_agri['Volume_Perdu_Tonnes'])
df_agri['Volume_Perdu_Predit_Tonnes'] = model_a.predict(X_reg_a).round(2)

print("\n[SUCCÈS] Tous les pipelines de données s'exécutent correctement.")
