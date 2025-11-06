import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Titel en beschrijving
st.title("🚢 Titanic Data Analyse")
st.markdown("""
Welkom bij mijn **Titanic-analyse**!  
Hier onderzoeken we wie er aan boord waren, hoeveel ze betaalden,  
en welke factoren invloed hadden op hun overlevingskans.
""")

# Data inladen
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

train = load_data()

st.header("1️⃣ Datasetverkenning")
st.write("Bekijk een paar rijen uit de dataset:")
st.dataframe(train.head())

# Beschrijving
st.write("Aantal rijen en kolommen:", train.shape)

# Missende waarden
st.subheader("Ontbrekende waarden")
missing = train.isna().sum()
st.bar_chart(missing)

# Verdeling van leeftijden
st.header("2️⃣ Demografische verdeling")
fig, ax = plt.subplots()
ax.hist(train["Age"].dropna(), bins=20)
ax.set_title("Verdeling van leeftijden")
st.pyplot(fig)

# Geslachtverdeling
st.subheader("Verdeling geslacht")
st.bar_chart(train["Sex"].value_counts())

# Overleving per klasse
st.header("3️⃣ Overleving per klasse")
survival_class = train.groupby("Pclass")["Survived"].mean()
st.bar_chart(survival_class)

# Ticketprijzen
st.header("4️⃣ Analyse van Ticketprijzen")
fig, ax = plt.subplots()
ax.hist(train["Fare"], bins=30)
ax.set_title("Verdeling van ticketprijzen")
st.pyplot(fig)

# Log(Fare)
fig, ax = plt.subplots()
ax.hist(np.log1p(train["Fare"]), bins=30)
ax.set_title("Log-verdeling van Fare")
st.pyplot(fig)

# Leeftijdsgroepen
st.header("5️⃣ Leeftijdsgroepen & kinderenanalyse")
train["AgeGroup"] = pd.cut(train["Age"], bins=[0, 1, 15, 100], labels=["Infant", "Child", "Adult"])

fare_by_group = train.groupby("AgeGroup")["Fare"].mean()
st.bar_chart(fare_by_group)

st.success("✅ Analyse afgerond!")