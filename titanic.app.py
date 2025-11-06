import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


pagina = st.sidebar.radio("Ga naar:", ["Introductie", "De 3e klasse", 'De 2e klasse', "De 1e klasse"])
if pagina == 'Introductie':
    st.title("Visual Analytics Dashboard") 
    st.subheader("Een interactieve data-analyse presentatie") 
    # Uitlegtekst 
    st.markdown("""Welkom bij mijn Visual Analytics project!  
                In deze applicatie laat ik zien hoe we data kunnen visualiseren en analyseren met behulp van **Streamlit**.
                
                **Wat je hier kunt verwachten:** 
                - Een overzicht van de dataset  
                - Interactieve grafieken  
                - Inzichten uit de analyses  
                
                Gebruik de sidebar om te navigeren tussen verschillende onderdelen van de presentatie.
                """)

                # Eventueel een afbeelding of logo
                # st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=200)

     # Footer of extra tekst
    st.info("Scroll naar beneden of gebruik het menu om verder te gaan.")

# pagina 2:
elif pagina == 'De 3e klasse':
    # Data inladen (zonder te tonen)
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Eenvoudige cleaning: verwijder rijen zonder Age
    train = train.dropna(subset=["Age"])
    test = test.dropna(subset=["Age"])

    # Eventueel extra preprocessing
    train["status"] = np.where(train["Survived"] == 1, "Overleeft", "Overleden")

    st.title('De 3e klasse')
    st.subheader('Dit was de eerste versie van de titanic case')

    tab1, tab2 = st.tabs(["Dataset", "Overleving Analyses"])

    # ---------------- Tab 1: Dataset ----------------
    with tab1:
        st.subheader("Voorbeeld van de dataset")
        st.dataframe(train.head())

    # ---------------- Tab 2: Overleving Analyses ----------------
    with tab2:
        st.subheader("Overleving per geslacht")
        sns.set_palette(["r","purple"])
        fig1, ax1 = plt.subplots()
        sns.barplot(data=train, x="Sex", y="Survived", hue="Sex", ax=ax1)
        ax1.set_title("Overlevingskans per geslacht")
        ax1.set_ylabel("Overlevingskans")
        st.pyplot(fig1)

        st.subheader("Overleving per leeftijd")
        sns.set_palette(["r", "black"])
        fig2, ax2 = plt.subplots()
        sns.histplot(data=train, x="Age", hue="status", alpha=0.6, multiple="dodge", ax=ax2)
        ax2.set_title("Verdeling van leeftijden met overlevingsstatus")
        ax2.set_xlabel("Leeftijd")
        ax2.set_ylabel("Aantal")
        st.pyplot(fig2)

        st.subheader("Overleving per klasse")
        sns.set_palette(["lime", "r"])
        fig3, ax3 = plt.subplots()
        sns.countplot(data=train, x="Pclass", hue="status", ax=ax3)
        ax3.set_title("Aantal passagiers per klasse met overlevingsstatus")
        ax3.set_xlabel("Klasse")
        ax3.set_ylabel("Aantal")
        st.pyplot(fig3)

# Pagina 3
elif pagina == 'De 1e klasse':
    tab3, tab4 = st.tabs(['Data verwerking', 'leeftijd en Gender'])
    with tab3:
    st.title('text')
    
    with tab4:
        fig4, ax = plt.subplots()
        sns.histplot(train["Age"], bins=30, kde=True, ax=ax)
        st.pyplot(fig4)
      
    




