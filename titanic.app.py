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


elif pagina == 'De 2e klasse':
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train['Age'] = train['Age'].fillna(train.groupby('Sex')['Age'].transform('mean'))
    # Kolom aangemaakt om te zien hoeveel pp betaald
    train['Ticket_Count'] = train.groupby('Ticket')['Ticket'].transform('count')
    train['Fare_per_person'] = train['Fare'] / train['Ticket_Count']
    train['Fare'] = train['Fare_per_person']

    # Kolom aangemaakt om te zien welke leeftijds groepen horen bij de PaassengerId
    train['Age_Group'] = np.select(
        [train['Age'] < 1, (train['Age'] >= 1) & (train['Age'] < 15), train['Age'] >= 15],
        ['Infant', 'Child', 'Adult'],
        default='Unknown')
    # Tarieven verschillen per leeftijdsgroep
    train['FareWeight'] = np.select(
        [train['Age_Group'] == 'Infant', train['Age_Group'] == 'Child', train['Age_Group'] == 'Adult'],
        [0.0, 0.5, 1.0]
    )

    # Berekening van het totale gewicht per ticket (gehele ticket)
    train['TotalFareWeight'] = train.groupby('Ticket')['FareWeight'].transform('sum')

    # Berekening van de waarde per gewichtseenheid (tarief per volwassene)
    train['Fare_per_weight'] = train['Fare'] / train['TotalFareWeight']

    # Berekening van de aangepaste fare (0.0 en 0.5 zodat ticket eerlijk verdeeld wordt)
    train['AdjustedFare'] = train['Fare_per_weight'] * train['FareWeight']

    # Oude kolom verwijderen (verkeerde data)
    train.drop(columns=['Fare_per_person'], inplace=True, errors='ignore')

    # Nieuwe kolom hernoemen naar juiste naam
    train.rename(columns={'AdjustedFare': 'Fare_per_person'}, inplace=True)

    train.drop(columns=['FareWeight', 'TotalFareWeight', 'Fare_per_weight'], inplace=True, errors='ignore')
    # Tarieven verschillen per leeftijdsgroep
    train['FareWeight'] = np.select(
        [train['Age_Group'] == 'Infant', train['Age_Group'] == 'Child', train['Age_Group'] == 'Adult'],
        [0.0, 0.5, 1.0]
    )

    # Berekening van het totale gewicht per ticket (gehele ticket)
    train['TotalFareWeight'] = train.groupby('Ticket')['FareWeight'].transform('sum')

    # Berekening van de waarde per gewichtseenheid (tarief per volwassene)
    train['Fare_per_weight'] = train['Fare'] / train['TotalFareWeight']

    # Berekening van de aangepaste fare (0.0 en 0.5 zodat ticket eerlijk verdeeld wordt)
    train['AdjustedFare'] = train['Fare_per_weight'] * train['FareWeight']

    # Oude kolom verwijderen (verkeerde data)
    train.drop(columns=['Fare_per_person'], inplace=True, errors='ignore')

    # Nieuwe kolom hernoemen naar juiste naam
    train.rename(columns={'AdjustedFare': 'Fare_per_person'}, inplace=True)

    train.drop(columns=['FareWeight', 'TotalFareWeight', 'Fare_per_weight'], inplace=True, errors='ignore')
    train.drop(columns=['Cabin'])
    train['Embarked'] = train['Embarked'].replace({
    'S': 'Southampton',
    'C': 'Cherbourg',
    'Q': 'Queenstown'
    })
    
    tab3, tab4, tab5 = st.tabs(['Data verwerking', 'Demografische verdeling', 'Invloedrijke factoren']) 
    with tab3:
        st.title('text')
    
    with tab4:
        # --- TITEL & INLEIDING ---
        st.title("Demografische verdeling van Titanic-passagiers")
        st.markdown("""
        Hieronder bekijken we de verdeling van leeftijden op de Titanic, uitgesplitst naar **geslacht** en **overlevingsstatus**.
        """)
        
        # --- PLOT 1: Leeftijdsverdeling per geslacht ---
        st.markdown("### Leeftijdsverdeling per geslacht")
        
        fig1, ax1 = plt.subplots(figsize=(8,5))
        sns.histplot(
            data=train,
            x='Age',
            hue='Sex',
            kde=True,
            bins=25,
            alpha=0.8,
            multiple='dodge',
            palette=["#FF8345", "#08675B88"],
            ax=ax1
        )
        ax1.set_title('Leeftijdsverdeling per geslacht')
        ax1.set_xlabel("Leeftijd")
        ax1.set_ylabel("Aantal passagiers")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)
        
        # Eventueel wat uitleg onder de plot
        st.caption("""
        Vrouwen lijken gemiddeld iets jonger in de dataset, en de verdeling is breder bij mannen.
        """)
        
        # --- PLOT 2: Leeftijdsverdeling per overleving ---
        st.markdown("### Leeftijdsverdeling per overleving")
        
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.histplot(
            data=train,
            x='Age',
            hue='Survived',
            kde=True,
            bins=25,
            alpha=0.8,
            multiple='dodge',
            palette=["#08675B88", "#FF8345"],  # kleuren omgedraaid voor contrast
            ax=ax2
        )
        ax2.set_title('Leeftijdsverdeling per overleving')
        ax2.set_xlabel("Leeftijd")
        ax2.set_ylabel("Aantal passagiers")
        ax2.legend(title="Overleefd", labels=["Nee", "Ja"])
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig2)

        st.markdown("### Overlevingspercentage per leeftijdsgroep en klasse")

        # Plot maken
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(
            data=train,
            x='Age_Group',
            y='Survived',
            hue='Pclass',
            palette=["#08675B", "#E3DF00FD", "#FF8345"],
            ax=ax
        )
        

        st.header("Relatie tussen leeftijd en ticketprijs (3e klasse)")
        
        # Filter enkel passagiers van de 3e klasse
        derde_klasse = train[train["Pclass"] == 3]
    
        # Plot maken
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='Age',
            y='Fare',
            data=train,
            hue=train['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'}),
            palette=["#08675B88", "#FF8345"],
            ax=ax
        )
        ax.set_xlabel('Leeftijd')
        ax.set_ylabel('Ticketprijs (£)')
        ax.set_title('Relatie tussen leeftijd en ticketprijs (3e klasse)')
        ax.grid(alpha=0.5, linestyle='--')
        
        st.pyplot(fig)

        g = sns.catplot(
        x='Embarked',
        hue='Survived',
        col='Pclass',
        kind='count',
        data=train,
        palette=["#08675B88", "#FF8345"],  # jouw kleuren
        hue_order=[1, 0],                  # zodat oranje (Nee) bovenop ligt
        height=5,
        aspect=0.9
        )
    
        # Titels en labels
        g.fig.suptitle('Aantal passagiers per haven, klasse en overleving', fontsize=14, y=1.03)
        g.set_axis_labels("Inschepingshaven", "Aantal passagiers")
    
        # Voeg getallen toe op de balken
        for ax in g.axes.flat:
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=9, color='black', padding=2)
    
        # Rasters toevoegen voor leesbaarheid
        for ax in g.axes.flat:
            ax.grid(axis='y', linestyle='--', alpha=0.6)
    
        # --- Plot tonen in Streamlit ---
        st.pyplot(g)

    with tab5:
        # Labels en layout
        ax.set_title('Overlevingspercentage per leeftijdsgroep en klasse')
        ax.set_xlabel('Leeftijdsgroep')
        ax.set_ylabel('Overlevingskans')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(title='Klasse', loc='upper right')
        plt.tight_layout()
        
        # Streamlit render
        st.pyplot(fig)
        train['Survived_label'] = train['Survived'].map({0: 'Niet overleefd', 1: 'Overleefd'})

        # Catplot maken
        g = sns.catplot(
            x='Embarked',
            hue='Survived_label',   # gebruik de gelabelde kolom
            kind='count',
            col='Pclass',
            data=train,
            palette=["#08675B", "#FF8345"]
        )
        
        # Titels en layout
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle('Aantal overlevenden per Embarked en Pclass', fontsize=16)
        g.set_axis_labels("Haven van inscheping", "Aantal passagiers")
        g._legend.set_title("Overleving")
        
        # Streamlit renderen
        st.pyplot(g.fig)
        
        # plot corr
        train_num = train[['Age', 'SibSp', 'Parch', 'Fare_per_person']]
        st.title('Invloedrijke factoren')
        
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(train_num.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlatie matrix van numerieke waarden")
        
        # Streamlit render
        st.pyplot(fig)
    
      
    
























