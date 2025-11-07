import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    


pagina = st.sidebar.radio("Ga naar:", ["Introductie", "Oude case", 'Nieuwe verkenning', "Ons model"])
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
elif pagina == 'Oude case':
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


elif pagina == 'Nieuwe verkenning':
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
    
    # plot catplot
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

    with tab5:
        st.title('Invloedrijke factoren')

        # --- Barplot Overleving per Age_Group en Pclass ---
        fig1, ax1 = plt.subplots(figsize=(8,5))
        sns.barplot(
            data=train,
            x='Age_Group',
            y='Survived',
            hue='Pclass',
            palette=["#08675B","#E3DF00","#FF8345"],
            ax=ax1
        )
        ax1.set_title('Overlevingspercentage per leeftijdsgroep en klasse')
        ax1.set_xlabel('Leeftijdsgroep')
        ax1.set_ylabel('Overlevingskans')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.legend(title='Klasse', loc='upper right')
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.title("Overlevingsanalyse per haven en klasse")
        g = sns.catplot(
        x='Embarked',
        hue='Survived',      # originele kolom 0/1
        kind='count',
        col='Pclass',
        data=train,          # geen aparte train_plot
        palette=["#08675B", "#FF8345"],
        height=5,
        aspect=0.9
        )
        
        # --- Pas legend labels aan (zonder extra kolom) ---
        for ax in g.axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=["Niet overleefd","Overleefd"], title="Overleving")
        
        # --- Titel en labels ---
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle('Aantal overlevenden per haven van inscheping en klasse', fontsize=16)
        g.set_axis_labels("Haven van inscheping", "Aantal passagiers")
        
        # --- Render in Streamlit ---
        st.pyplot(g.fig)

        # Correlatie matrix
        st.title("Correlatie matrix van numerieke waarden")
        train_num = train[['Age','SibSp','Parch','Fare']]
        # --- Heatmap plotten ---
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(train_num.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlatie matrix van numerieke waarden")
        
        # --- Render in Streamlit ---
        st.pyplot(fig)

elif pagina == 'Ons model':
    # --- Data inladen ---
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # --- Preprocessing ---
    train['Age'] = train.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mean()))
    test['Age'] = test.groupby('Sex')['Age'].transform(lambda x: x.fillna(x.mean()))
    
    # Drop irrelevante kolommen
    train = train.drop(columns=['Cabin', 'Name', 'Ticket'])
    test = test.drop(columns=['Cabin', 'Name', 'Ticket'])
    
    # Sex numeriek
    train['Sex'] = train['Sex'].map({'male':0,'female':1})
    test['Sex'] = test['Sex'].map({'male':0,'female':1})
    
    # Travel_Alone feature
    train['Travel_Alone'] = np.where((train['SibSp'] + train['Parch']) == 0, 1, 0)
    test['Travel_Alone'] = np.where((test['SibSp'] + test['Parch']) == 0, 1, 0)
    
    # Embarked one-hot encoding
    train = pd.get_dummies(train, columns=['Embarked'], drop_first=False)
    test = pd.get_dummies(test, columns=['Embarked'], drop_first=False)

    # Alle kolommen van train aan test toevoegen (die ontbreken) met 0
    for col in features:
        if col not in test.columns:
            test[col] = 0

    # Sorteer kolommen zodat de volgorde exact overeenkomt
    X_test = test[features]
    # --- Streamlit UI ---
    st.title("Logistic Regression Model - Titanic")
    st.markdown("""
    We trainen een **logistisch regressiemodel** om te voorspellen wie de Titanic heeft overleefd.  
    Hier zie je dataset preprocessing, modeltraining, evaluatie en feature importance.
    """)
    
    # Features en target
    target = "Survived"
    features = [col for col in train.columns if col != target and col != "PassengerId"]
    X = train[features]
    y = train[target]
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model trainen
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # --- Evaluatie ---
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    st.subheader("Accuracy op validation set")
    st.success(f"{acc:.3f}")
        
    # Feature importance
    st.subheader("Feature importance (coefficients)")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)
    st.dataframe(coef_df)
    
    # Visualisatie coefficients
    fig_coef, ax_coef = plt.subplots(figsize=(8,6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, palette="viridis", ax=ax_coef)
    ax_coef.set_title("Feature impact op overleving")
    st.pyplot(fig_coef)
    
    # --- Voorspelling testset ---
    X_test = test[features]
    test_pred = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_pred.astype(int)
    })
    
    st.subheader("Download voorspellingen")
    st.download_button(
        label="Download submission.csv",
        data=submission.to_csv(index=False).encode('utf-8'),
        file_name="submission.csv",
        mime="text/csv"
    )
    











