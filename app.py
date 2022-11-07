import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

df = pd.read_csv("iris.csv")

y = df['species']
X = df.drop ('species', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X_train, y_train)

st.title(" Les iris ")

sepal_length = st.slider("sepal_length : ", min_value= 4., max_value= 7., step= 0.1 )
sepal_width = st.slider("sepal_width : ", min_value= 2., max_value= 4., step= 0.1 )
petal_length = st.slider("petal_length : ", min_value= 1., max_value= 5., step= 0.1 )
petal_width = st.slider("petal_width : ", min_value= 0.1, max_value= 3., step= 0.1 )

choix_en_cours = {
    "sepal_length": [sepal_length],
    "sepal_width" : [sepal_width],
    "petal_length" : [petal_length],
    "petal_width" : [petal_width],
    "species" : "Votre iris"
    }

choix_en_cours = pd.DataFrame(choix_en_cours)
choix_en_cours = knn.predict([[sepal_length, sepal_width , petal_length, petal_width]])

if st.button('Calcul'):
    st.success(choix_en_cours[0])