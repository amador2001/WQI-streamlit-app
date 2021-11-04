import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import pickle

# necesario para bajar de github
import requests
import io

#importamos los dataframes features y target de github
url_features = "https://raw.githubusercontent.com/amador2001/WQI-streamlit-app/master/features.csv?token=ACPSY65VY4AVIMWMCTYAEYDBQKXZI" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url_features).content
features = pd.read_csv(io.StringIO(download.decode('utf-8')))

url_target ="https://raw.githubusercontent.com/amador2001/WQI-streamlit-app/master/target.csv?token=ACPSY63PWCLKYMNAE26KMBDBQK6DY"
download_2 = requests.get(url_target).content
target = pd.read_csv(io.StringIO(download_2.decode('utf-8')))

data = features.merge(target, how='inner', left_index=True, right_index=True)

lista_cat = ["Poor", "Marginal","Fair","Good", "Excellent" ]  # para corresponderse con 0,1,2,3,4

# # dividimos ambos dataframes en sus sibconjuntos train and test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

print(len(features))
print(len(target))

class StreamlitApp:

    def __init__(self):

        self.model = RandomForestClassifier()

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        # Para cada uno de los inputs vamos extrayendo los valores UNICOS 

        st.sidebar.markdown(
            '<p class="subheader-style">Please, set the observable values</p>',
            unsafe_allow_html=True
        )
        FC = st.sidebar.selectbox(
            #f"Select {cols[0]}",
            f"Select Phosphorus (FC) Level",
            sorted(features[cols[0]].unique())
        )

        Oxy = st.sidebar.selectbox(
            #f"Select {cols[1]}",
            f"Select Oxygene Level",
             
            sorted(features[cols[1]].unique())
        )

        pH = st.sidebar.selectbox(
            #f"Select {cols[2]}",
            f"Select pH",
            sorted(features[cols[2]].unique())
        )

        TSS = st.sidebar.selectbox(
            #f"Select {cols[3]}",
            f"Select Total Suspended Sediment",
            sorted(features[cols[3]].unique())
        )

        Temperature = st.sidebar.selectbox(
            #f"Select {cols[4]}",
            f"Select Temperature",
            sorted(features[cols[4]].unique())
        )

        TPN = st.sidebar.selectbox(
            #f"Select {cols[5]}",
             f"Select Nitrogen (TPN)",
            sorted(features[cols[5]].unique())
        )

        TP = st.sidebar.selectbox(
            #"Select {cols[6]}",
            f"Select  TP",
            sorted(features[cols[6]].unique())
        )

        Turb = st.sidebar.selectbox(
            #f"Select {cols[7]}",
            f"Select  Turbidity",
            sorted(features[cols[7]].unique())
        )

        values = [FC, Oxy, pH, TSS, Temperature, TPN, TP,Turb ]

        return values

    def plot_pie_chart(self, probabilities):
        fig = go.Figure(
            data=[go.Pie(
                    labels=lista_cat,
                    values=probabilities[0]
            )]
        )
        fig = fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value',
            textfont_size=15
        )
        return fig   

    def construct_app(self):

        st.image("images/elgressy_one_logo-2.png")

        # --------- ejecutamos funcion de entrenar datos del CLASIFICADOR -----------
        self.train_data()
        # recogemos los 8 valores de los inputs del sidebar
        values = self.construct_sidebar()
        print("values = ", values)

        # los metemos dentro de una lista [] para pasarlos al clasificador
        values_to_predict = np.array(values).reshape(1, -1) 
        #values_to_predict = [[91, 85, 95, 78, 81, 99, 84, 69]]  # para chequeo
        print("values_to_predict = ", values_to_predict)  # ya metidos en una lista los inputs

        prediction = self.model.predict(values_to_predict)  # la prediccion
        print("prediction = ", prediction)

        prediction_str = lista_cat[prediction[0]]
        print("prediction_str = ", prediction_str)

        probabilities = self.model.predict_proba(values_to_predict)

        # -------  ejecutamos funcion de entrenar datos del REGRESOR WQI -----------
        xgb = pickle.load(open('modelo_XGBRegressor.pkl', 'rb'))
        #prediccion_QWI = round(xgb.predict([[90, 95, 63, 83, 97, 96, 80, 80]])[0][0])  # Asi se usa
        prediccion_QWI = round(xgb.predict(values_to_predict)[0][0])
        print("prediccion_QWI = ", prediccion_QWI)



        # ---  FIN PREDICCIONES ----------------------------------------------------- 


        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Water Quality Prediction  </p>',
            unsafe_allow_html=True
        )

        #column_1, column_2 = st.columns(2)
        column_1, column_2, column_3 = st.columns(3)

        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")


        column_3.markdown(
            '<p class="font-style" >WQI</p>',
            unsafe_allow_html=True
        )
        column_3.write(f"{prediccion_QWI}")

        fig = self.plot_pie_chart(probabilities)
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

        return self
# --------------------------------------------------------------
# import plotly.express as px
# fig = px.scatter(data, x="WQI TPN", y="Overall WQI")
# fig.show()




sa = StreamlitApp()
sa.construct_app()

