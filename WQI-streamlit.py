
import pandas as pd
import streamlit as st
#import plotly.express as px
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import base64
import random
from random import randrange, uniform
import plotly.graph_objects as go

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


#from xgboost import XGBRegressor

#from streamlit.ScriptRunner import StopException, RerunException



#if st.button():
  # do other stuff
  #raise RerunException()  # This causes the app to rerun



st.set_page_config(
	page_title="Ex-stream-ly Cool App",
	page_icon="",
	layout="wide",
	initial_sidebar_state="auto",
)



st.markdown("""
     <style>
    body {
      color: #333;
       background-color: #ffffff;
     }
   </style>
      """, unsafe_allow_html=True)

# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    '''
        <style>
            .sidebar.sidebar-content {{
                width: 20px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

# st.image("logo_cemex.jpg")


html_temp = """
<div style="background-color:black;padding:10px">
<h1 style="color:white;text-align:center;">WATER QUALITY PREDICTOR </h1>
</div>
"""
#st.image("images/LVR_tech_logo.png")
st.markdown(html_temp,unsafe_allow_html=True)

boton="""<button style="height:50px; width=100px;">Predict</button>"""
st.subheader("Titulo")

grafica = st.image("images/scatter_plots.png")

st.sidebar.header('Please, set observable water measurements')

st.markdown("<h1 style='text-align: center; color: black;'>Water samples dataset</h1>", unsafe_allow_html=True)

data = pd.read_csv("ALH_w_dataset.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
# df.drop(columns=['Unnamed: 0'])
st.text("")
# st.title("Water samples dataset")  # add a title
# st.text(" \n")
st.write(data)


# -----------------------------  water -------
# estos son los INPUTS ---> ['WQI FC', 'WQI Oxy','WQI pH', 'WQI TSS', 'WQI Temp', 'WQI Turb', 'WQI TPN', 'WQI TP']



st.sidebar.subheader("WQI FC" + ": " + "Random Inizialized")
valueFC = st.sidebar.slider("", 0.0 , 100.0 ,float(random.uniform(0,100)))

st.sidebar.subheader("Oxygene" + ": " + "1234")
valueOxy = st.sidebar.slider("", 0.0 , 100.0, float(random.uniform(0,100)))

st.sidebar.subheader("pH" + ": " + "1234")
valuepH = st.sidebar.slider("", 0.0 , 100.0, float(random.uniform(0,100)))

st.sidebar.subheader("TSS" + ": " + "1234")
valueTSS = st.sidebar.slider("", 0.0 , 100.0 , float(random.uniform(0,100)))

st.sidebar.subheader("Temperature" + ": " + "1234")
valueTemperature = st.sidebar.slider("",0.0 , 100.0, float(random.uniform(0,100)))

st.sidebar.subheader("Turbidity" + ": " + "1234")
valueTurb = st.sidebar.slider("", 0.0 , 100.0 , float(random.uniform(0,100)))

st.sidebar.subheader("TPN" + ": " + "1234")
valueTPN = st.sidebar.slider("", 0.0 , 100.0 , float(random.uniform(0,100)))

st.sidebar.subheader("TP" + ": " + "1234")
valueTP = st.sidebar.slider("", 0.0 , 100.0 , float(random.uniform(0,100)))


#st.sidebar.subheader("Target Free Lime" + ": " + "1234")
#st.sidebar.subheader(": ".format(float(random.uniform(315.62,452.59))))
#st.sidebar.write(0234)
#valuefreelime=st.sidebar.slider(" ", 0.00,2.36 ,float(random.uniform(0.00,2.36)), step=0.1)


st.sidebar.header('Compute Water Quality Index of the Measurement')

#v_custom=st.custom_slider('Hello world',0,100,50,key="slider1")
#st.write(v_custom)

# recogemos los 8 valores

FC=valueFC
Oxy=valueOxy
pH=valuepH

TSS=valueTSS
Temperature=valueTemperature
Turb=valueTurb

TPN=valueTPN
TP=valueTP


# el modelo y el scaler 
best_model = pickle.load(open('modelo_normalizado.pkl', 'rb'))
min_max_scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))
#min_max_scaler_free_lime = pickle.load(open('min_max_scaler_free_lime.pkl', 'rb'))
#modelo_freelime=pickle.load(open('modelo_freelime.pkl','rb'))


# en el formulario aparecen en DIFERENTE ORDEN DE INPUT VARS
#list_features = [LSF,feed,free_lime_cat,fin,MS_rm]

# recogemos los valores para meterlos al regresor best_model
list_features = [FC, Oxy, pH, TSS, Temperature, Turb, TPN, TP ]
#list_features = [0.5, 0.5,0.5,0.5,0.5,0.5,0.5,0.5 ]

print("list_features = ", list_features)
	# hacemos la predicción a partir de las FEATURES RECOGIDAS DEL FORMULARIO index.html
WQI = best_model.predict(min_max_scaler.transform([list_features]))[0]




#DESPLEGAR LOS RESULTADOS

if st.sidebar.button("Predict"):


	col1, col2 ,col3= st.columns([1, 1,1])
	grafica.empty()

	
	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number",
    	value = WQI,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "QWI", 'font': {'size': 32}},
    	delta = {'reference':0, 'increasing': {'color': "RebeccaPurple"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "red"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0,WQI], 'color': 'red'},
            	#{'range': [0,100], 'color': 'red'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col2.plotly_chart(fig)



	