
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import base64
import random
from random import randrange, uniform
import plotly.graph_objects as go
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
<h1 style="color:white;text-align:center;">KILN ENERGY OPTIMIZATION MODEL </h1>
</div>
"""
#st.image("logo_cemex.jpg")
st.markdown(html_temp,unsafe_allow_html=True)

boton="""<button style="height:50px; width=100px;">Predict</button>"""

st.subheader("Please, press Predict to start or update optimal scenarios")

grafica = st.image("costes-med-3.jpg")

st.sidebar.header('Choose Operation Constraints')

#st.markdown("<h1 style='text-align: center; color: black;'>DATAFRAME</h1>", unsafe_allow_html=True)

#df = pd.read_csv("./datos2018.csv")  # read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
# df.drop(columns=['Unnamed: 0'])
# st.text("")
# st.title("DATAFRAME")  # add a title
# st.text(" \n")
# st.text(" \n")
# st.text(" \n")
# st.write(df)
#df = pd.read_csv("./dfoptimo.csv")

st.sidebar.subheader("LSF" + ": " + "1234")


valueLSF = st.sidebar.slider("", 100.09,108.04,float(random.uniform(100.09,108.04)))

st.sidebar.subheader("Feed" + ": " + "1234")
valuefeed=st.sidebar.slider("", 315.62,452.69 , float(random.uniform(315.62,452.59)))
st.sidebar.subheader("Target Free Lime" + ": " + "1234")
#st.sidebar.subheader(": ".format(float(random.uniform(315.62,452.59))))
#st.sidebar.write(0234)

valuefreelime=st.sidebar.slider(" ", 0.00,2.36 ,float(random.uniform(0.00,2.36)), step=0.1)


st.sidebar.subheader("Fineness" + ": " + "1234")
#HAY QUE MODIFICAR ESTE VALOR
fineness=st.sidebar.slider(" ",17.98,21.52 ,float(random.uniform(17.98,21.52)))
st.sidebar.subheader("MSI" + ": " + "1234")
MSI=st.sidebar.slider(" ",2.25,2.71 ,float(random.uniform(2.25,2.71)))


st.sidebar.header('Compute Optimal Values')

#v_custom=st.custom_slider('Hello world',0,100,50,key="slider1")
#st.write(v_custom)


LSF=valueLSF
feed=valuefeed
free_lime_cat=valuefreelime
fin=fineness
MS_rm=MSI


best_model = pickle.load(open('modelo_normalizado.pkl', 'rb'))
min_max_scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))
min_max_scaler_free_lime = pickle.load(open('min_max_scaler_free_lime.pkl', 'rb'))
modelo_freelime=pickle.load(open('modelo_freelime.pkl','rb'))


# en el formulario aparecen en DIFERENTE ORDEN DE INPUT VARS
list_features = [LSF,feed,free_lime_cat,fin,MS_rm]


	# hacemos la predicción a partir de las FEATURES RECOGIDAS DEL FORMULARIO index.html
Result_pred = best_model.predict(min_max_scaler.transform([list_features]))


#VALORES RANDOM ESTOS SE VAN A SUSTITUIR POR LOS QUE HAYA EN ESE MOMENTO

valorprim_real=(round(random.uniform(6.071066,32.38845), 5))
valorsec_real=(round(random.uniform(3.033211,32.26933), 5))
prod_rate=(round(random.uniform(192.4504,275.6052), 5))
energia_total_real=round((8180 * valorprim_real + 3800 * valorsec_real) / prod_rate ,2)
costo_real= 1*valorprim_real+ .75*valorsec_real

# creamos una columna ponderada
cost_weighted_real= costo_real / prod_rate
kilnspeed=(round(random.uniform(2.43,3.53), 5))
VTI=(round(random.uniform(673.43,821.60), 5))




#VALORES OPTIMOS
output=(Result_pred[0].tolist())
prim_pred=output[0]
sec_pred=output[1]
pred_cost_weighted=1*prim_pred+ .75*sec_pred/prod_rate
energia_total_pred=output[2]

#boton3="""<button style="display:inline-block;border:1px solid rgb(255, 255, 255);border-radius:16px;padding:28px;background:linear-gradient(to bottom,rgb(21, 17, 17),rgb(8, 7, 7));color:rgb(255, 255, 255)">Predict</button>"""


#botonj=st.markdown(boton3,unsafe_allow_html=True)

#MODELO DE FREE LIME
prim=valorprim_real
sec=valorsec_real
fan=VTI
k_rot=kilnspeed
list2_features=[prim,sec,fan,k_rot]
#list2_features=[28.288,9.034,758.833,3.042]

free_lime_pred = modelo_freelime.predict(np.array(list2_features).reshape((1,-1)))[0]
print("free_lime_pred = ", free_lime_pred)
#st.write(freelime_pred)


#DESPLEGAR LOS RESULTADOS

if st.sidebar.button("Predict"):


	col1, col2 ,col3= st.beta_columns([1, 1,1])
	grafica.empty()

	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number+delta",
    	value = prim_pred,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Primary fuel", 'font': {'color':"darkblue",'size': 32}},
    	delta = {'reference': valorprim_real, 'increasing': {'color': "red"}},
    	gauge = {
        	'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "darkblue"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0, valorprim_real], 'color': 'cyan'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	'threshold': {
            	'line': {'color': "red", 'width': 6},
            	'thickness': 0.75,
            	'value': prim_pred}}))

	fig.update_layout(paper_bgcolor = "white", width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	#st.write(valorprim_real)

	col1.plotly_chart(fig)


	#st.write('valor real',valorsec_real)
	#st.write(output[1])
	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number+delta",
    	value = sec_pred,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Altern fuel", 'font': {'color':"darkblue",'size': 32}},
    	delta = {'reference':(valorsec_real), 'increasing': {'color': "red"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "darkblue"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0, valorsec_real], 'color': 'cyan'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	'threshold': {
            	'line': {'color': "red", 'width': 6},
            	'thickness': 0.75,
            	'value': sec_pred}}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col2.plotly_chart(fig)
	#st.write(valorsec_real)


	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number+delta",
    	value = energia_total_pred,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Total Energy", 'font': {'color':"darkblue",'size': 32}},
    	delta = {'reference':(energia_total_real), 'increasing': {'color': "red"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 1200], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "darkblue"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0, energia_total_real], 'color': 'cyan'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	'threshold': {
            	'line': {'color': "red", 'width': 6},
            	'thickness': 0.75,
            	'value': energia_total_pred}}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col3.plotly_chart(fig)

	#st.write('Average Cost Saving is = ',  round(df_savings["SAVINGS-%"].mean(),1), '%')
	SAVINGS=( cost_weighted_real - (prim_pred +  0.301 *sec_pred) / prod_rate) / cost_weighted_real * 100
	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number",
    	value = kilnspeed,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Kiln Speed", 'font': {'size': 32}},
    	delta = {'reference':0, 'increasing': {'color': "RebeccaPurple"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 4], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "darkblue"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0,kilnspeed], 'color': 'cyan'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col1.plotly_chart(fig)
	#st.write(SAVINGS)

	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number",
    	value = SAVINGS,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Savings (%)", 'font': {'size': 32}},
    	delta = {'reference':0, 'increasing': {'color': "RebeccaPurple"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "red"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0,SAVINGS], 'color': 'red'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col2.plotly_chart(fig)



	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number",
    	value = VTI,
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "VTI", 'font': {'size': 32}},
    	delta = {'reference':0, 'increasing': {'color': "RebeccaPurple"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 900], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "darkblue"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0,VTI], 'color': 'cyan'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	}))

	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col3.plotly_chart(fig)

    #FREE LIME



	fig = go.Figure(go.Indicator(
   	 	mode = "gauge+number+delta",
    	value = float(free_lime_pred),
    	domain = {'x': [0, 1], 'y': [0, 1]},
    	title = {'text': "Predicted Free Lime", 'font': {'color':"darkblue",'size': 32}},
    	delta = {'reference':(free_lime_cat), 'increasing': {'color': "red"}},
    	#delta=valorprim_real,
    	gauge = {
        	'axis': {'range': [None, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
        	'bar': {'color': "red"},
        	'bgcolor': "white",
        	'borderwidth': 2,
        	'bordercolor': "gray",
        	'steps': [
            	{'range': [0, free_lime_cat], 'color': 'red'},
            	#{'range': [250, 400], 'color': 'royalblue'}
            	],
        	'threshold': {
            	'line': {'color': "red", 'width': 6},
            	'thickness': 0.75,
            	'value': float(free_lime_pred)}}))
	fig.update_layout(paper_bgcolor = "white",  width=500,
    height=300,font = {'color': "darkblue", 'family': "Arial"})
	col1.plotly_chart(fig)
