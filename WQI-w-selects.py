import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# necesario para bajar de github
import requests
import io


# iris_data = load_iris()

#importamos los dataframes features y target de github
url_features = "https://raw.githubusercontent.com/amador2001/WQI-streamlit-app/master/features.csv?token=ACPSY65VY4AVIMWMCTYAEYDBQKXZI" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url_features).content
features = pd.read_csv(io.StringIO(download.decode('utf-8')))

url_target ="https://raw.githubusercontent.com/amador2001/WQI-streamlit-app/master/target.csv?token=ACPSY63PWCLKYMNAE26KMBDBQK6DY"
download_2 = requests.get(url_target).content
target = pd.read_csv(io.StringIO(download_2.decode('utf-8')))

# # dividimos ambos dataframes en sus sibconjuntos train and test
#x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)

print(len(features))
print(len(target))

