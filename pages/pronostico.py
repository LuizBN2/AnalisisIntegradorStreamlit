import pandas as pd
import utilidades as util
import streamlit as st

util.generarMenu()

st.title('Síndrome Metabólico de Enfermedad Cardiovascular')
df = pd.read_csv('data/Datos_Pacientes.csv', index_col=0)

util.modelo_rf(df)