import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generarMenu():
    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("media/icono_pag2.png")
            st.image(image, use_container_width=False)
        with col2:
            st.header('SMEC')

        st.page_link('app.py', label='Inicio')
        st.page_link('pages/pronostico.py', label='Pronóstico')

#Función del modelo predictivo

def modelo_rf(df_p):    
    st.markdown('## Datos enfermedades de pacientes')
    st.write(df_p.head())
    st.subheader('Resultado del modelo Random Forest')
    #Variable a predecir
    Y = df_p.iloc[:,0]
    #Variables predictoras
    X = df_p.iloc[:,1:]
    #Variables de prueba -> prueba
    #variables de entrenamiento -> entrenar
    X_entrenar, X_prueba, Y_entrenar, Y_prueba = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

    #st.markdown('### Separamos los datos')
    st.write('Datos de entrenamiento')
    st.info(f'Muestra de las variables predictoras de entrenamiento: {X_entrenar.shape[0]} datos')
    st.info(f'Muestra de las variables predictoras de prueba: {X_prueba.shape[0]} datos')

    st.write('Datos de prueba')
    st.info(f'Muestra de la variable a predecir de entrenamiento: {Y_entrenar.shape[0]} datos')
    st.info(f'Muestra de la variable a predecir de  prueba: {Y_prueba.shape[0]} datos')


    st.write('Variables Predictoras')
    lista = list(X.columns)  
    delim = ", "
    text = delim.join(lista)
    st.info(text)

    st.write('Variable a Predecir')
    st.info(Y.name)

    #Creamos el bosque
    bosque = RandomForestClassifier()
    #Entrenar el bosque
    bosque.fit(X_entrenar,Y_entrenar)

    #Hacemos la predicción
    Y_prediccion = bosque.predict(X_prueba)
    accuracy = accuracy_score(Y_prueba,Y_prediccion)
    st.write('Métrica de precisión de puntos obtenidos para Predicción de la enfermedad')
    st.info(accuracy)
