import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#streamlit run Dashboard_app_choferes.py


st.title('Dashboard app choferes')
st.markdown('Dashboard para el siguimiento del proyecto **App choferes**')
st.sidebar.header('Filtros de los dashboards')

#https://docs.google.com/spreadsheets/d/e/2PACX-1vSy_eJ0cnDhdoT2NhTDLvld0KPFfLmMpEN1HgFJNMiORBU_5nd1u4Qf1EcZOrrH5Q/pubhtml

@st.cache_data
def load_data():
    url="https://docs.google.com/spreadsheets/d/e/2PACX-1vSy_eJ0cnDhdoT2NhTDLvld0KPFfLmMpEN1HgFJNMiORBU_5nd1u4Qf1EcZOrrH5Q/pubhtml"
    html=pd.read_html(url, header=4)
    df=html[0]
    df=df.dropna(subset=['No','CLAVE','UNIDAD','PLACAS', 'OPERADOR LOCAL','PESO UNIDAD','Foraneo/local'])
    df = df[['No','CLAVE','UNIDAD','PLACAS', 'OPERADOR LOCAL','PESO UNIDAD','Foraneo/local']]
    return df

def load_DBchoferes():
    url="https://docs.google.com/spreadsheets/d/e/2PACX-1vSy_eJ0cnDhdoT2NhTDLvld0KPFfLmMpEN1HgFJNMiORBU_5nd1u4Qf1EcZOrrH5Q/pubhtml"
    html=pd.read_html(url, header=1)
    DBchoferes=html[1]
    return DBchoferes

df=load_data()
DBchoferes=load_DBchoferes()

DBchoferes = pd.merge(DBchoferes, df[['OPERADOR LOCAL', 'Foraneo/local']], left_on='Nombre del chofer', right_on='OPERADOR LOCAL', how='left')
DBchoferes=DBchoferes.drop(columns=['1','Referencia del domicilio','Chofer de llegada','Chofer de salida','Etapa','Observaciones'])
DBchoferes=DBchoferes.drop(columns=['Lat. Llegada','Long. Llegada', 'Lat. Salida', 'Long. Salida','Desv. salida mt.','Desv. llegada mt.'])
DBchoferes[['Horas_llegada', 'Minutos_llegada', 'Segundos_llegada']] = DBchoferes['Hora llegada'].str.split(':', expand=True)
DBchoferes[['Horas_Salida', 'Minutos_Salida', 'Segundos_Salida']] = DBchoferes['Hora Salida'].str.split(':', expand=True)
DBchoferes=DBchoferes.drop(columns=['Hora llegada','Hora Salida','Segundos_llegada','Segundos_Salida'])


DBchoferes['Horas_llegada']=DBchoferes['Horas_llegada'].astype(float)
DBchoferes['Horas_Salida']=DBchoferes['Horas_Salida'].astype(float)
DBchoferes['Hora llegada min']=DBchoferes['Horas_llegada']*60
DBchoferes['Hora Salida min']=DBchoferes['Horas_Salida']*60
DBchoferes=DBchoferes.drop(columns=['Horas_llegada','Horas_Salida'])


DBchoferes['Minutos_llegada']=DBchoferes['Minutos_llegada'].astype(float)
DBchoferes['Minutos_Salida']=DBchoferes['Minutos_Salida'].astype(float)
DBchoferes['Hora llegada']=DBchoferes['Hora llegada min']+DBchoferes['Minutos_llegada']
DBchoferes['Hora Salida']=DBchoferes['Hora Salida min']+DBchoferes['Minutos_Salida']
DBchoferes['Tiempo total']=DBchoferes['Hora Salida']-DBchoferes['Hora llegada']
DBchoferes=DBchoferes.drop(columns=['Minutos_llegada','Minutos_Salida','Hora llegada min','Hora Salida min','Hora llegada','Hora Salida'])
DBchoferes['Fecha de entrega'] = pd.to_datetime(DBchoferes['Fecha de entrega'], format='%d/%m/%Y')


Fecha_min=DBchoferes['Fecha de entrega'].min().date()
Fecha_max=DBchoferes['Fecha de entrega'].max().date()
Start_date = st.sidebar.date_input('Fecha de inicio', value=Fecha_min, min_value=Fecha_min, max_value=Fecha_max)
End_date = st.sidebar.date_input('Fecha de fin', value=Fecha_max, min_value=Fecha_min, max_value=Fecha_max)
tipo_entrega = st.sidebar.selectbox('Tipo de entrega', ['Todos', 'local', 'Foraneo'])
filtered_DB = DBchoferes[(DBchoferes['Fecha de entrega'] >= pd.to_datetime(Start_date)) & (DBchoferes['Fecha de entrega'] <= pd.to_datetime(End_date))]


filtered_DB['Uso_app'] = 0
filtered_DB.loc[filtered_DB['Tiempo total'] > 0, 'Uso_app'] = 1


if tipo_entrega != 'Todos':
    filtered_df = filtered_DB[filtered_DB['Foraneo/local'] == tipo_entrega]
    
    Uso_de_la_app=filtered_df['Uso_app'].sum()
    Datos_df=filtered_df['Uso_app'].count()
    Porcentaje_uso=round(((Uso_de_la_app/Datos_df)*100),2)
    Porcentaje_max=(Datos_df/Datos_df)*100
    color_gauge = "#522d6d"
    color_gray = "#e5e1e6"
    color_threshold = "red"
    fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=Porcentaje_uso,
            title={'text': "Indicador de Medidor"},
            gauge={
                'axis': {'range': [0, Porcentaje_max]},
                'bar': {'color': color_gauge},
                'steps': [
                    {'range': [0, 50], 'color': color_gray},
                    {'range': [50, 100], 'color': color_gray}],
                'threshold': {
                    'line': {'color': color_threshold, 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
    #st.plotly_chart(fig)
    df_con_1 = filtered_df[filtered_df['Uso_app'] == 1]
    df_con_0 = filtered_df[filtered_df['Uso_app'] == 0]

    conteo_con_1 = df_con_1['OPERADOR LOCAL'].value_counts().reset_index()
    df_con_1=df_con_1.drop_duplicates()
    conteo_con_1.columns = ['Nombre', 'Conteo_1']

    conteo_con_0 = df_con_0['OPERADOR LOCAL'].value_counts().reset_index()
    df_con_0=df_con_0.drop_duplicates()
    conteo_con_0.columns = ['Nombre', 'Conteo_0']
    

    col1, col2, col3=st.columns(3)
    with col1:
        st.title('Choferes que usan la app')
        conteo_con_1

    with col2:
        st.title('Porcentaje que usan la app')
        st.plotly_chart(fig)

    with col3:
        st.title('Choferes que no usan la app')
        conteo_con_0

else:
    
    Uso_de_la_app=filtered_DB['Uso_app'].sum()
    Datos_df=filtered_DB['Uso_app'].count()
    Porcentaje_uso=round(((Uso_de_la_app/Datos_df)*100),2)
    Porcentaje_max=(Datos_df/Datos_df)*100
    color_gauge = "#522d6d"
    color_gray = "#e5e1e6"
    color_threshold = "red"
    fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=Porcentaje_uso,
            title={'text': "Indicador de Medidor"},
            gauge={
                'axis': {'range': [0, Porcentaje_max]},
                'bar': {'color': color_gauge},
                'steps': [
                    {'range': [0, 50], 'color': color_gray},
                    {'range': [50, 100], 'color': color_gray}],
                'threshold': {
                    'line': {'color': color_threshold, 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
    #st.plotly_chart(fig)
    df_con_1 = filtered_DB[filtered_DB['Uso_app'] == 1]
    df_con_0 = filtered_DB[filtered_DB['Uso_app'] == 0]

    conteo_con_1 = df_con_1['OPERADOR LOCAL'].value_counts().reset_index()
    df_con_1=df_con_1.drop_duplicates()
    conteo_con_1.columns = ['Nombre', 'Conteo_1']


    conteo_con_0 = df_con_0['OPERADOR LOCAL'].value_counts().reset_index()
    df_con_0=df_con_0.drop_duplicates()
    conteo_con_0.columns = ['Nombre', 'Conteo_0']
    
    col1, col2, col3=st.columns(3)
    with col1:
        st.title('Choferes que usan la app')
        conteo_con_1

    with col2:
        st.title('Porcentaje que usan la app')
        st.plotly_chart(fig)

    with col3:
        st.title('Choferes que no usan la app')
        conteo_con_0


