#Streamlit_choferes.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#streamlit run Dashboard_app_choferes.py
st.set_page_config(layout='wide')


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
filtered_DB.loc[filtered_DB['Tiempo total'] > 4, 'Uso_app'] = 1
#filtered_DB

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
    

    column1, column2, column3=st.columns(3)
    with column1:
        st.markdown("<h1 style='color: green;'>Choferes que usan la app</h1>", unsafe_allow_html=True)
        conteo_con_1

    with column2:
        st.markdown("<h1>Porcentaje que usan la app</h1>", unsafe_allow_html=True)
        st.plotly_chart(fig)

    with column3:
        st.markdown("<h1 style='color: red;'>Choferes que no usan la app</h1>", unsafe_allow_html=True)
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
    
    column1, column2, column3=st.columns(3)
    with column1:
        st.markdown("<h1 style='color: green;'>Choferes que usan la app</h1>", unsafe_allow_html=True)
        conteo_con_1

    with column2:
        st.markdown("<h1>Porcentaje que usan la app</h1>", unsafe_allow_html=True)
        st.plotly_chart(fig)

    with column3:
        st.markdown("<h1 style='color: red;'>Choferes que no usan la app</h1>", unsafe_allow_html=True)
        conteo_con_0

#filtered_DB
filtered_DB = filtered_DB[filtered_DB['Tiempo total'] > 0]
DBchoferes['conteo'] = 1
nombres_unicos = filtered_DB['OPERADOR LOCAL'].unique()
Grupos = filtered_DB.groupby(['OPERADOR LOCAL', 'Fecha de entrega'])[['Uso_app']].sum().reset_index()
conteo_fechas=DBchoferes.groupby(['OPERADOR LOCAL', 'Fecha de entrega'])[['conteo']].count().reset_index()
#conteo_fechas
Grupos = pd.merge(Grupos, conteo_fechas, on=['OPERADOR LOCAL', 'Fecha de entrega'])
#Grupos
cols = st.columns(3)

Grupos['Fecha de entrega numero'] = pd.to_datetime(Grupos['Fecha de entrega']).astype(int) // 10**9
Grupos['Uso_app'] = pd.to_numeric(Grupos['Uso_app'], errors='coerce')
resultados_correlacion = []

for i, nombre in enumerate(nombres_unicos):
    df_filtrado = filtered_DB[filtered_DB['OPERADOR LOCAL'] == nombre]
    x = Grupos.loc[Grupos['OPERADOR LOCAL'] == nombre, 'Fecha de entrega numero']
    y = Grupos.loc[Grupos['OPERADOR LOCAL'] == nombre, 'Uso_app']
    x = x.dropna()
    y = y.dropna()

    if len(x) < 2 or len(y) < 2:
        st.write(f'No hay suficientes datos para calcular la correlación para {nombre}.')
        continue

    #fig, ax = plt.subplots()
    #ax.scatter(Grupos['Fecha de entrega'], Grupos['Uso_app'])
    #ax.set_title(nombre, color='#a7a1c2')
    
    #ax.set_xlabel('Fecha')
    #ax.set_ylabel('Uso app')
    #ax.tick_params(axis='x', colors='#a7a1c2')
    #ax.tick_params(axis='y', colors='#a7a1c2')
    #ax.set_facecolor('none')
    #fig.patch.set_alpha(0)
    #plt.xticks(rotation=90)

    correlation_coefficient, _ = pearsonr(x, y)
    resultados_correlacion.append({'OPERADOR LOCAL': nombre, 'Coeficiente de Pearson': correlation_coefficient})

    #cols[i % 3].pyplot(fig)

resultados_correlacion_df = pd.DataFrame(resultados_correlacion)

def color_and_text(value):
    if value < -0.5:
        color = 'red'
        text = 'Inadecuado'
    elif -0.5 <= value <= 0.5:
        color = '#f6c624'
        text = 'Moderado'
    else:
        color = 'green'
        text = 'Excelente'
    return f'<span style="color: {color}">{text}</span>'

def texto(value):
    if value < -0.5:
        text = 'Inadecuado'
    elif -0.5 <= value <= 0.5:
        text = 'Moderado'
    else:
        text = 'Excelente'
    return text

resultados_correlacion_df['Interpretación'] = resultados_correlacion_df['Coeficiente de Pearson'].apply(color_and_text)
resultados_correlacion_df['Texto'] = resultados_correlacion_df['Coeficiente de Pearson'].apply(texto)
html_df = resultados_correlacion_df.copy()
html_df = html_df.drop(columns=['Coeficiente de Pearson'])
html_table = html_df.to_html(escape=False, index=False)

weights = [1, 3, 1]

col1, col2, col3 = st.columns(weights)
with col1:
    st.write("")

with col2:
    st.markdown(html_table, unsafe_allow_html=True)

with col3:
    st.write("")

df_inadecuado = html_df[html_df['Texto'] == 'Inadecuado']

Grupos['Fecha de entrega'] = pd.to_datetime(Grupos['Fecha de entrega'])
#df_inadecuado['semana'] = df_inadecuado['fecha'].dt.strftime('%U')  
Grupos['semana'] = Grupos['Fecha de entrega'].dt.strftime('%U')  
#Grupos
#html_df
#df_inadecuado
df_agrupado = df_inadecuado.merge(Grupos, on='OPERADOR LOCAL')  
df_agrupado = df_agrupado.drop(columns=['Interpretación'])

df_suma_semanal = df_agrupado.groupby(['OPERADOR LOCAL','semana']).agg({
    'Uso_app': 'sum',
    'conteo': 'sum'
}).reset_index()

df_suma_semanal['Porcentaje']=(df_suma_semanal['Uso_app']*100)/(df_suma_semanal['conteo'])

weights = [1, 3, 1]

col1, col2, col3 = st.columns(weights)



with col2:
    df_suma_semanal



weights = [1, 3, 1]

col1, col2, col3 = st.columns(weights)


for nombre, group in df_suma_semanal.groupby('OPERADOR LOCAL'):
    fig, ax = plt.subplots()
    ax.plot(group['semana'], group['Porcentaje'], label=nombre, marker='o')
    ax.set_title(nombre, color='#a7a1c2')
    
    ax.set_xlabel('Semana')
    ax.set_ylabel('Porcentaje de uso')
    ax.tick_params(axis='x', colors='#a7a1c2')
    ax.tick_params(axis='y', colors='#a7a1c2')
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.set_ylim(0, 110) 
    plt.xticks(rotation=90)

    col2.pyplot(fig)



choferes_faltantes = df[~df['OPERADOR LOCAL'].isin(DBchoferes['Nombre del chofer'])]
choferes_faltantes=choferes_faltantes.reset_index()


choferes_faltantes

#df_agrupado





#Grupos
#filtered_DB
    #with col3:
        #st.markdown("<h1 style='color: red;'>Choferes que no usan la app</h1>", unsafe_allow_html=True)
        #conteo_con_0


