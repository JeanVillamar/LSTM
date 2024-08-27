import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import datetime, timedelta
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
import folium
import geopandas as gpd
from streamlit_folium import st_folium
import time
import plotly.express as px
import pydeck as pdk
import altair as alt
 
 
 
st.set_page_config(
    page_title="Aplicación FULL AP!",
    page_icon="🤡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.tu-ayuda.com',
        'Report a bug': 'https://www.tu-reporte.com',
        'About': '## Aplicación de Pronóstico de Ventas\nSe podrá visualizar el pronóstico de 10 productos junto a sus abastecimientos'
    }
)
 
def validar_y_actualizar(fila):
    if  fila["cantidad_unid"] >= 1:
        if (fila['id_item'] == 13887): #JERINGA MEGA INSUL 1MLx29Gx1/2x100
            fila["cantidad_frac"] += (100 * int(fila["cantidad_unid"]))
            #datos = datos = datos.rename(columns={'cantidad_unid': 'cantidad_frac'})
            #fila["cantidad_unid"] = 0
        elif fila['id_item'] in {90765, 79680, 27112, 1669, 101609}: #x'unidad  
            fila["cantidad_frac"] += (int(fila["cantidad_unid"]))
 
        elif(fila['id_item'] == 54122): #XARELTO COM-RECx10MGx10
            fila["cantidad_frac"] += (10 * int(fila["cantidad_unid"]))
       
        elif(fila['id_item'] == 88275): #MICARDIX
            fila['cantidad_frac'] += (28 * int(fila["cantidad_unid"]))
    return fila  
 
# Preparar los datos
def preparar_datos(datos):
    # Reemplazar '#N/D' con NaN para unificar el manejo de valores faltantes
    datos.replace('#N/D', np.nan, inplace=True)
    
    # Seleccionar columnas específicas
    columnas_especificas = ['Fecha', 'id_item', 'cantidad_unid', 'cantidad_frac', 'provincia', 'latitud', 'longitud']
    datos = datos[columnas_especificas]
 
    # Aplicar la función a cada fila
    datos = datos.apply(validar_y_actualizar, axis=1)
    datos = datos.drop(columns=["cantidad_unid"])
   
    # Convertir la columna Fecha a formato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%d/%m/%Y %H:%M')
    # Establecer la hora y el minuto a 0
    datos['Fecha'] = datos['Fecha'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))
    
    # Reemplazar NaN en las columnas con valores predeterminados
    datos['provincia'].fillna('Desconocido', inplace=True)
    datos['latitud'].fillna(0, inplace=True)
    datos['longitud'].fillna(0, inplace=True)

    # Convertir columnas a string y reemplazar comas si necesario, luego convertir a float
    if datos['latitud'].dtype == 'object':
        datos['latitud'] = datos['latitud'].str.replace(',', '.').astype(float)
    else:
        datos['latitud'] = datos['latitud'].astype(float)

    if datos['longitud'].dtype == 'object':
        datos['longitud'] = datos['longitud'].str.replace(',', '.').astype(float)
    else:
        datos['longitud'] = datos['longitud'].astype(float)
 
    # Agrupar por Fecha, id_item, provincia, latitud y longitud, sumando cantidad_frac
    datos = datos.groupby(['Fecha', 'id_item', 'provincia', 'latitud', 'longitud'], as_index=False).sum()
    
    # Filtrar por una fecha y un id_item específicos
    fecha_deseada = pd.to_datetime('2024-07-12')
    result = datos[(datos['Fecha'] == fecha_deseada) & (datos['id_item'] == 88275)]
 
    # Ordenar el dataset por Fecha
    datos.sort_index(inplace=False)

    return datos

def preparar_datos1(datos):
 
    # Seleccionar columnas específicas
    columnas_especificas = ['Fecha','id_item','cantidad_unid','cantidad_frac','provincia', 'latitud', 'longitud']
    datos = datos[columnas_especificas]
 
    # Aplicar la función a cada fila
    datos = datos.apply(validar_y_actualizar, axis=1)
    datos = datos.drop(columns=["cantidad_unid"])
   
    # Convertir la columna Fecha a formato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%d/%m/%Y %H:%M')
    # Establecer la hora y el minuto a 0
    datos['Fecha'] = datos['Fecha'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))
 
    # Agrupar por Fecha e id_item y sumar cantidad_frac
    datos = datos.groupby(['Fecha', 'id_item', 'provincia', 'latitud', 'longitud'], as_index=False).sum()
 
    # Ordenar el dataset por Fecha
    datos.sort_index(inplace=False)
    #datos = datos.set_index('Fecha')
 
    return datos

# Definir la función de pérdida RMSE
def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))
 
# Cargar el modelo y el escalador guardados
def cargar_modelo_y_scaler(id_item):
    modelo_path = f'../data/modelo_{id_item}.keras'
    scaler_path = f'../data/scaler_{id_item}.pkl'
 
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    modelo = load_model(modelo_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
 
    return modelo, scaler
 
# Predecir futuros valores
def predecir(x, model, scaler):
    y_pred_s = model.predict(x, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_s)
    return y_pred.flatten()
 
# Reinterpolar los datos para cada id_item sin cambiar el índice
def reinterpolar_datos_por_id(datos):
    datos_reinterpolados = pd.DataFrame()
 
    for id_item in datos['id_item'].unique():
        df_item = datos[datos['id_item'] == id_item].copy()
 
        # Reinterpolar con frecuencia diaria
        df_item = df_item.asfreq(freq='D', fill_value=0)
 
        # Volver a agregar el id_item
        df_item['id_item'] = id_item
 
        # Concatenar los resultados
        datos_reinterpolados = pd.concat([datos_reinterpolados, df_item])
 
    return datos_reinterpolados.reset_index()
 
def generar_predicciones_futuras(df, id_item, input_length, num_predicciones):
    modelo, scaler = cargar_modelo_y_scaler(id_item)
 
    ultima_fecha = df['Fecha'].iloc[-1]  # Acceder a la última fecha de la columna 'Fecha'
   
    fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, num_predicciones + 1)]
 
    ultimo_segmento = df['cantidad_frac'][-input_length:].values
    ultimo_segmento = ultimo_segmento.reshape((1, input_length, 1))
 
    predicciones_futuras = []
    segmento_actual = ultimo_segmento
 
    for _ in range(num_predicciones):
        prediccion = predecir(segmento_actual, modelo, scaler)
        prediccion_redondeada = round(prediccion[0])  # Redondear la predicción a un entero
        predicciones_futuras.append(prediccion_redondeada)
 
        nuevo_valor = np.array(prediccion_redondeada).reshape(1, 1, 1)
        segmento_actual = np.append(segmento_actual[:, 1:, :], nuevo_valor, axis=1)
 
    # Crear un DataFrame con las predicciones futuras, fechas y el id_item correspondiente
    resultados_futuros = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Predicción': predicciones_futuras,
        'id_item': id_item  # Agregar el id_item correspondiente
    })
 
    return resultados_futuros
# Ejecutar todo el proceso para todos los id_item
def predecir_para_todos_los_items(datos, input_length, num_predicciones, lista):
    resultados_totales = pd.DataFrame()

    for id_item in lista:
        # Filtrar los datos para el id_item actual
        df_item = datos[datos['id_item'] == id_item]
        # Generar predicciones para este id_item
        resultados_item = generar_predicciones_futuras(df_item, id_item, input_length, num_predicciones)
        # Concatenar los resultados al DataFrame total
        resultados_totales = pd.concat([resultados_totales, resultados_item])
 
    return resultados_totales
 
def plot_ventas_promedios(datos, id_item):
    datos_item = datos[datos['id_item'] == id_item]
 
    # Ventas promedio por Mes
    plt.figure(figsize=(10, 5))
    axis = datos_item.groupby(datos_item['Fecha'].dt.month)[['cantidad_frac']].mean().plot(marker='o', color='r')
    axis.set_title(f'Ventas promedio por Mes para id_item {id_item}')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad Promedio')
    st.pyplot(plt)
 
    # Ventas promedio por Año
    plt.figure(figsize=(10, 5))
    axis = datos_item.groupby(datos_item['Fecha'].dt.year)[['cantidad_frac']].mean().plot(marker='o', color='b')
    axis.set_title(f'Ventas promedio por Año para id_item {id_item}')
    plt.xlabel('Año')
    plt.ylabel('Cantidad Promedio')
    st.pyplot(plt)
 
    # Ventas promedio por Día
    plt.figure(figsize=(10, 5))
    axis = datos_item.groupby(datos_item['Fecha'].dt.day)[['cantidad_frac']].mean().plot(marker='o', color='g')
    axis.set_title(f'Ventas promedio por Día para id_item {id_item}')
    plt.xlabel('Día')
    plt.ylabel('Cantidad Promedio')
    st.pyplot(plt)
 
def mostrar_ventas_promedio_diarias(datos, id_item):
    datos_item = datos[datos['id_item']==id_item]
    # Añadir una columna para el día de la semana
    datos_item['Día'] = datos_item['Fecha'].dt.day_name()
   
    # Calcular el promedio de ventas por día
    promedio_diario = datos_item.groupby('Día')['cantidad_frac'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
 
    ])
 
    # Crear la gráfica utilizando st_echarts
    option = {
        "xAxis": {
            "type": "category",
            "data": promedio_diario.index.tolist(),  # Días de la semana
        },
        "yAxis": {"type": "value"},
        "series": [{
            "data": promedio_diario.values.tolist(),  # Promedios diarios
            "type": "line"
        }],
    }
    
    st_echarts(options=option, height="400px")
 
def mostrar_ventas_futuras(datos, id_item):
    import pandas as pd
    from streamlit_echarts import st_echarts
    
    # Filtrar los datos para el id_item específico
    datos_item = datos[datos['id_item'] == id_item]
    
    # Verificar que la columna 'Fecha' sea de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(datos_item['Fecha']):
        datos_item['Fecha'] = pd.to_datetime(datos_item['Fecha'])
    
    # Añadir una columna para el día de la semana
    datos_item['Día'] = datos_item['Fecha'].dt.day_name()
    
    # Calcular el promedio de ventas por día
    promedio_diario = datos_item.groupby('Día')['Predicción'].mean()
    
    # Definir el orden tradicional de los días de la semana
    dias_semana_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Filtrar y ordenar los días presentes en los datos
    dias_presentes = [dia for dia in dias_semana_ordenados if dia in promedio_diario.index]
    promedio_diario = promedio_diario.reindex(dias_presentes)
    
    # Configurar las opciones para el gráfico
    options = {
        "title": {
            "text": f"Predicción de Ventas para Item ID: {id_item}",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "xAxis": {
            "type": "category",
            "data": promedio_diario.index.tolist(),
            "axisLabel": {
                "rotate": 45
            }
        },
        "yAxis": {
            "type": "value",
            "name": "Ventas Predichas"
        },
        "series": [{
            "data": promedio_diario.values.tolist(),
            "type": "line",
            "smooth": True,
            "lineStyle": {
                "width": 3
            },
            "marker": {
                "symbol": "circle",
                "size": 8
            },
            "areaStyle": {
                "opacity": 0.2
            }
        }],
        "color": ["#4E79A7"]
    }
    
    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="400px")

def mostrar_ventas_futuras_todos_items(datos):
    import pandas as pd
    from streamlit_echarts import st_echarts
    
    # Verificar que la columna 'Fecha' sea de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(datos['Fecha']):
        datos['Fecha'] = pd.to_datetime(datos['Fecha'])
    
    # Añadir una columna para el día de la semana en español
    dias_semana_es = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    datos['Día'] = datos['Fecha'].dt.day_name().map(dias_semana_es)
    
    # Crear una lista para almacenar las series de datos para cada item
    series = []
    
    for id_item in datos['id_item'].unique():
        # Filtrar los datos para el id_item específico
        datos_item = datos[datos['id_item'] == id_item]
        
        # Ordenar los datos según la fecha para mantener coherencia visual
        datos_item = datos_item.sort_values(by='Fecha')
        
        # Añadir los datos del item a la lista de series
        series.append({
            "name": f"Item {id_item}",
            "data": datos_item['Predicción'].tolist(),  # Predicciones diarias
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.2},
        })
    
    # Configurar las opciones para el gráfico
    options = {
        "title": {
            "text": "Predicción de Ventas por Día",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "legend": {
            "data": [f"Item {id_item}" for id_item in datos['id_item'].unique()],
            "top": "bottom"
        },
        "xAxis": {
            "type": "category",
            "data": datos_item['Día'].tolist(),  # Usar los días presentes en los datos
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "value", "name": "Ventas Predichas"},
        "series": series,
        "color": ["#4E79A7", "#59A14F", "#9C755F", "#E15759", "#F28E2B", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7", "#9D7660"]
    }
    
    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="500px")

def pydeck_ecuador_barra(datos2, col):
   
    # Crear el DataFrame para el mapa utilizando las columnas de latitud y longitud
    map_data = datos2[['latitud', 'longitud', 'cantidad_frac']].copy()
    map_data = map_data.rename(columns={'latitud': 'lat', 'longitud': 'lon', 'cantidad_frac': 'elevation'})
    # Filtrar valores inválidos (#N/D) y convertir a float
    map_data = map_data.replace('#N/D', float('nan'))
    map_data = map_data.dropna(subset=['lat', 'lon'])  # Eliminar filas con valores NaN
    map_data['lat'] = map_data['lat'].astype(float)
    map_data['lon'] = map_data['lon'].astype(float)
    # Configuración del mapa centrado en Ecuador
    view_state = pdk.ViewState(latitude=-1.831239, longitude=-78.183406, zoom=6, bearing=0, pitch=45)
    # Capa de columnas 3D
    column_layer = pdk.Layer(
    "ColumnLayer",
    map_data,
    get_position=["lon", "lat"],
    get_elevation="elevation",
    radius=10000,  # Puedes ajustar el radio según la densidad de datos
    elevation_scale=100,  # Ajusta la escala de elevación para mejor visualización
    get_fill_color=[0, 128, 255],)
    # Crear el mapa y mostrarlo en Streamlit
    r = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
    col.pydeck_chart(r)
 
@st.cache_data
def cargar_datos_geojson():
    # Cargar el archivo GeoJSON
    geojson_file = 'provinces.geojson'
    return gpd.read_file(geojson_file)
 
@st.cache_data
def contar_provincias(df, items_seleccionados):
    # Filtrar el DataFrame según los items seleccionados
    df_filtrado = df[df['id_item'].isin(items_seleccionados)]
    return df_filtrado.groupby('provincia').size().reset_index(name='Valor')
 
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    input_df['Anio'] = pd.to_datetime(input_df['Fecha']).dt.year
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap
 
def mostrar_top_10_provincias(datos, col):
    # Agrupar por provincia y sumar la cantidad_frac
    top_provincias = datos.groupby('provincia')['cantidad_frac'].sum().reset_index()
 
    # Ordenar en orden descendente y seleccionar el top 10
    top_provincias_sorted = top_provincias.sort_values(by='cantidad_frac', ascending=False).head(10)
 
    # Mostrar el top 10 con barras de progreso
    col.markdown('#### Top 10 Provincias por Cantidad')
 
    col.dataframe(top_provincias_sorted,
                 hide_index=True,
                 width=None,
                 column_config={
                    "provincia": st.column_config.TextColumn(
                        "Provincia",
                    ),
                    "cantidad_frac": st.column_config.ProgressColumn(
                        "Cantidad",
                        format="%f",
                        min_value=0,
                        max_value=max(top_provincias_sorted.cantidad_frac),
                     )}
                 )
   
def mostrar_historico_y_predicciones(datos_ventas, datos_abastecimiento, datos_predicciones):   

    
    print('VENTAS!!')
    print(datos_ventas.tail())

    print('ABAST!!')
    print(datos_abastecimiento.tail())

    print('PREDICCIONES!!')
    print(datos_predicciones)
    
    datos_ventas = datos_ventas.drop(columns=["latitud", "longitud","provincia","id_item"])
    datos_abastecimiento = datos_abastecimiento.drop(columns=["latitud", "longitud","id_item",'provincia'])
    datos_predicciones = datos_predicciones.drop(columns=["Día","id_item"])
   


    datos_ventas = datos_ventas.groupby(['Fecha'], as_index=False).sum()
    datos_abastecimiento = datos_abastecimiento.groupby(['Fecha'], as_index=False).sum()
    datos_predicciones = datos_predicciones.groupby(['Fecha'], as_index=False).sum()



    # Verificar que las columnas 'Fecha' sean de tipo datetime
    # if not pd.api.types.is_datetime64_any_dtype(datos_ventas['Fecha']):
    datos_ventas['Fecha'] = pd.to_datetime(datos_ventas['Fecha'])
    # if not pd.api.types.is_datetime64_any_dtype(datos_abastecimiento['Fecha']):
    datos_abastecimiento['Fecha'] = pd.to_datetime(datos_abastecimiento['Fecha'])
    # if not pd.api.types.is_datetime64_any_dtype(datos_predicciones['Fecha']):
    datos_predicciones['Fecha'] = pd.to_datetime(datos_predicciones['Fecha'])


    # Asegurarse de que todos los datasets estén ordenados por fecha
    datos_ventas = datos_ventas.sort_values(by='Fecha')
    datos_abastecimiento = datos_abastecimiento.sort_values(by='Fecha')
    datos_predicciones = datos_predicciones.sort_values(by='Fecha')



    # Crear una serie para las ventas históricas
    series_ventas = {
        "name": "Ventas Históricas",
        "data": datos_ventas['cantidad_frac'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2},
        "areaStyle": {"opacity": 0.2},
    }

    # Crear una serie para los abastecimientos históricos
    series_abastecimiento = {
        "name": "Abastecimientos Históricos",
        "data": datos_abastecimiento['cantidad_frac'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2, "type": "dashed"},  # Línea discontinua para distinguir del histórico de ventas
        "areaStyle": {"opacity": 0.2},
    }

    # Crear una serie para las predicciones de ventas futuras
    series_predicciones = {
        "name": "Predicciones Futuras",
        "data": datos_predicciones['Predicción'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2, "color": "red"},  # Línea roja para destacar las predicciones
        "areaStyle": {"opacity": 0.2},
    }


    # Configurar las opciones para el gráfico
    options = {
        "title": {
            "text": "Histórico de Abastecimientos, Ventas y Predicciones Futuras",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "legend": {
            "data": ["Ventas Históricas", "Abastecimientos Históricos", "Predicciones Futuras"],
            "top": "bottom"
        },
        "xAxis": {
            "type": "category",
            "data": datos_ventas['Fecha'].dt.strftime('%Y-%m-%d').tolist(),  # Usar las fechas como etiquetas en el eje X
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "value", "name": "Cantidad"},
        "series": [series_ventas, series_abastecimiento, series_predicciones],
        "color": ["#4E79A7", "#59A14F", "red"]
    }
    
    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="500px")

def calcular_kpis_y_mostrar(datos_ventas, datos_abastecimiento, datos_predicciones):
    import pandas as pd
    import streamlit as st
    from streamlit_echarts import st_echarts
    
    # Verificar que las columnas 'Fecha' sean de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(datos_ventas['Fecha']):
        datos_ventas['Fecha'] = pd.to_datetime(datos_ventas['Fecha'])
    if not pd.api.types.is_datetime64_any_dtype(datos_abastecimiento['Fecha']):
        datos_abastecimiento['Fecha'] = pd.to_datetime(datos_abastecimiento['Fecha'])
    if not pd.api.types.is_datetime64_any_dtype(datos_predicciones['Fecha']):
        datos_predicciones['Fecha'] = pd.to_datetime(datos_predicciones['Fecha'])
    
    # Filtrar datos para incluir solo el último mes
    fecha_hoy = pd.to_datetime("today")
    fecha_inicio = fecha_hoy - pd.DateOffset(days=30)
    
    datos_ventas_mes = datos_ventas[(datos_ventas['Fecha'] >= fecha_inicio) & (datos_ventas['Fecha'] <= fecha_hoy)]
    datos_abastecimiento_mes = datos_abastecimiento[(datos_abastecimiento['Fecha'] >= fecha_inicio) & (datos_abastecimiento['Fecha'] <= fecha_hoy)]
    
    # KPI 1: Tasa de Cobertura de Abastecimiento
    dias_cubiertos = (datos_abastecimiento_mes['cantidad_frac'] >= datos_ventas_mes['cantidad_frac']).sum()
    tasa_cobertura_abastecimiento = dias_cubiertos / len(datos_ventas_mes) * 100
    
    # KPI 2: Tasa de Crecimiento de Ventas
    ventas_mes_anterior = datos_ventas[(datos_ventas['Fecha'] >= fecha_inicio - pd.DateOffset(days=30)) & 
                                       (datos_ventas['Fecha'] < fecha_inicio)]['cantidad_frac'].sum()
    ventas_mes_actual = datos_ventas_mes['cantidad_frac'].sum()
    tasa_crecimiento_ventas = ((ventas_mes_actual - ventas_mes_anterior) / ventas_mes_anterior) * 100
    
    # KPI 3: Promedio de Ventas Diarias
    promedio_ventas_diarias = datos_ventas_mes['cantidad_frac'].mean()
    
    # KPI 4: Promedio de Abastecimientos Diarios
    promedio_abastecimientos_diarios = datos_abastecimiento_mes['cantidad_frac'].mean()
    
    # KPI 5: Desviación Estándar de Ventas
    desviacion_ventas = datos_ventas_mes['cantidad_frac'].std()
    
    # KPI 6: Ratio de Ventas a Abastecimiento
    ratio_ventas_abastecimiento = ventas_mes_actual / datos_abastecimiento_mes['cantidad_frac'].sum()
    
    # Mostrar los KPIs
    st.write("### KPIs del Último Mes")
    st.metric("Tasa de Cobertura de Abastecimiento", f"{tasa_cobertura_abastecimiento:.2f}%")
    st.metric("Tasa de Crecimiento de Ventas", f"{tasa_crecimiento_ventas:.2f}%")
    st.metric("Promedio de Ventas Diarias", f"{promedio_ventas_diarias:.2f}")
    st.metric("Promedio de Abastecimientos Diarios", f"{promedio_abastecimientos_diarios:.2f}")
    st.metric("Desviación Estándar de Ventas", f"{desviacion_ventas:.2f}")
    st.metric("Ratio de Ventas a Abastecimiento", f"{ratio_ventas_abastecimiento:.2f}")
    
    # Aquí se puede incluir el gráfico previamente implementado
    mostrar_historico_y_predicciones(datos_ventas_mes, datos_abastecimiento_mes, datos_predicciones)

def main():
    st.title('Predicción de Ventas')
    col1, col2, col3 = st.columns([1.5, 4.5, 2], gap='medium')
   
    datos = pd.DataFrame()
 
    uploaded_file = col1.file_uploader("Subir archivo de Ventas CSV", type="csv")
    uploaded_file2 = col1.file_uploader("Subir archivo de Abastecimientos CSV", type="csv")
   
    if (uploaded_file and uploaded_file2) is not None:
        # Barra de progreso
        progress_bar = col1.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)  # Simula un pequeño retardo
            progress_bar.progress(percent_complete + 1)
        if(percent_complete == 99):
            # Mostrar mensaje de archivo subido
            col1.success("Archivos subidos con éxito")
 
        # Cargar los datos
        datos = pd.read_csv(uploaded_file, delimiter=';')
        datos2 = datos.copy()
        datos = preparar_datos1(datos)
        abast = pd.read_csv(uploaded_file2, delimiter=';')
        abast = preparar_datos(abast)
        datos2 = preparar_datos(datos2)
        #abast = reinterpolar_datos_por_id(abast)
        


        #datos = datos.drop(columns=["provincia"])
        datos = datos.groupby(['Fecha', 'id_item'], as_index=False).sum()
        datos = datos.set_index('Fecha')
        datos = reinterpolar_datos_por_id(datos)
            
    
        abast = abast.groupby(['Fecha', 'id_item'], as_index=False).sum()
        abast = abast.set_index('Fecha')
        abast = reinterpolar_datos_por_id(abast)


    
        # Obtener fechas mínima y máxima del dataset
        min_date = datos2['Fecha'].min().to_pydatetime()
        max_date = datos2['Fecha'].max().to_pydatetime()
 
        # Configuración de la barra lateral
        with st.sidebar:
            ItemsId = datos['id_item'].unique()
            lista = st.multiselect(
                'Which product would you like to view?',
                ItemsId,
                [90765, 27112, 13887, 79680, 1669, 101609, 54122, 88275]
            )
 
            # Crear un slider para el rango de fechas
            selected_date = st.slider(
                "Select a date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                step=timedelta(days=1),
                format="DD/MM/YYYY"
            )

            
 
        # Filtrar los datos por el rango de fechas y los `id_item` seleccionados
        vent_filtrado = datos2[
            (datos2['Fecha'] >= selected_date[0]) &
            (datos2['Fecha'] <= selected_date[1]) &
            (datos2['id_item'].isin(lista))     


        ]

        # Filtrar los datos de abastecimiento por el rango de fechas y los `id_item`
        abast_filtrado = abast[
            (abast['Fecha'] >= selected_date[0]) &
            (abast['Fecha'] <= selected_date[1]) &
            (abast['id_item'].isin(lista))     
        ]

        #Filtrar los datos de predicciones por 'id_item' seleccionados
        datos_filtradosFut = datos[
            (datos['id_item'].isin(lista))

        ]
        
        vent_filtrado_org = vent_filtrado.copy()
        abast_filtrado_org = abast_filtrado.copy()
        datos_filtradosFut_org = datos_filtradosFut.copy()


        if vent_filtrado.empty:
            col1.warning("No hay datos disponibles para los filtros seleccionados.")
        else:
            # Generar las predicciones solo para los datos filtrados
            resultados_futuros = predecir_para_todos_los_items(datos_filtradosFut, 24, 4, lista)
            col1.write("Resultados de Predicción:")
            col1.dataframe(resultados_futuros)
            
            # Supongamos que tienes una Serie o una columna de un DataFrame que se llama 'fecha'
            #resultados_futuros['Fecha'] = pd.to_datetime(resultados_futuros['Fecha'])

            # with col1:
            #     for id_item in resultados_futuros['id_item'].unique():
            #         mostrar_ventas_futuras(resultados_futuros, id_item)
 
            # Mostrar la visualización de Pydeck con los datos filtrados
            pydeck_ecuador_barra(vent_filtrado, col2)
           

            # Extraer el año de la columna Fecha
            
           
            # Crear y mostrar el heatmap utilizando Altair
            col2.write("Heatmap de Cantidad de Registros por Año y Provincia")
            heatmap = make_heatmap(vent_filtrado, 'Anio', 'provincia', 'cantidad_frac', 'blues')
            col2.altair_chart(heatmap, use_container_width=True)



            with col2:
                #calcular_kpis_y_mostrar(datos2, abast,resultados_futuros )
                mostrar_ventas_futuras_todos_items(resultados_futuros)


            mostrar_historico_y_predicciones(vent_filtrado_org, abast_filtrado_org, resultados_futuros)

            # KPI 1: Total de Unidades Vendidas
            total_ventas = vent_filtrado_org['cantidad_frac'].sum()
            ventas_anteriores = vent_filtrado_org['cantidad_frac'].shift(1).sum()
            delta_ventas = total_ventas - ventas_anteriores
            st.metric(label="Total de Unidades Vendidas", value=total_ventas, delta=delta_ventas)

            # KPI 2: Total de Unidades Abastecidas
            total_abast = abast_filtrado_org['cantidad_frac'].sum()
            abast_anteriores = abast_filtrado_org['cantidad_frac'].shift(1).sum()
            delta_abast = total_abast - abast_anteriores
            st.metric(label="Total de Unidades Abastecidas", value=total_abast, delta=delta_abast)

            # KPI 3: Predicciones de Ventas Totales
            total_predicciones = resultados_futuros['Predicción'].sum()
            predicciones_anteriores = resultados_futuros['Predicción'].shift(1).sum()
            delta_predicciones = total_predicciones - predicciones_anteriores
            st.metric(label="Total de Predicciones de Ventas", value=total_predicciones, delta=delta_predicciones)

            # KPI 4: Tasa de Cumplimiento de Abastecimiento
            if total_predicciones > 0:
                tasa_cumplimiento = (total_abast / total_predicciones) * 100
            else:
                tasa_cumplimiento = 0
            st.metric(label="Tasa de Cumplimiento de Abastecimiento (%)", value=f"{tasa_cumplimiento:.2f}")

            # KPI 5: Cobertura de Inventario (días)
            promedio_ventas_diarias = total_ventas / vent_filtrado_org['Fecha'].nunique()  # Calcula el promedio de ventas diarias
            if promedio_ventas_diarias > 0:
                cobertura_inventario = total_abast / promedio_ventas_diarias
            else:
                cobertura_inventario = 0
            st.metric(label="Cobertura de Inventario (días)", value=f"{cobertura_inventario:.2f}")

            # Gráfico combinando ventas históricas y predicciones futuras
            st.header("Gráfico de Ventas y Predicciones Futuras")

            # Preparar datos para el gráfico
            ventas_para_grafico = vent_filtrado_org.groupby('Fecha')['cantidad_frac'].sum().reset_index()
            ventas_para_grafico.rename(columns={'cantidad_frac': 'Ventas'}, inplace=True)

            # Combinar con las predicciones
            predicciones_para_grafico = resultados_futuros[['Fecha', 'Predicción']].rename(columns={'Predicción': 'Ventas'})
            ventas_futuras_combinadas = pd.concat([ventas_para_grafico, predicciones_para_grafico], ignore_index=True)

            # Ordenar por fecha
            ventas_futuras_combinadas.sort_values(by='Fecha', inplace=True)

            # Mostrar el gráfico
            st.line_chart(ventas_futuras_combinadas.set_index('Fecha'))

           
            # Crear el expander
            with st.expander('Ventas por día'):
                # Aquí puedes agregar el contenido relacionado con las ventas por día

                # Agregar información del contacto del creador
                st.markdown("**Contactos**")

                # Cargar y mostrar ícono de GitHub
                st.image("../icons/github.png", width=24)
                st.markdown("[GitHub](https://github.com/JeanVillamar)")

                # Cargar y mostrar ícono de LinkedIn
                st.image("../icons/linkedin.png", width=24)
                st.markdown("[LinkedIn](www.linkedin.com/in/jean-villamar)")
                        
            
            # Llamar a la función para mostrar el top 10 de provincias
            mostrar_top_10_provincias(vent_filtrado, col3)
 
if __name__ == '__main__':
    main()