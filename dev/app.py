import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import timedelta
import pickle
import streamlit as st

def validar_y_actualizar(fila):
    if  fila["cantidad_unid"] >= 1:
        if (fila['id_item'] == 13887): #JERINGA MEGA INSUL 1MLx29Gx1/2x100
          fila["cantidad_frac"] += 100 * int(fila["cantidad_unid"])
          #datos = datos = datos.rename(columns={'cantidad_unid': 'cantidad_frac'})
          #fila["cantidad_unid"] = 0
        elif fila['id_item'] in {90765, 79680, 27112}:
          fila["cantidad_frac"] += int(fila["cantidad_unid"])

        elif(fila['id_item'] == 54122): #XARELTO COM-RECx10MGx10
          fila["cantidad_frac"] += 10 * int(fila["cantidad_unid"])
    return fila
    pass    

# Preparar los datos
def preparar_datos(datos):
    # Seleccionar columnas específicas
    columnas_especificas = ['Fecha', 'id_item', 'cantidad_unid', 'cantidad_frac']
    datos = datos[columnas_especificas]

    # Aplicar la función a cada fila
    datos = datos.apply(validar_y_actualizar, axis=1)
    datos = datos.drop(columns=["cantidad_unid"])

    print(datos)
    # Convertir la columna Fecha a formato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%d/%m/%Y %H:%M')
    # Establecer la hora y el minuto a 0
    datos['Fecha'] = datos['Fecha'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))

    # Agrupar por Fecha e id_item y sumar cantidad_frac
    datos = datos.groupby(['Fecha', 'id_item'], as_index=False).sum()

    # Ordenar el dataset por Fecha
    datos.sort_index(inplace=False)
    datos = datos.set_index('Fecha')

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

    ultima_fecha = df['Fecha'].iloc[-1] # Access the last date from the 'Fecha' column
    print(ultima_fecha)
    fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, num_predicciones + 1)]

    ultimo_segmento = df['cantidad_frac'][-input_length:].values
    ultimo_segmento = ultimo_segmento.reshape((1, input_length, 1))

    predicciones_futuras = []
    segmento_actual = ultimo_segmento

    for _ in range(num_predicciones):
        prediccion = predecir(segmento_actual, modelo, scaler)
        predicciones_futuras.append(prediccion[0])

        nuevo_valor = np.array(prediccion[0]).reshape(1, 1, 1)
        segmento_actual = np.append(segmento_actual[:, 1:, :], nuevo_valor, axis=1)

    # Crear un DataFrame con las predicciones futuras, fechas y el id_item correspondiente
    resultados_futuros = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Predicción': predicciones_futuras,
        'id_item': id_item  # Agregar el id_item correspondiente
    })

    return resultados_futuros



# Ejecutar todo el proceso para todos los id_item
def predecir_para_todos_los_items(datos, input_length, num_predicciones):
    resultados_totales = pd.DataFrame()
    lista = [90765, 27112]
    for id_item in lista:
        # Filtrar los datos para el id_item actual
        df_item = datos[datos['id_item'] == id_item]
        # Generar predicciones para este id_item
        resultados_item = generar_predicciones_futuras(df_item, id_item, input_length, num_predicciones)
        # Concatenar los resultados al DataFrame total
        resultados_totales = pd.concat([resultados_totales, resultados_item])

    return resultados_totales



def main():
    st.title('Predicción de Ventas')

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
    if uploaded_file is not None:
        datos = pd.read_csv(uploaded_file, delimiter=';')
        datos = preparar_datos(datos)
        datos = reinterpolar_datos_por_id(datos)
        resultados_futuros = predecir_para_todos_los_items(datos, 24, 4)
        st.write("Resultados de Predicción:")
        st.dataframe(resultados_futuros)



if __name__ == '__main__':
    main()