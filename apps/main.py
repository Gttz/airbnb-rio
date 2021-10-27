import pandas as pd
import numpy as np
import streamlit as st
import joblib

def classification_regression_features():

    regiao = st.sidebar.selectbox('Região', ['Zona Sul', 'Zona Oeste', 'Zona Norte', 'Zona Central'])

    classe_social = st.sidebar.selectbox('Classe-Social do Bairro', ['Classe Média-Alta', 'Classe Média', 'Classe Alta',
                                                                     'Classe Média-Baixa', 'Classe Baixa'])

    banheiros = st.sidebar.selectbox('Banheiros', ['1', '2', '3', '4+'])

    quartos = st.sidebar.selectbox('Quartos', ['1', '2', '3', '4+'])

    camas = st.sidebar.selectbox('Camas', ['1', '2', '3', '4+'])

    noites = st.sidebar.selectbox('Minimo de noites', ['1', '2', '3', '4', '5+'])

    acomodacoes = st.sidebar.selectbox("Limite de número de pessoas", ['1', '2', '3', '4', '5', '6', '7+'])

    review_scores_rating = st.sidebar.selectbox('Pontuação da estadia em geral',
                                                ['Prefiro não responder', 'Razoavel', 'Bom', 'Otimo', 'Perfeito'])

    review_scores_accuracy = st.sidebar.selectbox('Pontuação da descrição da estadia',
                                                  ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    review_scores_cleanliness = st.sidebar.selectbox('Pontuação da limpeza da estadia',
                                                     ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    review_scores_checkin = st.sidebar.selectbox('Pontuação do check-in da estadia',
                                                 ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    review_scores_communication = st.sidebar.selectbox('Pontuação da comunicação da estadia',
                                                       ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    review_scores_location = st.sidebar.selectbox('Pontuação da localização da estadia',
                                                  ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    review_scores_value = st.sidebar.selectbox('Pontuação do preço diário de estadia',
                                               ['Prefiro não responder', 'Razoavel', 'Regular', 'Bom', 'Otimo'])

    latitude = st.sidebar.slider('Latitude', -01.00000, -99.00000)
    longitude = st.sidebar.slider('Longitude', -01.00000, -99.00000)

    reserva_30 = st.sidebar.slider('Reserva disponivel nos próximos 30 dias', 0, 30)
    reserva_60 = st.sidebar.slider('Reserva disponivel nos próximos 60 dias', 0, 60)
    reserva_90 = st.sidebar.slider('Reserva disponivel nos próximos 90 dias', 0, 90)
    reserva_360 = st.sidebar.slider('Reserva disponivel nos próximos 365 dias', 0, 365)

    dataset = {
        'neighbourhood_region': regiao,
        'district_social_class': classe_social,
        'latitude': latitude,
        'longitude': longitude,
        'accommodates': acomodacoes,
        'bathrooms_text': banheiros,
        'bedrooms': quartos,
        'beds': camas,
        'minimum_nights': noites,
        'availability_30': reserva_30,
        'availability_60': reserva_60,
        'availability_90': reserva_90,
        'availability_365': reserva_360,
        'review_scores_rating': review_scores_rating,
        'review_scores_accuracy': review_scores_accuracy,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_checkin': review_scores_checkin,
        'review_scores_communication': review_scores_communication,
        'review_scores_location': review_scores_location,
        'review_scores_value': review_scores_value
    }
    # Juntando as informações descritas pelo usuário em um DataFrame
    df_classreg_simulation = pd.DataFrame(dataset, index=[0])
    return df_classreg_simulation


def classification_features():
    # Classificação com a feature 'price'
    df = classification_regression_features()
    preco = st.sidebar.slider('Preço diário da estadia', 0, 1150)
    df['price'] = preco
    return df


def regression_features():
    # Regressão com a feature 'property_type'
    df = classification_regression_features()
    propriedade = st.sidebar.selectbox('Tipo de propriedade', ['Apartamento', 'Quarto/Casa'])
    df['property_type'] = propriedade
    return df

st.title('Previsão do tipo de propriedade e do preço diário de estadia - Airbnb')
st.write('Selecione os valores no menu lateral esquerdo e o algoritmo definirá o tipo de propriedade e o preço diário '
         'do airbnb')
st.sidebar.header('Dados da estadia:')

app = st.selectbox('Selecione o tipo', ('Classificação', 'Regressão', 'Ambos'))

if app == 'Classificação':
    df_class = classification_features()
    X_class = df_class.values
    model_c = joblib.load('models/XGboost_class_airbnb.pkl')
    previsao_c = model_c.predict(X_class)
    st.subheader('Resultado do tipo de propriedade')
    if previsao_c == 'Room/House':
        st.write('O tipo de propriedade previsto é Quarto/Casa!')
        st.image('images/airbnb.png')
    else:
        st.write('O tipo de propriedade previsto é Apartamento!')
        st.image('images/airbnb.jpg')

elif app == 'Regressão':
    df_reg = regression_features()
    X_reg = df_reg.values
    model_r = joblib.load('models/LGBM_reg_airbnb.pkl')
    # Realizando a previsão do modelo
    previsao_r = model_r.predict(X_reg)
    # Mostrando o resultado
    st.subheader('Resultado do Preço diário de estadia')
    st.write('De acordo com as caracteristicas do imovel o preço da diária prevista é:')
    st.write(previsao_r[0])

else:
    df_classreg = classification_regression_features()
    X_classreg = df_classreg.values
    model_c = joblib.load('models/XGboost_classreg_airbnb.pkl')
    model_r = joblib.load('models/LGBM_classreg_airbnb.pkl')

    # Realizando a previsão do modelo
    previsao_c = model_c.predict(X_classreg)
    previsao_cp = model_c.predict_proba(X_classreg)
    previsao_r = model_r.predict(X_classreg)

    # Mostrando o resultado
    st.subheader('Resultado do Preço diário de estadia')
    st.write('De acordo com as caracteristicas do imovel o preço da diária prevista é:')
    st.write(previsao_r[0])
    st.subheader('Resultado do tipo de propriedade')
    if previsao_c == 'Room/House':
        st.write('O tipo de propriedade previsto é Quarto/Casa!')
        st.write('A probabilidade de acerto é:')
        st.write(previsao_cp[0][1] * 100)
        st.image('images/airbnb.png')
    else:
        st.write('O tipo de Propriedade previsto é Apartamento!')
        st.write('A probabilidade de acerto é:')
        st.write(previsao_cp[0][0] * 100)
        st.image('images/airbnb.jpg')
