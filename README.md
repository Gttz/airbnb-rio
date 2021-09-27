<h1 align="center">Airbnb Rio de Janeiro - Análise e predição do tipo de propriedade e do preço diário de estadia</h1>
<p align="center">
  <img src="https://github.com/Gttz/airbnb-rio/blob/main/images/airbnb-rio.jpg">
</p>

## :link: Deploy do modelo

![deploy airbnb-rio gif](https://github.com/Gttz/airbnb-rio/blob/main/images/streamlit.gif)

- Deploy do modelo - [Streamlit]()
- Jupyter Notebook - [Airbnb-rio.ipynb]()

## :bookmark_tabs: Introdução sobre o projeto

O Airbnb é um serviço online comunitário para as pessoas anunciarem, descobrirem e reservarem acomodações e meios de hospedagem. Neste dataset foi disponibilizado dados referentes a estadia (número de banheiros, número de quartos, número de camas entre outros)

O objetivo deste projeto era a realização de uma predição do tipo de propriedade e do preço diário da estadia utilizando modelos de Machine Learning e Deep Learning. Levando isso em consideração, há dois tipos de problemas neste projeto, chamados de classificação e regressão.

## :bar_chart: Resultado dos modelos

#### Classificação (Previsão do tipo de propriedade)

*Baseline de Classificação*

* **Modelo**: Decision Tree
* **Acurácia**: 0.80
* **ROC**: 0.78
* **F1-Score**: 0.70

*Modelo final de classificação*

* **Modelo**: XGBoost
* **Acurácia**: 0.85
* **ROC**: 0.84
* **F1-Score**: 0.78

*Modelo final do Pycaret de classificação*

* **Modelo**: Light Gradient Boosting
* **Acurácia**: 0.84
* **AUC**: 0.91
* **F1-Score**: 0.75


#### Regressão (Previsão do preço diário da estadia)

*Baseline de regressão*

* **Modelo**: Linear Regression
* **MAE**: 145.46
* **RMSE**: 198.55
* **R²**: 0.3732

*Modelo final de regressão*

* **Modelo**: Light Gradient Boosting
* **MAE**: 130.86
* **RMSE**: 184.62
* **R²**: 0.4170

*Modelo final do Pycaret de regressão*

* **Modelo**: Light Gradient Boosting
* **MAE**: 131.43
* **RMSE**: 183.89
* **R²**: 0.4216

## :grey_question: Perguntas propostas

###### Como foi a definição da sua estratégia de modelagem?

Busquei primeiramente entender os dados para realizar o pré processamento dos dados, como remoção de colunas irrelevantes, tratamento de valores faltantes (NaN), tratamento de valores inconsistentes, filtragem de dados e por fim, a criação dos modelos de Machine Learning tanto para Classificação, quanto para Regressão.

###### Como foi definida a função de custo utilizada?

Para o modelo de Classificação a função de custo foi utilizada avaliando a performance dos modelos, buscando a maximização principalmente das métricas de Acurácia, F1-Score e curva ROC.

Já para o modelo de Regressão a função de custo foi utilizada novamente avaliando a performance dos modelos, bucando a redução principalmente das métricas MAE (Mean Absolute Error) e MSE (Mean Squared Error) e buscando a maximização do coeficiente de determinação (R²).

###### Qual foi o critério utilizado na seleção do modelo final?

Para a escolha do modelo final tanto para os problemas de Classificação e Regressão, foi realizado uma comparação entre os modelos utilizados e o ajuste de hiperparamêtros do melhor modelo. Logo após, foi realizado a validação cruzada e o teste final no conjunto de dados que o modelo não havia tido contato ainda, com o intuito de simular uma situação mais próxima do mundo real. Desta forma obtive resultados parecidos com os testes realizados na base de treinamento e validação e assim, tendo certeza que não houve nem Overfitting e nem Underfitting nos testes.

###### Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Levando em consideração o tamanho do nosso conjunto de dados, inicialmente foi realizado a utilização de 75% dos dados para treinamento e 25% dos dados para o teste dos modelos, Após isso foi realizado a Validação Cruzada para garantir a representatividade dos cenários nas bases de treinamento e bases de teste e, como foi dito na questão anterior, foi realizado o teste final do melhor modelo utilizando dados que o modelo não havia tido contato, assim simulando as predições dos modelos e garantindo a sua capacidade.

###### Quais evidências você possui de que seu modelo é suficientemente bom?

Os modelos de Classificação e Regressão tiveram de fato um bom desempenho em relação a sua baseline porém, não há evidências que estes modelos são suficientemente bons para alguma aplicação, porém é possível realizar algumas melhorias ao modelo:

- Realizar mais análises e transformações no conjunto de dados
- Utilização de mais features (colunas) no conjunto de dados
- Utilização de hiperparâmetros nos modelos testados inicialmente
- Utilização da validação cruzada nos modelos testados incialmente

## :memo: Licença

Esse projeto está utilizando a licença MIT. clique em [LICENSE](https://github.com/Gttz/airbnb-rio/blob/main/LICENSE) para visualizar mais detalhes.

---
