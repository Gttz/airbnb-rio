<h1 align="center">Airbnb Rio de Janeiro - Análise e predição do tipo de propriedade e do preço diário de estadia</h1>
<p align="center">
  <img src="https://github.com/Gttz/airbnb-rio/blob/main/images/airbnb-rio.jpg">
</p>

## :link: Deploy do modelo

![deploy airbnb-rio gif](https://github.com/Gttz/airbnb-rio/blob/main/images/streamlit.gif)

- Deploy do modelo - [Streamlit](https://share.streamlit.io/gttz/airbnb-rio/main/apps/main.py)
- Jupyter Notebook - [Airbnb-rio.ipynb](https://nbviewer.jupyter.org/github/Gttz/airbnb-rio/blob/main/notebooks/Airbnb-rio.ipynb)

## :paperclips: Sumário do READ-ME

1. Introdução sobre o projeto
2. Dicionário dos dados
3. Pré-processamento e Manipulação dos dados
4. Análise Exploratória dos dados
5. Pré-processamento para os modelos
6. Resultado dos modelos
7. Perguntas Propostas
8. Licença

## :bookmark_tabs: 1. Introdução sobre o projeto

O Airbnb é um serviço online comunitário para as pessoas anunciarem, descobrirem e reservarem acomodações e meios de hospedagem. Neste dataset foi disponibilizado dados referentes a estadia (número de banheiros, número de quartos, número de camas entre outros)

O objetivo deste projeto foi a realização de uma predição do tipo de propriedade e do preço diário da estadia onde primeiramente busquei entender os dados para realizar o pré processamento dos dados (remoção de colunas irrelevantes, tratamento de valores faltantes (NaN), tratamento de valores inconsistentes, filtragem de dados, etc), análise exploratória dos dados, criação dos modelos de classificação e regressão, comparação entre os modelos, cross-validation, ajuste de hiperparâmetros (RandomizerSearchCV e GridSearchCV) e por fim, o teste final na base de teste. Após a escolha do melhor modelo, foi realizado um Deploy do projeto utilizando a biblioteca Streamlit.


## :book: 2. Dicionário dos dados

O nosso dataset inicialmente tinha 74 colunas presentes, optei em selecionar 22 destas variáveis características que eram relevantes (que seriam os dados de entrada) e 2 dessas variáveis são as nossas variáveis-alvos (property_type e price). Antes de começarmos, é necessário entender oque cada variável significa e, para isso realizei uma descrição de cada coluna abaixo:

    neighbourhood_cleansed      - Permite extrair a região / área dentro de cada bairro
    latitude                    - coordenada da latitude da estadia
    longitude                   - coordenada da longitude da estadia
    accommodates                - O número de pessoas que cada airbnb pode acomodar
    bathrooms                   - Número de banheiros em formato numérico
    bathrooms_text              - Número de banheiros em formato textual
    bedrooms                    - Número de quartos em cada estadia
    beds                        - Número de camas em cada estadia
    minimum_nights              - Número minímo de noites no airbnb
    availability_30             - O número de dias que um airbnb está disponível para reserva nos próximos 30 dias
    availability_60             - O número de dias que um airbnb está disponível para reserva nos próximos 60 dias
    availability_90             - O número de dias que um airbnb está disponível para reserva nos próximos 90 dias
    availability_365            - O número de dias que um airbnb está disponível para reserva nos próximos 365 dias
    review_scores_rating        - os hóspedes podem pontuar a propriedade em geral de 1 a 100
    review_scores_accuracy      - os hóspedes podem pontuar a precisão da descrição de uma propriedade de 0 a 10
    review_scores_cleanliness   - os hóspedes podem pontuar a limpeza de uma propriedade de 1 a 10
    review_scores_checkin       - os hóspedes podem pontuar seu check-in de 1 a 10
    review_scores_communication - os hóspedes podem pontuar a comunicação de um host de 1 a 10
    review_scores_location      - os hóspedes podem pontuar a localização de uma propriedade de 1 a 10
    review_scores_value         - os hóspedes podem pontuar o valor do dinheiro de uma reserva de 1 a 10
    property_type               - Tipo de propriedade (onde seria a nossa variável-alvo de classificação)
    Price                       - Preço da diária de estadia (onde seria a nossa variável-alvo de regressão)

## :bookmark_tabs: 3. Pré-processamento e Manipulação dos dados

Na etapa de pré-processamento é preciso realizar a manipulação e tratamento dos dados. Esta etapa é considerada uma das mais importantes em projetos de Data Science, pois considerando casos reais os dados dificilmente virão tratados, limpos e organizados. Considerando o nosso resultado final, os modelos de Machine Learning podem variar de resultado dependendo de como o Cientista de Dados tratou e manipulou estes dados. Esta etapa é considerada a mais trabalhosa em um projeto e, devemos ter atenção a todos os detalhes possíveis.

### 3.1 Verificação e tratamento dos Valores faltantes (NaN)

Nesta parte, irei realizar a verificação e o tratamento dos dados ausentes no nosso conjunto de dados, muitas das vezes estes dados ocorrem pelos dados que não são preenchidos corretamente ou ocorrem devido a nenhuma informação ser registrada para cada coluna. Estes dados devem ser tratados e há inúmeras formas de realizar este tratamento, podemos simplesmente apagar todos os valores ausentes, trocar os valores pelo maior valor ou pela média da coluna entre outros.

![image](https://user-images.githubusercontent.com/61298783/135503504-fbd6deb0-3948-476b-8757-46a526ac4fd6.png)

Podemos observar que temos altos valores faltantes em algumas colunas e, iremos visualiza-las de forma individual a seguir:

###### Colunas bathrooms & bathrooms_text

Como podemos observar todos os dados da coluna bathrooms são valores nulos e, por conta disto irei deletar a coluna. Para não perdermos a informação de quantos banheiros tem em cada estadia, podemos utilizar a coluna bathrooms_text e trata-la posteriormente. Porém também devemos tratar os valores ausentes da coluna bathrooms_text, podemos observar que temos 73 valores ausentes que serão alterados para o seu maior valor.

###### Colunas bedrooms & beds

Na coluna bedrooms temos o total de 1746 valores ausentes e na coluna beds temos o total de 252 valores ausentes e ambas as coluans serão substituidos pelo maior valor, que será 1.0.

###### Colunas review_scores

Nas colunas review_scores nós temos quase 11000 de valores faltantes em cada coluna e provavalmente os dados faltantes foram em consideração de que os usuários não deram nenhuma resposta ao ser perguntado as review's de cada um, e por isso, ficaram com valores NaN. Estas colunas possuem bastantes informações e, uma das formas de que eu encontrei para tratar estes dados é trocar os valores ausentes para 0, e depois os transformar para 'Sem Resposta', que será realizado mais para frente.

##### 3.1.1 Tratamento dos valores faltantes (NaN)

![image](https://user-images.githubusercontent.com/61298783/135505305-d5206b55-56a2-438f-9278-c6a6bedae13c.png)

Agora que realizamos as transformações, vamos visualizar novamente os nossos valores faltantes:

![image](https://user-images.githubusercontent.com/61298783/135505132-aa499e99-b2e4-454d-a158-5806f790ef85.png)

Dados faltantes foram tratados com sucesso :D

### 3.2 Valores duplicados

Outra das tarefas de pré-processamento e limpeza dos dados é realizar o tratamento de dados duplicados, que muitas das vezes podem influenciar negativamente nos resultados finais dos modelos. Para realizarmos o tratamento, utilizaremos o método drop_duplicates() que apenas deleta os dados repetidos.

![image](https://user-images.githubusercontent.com/61298783/135505480-d7a04dcf-80b9-47e7-bd6a-71fc042dee48.png)

Os valores duplicados foram tratados com sucesso! :D

### 3.3 Tipo das colunas

Assim como pode ser feito registro de dados sem nenhum significado, por quem constroi o conjunto de dados, as colunas também podem ser preenchidas de forma incorreta, ou seja, uma coluna que trata-se de uma coluna do tipo inteira, pode estar como coluna do tipo float ou até mesmo string.

![image](https://user-images.githubusercontent.com/61298783/135505640-a4eaa482-dde1-4624-a231-54658f667226.png)

Podemos observar que algumas colunas nós podemos realizar a alteração do tipo, realizei as alterações descritas a seguir:

    bedrooms - Alteração para o tipo int
    beds - Alteração para o tipo int
    bathrooms_text - Alteração para o tipo int
    review_scores_rating - Alteração para o tipo int
    review_scores_accuracy - Alteração para o tipo int
    review_scores_cleanliness - Alteração para o tipo int
    review_scores_checkin - Alteração para o tipo int
    review_scores_communication - Alteração para o tipo int
    review_scores_location - Alteração para o tipo int
    review_scores_value - Alteração para o tipo int
    price - Alteração para o tipo float

Os tipos das colunas foram tratados com sucesso! :D

### 3.4 Dados inconsistentes

Os outliers são considerados registros de dados que apresentam valores fisicamente inconsistentes. A identificação e o tratamento destes dados é de fundamental importância para a obtenção de uma série confiável de dados de fluxos. Após a identificação, o ideal é reallizar a remoção ou transformação deste dados, porém, alguns outliers são considerados dados autenticos e nem sempre é necessário realizar a remoção dos mesmos.

![image](https://user-images.githubusercontent.com/61298783/135528224-7a078020-c247-4144-bf6c-0214a803ce40.png)

### 3.4.1 Variáveis númericas

Algumas mudanças no qual realizei foram:

    price - Valores filtrados para >= 0 e <= 1150
    accommodates - Alterado o tipo da coluna para categórico e as acomodações maiores que 7 foram filtradas.
    bathrooms_text -  Alterado o tipo da coluna para categórico e os banheiros maiores que 4 foram filtrados.
    bedrooms -  Alterado o tipo da coluna para categórico e os quartos maiores que 4 foram filtrados.
    beds - Alterado o tipo da coluna para categórico e o número de camas maiores que 4 foram filtrados.
    minimum_nights - Valores filtrados para <= 360 e logo após tipo da coluna alterado para categórico sendo maiores que 5.

Já para todas as outras colunas review_scores também realizei um filtro sendo:

    0: Sem Resposta (que seriam os valores faltantes das colunas)
    1 até 5: Razoavel
    5 até 7: Regular
    7 até 9: Bom
    10: Ótimo
    
### 3.4.2 Variáveis categóricas

![image](https://user-images.githubusercontent.com/61298783/135529511-8947e7ce-8e45-41bd-bfe5-c4109f7e5fa6.png)

Algumas observações:

- Na coluna ``neighbourhood_cleansed`` temos 149 bairros diferentes (MUITOS bairros)
- Na coluna ``property_type`` temos 83 tipos de propriedades diferentes (MUITOS tipos de propriedades)

Na coluna neighbourhood_cleansed temos 149 bairros diferentes. Após realizar algumas pesquisas, tive a idéia de criar uma nova coluna com o nome de neighbourhood_region com as zonas de cada bairro. As zonas estão na imagem abaixo:

![image](https://user-images.githubusercontent.com/61298783/135529633-9bfbeff9-7354-4eac-b351-7c8782bd1694.png)

Após as transformações, as zonas ficaram como na imagem abaixo.

![image](https://user-images.githubusercontent.com/61298783/135529756-b33fbaca-b022-4976-b79e-299030fa5333.png)

Sendo a coluna property_type a que queremos realizar a previsão temos muitos números de categorias (83 tipos de propriedades diferentes). Podemos observar que os dados estão muito variados pois há um grande número de categorias com poucas listagens, iremos realizar o filtro de cada coluna para: 'Apartment' e 'Room/House' tornando assim, a coluna binária. Após realizar as transformações, o tipo de propriedade ficou como na imagem abaixo.

![image](https://user-images.githubusercontent.com/61298783/135529919-3bbb024d-9fe0-43a7-a417-111776d6594c.png)

## :bar_chart: 4. Análise Exploratória dos dados

A Análise Exploratória dos dados (EDA) tem como objetivo realizar um entedimento mais aprofundado dos dados utilizando ferramentas e técnicas de visualização, assim, tornando mais fácil a construção de hipóteses e suposições nas análises. Esta etapa é considerada bastante importante e geralmente é realizada após a coleta de dados e do pré-processamento dos dados.

### 4.1 Histogramas univariados

Uma das formas de que se pode ser aplicado na EDA é utilizando a análise univariada (análise de 1 variável). Que será observada logo abaixo:

![image](https://user-images.githubusercontent.com/61298783/135530061-f36ccf6a-b0ae-4a80-a0e7-cc12a0032c03.png)

Podemos observar que, após fizermos os pré-processamento e transformações no nosso conjunto de dados ficamos com bem poucas colunas do tipo númerico e assim, tratamos a maioria dos valores discrepantes (mas nem todos). 

Já para as nossas variavéis categóricas, ficamos como nas imagens a seguir.

![image](https://user-images.githubusercontent.com/61298783/135530171-51a51f9f-8eba-42dc-96ed-f43a5094437a.png)

![image](https://user-images.githubusercontent.com/61298783/135530226-1705255e-de91-423b-8e6d-2a9f0e7c91ea.png)

Podemos notar que agora temos altos valores categóricos no nosso conjunto de dados e, quando fizemos isso infelizmente aumentamos o desbalanceamento das nossas variáveis categóricas.

### 4.2 Matriz de Correlação Bivariada

A matriz de correlação bivariada é uma relação entre duas características que ajuda a entender de forma rápida sobre as informações correlacionadas no nosso conjunto de dados. 

Para isso, utilizaremos o ``seaborn`` e utilizaremos o mapa de calor que resumindo, as cores mais avermelhadas possuem uma alta correlação positiva (próximas de 1), enquanto os mais azulados possuem uma correlação negativa (próximas de -1).

![image](https://user-images.githubusercontent.com/61298783/135530347-0414699d-b6b7-4a76-ac9b-e28470f43475.png)

Analisando apenas a nossa váriavel-alvo price, podemos notar que as correlações com as colunas númericas estão bem baixas. Agora vamos ver as nossas variáveis-categóricas:

![image](https://user-images.githubusercontent.com/61298783/135530463-4ec1d0b2-c328-412e-b265-114851f3c3d4.png)

Podemos observar no mapa de calor acima que a nossa coluna ``price`` possui uma correlação positiva com o tipo de propriedade de apartamento, já para casa/casa possui uma correlação negativa. Já para a correlação de ``price`` para a região, temos uma correlação positiva apenas na Zona Oeste (porém baixa).

![image](https://user-images.githubusercontent.com/61298783/135530513-fc99da6c-db03-4c87-9454-aed0cf1e8a9e.png)

Na correlação de price para as colunas categóricas a maioria possui valores negativos, e os que estão positivos possuem uma correlação bem baixa.

### 4.3 Dados Geoespaciais Mulivariados

Para a nossa visualização se tornar mais completa, vamos utilizar dados geoespaciais utilizando as informações das colunas latitude e longitude. O interessante aqui é entender se a geografia influencia nas características. Para realizarmos esta análise utilizaremos a biblioteca ``plotly``.

![image](https://user-images.githubusercontent.com/61298783/135530754-4cd5612e-61f6-4091-bdf7-84193b54effb.png)

Como podemos observar no gráfico acima é que os tipos de propriedade de apartamentos tem uma maior concentração nas praias, fazendo total sentido pois as pessoas que alugam os airbnb's geralmente estão passando um tempo de férias. Já os do tipo casa/ quartos estão mais espalhados pelo mapa.

Observação: Também foram realizados outros plot's geógraficos que, optei em não colocar no README do projeto, para mais informações acesse o Jupyter Notebook :)

### 4.3 Construção de hipóteses e suposições

Nesta etapa será realizado a construção de hipóteses e suposições, as perguntas estão logo a seguir:

![image](https://user-images.githubusercontent.com/61298783/135531023-012978d1-f396-4c47-8cd7-74c8f057d4d2.png)

![image](https://user-images.githubusercontent.com/61298783/135531043-0a58e7c5-81b4-4c23-8bda-314a347c6ef3.png)

Podemos observar que os tipos de propriedades mais alugados são os apartamentos, que possuem um preço diário em média de R$374.

![image](https://user-images.githubusercontent.com/61298783/135531087-2d67ae8a-4b36-4954-85f8-815468edde77.png)

Como podemos observar, os bairros com os maiores preços são: Osvaldo Cruz, Inhoaíba, Grumari, Complexo do Alemão e Manguinhos.

![image](https://user-images.githubusercontent.com/61298783/135531119-5858a194-ffcb-4e6f-a54b-b062d3f23aa4.png)

Já os bairros com os menores preços são: Vila Kosmos, Senador Camará, Honório Gurgel, Galeão e Coelho Neto.

![image](https://user-images.githubusercontent.com/61298783/135531152-c80da067-e7e6-43a1-9707-25ed2a754735.png)

Podemos observar que as zonas Sul e Oeste possuem uma maior média de preços, porém também devemos levar em consideração que temos altos números de registros nestas zonas e, talvez esteja influenciando os preços.

Observação: Também houve outras perguntas que, para não ficar muito extenso optei em não colocar no README.

## :hammer_and_wrench: 5. Pré-processamento para os modelos

Tanto para os modelos de Classificação quanto de Regressão, tive que realizar algum pré-processamento dos dados para tornar mais fácil para o algoritmo ajustar um modelo a eles. Há uma grande variedade de transformações de pré-processamento que é possivel realizar para preparar os dados para modelagem, porém utilizaremos algumas técnicas comuns logo a seguir.

### 5.1 Codificando variáveis categóricas

Como a maioria dos algoritmos de aprendizado estatístico supervisionado só funcionam melhor com valores numéricos como entrada, é necessário então o pré-processamento das variáveis do tipo ``object`` antes de usar esse dataset como entrada para o treinamento de um modelo. Por exemplo, a nossa coluna ``neighbourhood_region`` possui recursos categóricos:

| neighbourhood_region |
| ---- |
|  Zona Sul   |
|  Zona Oeste   |
|  Zona Norte   |
|  Zona Central   |

Logo, iremos aplicar a * codificação ordinal * para substituir um valor inteiro exclusivo para cada categoria, mostrado logo abaixo:

| neighbourhood_region |
| ---- |
|  0   |
|  1   |
|  2   |
|  3   |

Porém, ao fazermos isto estamos simplesmente dizendo para o nosso modelo que a categoria 3(Central) é melhor do que a categoria 2 (Norte), 1 (Oeste) e 0 (Sul) quando isto não é verdade. 
E por isto, iremos utilizar outra técnica comum que é a *codificação dinâmica* que cria recursos binários individuais (0 ou 1) para cada valor de categoria possível. Por exemplo, a codificação one-hot-encoder traduz as categorias possíveis em colunas binárias como está logo abaixo:

|  neighbourhood_region_Sul  |  neighbourhood_region_Oeste  |  neighbourhood_region_Norte  | neighbourhood_region_Central
| -------  | -------- | -------- | -------- |
|    1     |     0    |    0     |    0     |
|    0     |     1    |    0     |    0     |
|    0     |     0    |    1     |    0     |
|    0     |     0    |    0     |    1     |

![image](https://user-images.githubusercontent.com/61298783/135532446-22118ee9-0849-4d35-9907-77d3bf8c7a61.png)

### 5.2 Dimensionando recursos numéricos

A normalização de recursos numéricos para que estejam na mesma escala evita que recursos com grandes valores produzam coeficientes que afetam desproporcionalmente as previsões. Por exemplo, suponha que seus dados incluam os seguintes recursos numéricos: (arrumar texto e checar se tá ok)

| availability_30 |  availability_60  |  availability_90  | availability_360  |
| - | --- | --- | --- |
| 12 | 41 | 76  | 304 |
    
Normalizar esses recursos para a mesma escala pode resultar nos seguintes valores (assumindo que availability_30 contém valores de 0 a 30, availability_60 contém valores de 0 a 60, availability_90 contém valores de 0 a 90 e availability_360 contém valores de 0 a 360):

|  availability_30  |  availability_60  |  availability_90  | availability_360 |
| --  | --- | --- | --- |
| 0.12 | 0.41| 0.76| 0.30|

Existem várias maneiras de dimensionar dados numéricos, como calcular os valores mínimo e máximo para cada coluna e atribuir um valor proporcional entre 0 e 1 ou usar a média e o desvio padrão de uma variável normalmente distribuída para manter o mesmo * spread * de valores em uma escala diferente.

![image](https://user-images.githubusercontent.com/61298783/135532505-2b6f2da8-31e2-4023-8834-2e047a448b12.png)

## :bar_chart: 5. Resultados Finais dos Modelos

Vale deixar claro que, antes de chegar aqui realizei também outras etapas, como construção dos modelos padrões, comparação dos resultados, ajuste dos hiperparamêtros, etc. Para uma visualização mais completa, acesse o Jupyter Notebook do projeto :)

Os resultados finais do projeto estão logo a seguir.

#### Problema de Classificação (Previsão do tipo de propriedade)

|Modelo|Acurácia|ROC|F1-Score
|-------|----|----|----|
|Decision Tree (Baseline) |80%|78%|70%|
|XGBoost |85%|84%|78%|
|Light Gradient Boosting (Pycaret) |84%|91%|75%|

#### Problema de Regressão (Previsão do preço diário da estadia)

|Modelo|MAE|RMSE|R²
|-------|----|----|----|
|Linear Regression (Baseline)|	145.46|198.55|0.37|
|Light Gradient Boosting |130.86|184.62|0.41|
|Light Gradient Boosting (Pycaret) |131.43|183.89|0.42|

Com isso, concluimos que primeiramente entendemos como so dados estavam e realizamos o pré-processamento dos dados logo em seguida, a remoção de colunas irrelevantes, tratamento de valores faltantes (NaN), tratamento de valores inconsistentes, filtragem de dados e por fim, a criação dos modelos de Classificação e Regressão. Os nossos modelos de Classificação de fato desempenharam bem mas também o modelo ainda poderia ter sido melhor. Já nos modelos de Regressão consegui atingir um bom desempenho mas nada de tão grandioso, onde também poderia ter sido melhor. Por hora, posso dizer que o projeto está finalizado e ele foi de bastante aprendizado e obrigado por me acompanhar neste projeto :D

## :grey_question: 6. Perguntas propostas

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

## :memo: 7. Licença

Esse projeto está utilizando a licença MIT. clique em [LICENSE](https://github.com/Gttz/airbnb-rio/blob/main/LICENSE) para visualizar mais detalhes.

---
