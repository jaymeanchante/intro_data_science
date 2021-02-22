% Aula 3 - aprendizado de máquina com scikit-learn
% Jayme Anchante
% 24 de fevereiro de 2021

# motivação

## carregando os dados

``` {.python .numberLines}
from sklearn.datasets import load_iris
iris = load_iris()
```

## descrição dos dados

``` {.python .numberLines}
print(iris.DESCR)
```

## montando base única

Crie um objeto chamado `df` que seja um pd.DataFrame com os atributos e as espécies de iris.

Dica: qual o tipo do objeto iris? veja as chaves.

## montando base

``` {.python .numberLines}
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.loc[:, "species"] = iris.target
def f(x):
    return iris.target_names[x]

df.loc[:, "species"] = df["species"].apply(f)
```

## desafio

Dada a iris

``` {.python .numberLines}
vals = [[2, 3, 2.5, 0.7]]
uma_iris = pd.DataFrame(vals, columns=iris.feature_names)
```

A qual espécie de íris esta flor pertence?

## algoritmo

Crie um algoritmo/função que receba como argumento `uma_iris` e retorno um texto com a espécie de iris que ela seria?

Dica: Comece escrevendo em linguagem natural (português) as etapas que você gostaria de seguir, depois tente implementar as etapas.

Dica: explore os dados, medidas de centralidade, dispersão, agrupamento etc.

## regras de negócios

``` {.python .numberLines}
col = "petal length (cm)"
print(df.groupby("species")[col].mean())
if uma_iris.loc[0, col] < 3:
    print("setosa")
elif uma_iris.loc[0, col] >= 3 and uma_iris.loc[0, col] < 4.5:
    print("versicolor")
elif uma_iris.loc[0, col] >= 4.5:
    print("virginica")
```

## algoritmo das médias

Regressão linear

``` {.python .numberLines}
medias = df.groupby("species").mean()
diff = medias - uma_iris.loc[0, ]
diff_abs = diff.abs()
s = diff_abs.idxmin()
vc = s.value_counts()
vc.index[0]
```

## vizinhos

``` {.python .numberLines}
df = df.set_index("species")
diff = df - uma_iris.loc[0]
diff_abs = diff.abs()
soma = diff_abs.sum(axis=1)
soma.idxmin()
# n é a quantidade de vizinhos
n = 3
soma.sort_values().head(n)
```

# teoria e conceitos

## bibliografia

Conteúdo baseado no capítulo 5 do livro [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html) de Jake VanderPlas

## aprendizado de máquina

O que é:

* Subcategoria da inteligência artificial
* Aprender as correlações de um fenômeno a apartir de dados históricos
* Série de heurísticas para entender o padrão dos dados

O que não é:

* Solução mágica para todos os problemas
* Sempre efetiva, a melhor solução, fácil de implementar
* Díficil de aplicar

## tipos de aprendizado de máquina

* Supervisionado: modela a relação entre dados de entrada (conhecidos como `features` ou `características`) e os dados de saída (conhecidos como `target` ou `alvo`)  

* Não supervisionado: modela os dados de entrada sem o conhecimento de seu rótulo (`label` em inglês, o alvo)

## aprendizado supervisionado

* Regressão: quando o problema envolve alvos contínuos ou numéricos. E.g. prever a idade dos usuários a partir de sua foto de perfil ou outras características; prever a receita de uma empresa a partir do balanço contábil do ano anterior; prever a renda de um indivíduo a partir de suas características demográficas e profissionais.  

* Classificação: quando o problema envolve alvos categóricos ou textuais. E.g. prever se uma transação é uma fraude ou não; recomendar uma música com base nas músicas anteriores escutadas; detectar a presença ou não de um cachorro em uma foto.

## aprendizado não supervisionado

* Clusterização: agrupamento das amostras por afinidade. E.g. encontrar ocorrências anômalas caso pertençam a um certo grupo; segmentar clientes automaticamente; separar textos por tópicos.

* Redução de dimensionalidade: representar os dados de maneira mais sucinta. E.g. reduzir a quantidade de features a serem utilizadas; encontrar representações mais eficientes dos dados (embedding)

## visualização de um problema de classificação

![](images/class_3/05.01-classification-1.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-classification-1.png)

## proposição

![](images/class_3/05.01-classification-2.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-classification-2.png)

## visualização de um problema de regressão

![](images/class_3/05.01-regression-1.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-regression-1.png)

## proposição

![](images/class_3/05.01-regression-3.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-regression-3.png)

## visualização de um problema de clusterização

![](images/class_3/05.01-clustering-1.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-clustering-1.png)

## proposição

![](images/class_3/05.01-clustering-2.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-clustering-2.png)

## visualização de um problema de redução de dimensionalidade

![](images/class_3/05.01-dimesionality-1.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-dimesionality-1.png)

## proposição

![](images/class_3/05.01-dimesionality-2.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.01-dimesionality-2.png)

# a arte do aprendizado de máquina

## método da redução

O [reducionismo](https://en.wikipedia.org/wiki/Reductionism) é uma forma de descrever fenômenos complexos em outros menores e simplificados

Em vez de 

- Fazer um modelo para aumentar as vendas do comércio eletrônico da empresa

Uma redução seria

- Melhorar a experiência do usuário ofertando produtos condizentes com seu perfil (recomendação de produtos)
- Prover uma melhor estimativa de quando o produto chegará (previsão de tempo de envio)
- Mandar um cupom de desconto apenas para aqueles usuários que não converteriam sem ele (algoritmo de retenção)

## modelo

[Modelo](https://en.wikipedia.org/wiki/Statistical_model) é uma forma simplificada de representar a realidade em termos matemáticos

No caso de uma previsão de renovação de uma subscrição (churn), para sermos 100% efetivos precisaríamos saber exatamente os planos futuros de uma pessoa quanto ao serviço.

Um modelo poderia levar em conta atrasos no pagamento, data da última entrada no sistema, quantidade de ações realizadas na plataforma, atualizações do aplicativo etc.

## na prática

As empresas querem aumentar vendas, entender e melhorar a experiênca do usuário, tomar decisões baseadas em dados.

Cabe a você cientista reduzir esses problemas genéricos em um problema de dados e explicitá-lo por meio de um modelo que possa ser utilizado e trazer retornos tangíveis.

1. Queremos tomar decisões baseadas em dados.  
2. Mesmo email promocional é enviado para a base toda.
3. Segmentar a base em um atributo simples (idade ou data do último login)
4. Usar os passos anteriores como benchmark e tentar introduzir um algoritmo que traga um resultado melhor

# scikit-learn: modelos

## o "atalho"

``` {.python .numberLines}
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(df.iloc[:, :4], df.iloc[:, 4])
model.predict(uma_iris)
```

## a biblioteca

* Simple and efficient tools for predictive data analysis
* Accessible to everybody, and reusable in various contexts
* Built on NumPy, SciPy, and matplotlib
* Open source, commercially usable - BSD license

Fonte: [site do scikit-learn](https://scikit-learn.org/)

## componentes

* Classificação
* Regressão
* Clusterização
* Redução de dimensionalidade
* Seleção de modelos
* Pré processamento

Fonte: [site do scikit-learn](https://scikit-learn.org/)

## empresas que utilizam

* J.P.Morgan
* Spotify
* Evernote
* Booking.com

Fonte: [testemunhos no site do scikit-learn](https://scikit-learn.org/stable/testimonials/testimonials.html)

## módulos

Podem ser visualizados em na [API](https://scikit-learn.org/stable/modules/)

Lógica básica consiste em:

1. Importar o estimador apropriado
2. Escolher os parâmetros
3. Organizar os dados como características `X` e opcionalmente o alvo `y`
4. Aplicar o método `.fit` do estimador nos dados históricos
5. Aplicar o método `.predict` do estimador ajustado em dados novos

## começando em scikit-learn

``` {.python .numberLines}
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict([[4, 5, 6]])  # predict classes of new data
```

## aviso legal

Este não é um curso completo de estatística e o tempo limitado, a base teórica de cada modelo não será muito discutida nem aprofundada.

Ainda, para o praticante de aprendizado de máquina não é estritamente necessário (ainda que recomendável) conhecer o funcionamento de cada algoritmo.

Para entender o funcionamento de cada algoritmo, consultar o livro canônico: [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) de Trevor Hastie, Robert Tibshirani e Jerome Friedman

## modelos lineares

Muito utilizando como baseline para um problema por (quase) não terem parâmetros

Oferecem grandes vantagens para interpretação e testes de hipótese

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)

``` {.python .numberLines}
sklearn.linear_model.LinearRegression(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=None,
    positive=False
)
```

## modelos de vizinhança

Não há aprendizado no estrito senso, mas uma previsão é realizada utilizando a interpolação do vizinhos da base de treinamento

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)

``` {.python .numberLines}
sklearn.neighbors.KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None
```

## modelos de árvore

Fácil de extrair regras lógicas e visualização destas regras

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)

``` {.python .numberLines}
sklearn.tree.DecisionTreeRegressor(
    criterion='mse',
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    ccp_alpha=0.0
)
```

## modelos conjuntos: bagging

Diversas árvores de decisão cada uma recebendo uma pequena porção dos dados

Grande poder preditivo, fácil paralelização

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)

``` {.python .numberLines}
sklearn.ensemble.RandomForestRegressor(
    n_estimators=100,
    criterion='mse',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    # ...
```

## modelos conjuntos: boosting

Várias árvores de decisão em sequencia consertando os erros das árvores anteriores

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)

``` {.python .numberLines}
sklearn.ensemble.GradientBoostingRegressor(
    loss='ls',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    #...
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=0.0001
```

## redes neurais

Algoritmo de aprendizado profundo (todos os anteriores pertencem a categoria de aprendizado raso) muito utilizado para problemas envolvendo dados não tabulares (image, áudio, vídeo, texto)

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)

``` {.python .numberLines}
sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=100,
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=200,
    tol=0.0001,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=10,
    # ...
    )
```

## clustering com kmeans

Escolha de grupos acontece definição posições para os centróides e tentando melhorar a posição deles de com a disperção de cada grupo.

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

``` {.python .numberLines}
sklearn.cluster.KMeans(
    n_clusters=8,
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=None,
    algorithm='auto'
    )
```

## redução de dimensionalidade com pca

Redução linear de dimensionalidade usando SVD para projeção em um espaço dimensional reduzido.

[Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

``` {.python .numberLines}
sklearn.decomposition.PCA(
    n_components=None,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    random_state=None
```

## exercícios

1. Aplique kmeans a base de dados de iris com `n_clusters` igual a 3 (pois sabemos de antemão que a base possui 3 espécies) e com `random_state` igual a 42.
2. Tente dar nome a cada um dos clusters encontrados.
3. Utilizando a base de dados [boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html), ajuste uma rede neural mlp com duas camadas de 25 neurônios cada uma (`hidden_layer_sizes=(25,25)`), com uma máximo de 1000 iterações (`max_iter=1000`) e com `early_stopping` como verdadeiro.
4. Aplique o método `.predict` da mlp ajustada nos dados, qual a diferença entre os valores previstos e os valores reais? Faça a média e a soma absoluta do erro.
5. Reduza para apenas duas dimensões a base de dados boston e ajuste novamente a rede neural, sua resposta mudou na pergunta anterior?

# scikit-learn: demais ferramentas

## dados

Para os seguintes slides iremos utilizar a base iris e boston

``` {.python .numberLines}
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
X_reg, y_reg = load_boston(True)
X_clf, y_clf = load_iris(True)
```

## seleção de modelos: teste

É uma boa prática separar uma porção do teste para podermos aferir se estamos sendo efetivos com nosso modelo

``` {.python .numberLines}
sklearn.model_selection.train_test_split(
    test_size=None, train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None
    )
```

``` {.python .numberLines}
from sklearn.model_selection import train_test_split
reg_train_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
    )
clf_train_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
    )
print(reg_train_test[0].shape, reg_train_test[1].shape)
```

## seleção de modelos: visualizando validação cruzada

![](images/class_3/cv.png)

[comment]: # (source: https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1*rgba1BIOUys7wQcXcL4U5A.png&f=1&nofb=1)

## seleção de modelos: dobras para descoberta de parâmetros

``` {.python .numberLines}
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
splits = kf.split(X_clf)
for train, test in splits:
    X_train, y_train = X_clf[train], y_clf[train]
    X_test, y_test = X_clf[test], y_clf[train]
    # do something
```

## seleção de modelos: busca exaustiva

``` {.python .numberLines}
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
parameters = {"max_depth": [2, 4], "splitter": ["best", "random"]}
grid = GridSearchCV(dt, parameters)
grid.fit(clf_train_test[0], clf_train_test[2])
```

##  métricas: classificação

* Acurácia é a quantidade de classificações corretas sobre o total de predições
* Top-k é quando acertamos a classificação nas k classificações com maior probabilidade

``` {.python .numberLines}
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    confusion_matrix
)
```

## visualização da matriz de confusão

![](images/class_3/confusion.png){width=150px}

[comment]: # (source: https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.researchgate.net%2Fprofile%2FMartin_Andreoni%2Fpublication%2F264741935%2Ffigure%2Fdownload%2Ffig1%2FAS%3A392377639424001%401470561527049%2FFigura-22-Matriz-de-confusao-dos-Sistemas-de-Deteccao-de-Intrusao.png&f=1&nofb=1)

``` {.python .numberLines}
y_true = clf_train_test[3]
y_pred = grid.predict(clf_train_test[1])
confusion_matrix(y_true, y_pred)
```

## métricas: regressão

* Erro absoluto médio: valor previsto menos valor efetivo de todas as linhas, dividido pelo número de linhas
* Erro quadrático médio: valor previsto menos valor efetivo ao quadrado, dividido pelo número de linhas. Penaliza mais os grandes erros. É a medida utilizada pela regressão linear para otimização.
* R2 (coeficiente de determinação): variância do alvo que é explicado pelas características de entrada.

``` {.python .numberLines}
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

## pré processamento

* Normalização: tirar a média e dividir pelo desvio padrão
* Codificador: "transformar" variáveis categóricas em numéricas

``` {.python .numberLines}
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder    
)
X = [["a"], ["b"], ["c"], ["a"]]
ohe = OneHotEncoder()
ohe.fit_transform(X).toarray()
```

## pipeline

Vamos encadenar uma série de transformações até chegar no modelo final

Tudo será armazenado em um único objeto, para ser fácil reproduzir todos as etapas

Seguiremos [este passo a passo](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

A base de dados do titanic pode ser encontrada [aqui](https://gist.githubusercontent.com/michhar/2dfd2de0d4f8727f873422c5d959fff5/raw/fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv)

## exercícios

``` {.python .numberLines}
X_train = reg_train_test[0]
X_test = reg_train_test[1]
y_train = reg_train_test[2]
y_test = reg_train_test[3]
```

1. Ajuste um modelo linear na base de treino.
2. Veja o erro quadrático médio na base de teste com o modelo anterior.
3. Faça uma busca exaustiva na base de treino usando um modelo de florestas aleatórias e variando pelo menos 2 parâmetros.
4. Você conseguiu melhorar o erro quadrático médio no teste?

# exercício

## dados

Use a base de dados [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) da UCI

A parte "Data Set Description" contém toda descrição da base de dados

O "Data Folder" contêm os dados de treinamento e teste

## lista

1. Remova os NaN e treine uma regressão logística.
2. Substitua os NaN pela média da coluna (dê uma olhada no [imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)) e faça uma busca exaustiva por pelo menos 4 parâmetros de um algoritmo de sua escolha.
3. Escreva um pipeline de processamento de dados e treinamento de modelo utilizando a classe `Pipeline` do scikit-learn.