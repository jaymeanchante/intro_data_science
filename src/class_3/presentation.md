% Aula 3 - aprendizado de máquina com scikit-learn
% Jayme Anchante
% 24 de fevereiro de 2021

# Motivação

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
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.loc[:, "species"] = iris.target
def f(x):
    return iris.target_names[x]
df.loc[:, "species"] = df["species"].apply(f)
```

## Desafio

Dada a iris

``` {.python .numberLines}
vals = [[2, 3, 2.5, 0.7]]
uma_iris = pd.DataFrame(vals, columns=iris.feature_names)
```

A qual espécie de íris esta flor pertence?

## Algoritmo

Crie um algoritmo/função que receba como argumento `uma_iris` e retorno um texto com a espécie de iris que ela seria?

Dica: Comece escrevendo em linguagem natural (português) as etapas que você gostaria de seguir, depois tente implementar as etapas.

Dica: explore os dados, medidas de centralidade, dispersão, agrupamento etc.

## Regras de negócios

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

## Algoritmo das médias

Regressão linear

``` {.python .numberLines}
medias = df.groupby("species").mean()
diff = medias - uma_iris.loc[0, ]
diff_abs = diff.abs()
diff_abs.idxmin()
vc = s.value_counts()
vc.index[0]
```

## Vizinhos

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

##

``` {.python .numberLines}
```

##

``` {.python .numberLines}
```

##

``` {.python .numberLines}
```

##

``` {.python .numberLines}
```

##

``` {.python .numberLines}
```

