% Aula 2 - numpy e pandas
% Jayme Anchante
% 23 de fevereiro de 2021

# montando ambiente

## software

* git
* anaconda
  - virtual environments
  - reproducibility
* jupyter

# numpy

## pacotes que usam numpy

![](images/class_2/numpy_based.png)

[comment]: # (source: https://numpy.org/)

>  The fundamental package for scientific computing with Python, [NumPy website](https://numpy.org/)

## fronteira da ciência

[Case Study: First Image of a Black Hole](https://numpy.org/case-studies/blackhole-image/)

![](images/class_2/black_hole.jpg)

[comment]: # (source: https://cdn.wccftech.com/wp-content/uploads/2019/04/106398636_mediaitem106398635.jpg)

## listas vs arrays vs numpy arrays

- listas aceitam qualquer tipo de dados (flexibilidade)  
- arrays tem tipo fixo (armazenamento eficiente)
- numpy arrays tem tipo fixo e otimizações (armazenamento e cálculo eficiente)

## lista vs numpy array

![](images/class_2/array_vs_list.png)

[comment]: # (source: https://jakevdp.github.io/PythonDataScienceHandbook/figures/array_vs_list.png)

## criando arrays

``` {.python .numberLines}
np.zeros(10)
np.ones((2,2))
np.full((3,1), 3.14)
np.arange(5)
np.linspace(0, 1, 5)
```

## gerador de números pseudo-aleatórios

``` {.python .numberLines}
np.random.RandomState(42)
np.random.seed(42)
np.random.<tab>
```

## acessando elementos

``` {.python .numberLines}
x1 = np.random.randint(10, size=6)      # 1 dim
x2 = np.random.randint(10, size=(3, 4)) # 2 dim
x1[0]
x1[-2]
x2[0, 0]
x2[2, -1]
x2[0, 0] = 12
```

## fatiamento de elementos

``` {.python .numberLines}
# x[start:stop:step]
x1[:5]   # primeiros cinco elementos
x1[::2]  # cada dois elementos
x1[::-1] # inversão dos elementos
```

## cópia de objetos

``` {.python .numberLines}
x2_sub = x2[:2, :2]
x2_sub[0, 0] = 99
# o que aconteceu com x2?
x2_sub_copy = x2[:2, :2].copy() # fazendo uma cópia
x2_sub_copy[0, 0] = 42
# o que aconteceu com x2?
```

## reformatação de objetos

``` {.python .numberLines}
x = np.array([1, 2, 3])
x.reshape((1, 3)) # row vector via reshape
x[np.newaxis, :]  # row vector via newaxis
x.reshape((3, 1)) # column vector via reshape
x[:, np.newaxis]  # column vector via newaxis
```

## junção e separação de objetos

``` {.python .numberLines}
np.concatenate
np.vstack
np.hstack
np.split
np.vsplit
np.hsplit
```

## operações com numpy

``` {.python .numberLines}
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
big_array = np.random.randint(1, 100, size=1000000)

%timeit compute_reciprocals(big_array)

%timeit 1.0 / big_array
```

## operações com numpy

``` {.python .numberLines}
# operações agregação
x = np.arange(1, 6)
np.add.reduce(x)        # soma dos elementos
np.add.accumulate(x)    # soma acumulada
np.multiply.outer(x, x) # produto cartesiano
```

## sumarizando np.array

``` {.python .numberLines}
a = np.random.random(100)
sum(a)
np.sum(a)
a.max()
# soma com dados faltantes
np.nansum(a)
```

## máscaras np.array

``` {.python .numberLines}
a = np.random.random(10)
a > 5
a[a>5]
a[(a>5) & (a<7)]
# suporte ao |/or e ~/not
```

## dado o seguinte np.array

``` {.python .numberLines}
a = np.arange(25).reshape(5,5)
```

1. Retorne os valores pares positivos menores que 14.
2. Qual a média da segunda coluna?
3. Qual a soma da quarta linha?
4. Separe a última linha e transforme em um vetor coluna.
5. Salve o objeto contendo o np.array em disco.

# pandas

## história

Iniciado por [Wes McKinney](https://wesmckinney.com/) em 2008 quando ele trabalhava no mercado financeiro

Começou como uma implementação em Python da API de dataframe do R

Código aberto em 2009 e posterior apoio pela [NumFocus](https://numfocus.org/sponsored-projects)

Livro base [Pyhon para Análise de dados](https://wesmckinney.com/pages/book.html)

## pd.Series

``` {.python .numberLines}
import pandas as pd
data = pd.Series([0, 2, 4, 6])
# valores e índice são np.array
data.values
data.index
# acessando elementos pelo índice
data[-1]
data[:2]
```

## pd.Series vs np.array

Uma das grandes diferenças está no índice. Ele pode ser não numérico, não sequencial.

``` {.python .numberLines}
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                index=['a', 'b', 'c', 'd'])
data['b']
```

## pd.Series vs dict

``` {.python .numberLines}
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
population['California':'Illinois']
```

## pd.DataFrame

Se o pd.Series pode ser comparados a um vetor unidimensional, o pd.DataFrame pode ser comparado a uma matrix bidimensional.

O pd.DataFrame é como uma sequencia de pd.Series que compatilham o mesmo índice.

``` {.python .numberLines}
area_dict = {'California': 423967, 'Texas': 695662,
             'New York': 141297, 'Florida': 170312,
             'Illinois': 149995}
area = pd.Series(area_dict)
```

## pd.DataFrame a partir de pd.Series

``` {.python .numberLines}
states = pd.DataFrame({'population': population,
                       'area': area})
```

# exercícios de casa

## lista

Façam um pipelines para processamento de uma dataset escolhido