% Aula 5 - Projeto final
% Jayme Anchante
% 1 de março de 2021

# construção do modelo: regressão linear

## conceito

> Modelo a relação linear entre uma variável resposta (alvo) e uma ou mais variáveis explicativas (características)

[Fonte](https://en.wikipedia.org/wiki/Linear_regression)

## tipos

Regressão linear simples modela a relação entre uma resposta e uma característica

Regressão linear múltipla modela a relação entre uma resposta e duas ou mais características

## otimização

* mínimos quadrados ordinários
* minimização do erro (absoluto ou quadrático)
* minimização de uma função de custo penalizadora (regressão ridge ou lasso)

## apresentação matemática

Sendo os dados $\{y_i, x_{i1}, ..., x_{ip} \} ^{n}_{i=1}$ de $n$ amostras/linhas e $p$ variáveis/colunas. A relação linear é modelada por meio de um termo de erro $\epsilon$, uma variável aleatória não observável que adiciona/controla o ruído na relação de regressores e regressando tal que

$$ y_i = \beta_0 + \beta_1 x_{i1} + ... + \beta_p x_{ip} + \epsilon_i = x^T_i\beta + \epsilon_i,\;\;i=1,...,n$$

## estimação por mínimos quadrados ordinários

Começando com a proposição inicial que 

$$ \hat{\beta} = arg min S(B) = || \boldsymbol{y} - \boldsymbol{X\beta} || ^2$$

Depois de algumas transformações, chegamos em

$$ \hat{\beta} = (\boldsymbol{X}^T\boldsymbol{X})^{-1} \boldsymbol{X}^T\boldsymbol{y}$$

## estimação por mínimos quadrados ordinários: regressão simples

![](images/class_5/simple_ols.png){width=200px}

[comment]: # (source: https://wikimedia.org/api/rest_v1/media/math/render/svg/817c4939058094674f0ef2787ef175b5c7170c07)

## estimação ingênua

$$ \hat{\beta} = || \boldsymbol{y} - \boldsymbol{X\beta} || ^2$$

1. Gerar valores (aleatórios) para os $\beta$
2. Calcular a perda quadrática
3. Se a perda for menor, substituir os melhores parâmetros pelos atuais
4. Voltar ao passo 1

Pode ser incluído um critério de parada opcional baseado em um dos seguintes critérios: i) número de iterações; ii) erro mínimo aceitável; iii) número de iterações sem uma melhora no erro; iv) melhora marginal menor que um mínimo aceitável

## exercício

Sendo os dados de $X$ e $y$:

``` {.python .numberLines}
import numpy as np
rng = np.random.RandomState(42)
e = rng.random(100) * rng.randint(1, 50)
beta_0 = 3
beta_1 = 1.5
X = np.linspace(1, 100, 100)
y = beta_0 + (beta_1 * X) + e
```

1. Qual são os parâmetros que você deverá encontrar?
2. Plote os dados para explorar suas relações.
2. Escolha um dos métodos de estimação vistos anteriormente e estime os parâmetros?
3. Compare suas respostas com `sklearn.linear_model.LinearRegression` (dica: utilize o método `coef_` do estimador ajustado).

# interpretação do modelo: regressão linear

## dados

Base de dados [Guerry do pacote HistData](https://cran.r-project.org/web/packages/HistData/HistData.pdf) do R. Andre-Michel Guerry (1833) foi o primeiro a sistematicamente coletar dados sociais. As variáveis utilizadas são Lottery (apostas per capita na loteria), Literacy (percentual de militares que são alfabetizados), Wealth (impostos recolhidos per capita), Region (região da França).

``` {.python .numberLines}
import statsmodels.api as sm
df = sm.datasets.get_rdataset("Guerry", "HistData").data
columns = ['Lottery', 'Literacy', 'Wealth', 'Region']
df = df[columns].dropna()
df.head()
```

## regressão linear

``` {.python .numberLines}
import statsmodels.formula.api as smf
formula = 'Lottery ~ Literacy + Wealth + Region'
mod = smf.ols(formula=formula, data=df)
res = mod.fit()
```

## resultados

``` {.python .numberLines}
print(res.summary())
```

## informações gerais

![](images/class_5/part1.png){width=200px}

Variável dependente (alvo), data número de linhas/observações, df quer dizer degrees of freedom (graud de liberdade), tipo de covariância (existem diferentes especificações de covariância)

## informações de ajuste

![](images/class_5/part2.png){width=200px}

* R2: coeficiente de determinação é a proporção da variância da variável dependente que é explicada pelas variáveis explicativas. [Ver mais](https://pt.wikipedia.org/wiki/Coeficiente_de_determina%C3%A7%C3%A3o).
* R2 ajustado: o mesmo que o anterior mas ajustado pelo número de colunas. [Ver mais](https://pt.wikipedia.org/wiki/Coeficiente_de_determina%C3%A7%C3%A3o)
* Estatística F: poder preditivo das variáveis explicativas. 
* Probabilidade F: hipótese nula de que o modelo com intercepto e o modelo ajustado são iguais. Um valor menor que 0.05 rejeita esta hipótese com um grau de confiança de 5%.
* [AIC](https://pt.wikipedia.org/wiki/Crit%C3%A9rio_de_informa%C3%A7%C3%A3o_de_Akaike) (critério de informação de Akaike) e o BIC (critério de informação de Schwarz) são critérios para seleção de modelo. Quanto menor, melhor.

## coeficientes

![](images/class_5/part3.png){width=400px}

* Coeficiente é o valor do beta, o parâmetro que estamos buscando
* t: teste t cuja hipótese nula é de que o verdadeiro valor do parâmetro é zero
* P>|t|: caso seja menor que 0.05 rejeitamos a hipótese de que o parâmetro é zero (insignificante)
* Poderíamos escrever a equação como sendo:

$$ y = 38.65 -15.4 Reg[=E] -10 Reg[=N] + ... - 0.18 Lit + 0.45 Wel $$

## coeficientes: intercepto

![](images/class_5/part4.png){width=400px}

* Intercepto é o valor do alvo caso todas as variáveis independentes sejam zero
* Interpretação: caso região, alfabetização e riqueza sejam 0, o valor de apostas per capita é 38.65

## coeficientes: região

![](images/class_5/part5.png){width=400px}

* Para fazermos interpretações, precisamos levar em conta sempre a categoria base (oculta)
* Pessoas da região S jogam, em média, -4.5 francos que pessoas da categoria base (região central), tudo o mais constante (ceteris paribus)

## coeficientes: riqueza

![](images/class_5/part6.png){width=400px}

* Para variáveis numéricas, a interpretação é em função da própria unidade da variável
* A cada um franco a mais de riqueza per capita, a quantidade de francos colocados em aposta per capita é, em média, de 0.45, tudo o mais constante.
* Vemos que a variável alfabetização não é significativa a 5%, ao contrário da riqueza.

# projeto final

## formato

1. 20min de apresentação por pessoa
2. 5min de perguntas da platéia
3. 10min de resposta do apresentador

# alunos

## a vez de vocês

![](images/class_5/ds_students.jpeg){width=350px}

[comment]: # (source: https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fnews.bryant.edu%2Fsites%2Fnews%2Ffiles%2F2019-10%2Fnews-data-science-co-op-1160x652.jpg&f=1&nofb=1)

# próximos passos

## assuntos pertinentes

* nuvem, devops, git, CI/CD, agilidade
* processamento distribuído, em GPUs
* aprendizado profundo para reconhecimento de image, vídeo, som e outras aplicações
* outras linguagens de programação, R, julia, shell
* praticar, praticar, praticar...