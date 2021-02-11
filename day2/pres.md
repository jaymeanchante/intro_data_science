# dia 2

numpy

pandas

---

# Environment

* git
* anaconda
  - virtual environments
  - reproducibility
* jupyter

---

## NumPy

>  The fundamental package for scientific computing with Python, [NumPy website](https://numpy.org/)

---

### pacotes que usam numpy

![](img/numpy_based.png)

---

### fronteira da ciência

[Case Study: First Image of a Black Hole](https://numpy.org/case-studies/blackhole-image/)

<img src="https://cdn.wccftech.com/wp-content/uploads/2019/04/106398636_mediaitem106398635.jpg" width="500">

---

### listas vs arrays vs numpy arrays

- listas aceitam qualquer tipo de dados (flexibilidade)  
- arrays tem tipo fixo (armazenamento eficiente)
- numpy arrays tem tipo fixo e otimizações (armazenamento e cálculo eficiente)

---

### lista vs numpy array

<img src="https://jakevdp.github.io/PythonDataScienceHandbook/figures/array_vs_list.png" width="700">

---

Criando arrays

<section>
  <pre><code data-trim data-noescape data-line-numbers>
np.zeros(10)
np.ones((2,2))
np.full((3,1), 3.14)
np.arange(5)
np.linspace(0, 1, 5)
  </code></pre>
</section>

---

Gerador de números pseudo-aleatórios

<section>
  <pre><code data-trim data-noescape data-line-numbers>
np.random.RandomState(42)
np.random.seed(42)
np.random.<tab>
  </code></pre>
</section>

---

Acessando elementos

<section>
  <pre><code data-trim data-noescape data-line-numbers>
x1 = np.random.randint(10, size=6)  # One-dimensional array
x2 = np.random.randint(10, size=(3, 4))
x1[0]
x1[-2]
x2[0, 0]
x2[2, -1]
x2[0, 0] = 12
  </code></pre>
</section>

---

Fatiamento de elementos

<section>
  <pre><code data-trim data-noescape data-line-numbers>
# x[start:stop:step]
x1[:5]   # primeiros cinco elementos
x1[::2]  # cada dois elementos
x1[::-1] # inversão dos elementos
  </code></pre>
</section>

---

Cópia de objetos

<section>
  <pre><code data-trim data-noescape data-line-numbers>
x2_sub = x2[:2, :2]
x2_sub[0, 0] = 99
# o que aconteceu com x2?
x2_sub_copy = x2[:2, :2].copy() # fazendo uma cópia
x2_sub_copy[0, 0] = 42
# o que aconteceu com x2?
  </code></pre>
</section>

---

Reformatação de objetos

<section>
  <pre><code data-trim data-noescape data-line-numbers>
x = np.array([1, 2, 3])
x.reshape((1, 3)) # row vector via reshape
x[np.newaxis, :]  # row vector via newaxis
x.reshape((3, 1)) # column vector via reshape
x[:, np.newaxis]  # column vector via newaxis
  </code></pre>
</section>

---

Junção e separação de objetos

<section>
  <pre><code data-trim data-noescape data-line-numbers>
np.concatenate
np.vstack
np.hstack
np.split
np.vsplit
np.hsplit
  </code></pre>
</section>

---

Operações com numpy

<section>
  <pre><code data-trim data-noescape data-line-numbers>
def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
big_array = np.random.randint(1, 100, size=1000000)

%timeit compute_reciprocals(big_array)

%timeit 1.0 / big_array
  </code></pre>
</section>

---

Operações com numpy

<section>
  <pre><code data-trim data-noescape data-line-numbers>
# suporte para  em np.array
# operações agregação
x = np.arange(1, 6)
np.add.reduce(x)        # soma dos elementos
np.add.accumulate(x)    # soma acumulada
np.multiply.outer(x, x) # produto cartesiano
  </code></pre>
</section>

---

Sumarizando np.array

<section>
  <pre><code data-trim data-noescape data-line-numbers>
a = np.random.random(100)
sum(a)
np.sum(a)
a.max()
# soma com dados faltantes
np.nansum(a)
  </code></pre>
</section>

---

Máscaras np.array

<section>
  <pre><code data-trim data-noescape data-line-numbers>
a = np.random.random(10)
a > 5
a[a>5]
a[(a>5) & (a<7)]
# suporte ao |/or e ~/not
  </code></pre>
</section>

---

# Exercícios de casa

Façam um pipelines para processamento de uma dataset escolhido