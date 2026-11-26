# Análisis de sentimiento y dimensiones semánticas en reseñas de hoteles de lujo

**Comparación entre representaciones dispersas (BoW / TF-IDF) y distribuidas (Word2Vec), proyecciones semánticas y modelado de tópicos**

| | |
|:--|:--|
| **Autores** | Isabel Cristina Morales Parra · Jaime Esteban Bejarano Caldas |
| **Curso** | Procesamiento de Lenguaje Natural |
| **Programa** | Maestría en Analítica Aplicada e Inteligencia Artificial · Universidad de La Sabana |
| **Profesor** | Andrés Rosso, Ph.D. |
| **Fecha** | Noviembre de 2026 |

---

## 1. Problema y objetivo

Las reseñas de huéspedes combinan una calificación numérica con un texto libre: la calificación dice *cuánto* le gustó la experiencia al huésped; el texto explica *por qué*. Este proyecto construye y evalúa un flujo de procesamiento de lenguaje natural que, a partir del texto, **predice la polaridad de la experiencia** e **identifica las dimensiones del servicio** asociadas a valoraciones positivas y negativas.

| # | Objetivo específico | Notebook | Estado |
|:--|:--|:--|:--|
| 1 | Caracterizar el corpus mediante análisis exploratorio textual | 01 | ✅ |
| 2 | Normalizar el texto y cuantificar su efecto sobre el vocabulario | 01 | ✅ |
| 3 | Comparar clasificadores con BoW, TF-IDF y Word2Vec | 02 · 03 | ✅ BoW / TF-IDF · ✅ Word2Vec |
| 4 | Evaluar de forma intrínseca el espacio Word2Vec del dominio | 03 | ✅ |
| 5 | Definir dimensiones del servicio mediante proyecciones semánticas | 03 | ✅ |
| 6 | Identificar temas mediante modelado de tópicos (NMF, LDA, SVD + KMeans) | 04 | ✅ |

---

## 2. Datos y depuración

`LuxuryHotels.csv`: 18.074 reseñas con calificación de 1 a 5 y año. La auditoría de calidad encontró tres problemas que no estaban documentados:

| Paso | Registros | Eliminados | Motivo |
|:--|--:|--:|:--|
| Archivo original | 18.074 | — | |
| Sin registros `#NAME?` | 18.071 | 3 | Error de exportación de Excel: el texto original se perdió |
| Sin duplicados | 17.976 | 95 | Textos idénticos |
| **Solo inglés** | **17.263** | 713 | Reseñas en 23 idiomas distintos del inglés (detección con `langdetect`) |

![Idiomas distintos del inglés](figures/01_idiomas_no_ingles.png)

**Variable objetivo (escenario principal A):** positivo = 4–5 estrellas, negativo = 1–2 estrellas (se excluye 3). Hay **16.438 reseñas, de las cuales solo el 4,0 % son negativas** (≈ 24 : 1), por lo que la exactitud no sirve como métrica: un modelo que siempre dice «positivo» acierta el 96 %.

**Supuestos declarados:**
- La columna `YY` corresponde al año (2014–2024).
- El corpus se concentra en hoteles de Nepal (*Kathmandu*, *Pokhara*); no hay identificador de hotel.
- El origen del conjunto de datos no está documentado; se usa exclusivamente con fines académicos.

---

## 3. Exploración del corpus

**Las reseñas negativas son casi el doble de largas** (mediana de 94–100 palabras frente a 55 en las de 5 estrellas), y **la proporción de negativas cae de 7,7 % (2017) a 1,5 % (2024)**: hay deriva temporal.

![Longitud y composición anual](figures/01_longitud_y_composicion_anual.png)

El vocabulario sigue la **ley de Zipf** (pendiente −1,38) y el **48,5 % de las palabras aparece una sola vez**, en su mayoría errores de digitación y nombres propios.

![Ley de Zipf](figures/01_zipf_y_frecuencias.png)

**Normalización** (spaCy: lematización y eliminación de palabras vacías **conservando las negaciones**): los tokens bajan 53,1 % y el vocabulario 17,3 %.

**Vocabulario distintivo.** 10 de los 15 bigramas más característicos de las reseñas negativas contienen una negación (*not worth*, *not clean*, *not recommend*). La insatisfacción se asocia a **cobros y gestión** (*refund*, *bill*, *charge*) y a **higiene** (*dirty*, *stain*); la satisfacción, a **comida, trato y paisaje** (*delicious*, *attentive*, *valley*, *sunset*).

![Bigramas distintivos](figures/01_bigramas_distintivos.png)

![Nubes de palabras distintivas](figures/01_nubes_distintivas.png)

---

## 4. Clasificación de sentimiento

Regresión logística con `class_weight="balanced"`, `C` ajustado por validación cruzada estratificada (5 particiones), evaluada en un conjunto de prueba de 3.288 reseñas (130 negativas).

| Modelo | F1-macro | Exactitud balanceada | *Recall* negativo | Precisión negativa |
|:--|--:|--:|--:|--:|
| Línea base (siempre «positivo») | 0,490 | 0,500 | 0,000 | 0,000 |
| BoW binario | 0,884 | 0,916 | 0,846 | 0,719 |
| BoW conteos | 0,886 | 0,891 | 0,792 | 0,769 |
| **TF-IDF (uni + bigramas)** | **0,918** | **0,927** | **0,862** | **0,824** |
| Word2Vec (promedio) | 0,806 | 0,945 | **0,931** | 0,482 |

**TF-IDF con bigramas** detecta 112 de las 130 reseñas negativas. Su ventaja proviene de la ponderación IDF y de los bigramas con negación.

![Comparación de representaciones dispersas](figures/02_comparacion_dispersas.png)

La **negación** (*not*) es la señal negativa más fuerte del modelo.

![Coeficientes TF-IDF](figures/02_coeficientes_tfidf.png)

---
## 5. Word2Vec y proyecciones semánticas

**Espacio vectorial del dominio** (skip-gram, 100 dimensiones, entrenado solo con las reseñas de entrenamiento): *dirty* → *filthy, mould, stain*; *refund* → *cancellation, agoda, payment*; la analogía *kathmandu − city + lake* devuelve **phewa**, el lago de Pokhara. El primer componente principal separa lo negativo de lo positivo.

![Word2Vec PCA](figures/03_word2vec_pca.png)

**Curva de aprendizaje.** Con pocas reseñas etiquetadas, **Word2Vec supera a TF-IDF** (F1-macro 0,837 frente a 0,646 con 250 reseñas); las curvas se cruzan alrededor de 1.000 y con el conjunto completo gana TF-IDF (0,918 frente a 0,806). Word2Vec aprende la semántica sin etiquetas, pero el promedio de vectores diluye la negación.

![Curva de aprendizaje](figures/03_curva_aprendizaje.png)

**Dimensiones del servicio.** Proyectar la reseña completa produce dimensiones redundantes (correlación 0,79–0,96: todas miden el sentimiento general); proyectar solo el contexto de cada aspecto las separa (0,32–0,59).

![Correlación entre métodos](figures/03_proyecciones_correlacion_metodos.png)

| Dimensión | Mencionada en negativas | d de Cohen | Hallazgo |
|:--|--:|--:|:--|
| Personal | 71 % | **2,72** | La que más separa experiencias positivas y negativas |
| Limpieza | 70 % | 2,03 | La peor valorada en las negativas; sobrerrepresentada |
| Comida | 57 % | 2,22 | |
| Precio | 40 % | 0,93 | 3,6 veces más frecuente en negativas; quejas de cobros, no de precio alto |
| Instalaciones | 43 % | 1,82 | |

![Dimensiones del servicio](figures/03_dimensiones_servicio.png)

## 6. Modelado de tópicos

Se compararon **NMF + TF-IDF**, **LDA + conteos** y **SVD + KMeans** (línea base) con K = 8 y K = 12, sobre el mismo vocabulario (4.088 términos).

| Método | K | NPMI ↑ | Diversidad@10 ↑ | Estabilidad Jaccard ↑ | NMI externo | Tiempo (s) ↓ |
|:--|--:|--:|--:|--:|--:|--:|
| **NMF + TF-IDF** | **8** | **0,160** | **0,888** | **0,955** | 0,066 | 5 |
| NMF + TF-IDF | 12 | 0,163 | 0,892 | 0,576 | 0,059 | 3 |
| LDA + conteos | 8 | 0,120 | 0,662 | 0,586 | 0,095 | 209 |
| LDA + conteos | 12 | 0,134 | 0,600 | 0,432 | 0,113 | 205 |
| SVD + KMeans | 8 | 0,141 | 0,625 | 0,648 | 0,092 | 3 |
| SVD + KMeans | 12 | 0,126 | 0,517 | 0,695 | 0,078 | 3 |

**Selección: NMF con K = 8**, el mejor en coherencia, diversidad y estabilidad. Con K = 12 la estabilidad cae a 0,576: los tópicos adicionales dependen de la semilla.

**Sin usar las calificaciones, el modelo aisló el tema de la insatisfacción:** el tópico *Problemas operativos* (*room, night, check, water, work*) ocupa el **54 % de las reseñas negativas y el 14 % de las positivas**. Los tópicos de destino (*Safari en Chitwan*, *Vistas de Pokhara*) y de trato (*Amabilidad del personal*) predominan en las positivas.

![Tópicos por sentimiento](figures/04_topicos_por_sentimiento.png)

**Validación humana.** En la prueba de intrusión de palabras, el evaluador identificó la intrusa en 2 de 8 tópicos (25 %, frente a 16,7 % por azar). Los aciertos ocurrieron en los tópicos de menor NPMI: la coherencia estadística y la interpretabilidad humana no coinciden en este dominio homogéneo.

## 7. Conclusiones

🔄 *Pendiente.*

---

## Estructura del repositorio

```
├── data/
│   ├── raw/            LuxuryHotels.csv (original)
│   └── processed/      Corpus depurado y normalizado
├── notebooks/          01 datos y exploración · 02 clasificación · 03 Word2Vec · 04 tópicos · 05 evaluación
├── src/                preprocesamiento.py · evaluacion.py · representaciones.py
├── results/            Tablas y métricas (CSV)
├── figures/            Gráficos
├── references/         Artículos de referencia
├── informe/            Informe final y anteproyecto
└── presentacion/       Diapositivas
```

## Cómo ejecutar

1. Clonar: `git clone https://github.com/isabelmorales2108-ux/pln-luxuryhotels-sentimiento.git`
2. Entorno Python 3.12 con: `pandas`, `numpy`, `scikit-learn`, `gensim`, `spacy`, `nltk`, `langdetect`, `seaborn`, `wordcloud`; modelo de spaCy: `python -m spacy download en_core_web_sm`
3. Ejecutar los notebooks en orden (01 → 05). El notebook 01 genera `data/processed/`, que usan los demás.

## Referencias

- Mikolov, T., Chen, K., Corrado, G. y Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. y Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. arXiv:1310.4546.
- Blei, D., Ng, A. y Jordan, M. (2003). *Latent Dirichlet Allocation*. JMLR, 3, 993–1022.
- Lee, D. y Seung, H. (1999). *Learning the parts of objects by non-negative matrix factorization*. Nature, 401, 788–791.
