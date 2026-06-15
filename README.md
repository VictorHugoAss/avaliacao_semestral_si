# Avaliação Semestral - Sistemas Inteligentes Avançados

Este repositório contém a implementação prática da Avaliação Semestral da disciplina de Sistemas Inteligentes Avançados. O projeto é composto por três módulos independentes que resolvem problemas distintos utilizando técnicas de Aprendizado de Máquina (Supervisionado e Não-Supervisionado).

## 📁 Estrutura do Repositório

O projeto está organizado da seguinte forma:

*   **`/data`**: Pasta contendo os arquivos `.csv` originais utilizados no treinamento.
*   **`/models`**: Pasta onde os modelos indutores, pipelines e normalizadores treinados (`.pkl`) são salvos para retenção de conhecimento.
*   **`Respostas_Teoricas.md`**: Arquivo de texto contendo as justificativas técnicas e teóricas exigidas no enunciado para cada questão.

### 🧠 Módulos Implementados

**Questão 1: Heart Failure (Agrupamento / K-Means)**
*   `Q1_treinamento.py`: Realiza o pré-processamento, descobre o número ótimo de clusters pelo método do cotovelo e treina o modelo K-Means [7, 8].
*   `Q1_inferencia.py`: Recebe os dados clínicos de um novo paciente e infere a qual grupo (cluster) ele pertence.

**Questão 2: Wine Quality (Classificação)**
*   `Q2_treinamento.py`: Unifica as bases de vinho tinto e branco, balanceia os dados utilizando SMOTE e treina três metaestimadores (Decision Tree, Random Forest e Logistic Regression), salvando o melhor.
*   `Q2_inferencia.py`: Recebe os dados de um novo vinho e classifica sua qualidade (nota de 3 a 9).

**Questão 3: Black Friday Sales (Classificação e Probabilidade)**
*   `Q3_treinamento.py`: Constrói um pipeline completo com `ColumnTransformer` (OneHotEncoder e StandardScaler) e treina modelos para prever a categoria do produto, método de pagamento e faixa etária do cliente.
*   `Q3_inferencia.py`: Simula uma venda e retorna não apenas a predição, mas o **grau de certeza (%)** da classificação utilizando probabilidades.

## ⚙️ Como Executar
1. Instale as dependências num ambiente Python: `pandas`, `numpy`, `scikit-learn` e `imblearn` (para o SMOTE).
2. Execute sempre o arquivo de **treinamento** de uma questão primeiro. Ele irá gerar os arquivos `.pkl` na pasta `/models`.
3. Em seguida, execute o arquivo de **inferência** correspondente para testar o sistema com dados novos.