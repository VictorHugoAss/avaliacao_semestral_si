Este documento detalha as decisões de projeto e o embasamento teórico para as abordagens adotadas nos algoritmos de aprendizado de máquina da Avaliação Semestral - Parte Prática.

---

## QUESTÃO 1 — Heart Failure

**a) Justificar a escolha do metaestimador**
Para esta tarefa, o objetivo exigido era indicar "a qual grupo esse paciente pertence", o que caracteriza um problema de agrupamento (clustering) por semelhança clínica, e não uma classificação preditiva. Por isso, o metaestimador escolhido foi o **K-Means** (Aprendizado Não-Supervisionado). O número ótimo de grupos foi calculado matematicamente através do método do cotovelo, garantindo que o modelo descobrisse os agrupamentos por perfil fisiológico.

**b) Demonstrar todos os procedimentos de pré-processamento**
1. **Limpeza de Atributos:** A coluna alvo `DEATH_EVENT` foi removida para garantir que o modelo agrupasse os pacientes pelo estado clínico e não pelo desfecho (óbito). A coluna temporal `time` também foi removida, pois é um dado de controle de acompanhamento e não um dado fisiológico.
2. **Tratamento de Variáveis Binárias:** A base possui informações binárias (ex: *anaemia*, *diabetes*, *sex*, etc.). A estratégia adotada foi aplicar o `MinMaxScaler` em **todas** as colunas. Como as variáveis binárias já operam nos limites de 0 e 1, a normalização não as distorce, mas reduz as variáveis contínuas de alta magnitude (como *creatinine_phosphokinase*) para a mesma escala. Isso garante que todas as *features* tenham o mesmo peso no cálculo de distância euclidiana do K-Means.

**c) Demonstrar a inferência funcionando**
A inferência foi implementada de forma a receber o dicionário de um novo paciente com dados clínicos desconhecidos, aplicar o mesmo transformador `MinMaxScaler` treinado originalmente e utilizar o preditor `.predict()` do K-Means para alocá-lo ao centroide (grupo) mais próximo, informando o resultado final.

---

## QUESTÃO 2 — Wine Quality

**a) O fluxo de procedimentos até o treinamento do modelo (pipeline)**
O pipeline consistiu em:
1. Carregar os dois arquivos originais (`winequality-red.csv` e `winequality-white.csv`) e adicionar uma nova coluna indicadora `is_red` (1 para tinto, 0 para branco) para não perder essa característica importante.
2. Concatenar os dados em um único arquivo de base de conhecimento.
3. Normalizar os dados numéricos utilizando o `StandardScaler`.
4. Aplicar o algoritmo **SMOTE** apenas nos dados de treinamento para gerar exemplos sintéticos por interpolação, resolvendo o extremo desbalanceamento da base original (que era amplamente dominada pelas notas 5 e 6).

**b) Acurácia global, acurácia por classes e a medida f1-score**
No módulo de treinamento, a função customizada extrai a matriz de confusão (`confusion_matrix`) normalizada para calcular e exibir a **acurácia por classe** individualmente. Após o treinamento dos 3 metaestimadores concorrentes exigidos (Decision Tree, Random Forest e Logistic Regression), o sistema imprime a **Acurácia Global** e o **F1-Score ponderado** (weighted) de cada um para embasar a decisão.

**c) Justificar qual é o modelo mais adequado para possível implantação**
O modelo selecionado para implantação foi o **Random Forest**.
O critério de seleção baseou-se no desempenho superior do *F1-Score ponderado* (e não apenas na acurácia global). Em bases muito desbalanceadas como esta, a acurácia global engana. O F1-Score é mais realista por penalizar os algoritmos que erram as classes mais difíceis (vinhos com notas muito altas ou muito baixas). O Random Forest equilibra um melhor desempenho para traçar fronteiras não lineares de predição e garante maior robustez contra *outliers*.

---

## QUESTÃO 3 — Black Friday Sales Dataset

**a) O fluxo de procedimentos até o treinamento do modelo (pipeline)**
Como a base contém uma mistura de características numéricas contínuas (ex: valor da compra, desconto) e dados categóricos (ex: gênero, cidade), construímos um pipeline utilizando um `ColumnTransformer`. O pré-processador aplicou o `OneHotEncoder` nas variáveis categóricas para transformá-las em matrizes binárias sem criar falsas ordens hierárquicas, e aplicou o `StandardScaler` nas colunas numéricas. Esse pipeline de limpeza e pré-processamento foi treinado de forma independente para cada um dos 3 alvos (*product category*, *payment_method* e *age_group*).

**b) Acurácia global, acurácia por classes e a medida f1-score / Especificidade / Sensibilidade**
Através da biblioteca `sklearn.metrics`, o script de avaliação extrai a matriz de confusão (*Confusion Matrix*). Através da extração algébrica das variáveis da matriz — *True Positives (TP)*, *False Negatives (FN)*, *False Positives (FP)* e *True Negatives (TN)* — o sistema calcula manualmente e exibe as exigências de Sensibilidade, Especificidade, Acurácia Global, e por classe.

**c) O funcionamento do sistema inteligente indicando categoria do produto, forma de pagamento e faixa etária**
O módulo de inferência final recebe as circunstâncias de uma venda pontual e submete essa nova entrada simultaneamente através dos três algoritmos gerados. Para cumprir o requisito de demonstrar um **grau de certeza**, o sistema substitui a saída tradicional e utiliza o método estatístico `.predict_proba()` do Scikit-Learn. Com isso, o código extrai não apenas a classificação, mas as chances associadas e imprime em tela a predição juntamente com a porcentagem exata de confiança do modelo em sua resposta.