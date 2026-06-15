QUESTÃO 1 — Heart Failure

a) Justificar a escolha do metaestimador
Como o objetivo da tarefa é indicar a qual grupo o paciente pertence, estamos lidando com um problema de agrupamento (clustering) por similaridade clínica, e não uma classificação preditiva. Por isso, optei pelo K-Means (Aprendizado Não-Supervisionado).

b) Procedimentos de pré-processamento

Limpeza: Primeiro, removi a coluna alvo DEATH_EVENT para garantir que o modelo agrupasse os pacientes pelo estado clínico no momento do exame, e não pelo óbito. Também retirei a coluna time, por ser apenas um dado de controle de acompanhamento.

Tratamento de Escalas e Binários: A base possui dados contínuos de alta magnitude (como plaquetas) e informações binárias (0 e 1, como diabetes e sexo). Apliquei o MinMaxScaler em toda a base. Isso reduziu os valores gigantes para a escala [0, 1] sem distorcer as variáveis binárias, garantindo que todas as features tivessem o mesmo peso no cálculo de distância euclidiana do K-Means.

c) A inferência funcionando
O módulo de inferência foi construído para receber um dicionário com os dados de um novo paciente. O código converte esses dados, aplica o .transform() do MinMaxScaler treinado previamente e usa a função .predict() do K-Means para alocar o paciente ao grupo (centroide) mais próximo.

QUESTÃO 2 — Wine Quality

a) O fluxo de procedimentos (Pipeline)
O pipeline desenvolvido seguiu estas etapas:

Carregamento dos arquivos winequality-red.csv e winequality-white.csv, adicionando a coluna is_red (1 = tinto, 0 = branco) para preservar a origem do vinho.

Concatenação dos dois arquivos em um único DataFrame.

Normalização dos atributos químicos com StandardScaler.

Balanceamento de classes usando SMOTE (aplicado estritamente aos dados de treino). Isso foi essencial para resolver o forte desbalanceamento da base, que era dominada por vinhos de nota 5 e 6.

b) Acurácia global, por classes e F1-Score
Treinei três algoritmos diferentes (Decision Tree, Random Forest e Logistic Regression). No código, extraí a Matriz de Confusão e utilizei a função classification_report para calcular a acurácia por classe, a acurácia global e o F1-Score ponderado (weighted) de cada modelo, exibindo tudo no terminal para comparação.

c) Justificativa do modelo adequado para implantação
O modelo escolhido para produção foi a Random Forest. A decisão foi baseada no F1-Score ponderado, e não apenas na acurácia global. Em bases desbalanceadas, a acurácia pode ser ilusória. O F1-Score penaliza os modelos que erram as classes minoritárias (vinhos excepcionais ou muito ruins). A Random Forest se mostrou muito mais robusta para lidar com as fronteiras não-lineares desses dados químicos, evitando o overfitting observado na Árvore de Decisão simples.

QUESTÃO 3 — Black Friday Sales Dataset

a) O fluxo de procedimentos (Pipeline)
Como a base mistura dados categóricos (gênero, cidade) e numéricos (valor da compra), utilizei um ColumnTransformer do Scikit-Learn. Apliquei o OneHotEncoder nas variáveis categóricas (para não criar uma falsa ordem hierárquica que o LabelEncoder geraria) e o StandardScaler nas numéricas. Esse pipeline foi instanciado e treinado três vezes, uma para cada modelo independente (Categoria, Pagamento e Idade).

b) Cálculo das Métricas (Sensibilidade, Especificidade, etc.)
Para problemas multiclasse, o cálculo dessas métricas não é direto. No script, construí uma lógica que itera sobre a Matriz de Confusão do modelo vencedor. A partir dela, o código isola matematicamente os Verdadeiros Positivos (TP), Falsos Negativos (FN), Falsos Positivos (FP) e Verdadeiros Negativos (TN) de cada classe, calculando a média global de Sensibilidade e Especificidade para cumprir a exigência do enunciado.

c) Funcionamento do sistema e Grau de Certeza
O módulo de inferência recebe o contexto (circunstância) de uma nova venda e o repassa simultaneamente aos três pipelines treinados. Para exibir o grau de certeza exigido, substituí a saída tradicional .predict() pelo método .predict_proba(). Assim, o sistema não joga apenas a classe final na tela, mas indica qual é a predição e a exata porcentagem de confiança do algoritmo naquela resposta.
