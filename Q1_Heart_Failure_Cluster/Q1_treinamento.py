import math, os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)

print("=" * 65)
print("QUESTÃO 1 — HEART FAILURE: TREINAMENTO K-MEANS")
print("=" * 65)

# 1. CARREGAMENTO

print("\n[1] CARREGAMENTO")
dados = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
print(f"  Shape: {dados.shape}")
print(f"  Colunas: {dados.columns.tolist()}")
print(f"  Nulos: {dados.isnull().sum().sum()}")
print(f"\n  Variáveis binárias encontradas:")
for c in dados.columns:
    if sorted(dados[c].unique().tolist()) == [0, 1]:
        print(f"    {c}: {dados[c].value_counts().to_dict()}")

# 2. PRÉ-PROCESSAMENTO

print("\n[2] PRÉ-PROCESSAMENTO")

atributos = dados.drop(columns=['DEATH_EVENT', 'time'])
print(f"  Colunas removidas: ['DEATH_EVENT', 'time']")
print(f"  Features para treino ({len(atributos.columns)}): {atributos.columns.tolist()}")

print(f"\n  Estatísticas ANTES da normalização:")
print(atributos.describe().round(2).to_string())

scaler = MinMaxScaler()
atributos_norm_array = scaler.fit_transform(atributos)
atributos_norm = pd.DataFrame(atributos_norm_array, columns=atributos.columns)

print(f"\n  Estatísticas APÓS normalização (todas as colunas em [0,1]):")
print(atributos_norm.describe().round(3).to_string())

print(f"\n  Verificação das binárias após MinMax (devem permanecer 0 ou 1):")
for c in ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']:
    vals = sorted(atributos_norm[c].unique().tolist())
    print(f"    {c}: {vals} ← sem alteração ✓")

pickle.dump(scaler, open('models/scaler_heart.pkl', 'wb'))
print(f"\n  Scaler salvo em models/scaler_heart.pkl")

# 3. MÉTODO DO COTOVELO — encontrar K ótimo

print("\n[3] MÉTODO DO COTOVELO — Encontrando K ótimo")

K_MAX = 10
K     = range(1, K_MAX + 1)
distorcoes = []

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(atributos_norm)
    dist = sum(
        np.min(cdist(atributos_norm, km.cluster_centers_, 'euclidean'), axis=1)
    ) / len(atributos_norm)
    distorcoes.append(dist)
    print(f"  k={k:2d}  distorção={dist:.4f}")

# Fórmula geométrica: distância perpendicular de cada ponto à reta
x0, y0 = K[0],    distorcoes[0]
xn, yn = K[-1],   distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    num = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    den = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    distancias.append(num / den)

k_otimo = K[distancias.index(max(distancias))]
print(f"\n  → K ótimo encontrado: {k_otimo}")

# 4. TREINAMENTO COM K ÓTIMO

print(f"\n[4] TREINAMENTO (K-Means com k={k_otimo})")
kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
kmeans.fit(atributos_norm)
print(f"  Inércia final: {kmeans.inertia_:.2f}")

pickle.dump(kmeans,  open('models/kmeans_heart.pkl',  'wb'))
pickle.dump(list(atributos.columns), open('models/colunas_heart.pkl', 'wb'))
print(f"  Modelos salvos em models/")

# 5. DESCRIÇÃO DOS CLUSTERS

print(f"\n[5] DESCRIÇÃO DOS CLUSTERS (centroides desnormalizados)")

# Desnormalizar os centroides para interpretar em escala real
centroides_norm = pd.DataFrame(kmeans.cluster_centers_, columns=atributos.columns)
centroides_nat  = pd.DataFrame(
    scaler.inverse_transform(centroides_norm),
    columns=atributos.columns
)

# Adicionar rótulo de cluster ao dataset original para análise
dados_com_cluster = dados.copy()
dados_com_cluster['cluster'] = kmeans.labels_

BINARIAS   = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
CONTINUAS  = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
              'platelets', 'serum_creatinine', 'serum_sodium']

print()
for i in range(k_otimo):
    grupo    = dados_com_cluster[dados_com_cluster['cluster'] == i]
    n        = len(grupo)
    obitos   = grupo['DEATH_EVENT'].sum()
    pct_obit = 100 * obitos / n
    row      = centroides_nat.iloc[i]

    print(f"{'─' * 60}")
    print(f"CLUSTER {i}  ({n} pacientes | {pct_obit:.1f}% óbitos)")
    print(f"{'─' * 60}")
    print(f"Idade média:{row['age']:.0f} anos")
    print(f"Fração de ejeção:{row['ejection_fraction']:.0f}%  (normal: 55-70%)")
    print(f"Creatinina sérica: {row['serum_creatinine']:.2f} mg/dL  (normal: 0.6-1.2)")
    print(f"Sódio sérico: {row['serum_sodium']:.0f} mEq/L  (normal: 135-145)")
    print(f"CPK:{row['creatinine_phosphokinase']:.0f} mcg/L")
    print(f"Plaquetas:{row['platelets']:.0f} kiloplaq/mL")
    print(f"Anemia:{'Sim' if row['anaemia']  > 0.5 else 'Não':5}  | "
          f"Diabetes: {'Sim' if row['diabetes'] > 0.5 else 'Não':5}  | "
          f"Hipertensão: {'Sim' if row['high_blood_pressure'] > 0.5 else 'Não'}")
    print(f"Fumante:   {'Sim' if row['smoking']  > 0.5 else 'Não':5}  | "
          f"Sexo: {'M' if row['sex'] > 0.5 else 'F'}")
    print()

print("=" * 65)
print("TREINAMENTO CONCLUÍDO")
print("=" * 65)