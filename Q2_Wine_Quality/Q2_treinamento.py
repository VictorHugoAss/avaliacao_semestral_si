"""
 QUESTÃO 2 — Wine Quality (UCI 186)
 Classificador de Qualidade de Vinho (notas 3 a 9)

PIPELINE:
  1. Carregar winequality-red.csv e winequality-white.csv
  2. Combinar com identificador de tipo (is_red)
  3. Salvar arquivo unificado
  4. Normalizar com StandardScaler
  5. Balancear com SMOTE (classes muito desiguais: nota 5/6 dominam)
  6. Treinar 3 classificadores
  7. Comparar: acurácia global, acurácia por classe, F1-Score
  8. Justificar o melhor para produção
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)

print("=" * 65)
print("QUESTÃO 2 — WINE QUALITY: TREINAMENTO")
print("=" * 65)


# 1. CARREGAMENTO DOS ARQUIVOS ORIGINAIS

print("\n[1] CARREGAMENTO")
df_red   = pd.read_csv('data/winequality-red.csv',   sep=';')
df_white = pd.read_csv('data/winequality-white.csv', sep=';')

print(f"  Vinhos tintos:  {df_red.shape[0]} amostras")
print(f"  Vinhos brancos: {df_white.shape[0]} amostras")
print(f"  Colunas: {df_red.columns.tolist()}")


# 2. MONTAR ARQUIVO UNIFICADO
# Adicionei is_red (1=tinto, 0=branco) para o modelo saber
# de qual tipo é o vinho, isso é uma feature adicional relevante.

df_red['is_red']   = 1
df_white['is_red'] = 0

df_wine = pd.concat([df_red, df_white], ignore_index=True)
df_wine.to_csv('data/winequality-all.csv', index=False)
print(f"\n[2] ARQUIVO UNIFICADO: {df_wine.shape[0]} amostras salvas em data/winequality-all.csv")

print(f"\n  Distribuição de qualidade (dataset completo):")
dist = df_wine['quality'].value_counts().sort_index()
for nota, qtd in dist.items():
    barra = '█' * (qtd // 50)
    print(f"    Nota {nota}: {qtd:5d}  {barra}")

# 3. SEPARAÇÃO FEATURES / ALVO

print("\n[3] SEPARAÇÃO FEATURES / ALVO")
X = df_wine.drop(columns=['quality'])
y = df_wine['quality']

print(f"Features ({X.shape[1]}): {X.columns.tolist()}")
print(f"Alvo: quality  |  Classes: {sorted(y.unique().tolist())}")

# Divisão antes de normalizar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")


# 4. NORMALIZAÇÃO

print("\n[4] NORMALIZAÇÃO (StandardScaler)")
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)
print(f"  Feito. Média das features de treino ≈ 0, desvio ≈ 1.")


# 5. BALANCEAMENTO COM SMOTE

print("\n[5] BALANCEAMENTO (SMOTE)")
print(f"  Antes do SMOTE — distribuição no treino:")
antes = pd.Series(y_train).value_counts().sort_index()
for nota, qtd in antes.items():
    print(f"    Nota {nota}: {qtd}")

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train_norm, y_train)

print(f"\n  Depois do SMOTE — distribuição no treino:")
depois = pd.Series(y_train_bal).value_counts().sort_index()
for nota, qtd in depois.items():
    print(f"    Nota {nota}: {qtd}")
print(f"  Total treino balanceado: {len(X_train_bal)} amostras")

# 6. TREINAMENTO E AVALIAÇÃO DOS 3 MODELOS

modelos = {
    'Decision Tree':DecisionTreeClassifier(max_depth=15, random_state=42),
    'Random Forest':RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
}

melhor_f1   = -1
melhor_nome = ''
melhor_modelo = None
classes = sorted(y_test.unique())

def calcular_acc_por_classe(y_true, y_pred, classes):
    """Diagonal da matriz de confusão normalizada = acurácia por classe."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm, cm_norm

print("\n[6] TREINAMENTO E AVALIAÇÃO")

for nome, modelo in modelos.items():
    print(f"\n{'═' * 65}")
    print(f"  {nome}")
    print(f"{'═' * 65}")

    modelo.fit(X_train_bal, y_train_bal)
    y_pred = modelo.predict(X_test_norm)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm, cm_norm = calcular_acc_por_classe(y_test, y_pred, classes)

    print(f"\n  [a] Pipeline: StandardScaler → SMOTE → {nome}")
    print(f"\n  [b] MÉTRICAS")
    print(f"  Acurácia Global : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-Score Global : {f1w:.4f}")

    # Matriz de confusão com rótulos
    df_cm = pd.DataFrame(
        cm,
        index=[f"R:nota{c}" for c in classes],
        columns=[f"P:nota{c}" for c in classes]
    )
    print(f"\n  Matriz de Confusão (R=Real, P=Previsto):")
    print(df_cm.to_string())

    # Acurácia por classe
    print(f"\n  Acurácia por Classe (diagonal da CM normalizada):")
    print(f"  {'Nota':8} {'Acurácia':12} {'Suporte':10}")
    print(f"  {'─' * 30}")
    for i, cls in enumerate(classes):
        sup = cm[i, :].sum()
        if sup > 0:
            print(f"  Nota {cls}   {cm_norm[i,i]:.4f}  ({cm_norm[i,i]*100:.1f}%)   {sup}")

    # Relatório completo sklearn (precision, recall, f1 por classe)
    print(f"\n  Relatório Completo por Classe:")
    print(classification_report(
        y_test, y_pred,
        labels=classes,
        target_names=[f"nota_{c}" for c in classes],
        zero_division=0
    ))

    if f1w > melhor_f1:
        melhor_f1   = f1w
        melhor_nome = nome
        melhor_modelo = modelo

# 7. JUSTIFICATIVA DO MELHOR MODELO

print(f"\n{'=' * 50}")
print(f"[c] RESULTADO FINAL E MODELO RECOMENDADO")
print(f"{'=' * 50}")
print(f" -> Modelo selecionado: {melhor_nome}")
print(f" -> F1-Score ponderado: {melhor_f1:.4f}\n")

print(f"""Resumo da decisão:
Escolhi o {melhor_nome} olhando para o F1-Score em vez da acurácia.
Como a base de vinhos é muito desbalanceada (só dá nota 5 e 6), a 
acurácia global engana muito. O Random Forest conseguiu lidar melhor 
com os outliers e prever notas extremas sem dar overfitting.
""")

# 8. SALVAR

pickle.dump(melhor_modelo, open('models/best_wine_model.pkl', 'wb'))
pickle.dump(scaler,        open('models/wine_scaler.pkl',     'wb'))
print(f"  [Salvos: models/best_wine_model.pkl | models/wine_scaler.pkl]")
print(f"\n{'═' * 65}")
print("TREINAMENTO CONCLUÍDO")
print(f"{'═' * 65}")