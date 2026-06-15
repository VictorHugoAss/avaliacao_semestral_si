"""
=============================================================
 QUESTÃO 3 — Black Friday Sales Dataset (Kaggle)
 Pipeline de Classificação: Produto, Pagamento, Faixa Etária
=============================================================
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

os.makedirs('models', exist_ok=True)

print("=" * 65)
print("QUESTÃO 3 — BLACK FRIDAY: TREINAMENTO")
print("=" * 65)

# 1. CARREGAMENTO

df = pd.read_csv('data/retail_black_friday_sales_100k.csv')
print(f"\n[1] Shape: {df.shape}")
print(f"Colunas: {list(df.columns)}")
print(f"\nNulos: {df.isnull().sum().sum()}")
print(f"\nDistribuição dos alvos:")
for alvo in ['product_category', 'payment_method', 'age_group']:
    print(f"\n  {alvo}:")
    print(df[alvo].value_counts().to_string())

# 2. FEATURES E ALVOS

FEATURES_CAT = ['gender', 'city', 'customer_segment']
FEATURES_NUM = [
    'purchase_amount', 'original_price', 'discount_pct',
    'final_price', 'quantity', 'purchase_hour',
    'is_weekend', 'is_black_friday'
]
FEATURES = FEATURES_CAT + FEATURES_NUM

X     = df[FEATURES]
y_cat = df['product_category']
y_pay = df['payment_method']
y_age = df['age_group']

# 3. DIVISÃO (mesmos índices para os 3 alvos)

X_train, X_test, \
y_tr_cat, y_te_cat, \
y_tr_pay, y_te_pay, \
y_tr_age, y_te_age = train_test_split(
    X, y_cat, y_pay, y_age,
    test_size=0.3, random_state=42
)
print(f"\n[2] Divisão: {len(X_train)} treino | {len(X_test)} teste")

# 4. PRÉ-PROCESSADOR

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), FEATURES_CAT),
    ('num', StandardScaler(), FEATURES_NUM)
])

# 5. MODELOS CANDIDATOS

modelos = {
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=15,
                                                   random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

# 6. FUNÇÕES DE AVALIAÇÃO

def calcular_sens_espec(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    resultados = []
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        espec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        resultados.append((cls, sens, espec, int(cm[i, :].sum())))
    return resultados, cm


def pipeline_alvo(nome_alvo, y_train, y_test, pkl):
    print(f"\n{'═' * 65}")
    print(f"  ALVO: {nome_alvo.upper()}")
    print(f"{'═' * 65}")

    classes = sorted(y_test.unique())

    # [a] Comparação dos modelos
    print(f"\n  [a] PIPELINE E COMPARAÇÃO DE MODELOS")
    print(f"  {'Modelo':25} {'Acurácia':10} {'F1 (w)':10}")
    print(f"  {'─' * 45}")

    melhor_f1, melhor_nome, melhor_pipe = -1, '', None

    for nome_m, clf in modelos.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        yp  = pipe.predict(X_test)
        acc = accuracy_score(y_test, yp)
        f1  = f1_score(y_test, yp, average='weighted', zero_division=0)
        print(f"  {nome_m:25} {acc:.4f}    {f1:.4f}")
        if f1 > melhor_f1:
            melhor_f1, melhor_nome, melhor_pipe = f1, nome_m, pipe

    print(f"\n  → CAMPEÃO: {melhor_nome}  (F1 weighted = {melhor_f1:.4f})")

    # [b] Métricas completas do campeão
    yp  = melhor_pipe.predict(X_test)
    acc = accuracy_score(y_test, yp)
    f1w = f1_score(y_test, yp, average='weighted', zero_division=0)
    metricas, cm = calcular_sens_espec(y_test, yp, classes)

    print(f"\n  [b] MÉTRICAS COMPLETAS — {melhor_nome}")
    print(f"  {'─' * 62}")
    print(f"  Acurácia Global : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  F1-Score Global : {f1w:.4f}")

    df_cm = pd.DataFrame(
        cm,
        index=[f"R:{c}" for c in classes],
        columns=[f"P:{c}" for c in classes]
    )
    print(f"\n  Matriz de Confusão (R=Real, P=Previsto):")
    print(df_cm.to_string())

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    print(f"\n  Acurácia por Classe:")
    for i, cls in enumerate(classes):
        if cm[i, :].sum() > 0:
            print(f"    {str(cls):25} {cm_norm[i, i]:.4f}  ({cm_norm[i, i] * 100:.1f}%)")

    print(f"\n  Sensibilidade e Especificidade por Classe:")
    print(f"  {'Classe':25} {'Sensib.':10} {'Especif.':10} {'Suporte':8}")
    print(f"  {'─' * 55}")
    for cls, sens, espec, sup in metricas:
        print(f"  {str(cls):25} {sens:.4f}    {espec:.4f}    {sup}")

    print(f"\n  Relatório por Classe (sklearn):")
    print(classification_report(
        y_test, yp, labels=classes,
        target_names=[str(c) for c in classes],
        zero_division=0
    ))

    pickle.dump(melhor_pipe, open(f'models/{pkl}', 'wb'))
    print(f"  [Salvo: models/{pkl}]")
    return melhor_pipe


# 7. EXECUÇÃO

pipeline_alvo("Categoria do Produto", y_tr_cat, y_te_cat, 'pipeline_cat.pkl')
pipeline_alvo("Forma de Pagamento",   y_tr_pay, y_te_pay, 'pipeline_pay.pkl')
pipeline_alvo("Faixa Etária",         y_tr_age, y_te_age, 'pipeline_age.pkl')

print(f"\n{'═' * 65}")
print("TREINAMENTO CONCLUÍDO — Pipelines salvos em models/")
print(f"{'═' * 65}")