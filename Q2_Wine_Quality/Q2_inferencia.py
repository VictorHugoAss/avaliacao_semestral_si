"""
QUESTÃO 2 — Wine Quality: Módulo de Inferência
"""
import pickle
import numpy as np
import pandas as pd


def inferir_qualidade_vinho(dados_vinho: dict):
    print("\n=== MÓDULO DE INFERÊNCIA: WINE QUALITY ===")

    scaler = pickle.load(open('models/wine_scaler.pkl',     'rb'))
    modelo = pickle.load(open('models/best_wine_model.pkl', 'rb'))

    df_novo = pd.DataFrame([dados_vinho])
    print(f"\nAtributos do vinho:")
    print(df_novo.to_string(index=False))

    # Normalizar com o scaler do treino
    vinho_norm = scaler.transform(df_novo)

    # Predição e probabilidades
    qualidade = modelo.predict(vinho_norm)[0]
    probas    = modelo.predict_proba(vinho_norm)[0]
    classes   = modelo.classes_

    print(f"\n[ RESULTADO ]")
    print(f"  Qualidade prevista: Nota {qualidade}")
    print(f"\n  Distribuição de probabilidade por nota:")
    for c, p in sorted(zip(classes, probas), key=lambda x: -x[1]):
        barra = '█' * int(p * 40)
        print(f"    Nota {c}:  {p:.3f}  {barra}")

    return qualidade


if __name__ == '__main__':
    # Vinho tinto — perfil típico de nota 5
    vinho_tinto = {
        'fixed acidity':        7.4,
        'volatile acidity':     0.70,
        'citric acid':          0.00,
        'residual sugar':       1.9,
        'chlorides':            0.076,
        'free sulfur dioxide':  11.0,
        'total sulfur dioxide': 34.0,
        'density':              0.9978,
        'pH':                   3.51,
        'sulphates':            0.56,
        'alcohol':              9.4,
        'is_red':               1,   # 1 = tinto, 0 = branco
    }
    inferir_qualidade_vinho(vinho_tinto)

    # Vinho branco — perfil de alta qualidade
    print()
    vinho_branco = {
        'fixed acidity':        6.8,
        'volatile acidity':     0.26,
        'citric acid':          0.42,
        'residual sugar':       1.2,
        'chlorides':            0.034,
        'free sulfur dioxide':  22.0,
        'total sulfur dioxide': 90.0,
        'density':              0.9890,
        'pH':                   3.10,
        'sulphates':            0.61,
        'alcohol':              12.8,
        'is_red':               0,
    }
    inferir_qualidade_vinho(vinho_branco)