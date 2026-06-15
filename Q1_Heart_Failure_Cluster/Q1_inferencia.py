"""
QUESTÃO 1 — Heart Failure: Módulo de Inferência
"""
import pickle
import numpy as np
import pandas as pd


def inferir_grupo_paciente(dados_paciente: dict):
    print("\n=== INFERÊNCIA — HEART FAILURE ===")

    scaler  = pickle.load(open('models/scaler_heart.pkl',  'rb'))
    kmeans  = pickle.load(open('models/kmeans_heart.pkl',  'rb'))
    colunas = pickle.load(open('models/colunas_heart.pkl', 'rb'))

    df = pd.DataFrame([dados_paciente])[colunas]

    print("Dados brutos do paciente:")
    print(df.to_string(index=False))

    # Normalizar com o scaler do treino
    df_norm = pd.DataFrame(scaler.transform(df), columns=colunas)

    # Predizer cluster
    grupo = kmeans.predict(df_norm)[0]

    # Distância do paciente a cada centroide
    centroides = kmeans.cluster_centers_
    distancias = np.linalg.norm(centroides - df_norm.values, axis=1)

    print(f"\n[ RESULTADO ]")
    print(f"  Grupo identificado: CLUSTER {grupo}")
    print(f"\n  Distância do paciente a cada cluster (menor = mais similar):")
    for i, d in enumerate(distancias):
        marcador = " ← GRUPO DO PACIENTE" if i == grupo else ""
        print(f"    Cluster {i}: {d:.4f}{marcador}")

    return grupo


if __name__ == '__main__':
    # Paciente de ALTO RISCO (fração de ejeção baixa, creatinina alta)
    print("\n--- Paciente 1: Perfil de alto risco ---")
    paciente_joao = {
        'age':                        75.0,
        'anaemia':                     1,
        'creatinine_phosphokinase':  582,
        'diabetes':                    0,
        'ejection_fraction':          20,
        'high_blood_pressure':         1,
        'platelets':               265000.0,
        'serum_creatinine':           1.9,
        'serum_sodium':              130,
        'sex':                         1,
        'smoking':                     0,
    }
    inferir_grupo_paciente(paciente_joao)

    # Paciente de BAIXO RISCO
    print("\n--- Paciente 2: Perfil de baixo risco ---")
    paciente_maria = {
        'age':                        45.0,
        'anaemia':                     0,
        'creatinine_phosphokinase':  200,
        'diabetes':                    0,
        'ejection_fraction':          60,
        'high_blood_pressure':         0,
        'platelets':               300000.0,
        'serum_creatinine':           0.9,
        'serum_sodium':              138,
        'sex':                         0,
        'smoking':                     0,
    }
    inferir_grupo_paciente(paciente_maria)