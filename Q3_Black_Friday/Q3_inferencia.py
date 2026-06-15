import pickle
import numpy as np
import pandas as pd

def inferir_venda(dados_venda: dict):
    print("\n=== MÓDULO DE INFERÊNCIA: BLACK FRIDAY ===")
    
    # 1. Transformar o dicionário de entrada em um DataFrame de uma linha
    df_novo = pd.DataFrame([dados_venda])
    
    # 2. Carregar os pipelines completos
    pipeline_cat = pickle.load(open('models/pipeline_cat.pkl', 'rb'))
    pipeline_pay = pickle.load(open('models/pipeline_pay.pkl', 'rb'))
    pipeline_age = pickle.load(open('models/pipeline_age.pkl', 'rb'))
    
    modelos = {
        'Categoria de Produto (product_category)': pipeline_cat,
        'Forma de Pagamento (payment_method)': pipeline_pay,
        'Faixa Etária (age_group)': pipeline_age
    }
    
    print("\nResultados da Inferência:")
    # 3. Fazer a inferência e extrair o grau de certeza
    for alvo, modelo in modelos.items():
        # A previsão da classe
        classe_predita = modelo.predict(df_novo)
        
        # O Grau de Certeza
        probabilidades = modelo.predict_proba(df_novo)
        grau_certeza = np.max(probabilidades) * 100
        
        print(f" -> {alvo}: {classe_predita} (Grau de Certeza: {grau_certeza:.2f}%)")

if __name__ == '__main__':
    # Exemplo de entrada
    venda = {
        'gender':           'Male',
        'city':             'San Francisco',
        'customer_segment': 'Loyal',
        'purchase_amount':  350.00,
        'original_price':   500.00,
        'discount_pct':     30,
        'final_price':      350.00,
        'quantity':         1,
        'purchase_hour':    14,
        'is_weekend':       1,
        'is_black_friday':  0,
    }
    inferir_venda(venda)