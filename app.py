import numpy as np
import pandas as pd
import streamlit as tf_streamlit  # Usando o alias padrão
import tensorflow as tf
from tensorflow.keras import layers, models


# Configuração da Página
tf_streamlit.set_page_config(
    page_title="Pet Feliz | Previsão de Bem-Estar", page_icon="🐾", layout="centered"
)

# Estilização CSS personalizada para um visual limpo e profissional (Estilo Empresarial)
tf_streamlit.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2b580c;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3b7a12;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Título e Introdução
tf_streamlit.title("🐾 Sistema Pet Feliz")
tf_streamlit.markdown(
    """
### Descubra o nível de felicidade do seu cachorro com base nos passeios diários!
*Esta ferramenta utiliza Inteligência Artificial (Rede Neural TensorFlow) para analisar a rotina e estimar o bem-estar do seu pet.*
---
"""
)

# 1. Preparação dos Dados (Conforme solicitado)
pets = pd.DataFrame(
    {"passeios": [1, 2, 3, 4, 5], "felicidade": [2, 4, 5, 8, 10]}
)

X = pets[["passeios"]].values
y = pets["felicidade"].values


# 2. Construção e Treinamento do Modelo TensorFlow (em cache para otimizar)
@tf_streamlit.cache_resource
def treinar_modelo():
  # Criando uma rede neural simples (Regressão Linear via Keras)
  modelo = models.Sequential([layers.Input(shape=(1,)), layers.Dense(units=1)])

  modelo.compile(optimizer="sgd", loss="mean_squared_error")

  # Treinando o modelo com os dados fornecidos (épocas ajustadas para aprendizado rápido)
  modelo.fit(X, y, epochs=500, verbose=0)
  return modelo


modelo_ia = treinar_modelo()

# 3. Interface de Entrada de Dados
tf_streamlit.subheader("📊 Insira a Rotina do seu Pet")

col1, col2 = tf_streamlit.columns([2, 1])

with col1:
  passeios_usuario = tf_streamlit.slider(
      "Quantos passeios diários o seu cachorro faz?",
      min_value=0,
      max_value=10,
      value=2,
      step=1,
      help="Arraste para selecionar a quantidade média de passeios por dia.",
  )

with col2:
  tf_streamlit.markdown("<br>", unsafe_allow_html=True)
  botao_prever = tf_streamlit.button("Calcular Felicidade")

# 4. Processamento e Exibição do Resultado
if botao_prever:
  # Realizando a predição com o TensorFlow
  entrada_usuario = np.array([[passeios_usuario]], dtype=float)
  predicao = modelo_ia.predict(entrada_usuario)

  # Limitando a nota de felicidade entre 0 e 10 para coerência didática
  nota_felicidade = float(np.clip(predicao[0][0], 0, 10))

  tf_streamlit.markdown("---")
  tf_streamlit.subheader("🎯 Resultado da Análise de IA")

  # Exibindo métrica em destaque
  col_res1, col_res2 = tf_streamlit.columns(2)

  with col_res1:
    tf_streamlit.metric(
        label="Nível Estimado de Felicidade (de 0 a 10)",
        value=f"{nota_felicidade:.1f} 🐾",
    )

  with col_res2:
    if nota_felicidade >= 8:
      feedback = "Excelente! Seu pet está com uma rotina altamente estimulada e feliz."
      emoji = "🐶🎉"
    elif nota_felicidade >= 5:
      feedback = "Bom! Mas adicionar mais um passeio pode elevar o bem-estar dele."
      emoji = "🐕✨"
    else:
      feedback = (
          "Atenção: O pet pode precisar de mais passeios para gastar energia e"
          " ficar alegre."
      )
      emoji = "⚠️🐾"

    tf_streamlit.markdown(f"### {emoji}")
    tf_streamlit.write(feedback)

  # Informação educativa sobre como a IA funciona
  with tf_streamlit.expander("💡 Como a Inteligência Artificial calculou isso?"):
    tf_streamlit.write(
        f"""
        O modelo TensorFlow analisou os padrões históricos onde **{passeios_usuario} passeios** 
        foram correlacionados matematicamente através de pesos sinápticos ajustados por aprendizado de máquina. 
        Quanto mais passeios na rotina, maior a pontuação de felicidade prevista pelo sistema!
        """
    )

# Rodapé profissional
tf_streamlit.markdown("---")
tf_streamlit.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.85rem;'>Sistema"
    " Pet Feliz • Desenvolvido com Streamlit e TensorFlow</div>",
    unsafe_allow_html=True,
)