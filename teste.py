import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
!pip install scikit-learn

from google.colab import files
upload = files.upload()

df = pd.read_excel(io.BytesIO(upload['Lista NPS Positivo_V4.xlsx']))

# Agrupar pela coluna 'target' e contar
contagem_target = df['target'].value_counts()

# Exibir a contagem
print("Contagem por target:")
print(contagem_target)

# Exibir os totais esperados
total_promotores = contagem_target.get('promotor', 0)
total_neutros = contagem_target.get('neutro', 0)
total_detratores = contagem_target.get('detrator', 0)

print(f"\nPromotores: {total_promotores}")
print(f"Neutros: {total_neutros}")
print(f"Detratores: {total_detratores}")

df = df.loc[df["mercado"] == "BRASIL"]

grupos_filtrados = ["Grupo 9", "Grupo 10"]

df_filtrado = df.loc[df["Grupo de Produto"].isin(grupos_filtrados)]

# Calculando a contagem de cada classe ('promotor', 'neutro', 'detrator') no grupo filtrado
target_counts_group = df_filtrado['target'].value_counts()

# Calculando o percentual de cada classe no grupo filtrado
target_percentages_group = df_filtrado['target'].value_counts(normalize=True) * 100

# Exibindo os resultados
print("Contagem de cada classe no grupo:")
print(target_counts_group)

print("\nPercentual de cada classe no grupo:")
print(target_percentages_group)

