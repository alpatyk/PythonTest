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

# Filtrando o DataFrame para o seu grupo, já feito anteriormente
df_group = df_filtrado  # Seu grupo filtrado (já criado)

# Calculando a volumetria por Safra
safra_volumetria = df_group.groupby(['safra', 'target']).size().unstack(fill_value=0)

# Calculando os percentuais
safra_volumetria_percentual = safra_volumetria.div(safra_volumetria.sum(axis=1), axis=0) * 100

# Adicionando uma linha de Total
safra_volumetria.loc['Total'] = safra_volumetria.sum(axis=0)
safra_volumetria_percentual.loc['Total'] = safra_volumetria_percentual.sum(axis=0)

# Exibindo a volumetria total (contagem e percentual)
print("Volumetria de Safra (Valores Absolutos):")
print(safra_volumetria)

print("\nVolumetria de Safra (Percentuais):")
print(safra_volumetria_percentual)

# Filtrando por região
regioes = df_group['regiao'].unique()

# Criando uma tabela de volumetria por região
regiao_volumetria = {}

for regiao in regioes:
    df_regiao = df_group[df_group['regiao'] == regiao]
    regiao_volumetria[regiao] = df_regiao.groupby(['safra', 'target']).size().unstack(fill_value=0)
    regiao_volumetria_percentual = regiao_volumetria[regiao].div(regiao_volumetria[regiao].sum(axis=1), axis=0) * 100
    regiao_volumetria[regiao].loc['Total'] = regiao_volumetria[regiao].sum(axis=0)
    regiao_volumetria_percentual.loc['Total'] = regiao_volumetria_percentual.sum(axis=0)

    print(f"\nVolumetria por Região: {regiao}")
    print("Contagem por Safra e Target:")
    print(regiao_volumetria[regiao])

    print("\nPercentuais por Safra e Target:")
    print(regiao_volumetria_percentual)

    # Filtrando por Período de Pesquisa
periodos = df_group['Periodo de Pesquisa'].unique()

# Criando uma tabela de volumetria por Período de Pesquisa
periodo_volumetria = {}

for periodo in periodos:
    df_periodo = df_group[df_group['Periodo de Pesquisa'] == periodo]
    periodo_volumetria[periodo] = df_periodo.groupby(['safra', 'target']).size().unstack(fill_value=0)
    periodo_volumetria_percentual = periodo_volumetria[periodo].div(periodo_volumetria[periodo].sum(axis=1), axis=0) * 100
    periodo_volumetria[periodo].loc['Total'] = periodo_volumetria[periodo].sum(axis=0)
    periodo_volumetria_percentual.loc['Total'] = periodo_volumetria_percentual.sum(axis=0)

    print(f"\nVolumetria por Período de Pesquisa: {periodo}")
    print("Contagem por Safra e Target:")
    print(periodo_volumetria[periodo])

    print("\nPercentuais por Safra e Target:")
    print(periodo_volumetria_percentual)

    # Filtrando por Período de Pesquisa e Região Centro-Oeste
periodos = df_group['Periodo de Pesquisa'].unique()

# Criando uma tabela de volumetria por Período de Pesquisa
periodo_volumetria = {}

for periodo in periodos:
    # Filtrando para o período e a região Centro-Oeste
    df_periodo = df_group[(df_group['Periodo de Pesquisa'] == periodo) & (df_group['regiao'] == 'Centro-Oeste')]

    # Verificando se há dados para o filtro
    if df_periodo.empty:
        continue

    periodo_volumetria[periodo] = df_periodo.groupby(['safra', 'target']).size().unstack(fill_value=0)

    # Calculando os percentuais
    periodo_volumetria_percentual = periodo_volumetria[periodo].div(periodo_volumetria[periodo].sum(axis=1), axis=0) * 100

    # Adicionando totais
    periodo_volumetria[periodo].loc['Total'] = periodo_volumetria[periodo].sum(axis=0)
    periodo_volumetria_percentual.loc['Total'] = periodo_volumetria_percentual.sum(axis=0)

    print(f"\nVolumetria por Período de Pesquisa: {periodo}")
    print("Contagem por Safra e Target:")
    print(periodo_volumetria[periodo])

    print("\nPercentuais por Safra e Target:")
    print(periodo_volumetria_percentual)

    # Verificando a coerência para a base total
total_base = df_group.groupby('target').size()

# Verificando a coerência para as safras
total_safra = df_group.groupby('safra')['target'].value_counts()

# Comparando as contagens
print("\nTotal Base (contagem):")
print(total_base)

print("\nTotal por Safra:")
print(total_safra)

