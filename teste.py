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

# Verificando as classes majoritária e minoritária para cada safra
for safra in safra_volumetria.index[:-1]:  # Excluindo a linha 'Total'
    print(f"\nAnalisando a Safra {safra}:")
    safra_data = safra_volumetria.loc[safra]
    majoritaria = safra_data.idxmax()  # Classe majoritária
    minoritaria = safra_data.idxmin()  # Classe minoritária
    print(f"Classe Majoritária: {majoritaria}")
    print(f"Classe Minoritária: {minoritaria}")

    # Verificando se alguma safra apresenta volumetria muito diferente das outras
    print(f"Distribuição para a Safra {safra}:")
    print(safra_data)

    # Definir as classificações
df['Classificacao'] = df['nota'].apply(lambda x: 'Promotor' if x >= 9 else ('Detrator' if x <= 6 else 'Neutro'))

# Filtrar apenas Promotores e Detratores
df_filtrado = df[df['Classificacao'].isin(['Promotor', 'Detrator'])]

# Agrupar por safra, região e classificação
resultado = (
    df_filtrado.groupby(['safra', 'regiao', 'Classificacao'])
    .size()
    .reset_index(name='Contagem')
)

# Encontrar a região com mais promotores e detratores por safra
promotores = resultado[resultado['Classificacao'] == 'Promotor'].sort_values(['safra', 'Contagem'], ascending=[True, False]).groupby('safra').first().reset_index()
detratores = resultado[resultado['Classificacao'] == 'Detrator'].sort_values(['safra', 'Contagem'], ascending=[True, False]).groupby('safra').first().reset_index()

# Resultados
print("Regiões com mais promotores por safra:")
print(promotores[['safra', 'regiao', 'Contagem']])

print("\nRegiões com mais detratores por safra:")
print(detratores[['safra', 'regiao', 'Contagem']])

# Lista das perguntas que você precisa utilizar
perguntas_necessarias = [
    'capacidade operacional (hectares por hora) (csat)',
    'adequação as diversas operações e implementos (csat)',
    'facilidade de operação (csat)',
    'conforto e ergonomia (csat)',
    'disponibilidade e confiabilidade mecânica  (csat)',
    'facilidade para realização de manutenções (csat)',
    'custo de manutenção (csat)',
    'consumo de combustível (litros por hectares) (csat)',
    'adaptabilidade as mais diversas condições de trabalho (csat)',
    'facilidade de uso do piloto automático (csat)',
    'geração e transmissão de dados para gestão da frota (csat)',
    'geração e transmissão de dados para gestão agrícola (csat)'
]

# Filtrando as perguntas no seu DataFrame
df_perguntas_filtrado = df[perguntas_necessarias]

# Exibindo o DataFrame filtrado
print(df_perguntas_filtrado.head())

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Função para calcular a correlação de Spearman e formatar as cores
def calcular_correlacao(df, perguntas_necessarias, grupo):
    # Filtrar as perguntas necessárias e a coluna 'nota'
    df_grupo = df[df["Grupo de Produto"].isin(grupo)]

    # Calcular a correlação de Spearman entre 'nota' e as perguntas
    correlacoes = df_grupo[perguntas_necessarias + ['nota']].corr(method='spearman')['nota'].drop('nota')

    # Ordenar as correlações da maior para a menor
    correlacoes = correlacoes.sort_values(ascending=False)

    # Criar um DataFrame com a correlação para aplicar o estilo
    correlacoes_df = pd.DataFrame(correlacoes).reset_index()
    correlacoes_df.columns = ['Pergunta', 'Correlação']

    # Colorir as correlações conforme sua intensidade
    def colorir_correlacao(val):
        if val >= 0.7:
            return 'background-color: red'  # Correlação forte
        elif 0.4 <= val < 0.7:
            return 'background-color: blue'  # Correlação média
        else:
            return 'background-color: green'  # Correlação fraca

    # Aplicar a formatação de cores nas correlações usando Styler.map
    correlacoes_style = correlacoes_df.style.map(lambda val: colorir_correlacao(val), subset=['Correlação'])

    return correlacoes_style

# Definindo as perguntas necessárias para o seu grupo
perguntas_necessarias = [
    'capacidade operacional (hectares por hora) (csat)',
    'adequação as diversas operações e implementos (csat)',
    'facilidade de operação (csat)',
    'conforto e ergonomia (csat)',
    'disponibilidade e confiabilidade mecânica  (csat)',
    'facilidade para realização de manutenções (csat)',
    'custo de manutenção (csat)',
    'consumo de combustível (litros por hectares) (csat)',
    'adaptabilidade as mais diversas condições de trabalho (csat)',
    'facilidade de uso do piloto automático (csat)',
    'geração e transmissão de dados para gestão da frota (csat)',
    'geração e transmissão de dados para gestão agrícola (csat)'
]

# Exemplo de grupo e de filtragem de dados
grupo = ["Grupo 9", "Grupo 10"]

# Calculando a correlação para o seu grupo inteiro
correlacao_grupo = calcular_correlacao(df, perguntas_necessarias, grupo)

# Exibindo a correlação do grupo
correlacao_grupo

# Caso precise, é possível salvar como um arquivo HTML para visualização interativa
# correlacao_grupo.to_html('correlacao_grupo.html')

# Para realizar o cálculo por região e por período de pesquisa, você pode replicar o processo
# Para cada região:
for regiao in df['regiao'].unique():
    correlacao_regiao = calcular_correlacao(df[df['regiao'] == regiao], perguntas_necessarias, grupo)
    print(f"Correlação para a região {regiao}:")
    display(correlacao_regiao)

# Para cada período de pesquisa:
for periodo in df['Periodo de Pesquisa'].unique():
    correlacao_periodo = calcular_correlacao(df[df['Periodo de Pesquisa'] == periodo], perguntas_necessarias, grupo)
    print(f"Correlação para o período de pesquisa {periodo}:")
    display(correlacao_periodo)

# Para calcular por safra, também podemos fazer da mesma forma:
for safra in df['safra'].unique():
    correlacao_safra = calcular_correlacao(df[df['safra'] == safra], perguntas_necessarias, grupo)
    print(f"Correlação para a safra {safra}:")
    display(correlacao_safra)

# Função para criar o target binário para detratores e neutros
def criar_target_binario(df, tipo_target):
    """
    tipo_target: 'detrator' ou 'neutro'
    """
    if tipo_target == 'detrator':
        # Detratores: 1 para detrator, 0 para neutro ou promotor
        return df['target'].apply(lambda x: 1 if x == 'detrator' else 0)
    elif tipo_target == 'neutro':
        # Neutros: 1 para neutro, 0 para promotor ou detrator
        return df['target'].apply(lambda x: 1 if x == 'neutro' else 0)

# Passo 2: Criar as variáveis de entrada (X) e os alvos (y) para os modelos
# Excluindo a coluna 'nota' e a coluna 'target' de 3 classes
perguntas = [
    'capacidade operacional (hectares por hora) (csat)',
    'adequação as diversas operações e implementos (csat)',
    'facilidade de operação (csat)',
    'conforto e ergonomia (csat)',
    'disponibilidade e confiabilidade mecânica  (csat)',
    'facilidade para realização de manutenções (csat)',
    'custo de manutenção (csat)',
    'consumo de combustível (litros por hectares) (csat)',
    'adaptabilidade as mais diversas condições de trabalho (csat)',
    'facilidade de uso do piloto automático (csat)',
    'geração e transmissão de dados para gestão da frota (csat)',
    'geração e transmissão de dados para gestão agrícola (csat)'
]

# Passo 3: Filtrar as variáveis de entrada (X) e as variáveis alvo (y)
X = df[perguntas]
y_detrator = criar_target_binario(df, 'detrator')  # Target binário para detratores
y_neutro = criar_target_binario(df, 'neutro')  # Target binário para neutros

# Passo 4: Divisão de dados em treino e teste
X_train, X_test, y_detrator_train, y_detrator_test = train_test_split(X, y_detrator, test_size=0.2, random_state=42)
X_train, X_test, y_neutro_train, y_neutro_test = train_test_split(X, y_neutro, test_size=0.2, random_state=42)

# Passo 5: Treinamento do Modelo de Detratores
modelo_detrator = RandomForestClassifier(random_state=42)
modelo_detrator.fit(X_train, y_detrator_train)

# Passo 6: Avaliação do Modelo de Detratores
y_detrator_pred = modelo_detrator.predict(X_test)
print("Relatório de Classificação para Detratores:")
print(classification_report(y_detrator_test, y_detrator_pred))

# Passo 7: Treinamento do Modelo de Neutros
modelo_neutro = RandomForestClassifier(random_state=42)
modelo_neutro.fit(X_train, y_neutro_train)

# Passo 8: Avaliação do Modelo de Neutros
y_neutro_pred = modelo_neutro.predict(X_test)
print("Relatório de Classificação para Neutros:")
print(classification_report(y_neutro_test, y_neutro_pred))

# Passo 1: Filtrar o DataFrame para a região Centro-Oeste
df_centro_oeste = df[df['regiao'] == 'Centro-Oeste']  # Substitua 'região' pelo nome da sua coluna de região

# Passo 2: Criar a função para o target binário (sem modificações)
def criar_target_binario(df, tipo_target):
    """
    tipo_target: 'detrator' ou 'neutro'
    """
    if tipo_target == 'detrator':
        # Detratores: 1 para detrator, 0 para neutro ou promotor
        return df['target'].apply(lambda x: 1 if x == 'detrator' else 0)
    elif tipo_target == 'neutro':
        # Neutros: 1 para neutro, 0 para promotor ou detrator
        return df['target'].apply(lambda x: 1 if x == 'neutro' else 0)

# Passo 3: Filtrar as variáveis de entrada (X) e as variáveis alvo (y)
perguntas = [
    'capacidade operacional (hectares por hora) (csat)',
    'adequação as diversas operações e implementos (csat)',
    'facilidade de operação (csat)',
    'conforto e ergonomia (csat)',
    'disponibilidade e confiabilidade mecânica  (csat)',
    'facilidade para realização de manutenções (csat)',
    'custo de manutenção (csat)',
    'consumo de combustível (litros por hectares) (csat)',
    'adaptabilidade as mais diversas condições de trabalho (csat)',
    'facilidade de uso do piloto automático (csat)',
    'geração e transmissão de dados para gestão da frota (csat)',
    'geração e transmissão de dados para gestão agrícola (csat)'
]

# Passo 4: Filtrar as variáveis de entrada (X) e as variáveis alvo (y) apenas para a região Centro-Oeste
X_centro_oeste = df_centro_oeste[perguntas]
y_detrator_centro_oeste = criar_target_binario(df_centro_oeste, 'detrator')  # Target binário para detratores
y_neutro_centro_oeste = criar_target_binario(df_centro_oeste, 'neutro')  # Target binário para neutros

# Passo 5: Divisão de dados em treino e teste
X_train, X_test, y_detrator_train, y_detrator_test = train_test_split(X_centro_oeste, y_detrator_centro_oeste, test_size=0.2, random_state=42)
X_train, X_test, y_neutro_train, y_neutro_test = train_test_split(X_centro_oeste, y_neutro_centro_oeste, test_size=0.2, random_state=42)

# Passo 6: Treinamento do Modelo de Detratores
modelo_detrator = RandomForestClassifier(random_state=42)
modelo_detrator.fit(X_train, y_detrator_train)

# Passo 7: Avaliação do Modelo de Detratores
y_detrator_pred = modelo_detrator.predict(X_test)
print("Relatório de Classificação para Detratores (Centro-Oeste):")
print(classification_report(y_detrator_test, y_detrator_pred))

# Passo 8: Treinamento do Modelo de Neutros
modelo_neutro = RandomForestClassifier(random_state=42)
modelo_neutro.fit(X_train, y_neutro_train)

# Passo 9: Avaliação do Modelo de Neutros
y_neutro_pred = modelo_neutro.predict(X_test)
print("Relatório de Classificação para Neutros (Centro-Oeste):")
print(classification_report(y_neutro_test, y_neutro_pred))

# Função para criar o target binário
def criar_target_binario(df, tipo_target):
    if tipo_target == 'detrator':
        return df['target'].apply(lambda x: 1 if x == 'detrator' else 0)
    elif tipo_target == 'neutro':
        return df['target'].apply(lambda x: 1 if x == 'neutro' else 0)
    else:
        raise ValueError(f"Tipo de target inválido: {tipo_target}. Use 'detrator' ou 'neutro'.")

# Função para treinar e avaliar um modelo para um dado filtro
def treinar_e_avaliar_modelo(df, perguntas, filtro_coluna, filtro_valor):
    # Filtrar os dados com base no filtro
    df_filtrado = df[df[filtro_coluna] == filtro_valor]

    # Verificar se o DataFrame está vazio
    if df_filtrado.empty:
        print(f"Warning: DataFrame is empty for filter {filtro_coluna}: {filtro_valor}. Skipping model training.")
        return

    # Criação do target binário
    y_detrator = criar_target_binario(df_filtrado, 'detrator')
    y_neutro = criar_target_binario(df_filtrado, 'neutro')

    # Verificar valores nulos nas variáveis de entrada
    X = df_filtrado[perguntas].dropna()
    y_detrator = y_detrator.loc[X.index]
    y_neutro = y_neutro.loc[X.index]

    # Divisão dos dados em treino e teste (mesma divisão para ambos os modelos)
    X_train, X_test, y_train_detrator, y_test_detrator = train_test_split(X, y_detrator, test_size=0.2, random_state=42)
    _, _, y_train_neutro, y_test_neutro = train_test_split(X, y_neutro, test_size=0.2, random_state=42)

    # Modelo para detratores
    modelo_detrator = RandomForestClassifier(random_state=42)
    modelo_detrator.fit(X_train, y_train_detrator)
    y_pred_detrator = modelo_detrator.predict(X_test)
    print(f"\nRelatório de Classificação - Detratores - Filtro {filtro_coluna}: {filtro_valor}")
    print(classification_report(y_test_detrator, y_pred_detrator))

    # Modelo para neutros
    modelo_neutro = RandomForestClassifier(random_state=42)
    modelo_neutro.fit(X_train, y_train_neutro)
    y_pred_neutro = modelo_neutro.predict(X_test)
    print(f"\nRelatório de Classificação - Neutros - Filtro {filtro_coluna}: {filtro_valor}")
    print(classification_report(y_test_neutro, y_pred_neutro))

# Definindo as colunas de filtro e perguntas
filtros = [
    ('Grupo de Produto', 'Grupo 9'),
    ('Grupo de Produto', 'Grupo 10'),
    ('regiao', 'Nordeste'),
    ('regiao', 'Norte'),
    ('regiao', 'Centro-Oeste'),
    ('regiao', 'Sudeste'),
    ('regiao', 'Sul'),
    ('Periodo de Pesquisa', '3 a 6 M'),
    ('Periodo de Pesquisa', '6 a 12 M'),
    ('Periodo de Pesquisa', '12 a 18 M'),
    ('Periodo de Pesquisa', '18 a 30 M')
]

# Executar o treinamento e avaliação para cada filtro
for filtro_coluna, filtro_valor in filtros:
    treinar_e_avaliar_modelo(df, perguntas, filtro_coluna, filtro_valor)

# Função para treinar e avaliar um modelo para um dado filtro
def treinar_e_avaliar_modelo(df, perguntas, filtro_coluna, filtro_valor):
    # Filtrar os dados com base no filtro
    df_filtrado = df[(df['regiao'] == 'Centro-Oeste') & (df[filtro_coluna] == filtro_valor)]

    # Verificar se o DataFrame está vazio
    if df_filtrado.empty:
        print(f"Warning: DataFrame is empty for filter {filtro_coluna}: {filtro_valor}. Skipping model training.")
        return

    # Criação do target binário
    y_detrator = criar_target_binario(df_filtrado, 'detrator')
    y_neutro = criar_target_binario(df_filtrado, 'neutro')

    # Verificar valores nulos nas variáveis de entrada
    X = df_filtrado[perguntas].dropna()
    y_detrator = y_detrator.loc[X.index]
    y_neutro = y_neutro.loc[X.index]

    # Divisão dos dados em treino e teste (mesma divisão para ambos os modelos)
    X_train, X_test, y_train_detrator, y_test_detrator = train_test_split(X, y_detrator, test_size=0.2, random_state=42)
    _, _, y_train_neutro, y_test_neutro = train_test_split(X, y_neutro, test_size=0.2, random_state=42)

    # Modelo para detratores
    modelo_detrator = RandomForestClassifier(random_state=42)
    modelo_detrator.fit(X_train, y_train_detrator)
    y_pred_detrator = modelo_detrator.predict(X_test)
    print(f"\nRelatório de Classificação - Detratores - Filtro {filtro_coluna}: {filtro_valor}")
    print(classification_report(y_test_detrator, y_pred_detrator))

    # Modelo para neutros
    modelo_neutro = RandomForestClassifier(random_state=42)
    modelo_neutro.fit(X_train, y_train_neutro)
    y_pred_neutro = modelo_neutro.predict(X_test)
    print(f"\nRelatório de Classificação - Neutros - Filtro {filtro_coluna}: {filtro_valor}")
    print(classification_report(y_test_neutro, y_pred_neutro))

# Definindo os filtros para 'Periodo de Pesquisa' (somente para 'Centro-Oeste')
periodos_de_pesquisa = [
    ('Periodo de Pesquisa', '3 a 6 M'),
    ('Periodo de Pesquisa', '6 a 12 M'),
    ('Periodo de Pesquisa', '12 a 18 M'),
    ('Periodo de Pesquisa', '18 a 30 M')
]

# Executar o treinamento e avaliação para cada filtro de período de pesquisa (região já é filtrada)
for filtro_coluna, filtro_valor in periodos_de_pesquisa:
    treinar_e_avaliar_modelo(df, perguntas, filtro_coluna, filtro_valor)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Função para criar o target binário
def criar_target_binario(df, tipo_target):
    if tipo_target == 'detrator':
        return df['target'].apply(lambda x: 1 if x.lower() == 'detrator' else 0)
    elif tipo_target == 'neutro':
        return df['target'].apply(lambda x: 1 if x.lower() == 'neutro' else 0)
    else:
        raise ValueError(f"Tipo de target inválido: {tipo_target}. Use 'detrator' ou 'neutro'.")

# Função para treinar o modelo e exibir as top variáveis
def treinar_e_exibir_variaveis(df, perguntas, filtro_coluna, filtro_valor, modelo_tipo='RandomForest'):
    # Filtrar os dados com base no filtro
    df_filtrado = df[df[filtro_coluna] == filtro_valor]

    # Verificar se o DataFrame está vazio
    if df_filtrado.empty:
        print(f"Warning: DataFrame is empty for filter {filtro_coluna}: {filtro_valor}. Skipping model training.")
        return

    # Criação do target binário
    y_detrator = criar_target_binario(df_filtrado, 'detrator')
    y_neutro = criar_target_binario(df_filtrado, 'neutro')

    # Verificar valores nulos nas variáveis de entrada
    X = df_filtrado[perguntas].dropna()
    y_detrator = y_detrator.loc[X.index]
    y_neutro = y_neutro.loc[X.index]

    # Divisão dos dados
    X_train, X_test, y_train_detrator, y_test_detrator = train_test_split(X, y_detrator, test_size=0.2, random_state=42)
    _, _, y_train_neutro, y_test_neutro = train_test_split(X, y_neutro, test_size=0.2, random_state=42)

    for target_name, y_train, y_test in zip(['Detrator', 'Neutro'], [y_train_detrator, y_train_neutro], [y_test_detrator, y_test_neutro]):
        # Seleção do modelo
        if modelo_tipo == 'RandomForest':
            modelo = RandomForestClassifier(random_state=42)
        elif modelo_tipo == 'XGBoost':
            modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        else:
            raise ValueError(f"Modelo inválido: {modelo_tipo}. Use 'RandomForest' ou 'XGBoost'.")

        # Treinamento do modelo
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Relatório de Classificação
        print(f"\nRelatório de Classificação - {target_name} - Filtro {filtro_coluna}: {filtro_valor}")
        print(classification_report(y_test, y_pred))

        # Extração das top 10 variáveis
        if modelo_tipo == 'RandomForest':
            importancias = modelo.feature_importances_
            top_variaveis = pd.Series(importancias, index=perguntas).nlargest(10)
        elif modelo_tipo == 'XGBoost':
            importancias = modelo.get_booster().get_score(importance_type='weight')
            top_variaveis = pd.Series(importancias).nlargest(10)

        # Exibir as top variáveis
        print(f"\nTop 10 variáveis mais importantes para {target_name} - Filtro {filtro_coluna}: {filtro_valor}")
        print(top_variaveis)

# Definindo filtros e perguntas
filtros = [
    ('Grupo de Produto', 'Grupo 9'),
    ('Grupo de Produto', 'Grupo 10'),
    ('regiao', 'Nordeste'),
    ('regiao', 'Norte'),
    ('regiao', 'Centro-Oeste'),
    ('regiao', 'Sudeste'),
    ('regiao', 'Sul'),
    ('Periodo de Pesquisa', '3 a 6 M'),
    ('Periodo de Pesquisa', '6 a 12 M'),
    ('Periodo de Pesquisa', '12 a 18 M'),
    ('Periodo de Pesquisa', '18 a 30 M'),
]

# Treinar e avaliar modelos
for filtro_coluna, filtro_valor in filtros:
    treinar_e_exibir_variaveis(df, perguntas, filtro_coluna, filtro_valor, modelo_tipo='RandomForest')

