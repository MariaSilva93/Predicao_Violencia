# -*- coding: utf-8 -*-

# =========================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =========================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================
# 2. APRESENTAÇÃO DO PROJETO
# =========================================================================

print("*" * 80)
print("ANÁLISE EXPLORATÓRIA - VIOLÊNCIA CONTRA A MULHER NO BRASIL")
print("*" * 80)

# =========================================================================
# 3. LEITURA DOS DADOS
# =========================================================================

csv_path = r'C:\Users\msilv\Downloads\ATV0007\SINAN-VIOL-2017-2019.csv'

df = pd.read_csv(
    csv_path,
    sep=',',
    encoding='latin1',
    low_memory=False
)

print("LEITURA REALIZADA COM SUCESSO!")

# =========================================================================
# 4. LEITURA DOS DADOS 
# =========================================================================

df = df.rename(columns={
    'level_0': 'UF',
    'level_1': 'ANO'
})

# Somente mulheres
df = df[df['CS_SEXO'] == 'F']

# Remover lesão autoprovocada (se existir)
if 'LES_AUTOP' in df.columns:
    df = df[df['LES_AUTOP'] != 1]

#copia para evitar fragmentação
df = df.copy()

# =========================================================================
# 5. FILTRAGEM DOS DADOS
# =========================================================================

df['DT_OCOR'] = pd.to_datetime(df['DT_OCOR'], errors='coerce')
df = df[df['DT_OCOR'].notna()]

# Filtrar período correto
if 'ANO' in df.columns:
    df['ANO'] = pd.to_numeric(df['ANO'], errors='coerce')
df = df[df['ANO'].between(2017, 2019)]

print("UF:")
print(df['UF'].value_counts(dropna=False).head())

# =========================================================================
# 6. TRATAMENTO DAS DATAS E CRIAÇÃO DA IDADE
# =========================================================================

df['DT_OCOR'] = pd.to_datetime(
    df['DT_OCOR'],
    errors='coerce'
)

df['DT_NASC'] = pd.to_datetime(
    df['DT_NASC'],
    errors='coerce'
)

# Criando variável idade
df['IDADE'] = (
    (df['DT_OCOR'] - df['DT_NASC'])
    .dt.days // 365
)

# Removendo idades inválidas
df = df[
    (df['IDADE'] >= 0) &
    (df['IDADE'] <= 110)
]

# =========================================================================
# 7. CRIAÇÃO DE FAIXA ETÁRIA
# =========================================================================

bins = [0, 12, 17, 29, 39, 59, 110]

labels = [
    'Criança',
    'Adolescente',
    'Jovem',
    'Adulta',
    'Meia Idade',
    'Idosa'
]

df['FAIXA_ETARIA'] = pd.cut(
    df['IDADE'],
    bins=bins,
    labels=labels
)

# =========================================================================
# 8. MAPEAMENTO DE CATEGORIAS
# =========================================================================

# =========================================================================
# LOCAL DE OCORRÊNCIA
# =========================================================================

map_local = {
    1: 'Residência',
    2: 'Habitação Coletiva',
    3: 'Escola',
    4: 'Local Esportivo',
    5: 'Bar',
    6: 'Via Pública',
    7: 'Comércio/Serviços',
    8: 'Indústria',
    9: 'Construção',
    99: 'Ignorado'
}

df['LOCAL_OCOR_NOME'] = df['LOCAL_OCOR'].map(map_local)

# =========================================================================
# SEXO DO AUTOR
# =========================================================================

map_autor = {
    1: 'Masculino',
    2: 'Feminino',
    3: 'Ambos',
    9: 'Ignorado'
}

df['AUTOR_SEXO_NOME'] = df['AUTOR_SEXO'].map(map_autor)

# =========================================================================
# RAÇA
# =========================================================================

map_raca = {
    1: 'Branca',
    2: 'Preta',
    3: 'Amarela',
    4: 'Parda',
    5: 'Indígena',
    9: 'Ignorado'
}

df['CS_RACA_NOME'] = df['CS_RACA'].map(map_raca)

# =========================================================================
# ESCOLARIDADE
# =========================================================================

map_escolaridade = {
    1: 'Analfabeto',
    2: '1ª a 4ª incompleta',
    3: '4ª série completa',
    4: '5ª a 8ª incompleta',
    5: 'Fundamental completo',
    6: 'Médio incompleto',
    7: 'Médio completo',
    8: 'Superior incompleto',
    9: 'Superior completo',
    10: 'Ignorado'
}

df['ESCOLARIDADE_NOME'] = df['CS_ESCOL_N'].map(
    map_escolaridade
)

# =========================================================================
# 9. GRÁFICOS EXPLORATÓRIOS
# =========================================================================

# =========================================================================
# TOTAL DE CASOS POR ANO
# =========================================================================

plt.figure(figsize=(10,5))

df.groupby('ANO').size().plot(kind='bar')

plt.title('Total de Casos por Ano')
plt.xlabel('Ano')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# TOTAL DE CASOS POR ESTADO
# =========================================================================

plt.figure(figsize=(12,5))

df.groupby('UF').size() \
    .sort_values(ascending=False) \
    .plot(kind='bar')

plt.title('Total de Casos por Estado')
plt.xlabel('Estado')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# ESTADOS COM MENOR NÚMERO DE CASOS
# =========================================================================

uf_counts = df.groupby('UF').size().sort_values()

plt.figure(figsize=(10,5))

uf_counts.head(10).plot(kind='bar')

plt.title('Estados com Menor Número de Casos')
plt.xlabel('Estado')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# ESTADOS COM MAIOR NÚMERO DE CASOS
# =========================================================================

plt.figure(figsize=(10,5))

uf_counts.tail(10).plot(kind='bar')

plt.title('Estados com Maior Número de Casos')
plt.xlabel('Estado')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# LOCAL DE OCORRÊNCIA
# =========================================================================

plt.figure(figsize=(12,5))

df['LOCAL_OCOR_NOME'] \
    .value_counts() \
    .head(10) \
    .plot(kind='bar')

plt.title('Locais com Maior Ocorrência de Violência')
plt.xlabel('Local')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# TIPOS DE VIOLÊNCIA
# =========================================================================

viol_cols = [
    'VIOL_FISIC',
    'VIOL_PSICO',
    'VIOL_TORT',
    'VIOL_SEXU',
    'VIOL_FINAN',
    'VIOL_NEGLI'
]

violencia_totais = {}

for col in viol_cols:
    violencia_totais[col] = (df[col] == 1).sum()

nomes_violencia = {
    'VIOL_FISIC': 'Física',
    'VIOL_PSICO': 'Psicológica',
    'VIOL_TORT': 'Tortura',
    'VIOL_SEXU': 'Sexual',
    'VIOL_FINAN': 'Financeira',
    'VIOL_NEGLI': 'Negligência'
}

serie_violencia = pd.Series(violencia_totais)

serie_violencia.index = serie_violencia.index.map(
    nomes_violencia
)

plt.figure(figsize=(10,5))

serie_violencia \
    .sort_values(ascending=False) \
    .plot(kind='bar')

plt.title('Tipos de Violência Mais Frequentes')
plt.xlabel('Tipo de Violência')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# SEXO DOS AUTORES DA VIOLÊNCIA
# =========================================================================

plt.figure(figsize=(7,7))

df['AUTOR_SEXO_NOME'] \
    .value_counts() \
    .plot(
        kind='pie',
        autopct='%1.1f%%'
    )

plt.title('Sexo dos Autores da Violência')
plt.ylabel('')

plt.tight_layout()
plt.show()

# =========================================================================
# DISTRIBUIÇÃO DA IDADE DAS VÍTIMAS
# =========================================================================

plt.figure(figsize=(10,5))

df['IDADE'].plot(
    kind='hist',
    bins=20
)

plt.title('Distribuição da Idade das Vítimas')
plt.xlabel('Idade')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# DETECÇÃO DE OUTLIERS - IDADE
# =========================================================================

plt.figure(figsize=(8,4))

sns.boxplot(x=df['IDADE'])

plt.title('Detecção de Outliers - Idade das Vítimas')
plt.xlabel('Idade')

plt.tight_layout()
plt.show()

# =========================================================================
# VIOLÊNCIA POR FAIXA ETÁRIA
# =========================================================================

plt.figure(figsize=(10,5))

df['FAIXA_ETARIA'] \
    .value_counts() \
    .sort_index() \
    .plot(kind='bar')

plt.title('Violência por Faixa Etária')
plt.xlabel('Faixa Etária')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# ESCOLARIDADE DAS VÍTIMAS
# =========================================================================

plt.figure(figsize=(12,5))

df['ESCOLARIDADE_NOME'] \
    .value_counts() \
    .head(10) \
    .plot(kind='bar')

plt.title('Distribuição das Vítimas por Escolaridade')
plt.xlabel('Escolaridade')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# DISTRIBUIÇÃO DAS VÍTIMAS POR RAÇA
# =========================================================================

plt.figure(figsize=(10,5))

df['CS_RACA_NOME'] \
    .value_counts() \
    .plot(kind='bar')

plt.title('Distribuição das Vítimas por Raça')
plt.xlabel('Raça')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# DISTRIBUIÇÃO DA VARIÁVEL ALVO
# =========================================================================

distribuicao = df['VIOL_FISIC'] \
    .replace({
        1: 'Física',
        2: 'Não Física',
        9: 'Ignorado'
    }) \
    .value_counts()

plt.figure(figsize=(6,5))

distribuicao.plot(kind='bar')

plt.title('Distribuição da Variável Alvo - Violência Física')
plt.xlabel('Classe')
plt.ylabel('Quantidade')

plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# =========================================================================
# MATRIZ DE CORRELAÇÃO
# =========================================================================

correlacao = df[
    [
        'IDADE',
        'VIOL_FISIC',
        'VIOL_PSICO',
        'VIOL_TORT',
        'VIOL_SEXU',
        'VIOL_FINAN',
        'VIOL_NEGLI'
    ]
].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    correlacao,
    annot=True,
    cmap='coolwarm',
    center=0
)

plt.title('Matriz de Correlação')

plt.tight_layout()
plt.show()

# =========================================================================
# TOP CORRELAÇÕES COM VIOLÊNCIA FÍSICA
# =========================================================================

top_corr = correlacao['VIOL_FISIC'] \
    .drop('VIOL_FISIC') \
    .sort_values()

plt.figure(figsize=(8,5))

top_corr.plot(kind='barh')

plt.title('Correlação com Violência Física')
plt.xlabel('Correlação')

plt.tight_layout()
plt.show()
