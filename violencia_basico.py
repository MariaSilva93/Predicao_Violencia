"""
Created on Sat May  2 13:43:51 2026

@author: duda
"""

# =========================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =========================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# =========================================================================
# 2. APRESENTAÇÃO DO PROJETO
# =========================================================================
print("\n" + "*" * 80)
print("ANÁLISE EXPLORATÓRIA - VIOLÊNCIA CONTRA A MULHER NO BRASIL")
print("PREDIÇÃO DA OCORRÊNCIA DE VIOLÊNCIA FÍSICA")
print("*" * 80)

# =========================================================================
# 3. LEITURA DOS DADOS
# =========================================================================

print("LER DADOS DOS ARQUIVOS")

# caminho do CSV
csv_path = r'C:\Users\msilv\Downloads\ATV0007\SINAN-VIOL-2017-2019.csv'
df = pd.read_csv(
    csv_path,
    sep=',',  
    encoding='latin1',
    low_memory=False,
    nrows=1000
)
print("LEITURA EFETUADA COM SUCESSO!")

# =========================================================================
# 4. CÓPIA DE SEGURANÇA
# OBJETIVO: PRESERVAR OS DADOS ORIGINAIS
# =========================================================================
df_original = df.copy() #esse dataframe será utilizado como base de backup

input("\nDIGITE ALGO PARA CONTINUAR: ")
print(df_original.columns)

# =========================================================================
# 5. RENOMEANDO COLUNAS
# ========================================================================

# Renomeando as colunas level_0 e level_1 para UF e ANO
df_original.rename(columns={'level_0': 'UF', 'level_1': 'ANO'}, inplace=True)

#  Verificando o tipo de índice e colunas (verificação das colunas e dados do dataframe)
print("\n--- DF_ORIGINAL INFO ---")
df_original.info()

# BASE PARA GRÁFICOS E COMPARAÇÕES
df_graficos = df_original.copy() #esse dataframe será utilizado para a construção dos gráficos e análises exploratórias

# BASE PARA TRATAMENTO
df_tratado = df_original.copy() #esse detaframe será utilizado para limpeza dos dados e modelagem

# =========================================================================
# 6. SELEÇÃO DAS VARIÁVEIS
# =========================================================================

# Selecionando as colunas relevantes para analise 
colunas_analise = [
    'UF', 'ANO',
    'DT_NOTIFIC', 'DT_OCOR', 'DT_NASC',
    'NU_IDADE_N', 'CS_SEXO', 'CS_RACA',
    'CS_ESCOL_N', 'SIT_CONJUG', 'ID_OCUPA_N',
    'ORIENT_SEX', 'IDENT_GEN', 'CICL_VID',
    'LOCAL_OCOR', 'HORA_OCOR', 'OUT_VEZES',
    'VIOL_MOTIV', 'LES_AUTOP',
    'VIOL_FISIC', 'VIOL_PSICO', 'VIOL_TORT',
    'VIOL_SEXU', 'VIOL_FINAN', 'VIOL_NEGLI',
    'SEX_ASSEDI', 'SEX_ESTUPR', 'SEX_EXPLO',
    'SEX_PORNO', 'SEX_OUTRO',
    'REL_PAI', 'REL_MAE', 'REL_PAD', 'REL_MAD',
    'REL_CONJ', 'REL_EXCON', 'REL_NAMO',
    'REL_EXNAM', 'REL_FILHO', 'REL_IRMAO',
    'REL_CONHEC', 'REL_DESCO',
    'AUTOR_SEXO', 'AUTOR_ALCO',
    'LESAO_NAT', 'LESAO_CORP',
    'CLASSI_FIN', 'EVOLUCAO',
    'ATEND_MULH', 'ASSIST_SOC', 'DELEG_MULH'
]

# Criando novo DataFrame apenas com as variáveis selecionadas para análise
df_tratado = df_tratado.reindex(columns=colunas_analise).copy()
df_tratado.info()

print("\nSELEÇÃO CONCLUÍDA COM SUCESSO!")

# =========================================================================
# 7. FILTRAGEM DOS DADOS
# =========================================================================

# Filtrando os dados para focar em casos de violência contra mulheres (CS_SEXO = 'F') e excluindo casos de lesão autoprovocada (LES_AUTOP != 1)
df_tratado = df_tratado[df_tratado['CS_SEXO']=='F']
df_tratado = df_tratado[df_tratado['LES_AUTOP']!=1]
df_tratado.drop(
    ['CS_SEXO', 'LES_AUTOP', 'DT_NOTIFIC'],
    axis=1,
    inplace=True,
    errors='ignore'
)
df_tratado.dropna(subset=['DT_OCOR'], inplace = True)

df_tratado['UF'] = df_tratado['UF'].astype('category')
df_tratado['ANO'] = df_tratado['ANO'].astype('category')
df_tratado['REL_PAI'] = pd.to_numeric(df_tratado['REL_PAI'], errors='coerce')
df_tratado['DT_NASC'] = pd.to_datetime(
    df_tratado['DT_NASC'],
    errors='coerce'
)     
df_tratado['DT_OCOR'] = pd.to_datetime(
    df_tratado['DT_OCOR'],
    errors='coerce'
)
df_tratado.info()

# =========================================================================
# 8. DADOS FALTANTES
# =========================================================================

# Quantidade total de registros
total_registros = len(df_tratado)
print(f"\nTotal de registros: {total_registros}")

# Quantidade e percentual de valores ausentes por coluna
valores_ausentes = pd.DataFrame({
    'Quantidade': df_tratado.isnull().sum(),
    'Percentual (%)': round(
        (df_tratado.isnull().sum() / total_registros) * 100, 2)
})

# Ordenar do maior para o menor percentual
valores_ausentes = valores_ausentes.sort_values(
    by='Percentual (%)',
    ascending=False
)

print("\nVALORES AUSENTES POR COLUNA:")
print(valores_ausentes)

# Exibir somente colunas que possuem dados faltantes
colunas_com_nulos = valores_ausentes[
    valores_ausentes['Quantidade'] > 0
]

print("\nCOLUNAS COM DADOS FALTANTES:")
print(colunas_com_nulos)

# Quantidade total de valores ausentes no dataset
total_ausentes = df_tratado.isnull().sum().sum()
print(f"\nTotal geral de valores ausentes: {total_ausentes}")

# identificando na base podemos citar que a LESAO_CORP, LESAO_NAT, CLASSI_FIN, EVOLUCAO, ID_OCUPA_N e HORA_OCOR
# possui grande parte de valores nulos, por isso será necessario a remoção delas da base - além de que não servem muito para analise da predição

# =========================================================================
# 9. REMOÇÃO DE COLUNAS COM ALTO PERCENTUAL DE AUSÊNCIA
# =========================================================================

#neste trecho citamos as colunas que são essenciais para analise - assim não podem ser removidas, enquanto as colunas que apresentarem percentual cima de 40% de valores nulos fazemos a remoção

# Definir o limite de remoção (40% de valores nulos)
eliminar = 0.40

# Calcular o número de registros
total_registros = len(df_tratado)

# Calcular o percentual de valores nulos por coluna
percentual_nulos = df_tratado.isnull().sum() / total_registros

# Variável alvo (a ser mantida mesmo que tenha alto percentual de nulos)
variavel_alvo = 'VIOL_FISIC'

# Identificar colunas a serem removidas (excluindo a variável alvo)
colunas_remover = percentual_nulos[
    (percentual_nulos > eliminar) &
    (percentual_nulos.index != variavel_alvo)
].index.tolist()

print("\nColunas removidas (>40% nulos):")
for col in colunas_remover:
    print(f"- {col} ({percentual_nulos[col]:.2%})")

df_tratado.drop(columns=colunas_remover, inplace=True)

# =========================================================================
# 10. TRATAMENTO DE DATAS E CRIAÇÃO DE VARIÁVEL IDADE
# =========================================================================

print("\nCRIANDO VARIÁVEL IDADE A PARTIR DAS DATAS")

# Garantir formato datetime e lidar com erros de conversão (coerce para NaT)
df_tratado['DT_OCOR'] = pd.to_datetime(df_tratado['DT_OCOR'], errors='coerce')
df_tratado['DT_NASC'] = pd.to_datetime(df_tratado['DT_NASC'], errors='coerce')

# Criar variável idade calculando a diferença entre a data de ocorrência e a data de nascimento, convertendo para anos
df_tratado['IDADE'] = (
    (df_tratado['DT_OCOR'] - df_tratado['DT_NASC'])
    .dt.days // 365
)

# Remover registros com idade inválida
df_tratado = df_tratado[
    (df_tratado['IDADE'] >= 0) &
    (df_tratado['IDADE'] <= 110)
]

# Tratar valores ausentes da idade
df_tratado['IDADE'] = df_tratado['IDADE'].fillna(
    df_tratado['IDADE'].median()
)

# Remover datas
df_tratado.drop(['DT_OCOR', 'DT_NASC'], axis=1, inplace=True)

# =========================================================================
# LIMPEZA ANTES DA CONVERSÃO DE VARIÁVEIS CATEGÓRICAS
# =========================================================================

print("\nREALIZANDO LIMPEZA DE TIPOS CATEGÓRICOS")

colunas_categoricas = [
    'CS_RACA',
    'CS_ESCOL_N',
    'SIT_CONJUG',
    'ORIENT_SEX',
    'LOCAL_OCOR'
]

for col in colunas_categoricas:
    if col in df_tratado.columns:
        df_tratado[col] = (
            df_tratado[col]
            .fillna('Ignorado')
            .astype(str)
            .str.strip()
        )

print("Limpeza das variáveis categóricas concluída com sucesso!")
# =========================================================================
# CONVERSÃO DE VARIÁVEIS CATEGÓRICAS - UF E ANO

print("\nCONVERTENDO VARIÁVEIS CATEGÓRICAS")
colunas_categoricas = [
    'UF',
    'ANO'
]

colunas_existentes = [
    col for col in colunas_categoricas
    if col in df_tratado.columns
]

print("UF e ANO convertidas com sucesso!")

# =========================================================================
# CONVERSÃO DE BOOLEANOS PARA INTEIRO
# =========================================================================

print("\nCONVERTENDO VARIÁVEIS BOOLEANAS PARA INTEIRO")
bool_cols = df_tratado.select_dtypes(include='bool').columns
df_tratado[bool_cols] = df_tratado[bool_cols].astype(int)
print("Colunas booleanas convertidas para int com sucesso!")

# =========================================================================
# VERIFICAÇÃO FINAL
# =========================================================================

print("\nTRANSFORMAÇÃO DE VARIÁVEIS CATEGÓRICAS/BOOLEANAS CONCLUÍDA COM SUCESSO.")

print("\nDimensão final:")
print(df_tratado.shape)

print("\nTipos de dados:")
print(df_tratado.dtypes)


# =========================================================================
# 11. TRATAMENTO DO ALVO
# =========================================================================

print("\nTRATANDO VARIÁVEL ALVO (VIOL_FISIC)")

df_modelo = df_tratado.copy()

# converter para numérico
df_modelo['VIOL_FISIC'] = pd.to_numeric(df_modelo['VIOL_FISIC'], errors='coerce')

# MAPEAMENTO CORRETO DO SINAN:
# 1 = SIM
# 2 = NÃO
# 9 = IGNORADO

df_modelo = df_modelo[df_modelo['VIOL_FISIC'].isin([1, 2])]

# transformar em binário real (OBRIGATÓRIO PARA ML)
df_modelo['VIOL_FISIC'] = df_modelo['VIOL_FISIC'].map({1: 1, 2: 0})

df_modelo = df_modelo.dropna(subset=['VIOL_FISIC'])

df_modelo['VIOL_FISIC'] = df_modelo['VIOL_FISIC'].astype(int)

print("\nDISTRIBUIÇÃO DO ALVO ANTES DO SPLIT:")
print(df_modelo['VIOL_FISIC'].value_counts())

# =========================================================================
# 12. SEPARAÇÃO X E Y
# =========================================================================

print("\nSEPARANDO VARIÁVEL ALVO")

y = df_modelo['VIOL_FISIC'].copy()
X = df_modelo.drop(columns=['VIOL_FISIC']).copy()

print("\nX:")
print(X.head())

print("\nY:")
print(y.head())


# =========================================================================
# 13. CONVERSÃO PARA NUMÉRICO
# =========================================================================

# manter UF separada apenas para análise/gráficos
uf_graficos = df_graficos['UF'].copy()

# remover UF do dataset de modelagem
X = X.drop(columns=['UF'], errors='ignore')

# transformar variáveis categóricas em numéricas
X = pd.get_dummies(X, drop_first=True)

# garantir tudo numérico (segurança final)
X = X.fillna(0)


# =========================================================================
# 14. IMPUTAÇÃO DE VALORES AUSENTES (CORREÇÃO DEFINITIVA)
# =========================================================================

imputer = SimpleImputer(strategy='median')

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# alinhamento seguro
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print("\nChecagem de dados faltantes:")
print("NaN em X:", X.isna().sum().sum())
print("NaN em y:", y.isna().sum().sum())

print("\nDimensão final X:", X.shape)
print("Dimensão final y:", y.shape)

# =========================================================================
# 15. ESTRUTURA DO DATASET
# =========================================================================

print("\nESTRUTURA DO DATASET")

print("\nDIMENSÃO:")
print(df_tratado.shape)

print("\nCOLUNAS:")
print(df_tratado.columns)

print("\nTIPOS DE DADOS:")
print(df_tratado.dtypes)

print("\nPRIMEIROS REGISTROS:")
print(df_tratado.head())

print("\nESTATÍSTICA DESCRITIVA:")
print(df_tratado.describe(include='all'))
# =========================================================================
# 16. ANÁLISE ESTATÍSTICA
# =========================================================================

print("\nESTATÍSTICA DESCRITIVA")
print(X.describe().T)

print("\nDISTRIBUIÇÃO DO ALVO")
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)


# =========================================================================
# 17. ANÁLISE EXPLORATÓRIA
# =========================================================================

# OUTLIERS (IDADE)
if 'IDADE' in X.columns:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=X[['IDADE']])
    plt.title("Detecção de Outliers - IDADE")
    plt.show()

# CORRELAÇÃO
df_corr = X.copy()
df_corr['VIOL_FISIC'] = y

corr = df_corr.corr(numeric_only=True)

top_corr = corr['VIOL_FISIC'].abs().sort_values(ascending=False).head(15).index

plt.figure(figsize=(12,8))

sns.heatmap(
    corr.loc[top_corr, top_corr],
    annot=True,
    cmap='coolwarm'
)

plt.title("Top Correlações")
plt.tight_layout()
plt.show()

if df_modelo['VIOL_FISIC'].nunique() < 2:
    raise ValueError("O dataset ficou com apenas uma classe no alvo. Revise filtros.")

print(df_modelo['VIOL_FISIC'].value_counts(normalize=True) * 100)

# =========================================================================
# 18. SPLIT TREINO / TESTE
# =========================================================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# encoding 
X_total = pd.concat([X_treino, X_teste], axis=0)
X_total = pd.get_dummies(X_total, drop_first=True)

# alinhamento obrigatório
X_treino = X_total.iloc[:len(X_treino), :]
X_teste = X_total.iloc[len(X_treino):, :]

X_treino = X_treino.reset_index(drop=True)
X_teste = X_teste.reset_index(drop=True)
y_treino = y_treino.reset_index(drop=True)
y_teste = y_teste.reset_index(drop=True)

print("\nSplit concluído")
print("Treino:", X_treino.shape)
print("Teste:", X_teste.shape)

print("\nDistribuição treino:")
print(y_treino.value_counts(normalize=True) * 100)

print("\nDistribuição teste:")
print(y_teste.value_counts(normalize=True) * 100)

# =========================================================================
# ÁRVORE DE DECISÃO
# =========================================================================

print("\nTREINANDO ÁRVORE DE DECISÃO")

modelo_tree = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    random_state=42
)

modelo_tree.fit(X_treino, y_treino)

# ==========================================================
# ÁRVORE SIMPLIFICADA 
# ==========================================================

modelo_tree_artigo = DecisionTreeClassifier(
    max_depth=3,
    criterion='entropy',
    random_state=42
)

modelo_tree_artigo.fit(X_treino, y_treino)

from sklearn.tree import plot_tree

plt.figure(figsize=(18,10))

plot_tree(
    modelo_tree_artigo,
    feature_names=X_treino.columns,
    class_names=['Não Física', 'Física'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Árvore de Decisão - Versão Simplificada")

plt.tight_layout()

plt.savefig(
    "Arvore_Decisao_Artigo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



y_pred_tree = modelo_tree.predict(X_teste)



print("\n==============================")
print("ÁRVORE DE DECISÃO")
print("==============================")

print("Acurácia:", accuracy_score(y_teste, y_pred_tree))
print(confusion_matrix(y_teste, y_pred_tree))
print(classification_report(y_teste, y_pred_tree))

# =========================================================================
# MATRIZ DE CONFUSÃO - ÁRVORE DE DECISÃO
# =========================================================================

plt.figure(figsize=(6,4))

sns.heatmap(
    confusion_matrix(y_teste, y_pred_tree),
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=['Não Física', 'Física'],
    yticklabels=['Não Física', 'Física']
)

plt.title("Matriz de Confusão - Árvore de Decisão")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")

plt.tight_layout()

plt.savefig(
    "Matriz_Confusao_Arvore_Decisao.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =========================================================================
# NORMALIZAÇÃO
# =========================================================================

scaler = StandardScaler()

X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# =========================================================================
# 19. MODELAGEM
# =========================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# REGRESSÃO LOGÍSTICA

modelo_lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
    )

modelo_lr.fit(X_treino_scaled, y_treino)

y_pred_lr = modelo_lr.predict(X_teste_scaled)

# RANDOM FOREST

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

modelo_rf.fit(X_treino, y_treino)

y_pred_rf = modelo_rf.predict(X_teste)


# =========================================================================
# 20. AVALIAÇÃO DOS MODELOS
# =========================================================================

print("\n==============================")
print("REGRESSÃO LOGÍSTICA")
print("==============================")

print("Acurácia:", accuracy_score(y_teste, y_pred_lr))
print(confusion_matrix(y_teste, y_pred_lr))
print(classification_report(y_teste, y_pred_lr))

# =========================================================================
# MATRIZ DE CONFUSÃO - REGRESSÃO LOGÍSTICA
# =========================================================================

plt.figure(figsize=(6,4))

sns.heatmap(
    confusion_matrix(y_teste, y_pred_lr),
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Não Física', 'Física'],
    yticklabels=['Não Física', 'Física']
)

plt.title("Matriz de Confusão - Regressão Logística")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")

plt.tight_layout()

plt.savefig(
    "Matriz_Confusao_Regressao_Logistica.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print("Acurácia:", accuracy_score(y_teste, y_pred_rf))
print(confusion_matrix(y_teste, y_pred_rf))
print(classification_report(
    y_teste,
    y_pred_rf,
    target_names=['Não Física', 'Física']
))

# =========================================================================
# MATRIZ DE CONFUSÃO VISUAL - RANDOM FOREST
# =========================================================================

plt.figure(figsize=(6,4))

sns.heatmap(
    confusion_matrix(y_teste, y_pred_rf),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Não Física', 'Física'],
    yticklabels=['Não Física', 'Física']
)

plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# =========================================================================
# CURVA ROC - RANDOM FOREST
# =========================================================================

print("\nGERANDO CURVA ROC")

# probabilidades da classe positiva
y_prob_rf = modelo_rf.predict_proba(X_teste)[:,1]

# cálculo ROC
fpr, tpr, thresholds = roc_curve(y_teste, y_prob_rf)

# cálculo AUC
roc_auc = roc_auc_score(y_teste, y_prob_rf)

print(f"ROC AUC: {roc_auc:.4f}")

# gráfico
plt.figure(figsize=(8,5))

plt.plot(
    fpr,
    tpr,
    label=f'Random Forest (AUC = {roc_auc:.4f})'
)

# linha diagonal
plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.title("Curva ROC - Random Forest")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.legend()

plt.tight_layout()
plt.show()

# =========================================================================
# 21. IMPORTÂNCIA DAS VARIÁVEIS
# =========================================================================

importances = pd.DataFrame({
   'variavel': X_treino.columns,
    'importancia': modelo_rf.feature_importances_
}).sort_values(by='importancia', ascending=False)

print("\nTOP 10 VARIÁVEIS MAIS IMPORTANTES")
print(importances.head(10))
plt.figure(figsize=(10,6))

sns.barplot(
    data=importances.head(10).sort_values(by='importancia'),
    x='importancia',
    y='variavel'
)

plt.title("Top 10 Variáveis Mais Importantes")
plt.tight_layout()
plt.show()


# =========================================================================
# 22. COMPARAÇÃO FINAL DOS MODELOS
# =========================================================================

print("\n==============================")
print("COMPARAÇÃO FINAL")
print("==============================")

print(f"Árvore de Decisão:  {accuracy_score(y_teste, y_pred_tree):.4f}")
print(f"Regressão Logística:{accuracy_score(y_teste, y_pred_lr):.4f}")
print(f"Random Forest:      {accuracy_score(y_teste, y_pred_rf):.4f}")

# =========================================================================
# 23. PREDIÇÕES
# =========================================================================

print("\n==============================")
print("PREDIÇÃO INTERATIVA")
print("==============================")

while True:

    try:

        idade = int(input("\nDigite a idade da vítima: "))

        autor_alco = int(input(
            "O agressor utilizou álcool? (1=Sim / 0=Não): "
        ))

        viol_psico = int(input(
            "Houve violência psicológica? (1=Sim / 0=Não): "
        ))

        rel_conj = int(input(
            "O agressor é cônjuge/companheiro? (1=Sim / 0=Não): "
        ))

        local_ocor = input(
            "Local da ocorrência (RESIDENCIA / VIA_PUBLICA): "
        ).upper()

        # dataframe do exemplo
        exemplo = pd.DataFrame({
            'IDADE': [idade],
            'AUTOR_ALCO': [autor_alco],
            'VIOL_PSICO': [viol_psico],
            'REL_CONJ': [rel_conj],
            'LOCAL_OCOR': [local_ocor]
        })

        # aplicar get_dummies
        exemplo = pd.get_dummies(exemplo)

        # alinhar colunas com treino
        exemplo = exemplo.reindex(
            columns=X_treino.columns,
            fill_value=0
        )

        # previsão
        resultado = modelo_rf.predict(exemplo)

        # probabilidade
        probabilidade = modelo_rf.predict_proba(exemplo)

        print("\nRESULTADO DA PREVISÃO")

        if resultado[0] == 1:
            print("BAIXA" \
            " OCORRÊNCIA DE VIOLÊNCIA FÍSICA")
        else:
            print("ALTA PROBABILIDADE DE VIOLÊNCIA FÍSICA")

        print(
            f"Probabilidade estimada: "
            f"{probabilidade[0][1] * 100:.2f}%"
        )

        continuar = input(
            "\nDeseja realizar outra previsão? (S/N): "
        ).upper()

        if continuar == 'N':
            break

    except Exception as erro:
        print("\nERRO AO REALIZAR PREVISÃO")
        print(erro)
