import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Neste exercício, vamos criar um classificador para prever a sobrevivência de passageiros do Titanic.
# Para isto, iremos ler o arquivo 'titanic.csv', que possuem os dados dos passageiros com algumas features
# que podemos utilizar no classificador.

# Dados lidos de: https://www.openml.org/d/40945
all_data = pd.read_csv('data/titanic.csv', sep=';')

# O arquivo possui a seguinte sequência de colunas:
# 00: survived  : Numérico  : 0 se faleceu, 1 se sobreviveu
# 01: pclass    : Numérico  : Classe no navio: 1, 2 ou 3
# 02: name      : Texto     : Nome do passageiro
# 03: sex       : Texto     : 'female' ou 'male'
# 04: age       : Numérico  : Idade do passageiro
# 05: sibsp     : Numérico  : Número de irmãos/esposos abordo
# 06: parch     : Numérico  : Número de pais/filhos abordo
# 07: ticket    : Texto     : Número do ticket
# 08: fare      : Numérico  : Preço da passagem
# 09: cabin     : Texto     : Número da cabine
# 10: embarked  : Texto     : Local de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)
# 11: boat      : Texto     : Número do bote de resgate
# 12: body      : Numérico  : Número do corpo resgatado
# 13: home.dest : Texto     : Cidade natal

classes = all_data[['survived']]
features = all_data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# O Pipeline nos ajuda a criar sequências de comandos que devem ser executados ao passar pelo classificador.
# O SimpleImputer é um transformador que preenche os dados de colunas faltantes. Como os dados do Titanic não são completos,
# esse passo é necessário para preencher as lacunas. Com a estratégia 'mean', o transformador completará com a média dos valores.
numeric_features_idx = ['pclass', 'age', 'sibsp', 'parch', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# A estratégia 'constant' no Imputer, colocará o valor 'missing' nas entradas categóricas em que não houver dados.
categorical_features_idx = ['sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# O ColumnTransformer especifica quais são os preprocessadores que irão atuar sobre qual coluna.
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numeric_transformer, numeric_features_idx),
        ('categorical', categorical_transformer, categorical_features_idx)])

# Colocando o classificador no final do Pipeline completa nosso fluxo
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', svm.SVC())])

# Aqui, dividimos os nossos dados de teste e treino
features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.2)

clf.fit(features_train, classes_train)
predictions = clf.predict(features_test)

precision = precision_score(classes_test, predictions)
recall = recall_score(classes_test, predictions)
f1_score = f1_score(classes_test, predictions)
print(confusion_matrix(classes_test, predictions))
print('Precisão: ' + str(precision))
print('Recall: ' + str(recall))
print('F1 Score: ' + str(f1_score))