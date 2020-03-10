from sklearn import svm
from sklearn import preprocessing 

# Como você pode ver, o pre-processamento dos dados faz parte do cotidiano da análise de dados...
# E não para por aí... 
# É recomendado que as características numéricas possuam sempre um formato parecido com uma gaussiana de média 0 e variância 1.

#   ------------- Cada característica (individualmente) deveria ter média 0 e variância 1
#   |   -------/
#   |   |   --/
#   |   |  |
# [ -2  0  2 ]
# [ -1  0  1 ]
# [ -1  1  1 ]
X = [[-2, 0, 2], [-1, 0, 1], [-1, 1, 1]]

# o StandardScaler é um dos vários pré-processadores no scikit... Este transforma os dados de entrada do sistema para média 0 e variância 1.
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))