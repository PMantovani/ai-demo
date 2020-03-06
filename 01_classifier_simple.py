# Para importar esta biblioteca, primeiro é necessário installar o sklearn. Execute o seguinte comando:
# pip install -U scikit-learn
from sklearn import svm

# A matriz X tem o formato de [n_amostras, n_caracteristicas]
# No caso abaixo, a matriz fica assim:

#   ------ Característica #1
#   |  --- Característica #2
#   |  |
# [ 0  0 ] <- amostra #1
# [ 1  1 ] <- amostra #2
X = [[0, 0], [1, 1]]

# O vetor Y contém as classificações para cada amostra diferente.
# Desta forma, a amostra #1 é classificada como '0', e a amostra #2 é classificada como '1'.
y = [0, 1]

# Iniciando o nosso classificador com as opções desejadas 
# (para mais informações: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
clf = svm.SVC()

# Fit é o comando que realiza o treinamento do nosso classificador, dadas as amostras, características e classificações.
clf.fit(X, y)

# Predict é o método que tenta prever a classificação de uma nova amostra, baseado no classificador treinado anteriormente.
prediction = clf.predict([[2., 2.]])

# Veja que como a nova amostra possui características mais próximas da amostra #2 do que da #1, o classificador previu que ela é da classe 1. Yey! :)
print(prediction)

# E se fosse mais próximo da amostra #1?
prediction = clf.predict([[0.3, 0.3]])
# Agora, como esperado, ele previu como '0'!
print(prediction)