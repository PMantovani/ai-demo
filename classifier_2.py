from sklearn import svm

# E se tivéssemos mais características (features)?
# Simplesmente aumentaríamos a dimensionalidade do problema, mas é tão simples quanto adicionar mais um campo :)

#   -------- Característica #1
#   |  ----- Característica #2
#   |  |  -- Característica #3
#   |  |  |
# [ 0  0  0 ] <- amostra #1
# [ 1  1  1 ] <- amostra #2
X = [[0, 0, 0], [1, 1, 1]]

# Veja que o vetor y continua tendo dois elementos somente, porque continuamos tendo apenas duas amostras
y = [0, 1]

clf = svm.SVC()
clf.fit(X, y)

# As amostras para o predict também devem ter 3 elementos.
prediction = clf.predict([[2., 2., 2.]])

print(prediction)