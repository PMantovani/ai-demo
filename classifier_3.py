from sklearn import svm

# Classificar as coisas como 0 e 1 é meio chato né?

X = [[0, 0], [1, 1]]

# Simplesmente podemos mudar as classificações para uma string qualquer
y = ['banana', 'melancia']

clf = svm.SVC()
clf.fit(X, y)

# As amostras para o predict também devem ter 3 elementos.
prediction = clf.predict([[2., 2.]])

print(prediction)