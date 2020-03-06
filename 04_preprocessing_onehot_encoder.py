from sklearn import svm
from sklearn import preprocessing 

# Já que as classes podem ser strings, as características também podem, certo? Afinal, sempre temos características em texto...

#   ------------- Característica #1: Cor
#   |       ----- Característica #2: Altura
#   |       |
#   |       |
# [ azul  alto  ] <- amostra #1 (avatar)
# [ azul  medio ] <- amostra #2 (blue man group)
# [ azul  baixo ] <- amostra #3 (smurf)
X = [['azul', 'alto'], ['azul', 'medio'], ['azul', 'baixo']]

# Simplesmente podemos mudar as classificações para uma string qualquer
y = ['avatar', 'blue man group', 'smurf']

clf = svm.SVC()
clf.fit(X, y)

# ERRO!!!! Não podemos usar strings nas características. Por quê?
# Porque o classificador nada mais é do que uma função matemática. Como ele poderia usar uma string em uma função matemática?

# Para resolver este problema podemos usar uma técnica chamada de One Hot Encoding. (quem já cursou eletrônica digital provavelmente já ouviu esse termo).
# É um encoding que transforma uma coluna em 'N', uma para cada valor diferente. O valor é 1 somente na coluna em que há o 'match', e 0 em todas as outras.
# Para mais info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

encoder = preprocessing.OneHotEncoder()
# O comando fit do encoder analisa sua matriz de características para codificar de acordo com o One Hot Encoding.
encoder.fit(X)

# O transform transformará de fato uma matriz de acordo com o encoding.
encoded_array = encoder.transform(X).toarray()
print(encoded_array)

#   ----------- O '1' da primeira coluna representa a cor 'azul'
#   |
#   |     ------- As três colunas a seguir representam a altura: a primeira representa 'alto', a segunda 'baixo', a terceira 'medio'
#   |   / | \
#   |  /  |  \
# [ 1  1  0  0 ]
# [ 1  0  0  1 ]
# [ 1  0  1  0 ]

clf = svm.SVC()
clf.fit(encoded_array, y)

result = clf.predict(encoder.transform([['azul', 'baixo']]).toarray())

# Smurf has been detected successfully! :)
print(result)