import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Vamos repetir o exemplo anterior, mas ao invés de usarmos uma SVM, trabalharemos com uma Rede Neural...
# ... ou neural network...
# ... ou ainda deep learning... pra vc parecer cool :)
# A diferença entre uma 'rede neural' e 'deep learning' é somente na complexidade das camadas de neurônios que a rede vai possuir
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Dividindo nossos dados em treino e teste. A primeira metade serão os dados de treino, a segunda os dados de teste
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
predicted = mlp.predict(X_test)

# Printando o resultado do nosso classificador 
print("Classification report for classifier %s:\n%s\n"
      % (mlp, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()

#####################################################################################################################################################

# Wow! Parece que alguma coisa deu errado! Mas o que seria?
# É por esses motivos que inteligência artificial pode ser muito complicada às vezes.
# Essas bibliotecas facilitam muito nossa tarefa, mas também precisamos entender um pouco de como as coisas funcionam por baixo do pano.
# ...
# Nesse caso, o nosso problema é que utilizamos o solucionador para otimizar os pesos de cada neurônio 'Adam'. Este solver não é muito adequado para
# datasets pequenos. Então, vamos tentar com o solver 'lbfgs', que é mais adequada para este cenário.

#####################################################################################################################################################

# mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
#                     solver='lbfgs', verbose=10, random_state=1,
#                     learning_rate_init=.1)

# mlp.fit(X_train, y_train)
# predicted = mlp.predict(X_test)

# # Printando o resultado do nosso classificador 
# print("Classification report for classifier %s:\n%s\n"
#       % (mlp, metrics.classification_report(y_test, predicted)))
# disp = metrics.plot_confusion_matrix(mlp, X_test, y_test)
# disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)

# plt.show()

#####################################################################################################################################################

# Muito melhor né? :)
# Aqui, infelizmente não há espaço para explicar o porquê um é melhor que o outro. 
# Por isso que esse mundo de IA e Análise de Dados pode ser tão fácil e difícil ao mesmo tempo.
#
# Aqui estamos explorando apenas a ponta do iceberg desse mundo novo. Tudo isso que estamos vendo aqui são implementações
# já prontas dos algoritmos de inteligência artificial, que possuem inúmeros parâmetros internos e detalhes de cada algoritmo.
#
# É um campo de estudos muito vasto, que fornece possibilidades muito interessantes para os experts, mas que também
# por vezes pode exigir um conhecimento um pouco mais avançado de matemática, estatística e até mesmo de algoritmos.
#
# Espero que tenha se interessado pelo assunto e pelos exemplos, e que essa seja uma boa porta de entrada para o mundo da inteligência artificial!