import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Beleza, mas e como fazemos tudo isso com imagens?

# A função abaixo carrega um dataset já pré-definido no sklearn de imagens 8x8 de números escritos a mão.
# digits.images contém as matrizes de imagens 8x8 (preto e branco)
# digits.target contém as classificações. São números de 0 a 9.
# Caso quiséssemos carregar alguma imagem de 'verdade', poderíamos usar a função matplotlib.pyplot.imread. (as imagens precisam ser do mesmo tamanho)
digits = datasets.load_digits()

# Vamos mostrar as imagens para ver como elas se parecem?
# O comando subplots da bilbioteca matplotlib cria uma janela com vários espaços para plotarmos essas imagens (2x4)
fig1, axes = plt.subplots(1, 8)

# Junta os vetores de classificação e as imagens em um único objeto
images_and_labels = list(zip(digits.images, digits.target))

# Plota as 8 primeiras imagens de treino em uma figura no matplotlib 
for ax, (image, label) in zip(axes, images_and_labels[:8]):
    ax.set_axis_off() # Desativa os eixos do gráfico
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest') # Printa a imagem em preto e branco
    ax.set_title('Training: %i' % label)

# Cada um de nossas imagens atualmente está no formato 8x8. Porém, a nossa SVM não aceita matrizes como entrada.
# Precisamos então "achatá-la", e transformá-la em um vetor 1x64. Isto é feito pelo comando reshape. 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Dividindo nossos dados em treino e teste. A primeira metade serão os dados de treino, a segunda os dados de teste
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Predizendo os valores da segunda metade (os dados de teste)
predicted = classifier.predict(X_test)

# Vamos mostrar as imagens testadas e a previsão do nosso classificador?
fig2, axes2 = plt.subplots(1, 8)
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

# Plota as primeiras 8 imagens que foram preditas pelo classificador.
for ax, (image, prediction) in zip(axes2, images_and_predictions[:8]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

# Printando o resultado do nosso classificador 
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()