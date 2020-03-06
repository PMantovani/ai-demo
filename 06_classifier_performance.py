import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

train = []
test = []
all_data = []

# Vamos ler o arquivo e dividir metade em dados de treino e metade em dados de teste
with open('data/random.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    add_to_test = False
    for row in csv_reader:
        all_data.append(row)
        if add_to_test:
            test.append(row)
            add_to_test = False
        else:
            train.append(row)
            add_to_test = True

test = np.array(test)
train = np.array(train)

# A notação [:,1:] serve para remover a primeira coluna da matriz (que é a coluna das classificações)
train_without_class = train[:,1:]
test_without_class = test[:,1:]
# A notação [:,0] serve para pegar apenas a primeira coluna da matriz (que é a coluna das classificações)
train_class = train[:,0]
test_class = test[:,0]

# Treinando o nosso classificador!
clf = svm.SVC()
clf.fit(train_without_class, train_class)

# Vamos colocar o classificador em prática para tentar "adivinhar" nossos dados de teste
predictions = clf.predict(test_without_class)

# # Agora, vamos analisar o desempenho do nosso classificador

# Os dados podem estar distribuídos na seguinte forma...
#
#                                  ----------------------------------------------------------------------
#                                  |                           Dados reais                              |
#                                  |--------------------------------------------------------------------|
#                                  |            Positivos          |              Negativos             |
#     -----------------------------X-------------------------------X------------------------------------|
#     |                | Positivos |    Verdadeiros Positivos      |           Falsos Positivos         |      <==== Esta tabela se chama matriz de confusão!
#     |   Predições    |-----------X-------------------------------X------------------------------------|
#     |                | Negativos |      Falsos Negativos         |         Verdadeiros Negativos      |
#     --------------------------------------------------------------------------------------------------|

# Verdadeiros positivos: dados classificados como '1' que são de fato '1'
# Falsos positivos: dados classificados como '1' que são na verdade '0'
# Falsos negativos: dados classificados como '0' que são na verdade '1'
# Verdadeiros negativos: dados classificados como '0' que são de fato '0'

# Nosso objetivo é maximizar os verdadeiros positivos e negativos, e minimizar os falsos positivos e negativos!

# Temos várias métricas para analisar o desempenho do classificador, mas principalmente: precision, recall, accuracy e f1_score.
# https://en.wikipedia.org/wiki/Precision_and_recall


# Precision = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)
# Recall    = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)
# Accuracy  = (Verdadeiros Positivos + Verdadeiros Negativos) / Total
# F1 Score  = 2 * precision * recall / (precision + recall)                            <- média harmônica da precisão e do recall


# Mas e qual é a melhor métrica??.... Depende do caso!!!

# Se estamos analisando um possível caso de câncer, é melhor termos um falso positivo (e realizar mais exames por via das dúvidas), do que um falso negativo.
# Nesse caso, o recall seria mais relevante, pois avaliaria melhor o quanto de falsos negativos nosso classificador teve.

# Se estamos analisando se uma digital corresponde a de um usuário para a autenticação no sistema, não podemos ter falsos positivos.
# Nesse caso, a precisão é mais relevante.

# Porém, é muito fácil fazer um classificador que tenha uma ótima precisão e um péssimo recall, e vice-versa... Esse classificador na verdade não ajudaria de nada!
# Portanto, é importante sempre manter um F1 score alto!




# No nosso conjunto de dados genérico deste exemplo, vamos aos números:
print(confusion_matrix(test_class, predictions))

#                                  -----------------------------
#                                  |  Dados reais              |
#                                  |---------------------------|
#                                  |  Positivos  |  Negativos  |
#     -----------------------------X---------------------------|
#     |                | Positivos |    23       |     7       |
#     |   Predições    |-----------X---------------------------|
#     |                | Negativos |     2       |    18       |
#     ---------------------------------------------------------|



# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
precision = precision_score(test_class, predictions)      # 23 / (23 + 7) = 0.76667
recall = recall_score(test_class, predictions)            # 23 / (23 + 2) = 0.92
f1_score = f1_score(test_class, predictions)              # 2 * 0.7667 * 0.92 / (0.7667 + 0.92) = 0.82
print('Precisão: ' + str(precision))
print('Recall: ' + str(recall))
print('F1 Score: ' + str(f1_score))