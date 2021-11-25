# EmergingTechnologies
#k-nearest neighbors algorithm usando Python

# Import LabelEncoder

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

#Atribuição de recursos e variáveis de rótulo

#Primeiro recurso

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']

#Segundo recurso

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

#Rótulo ou variável de destino

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

# Neste conjunto de dados, temos dois recursos (clima e temperatura) e um rótulo (reprodução).

#creating labelEncoder: representar colunas categóricas em uma coluna numérica

le = preprocessing.LabelEncoder() #codificação de rótulo, e o sklearn convenientemente fará isso

# Converter rótulos de clima em números.

weather_encoded=le.fit_transform(weather) #objeto LabelEncoder ajusta e transformar a coluna "clima" na coluna numérica

print(weather_encoded)

# converter rótulos de temperatura em números

temp_encoded=le.fit_transform(temp) #objeto LabelEncoder ajusta e transformar a coluna "clima" na coluna numérica
label=le.fit_transform(play) #objeto LabelEncoder do rótulo ou variável desitno
print(temp_encoded)
print(label)

#várias colunas ou recursos em um único conjunto de dados usando a função "zip"

features=list(zip(weather_encoded,temp_encoded))

print(features)

#construindo o modelo do classificador KNN

model = KNeighborsClassifier(n_neighbors=3) #objeto classificador KNN e  número do argumento de vizinhos

# Treinando o modelo usando os conjuntos de treinamento

model.fit(features,label)

#previsão no conjunto de teste usando Predict ()

predicted= model.predict([[0,2]])
#primeira coluna corresponde ao clima
#Clima = 0 (nublado)
#clima = 1 (chuvoso)
#clima = 2 (ensolarado)
#segunda coluna corresponde a temperatura 
#tempo = 0 (suave)
#tempo = 1 (quente)
#tempo = 2 (moderado/ameno)

print(predicted)
#Rótulo play: não = 0 e sim = 1


#Logistic Regression algorithm usando Python
​
#Importando bibliotecas

import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML


#Importa o dataset

dataset = pd.read_csv("C:/Users/breno/desktop/Tec Emergentes/heart_disease.csv")
dataset = dataset.dropna() #removendo valores nulos 
X = dataset.iloc[:, :-1].values #todas as colunas menos a última
y = dataset.iloc[:, -1].values #Y é o que queremos: tem doença ou não?
#pd.options.display.max_columns = None
#display(dataset)
X = pd.DataFrame(X)
X

# Separar dados em Treino e Teste

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 1/5, random_state = 0)

X_train = pd.DataFrame(X_train)
X_train

X_test = pd.DataFrame(X_test)
X_test

# Treinando o modelo

classifier = LogisticRegression() #O nome está classifier porque estamos classificando se tem ou não doença cardiaca.
classifier.fit(X_train, y_train)

# Prever um Valor Especifico

print(classifier.predict([[1,25,4,0,0,0,0,0,0,195,130,70,80,20,56]]))

# Previsao

y_pred = classifier.predict(X_test) #para Classificação
y_pred_prob = classifier.predict_proba(X_test) #para a probabilidade 
y_pred_prob = y_pred_prob[:,1]
y_result_prob = np.concatenate((y_pred.reshape(len(y_pred),1), y_pred_prob.reshape(len(y_pred_prob),1)),1)
y_result_prob = pd.DataFrame(y_result_prob)
y_result_prob

#Probabilidade de ter doenças cardiácas e a classificação do classifier ao lado (coluna 0)

# Matrix de confusao

cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm)
cm

print(accuracy_score(y_test, y_pred))

#Visualizar o que quer dizer a Confusion Matrix em forma de tabela

y_result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1) #cancatenamos y prev com y test
y_result = pd.DataFrame(y_result)
y_result
#Ou seja, primeira coluna Y previsto e segunda Coluna Y de test
