# EmailSpamClassifier.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Cette fonction vous permet de diviser vos données en deux ensembles : un ensemble d'entraînement et un ensemble de test.
from sklearn.feature_extraction.text import TfidfVectorizer
#Lorsque vous travaillez avec des données textuelles (comme des commentaires, des articles de blog, etc.),
# vous devez convertir ces textes en un format numérique que les modèles
#d'apprentissage automatique peuvent comprendre. C'est là que le TfidfVectorizer intervient.
from sklearn.linear_model import LogisticRegression
#La régression logistique est un algorithme d'apprentissage automatique utilisé pour la classification(spam ou non spam "ham")
from sklearn.metrics import accuracy_score
#Lorsque vous avez entraîné un modèle de classification, il est essentiel d'évaluer ses performances.
#L'exactitude (accuracy) est l'une des mesures courantes pour évaluer la performance d'un modèle de classification.
from google.colab import drive
drive.mount('/content/drive')
#path = '/content/drive/MyDrive/CSV/mail_data.csv'
df = pd.read_csv('/content/drive/MyDrive/CSV/mail_data.csv')
#print(df)
data = df.where((pd.notnull(df)),'') #remplace toutes les valeurs nulles (NaN) dans le DataFrame df par des chaînes
#de caractères vides ('') tout en maintenant les valeurs non-nulles intactes.
data.head() #five first lignes if you want the last 5 lignes use tail
#data.info()
#data.shape
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']
print(x)
print(y)
x_train , x_test , y_train , y_test = train_test_split (x , y , test_size=0.2 ,random_state= 3) #80% trainning 20% test
#random_state : Ce paramètre est utilisé pour initialiser le générateur de nombres aléatoires pour mélanger les données
#avant la division. En fournissant une valeur fixe (dans ce cas, 3), vous garantissez que la division des données sera
#la même chaque fois que vous exécutez votre code avec le même jeu de données.
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)
feature_extraction = TfidfVectorizer(min_df = 1 , stop_words= 'english')
x_train_featrues = feature_extraction.fit_transform(x_train)
x_test_featrues = feature_extraction.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
model = LogisticRegression() #train the model
model.fit(x_train_featrues , y_train)
prediction_on_training_data = model.predict(x_train_featrues)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('accuracy on test data = ', accuracy_on_training_data)

input_your_mail =["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction =model.predict(input_data_features)
print(prediction)
if (prediction[0]==1) :
  print('Ham mail')
else:
  print('Spam mail')
