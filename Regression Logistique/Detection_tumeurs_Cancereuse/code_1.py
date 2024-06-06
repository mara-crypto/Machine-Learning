import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('breast-cancer.csv')

# Remplacer les valeurs dans la colonne 'diagnosis'
data['diagnosis'] = data['diagnosis'].replace({'M': 0, 'B': 1})

# Séparer les variables de caractéristiques (X) et la variable cible (y)
X = data.drop(columns=['id', 'diagnosis'])  
y = data['diagnosis']

# Divisez les données en ensembles d'apprentissage (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur l'ensemble d'apprentissage
model.fit(X_train, y_train)

# Prédire les étiquettes sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Précision du modèle : {:.2f}%".format(accuracy * 100))
print("Matrice de confusion :")
print(conf_matrix)
print("Rapport de classification :")
print(class_report)
