import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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


# Obtenez les probabilités prédites pour la classe positive (tumeurs bénignes)
y_scores = model.predict_proba(X_test)[:, 1]

# Calculez la courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Tracez la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC de la régression logistique')
plt.legend(loc='lower right')
plt.show()
