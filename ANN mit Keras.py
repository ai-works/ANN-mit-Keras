### Schritt 1. Importieren der libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras #Framework für NNs per "Drag'n'Drop" ;)
from keras.models import Sequential # Für die Initialisierung, wir bauen das ANN als Sequenz von Schichten
from keras.layers import Dense


### Schritt 2. Importieren des Datensets
dataset = pd.read_csv('Your.csv')

### Schritt 3. Festlegen der abhängigen und unabhängigen Variablen
X = dataset.iloc[:, ?:??].values # ? = alle Features die in das NN einfließen sollen
y = dataset.iloc[:, ?].values # Letzte Variable als abhängige/Label

### Schritt 4. Falls kategoriale Daten in numerische umzuwandeln sind
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X_1 = LabelEncoder()
# X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# labelencoder_X_2 = LabelEncoder()
# X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features = [1])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]

### Schritt 5. Aufteilen des Datensets in Trainings- und Testdaten,  80% und 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#### Schritt 6.Skalieren der Daten, da nur das Verhältnis zwischen den Datenpunkten relevant ist, nicht der Abstand
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Schritt 7. Initialisieren = Kreieren eines Objektes der Klasse aus Keras = Sequential
classifier = Sequential()

### Schritt 8. Hinzufügen des ersten bzw. des Input-Layers
# Input_Dimension = Anzahl der Input Neuronen, die der Anzahl der Features entspricht
# Die Output Dimension kommt aber auf das Problem und verfügbaere Daten an
classifier.add(Dense(output_dim = ?, init = 'uniform', activation = 'relu', input_dim = ??))

### Schritt 9. Hinzufügen des Hidden Layers
# Anzahl der Neuronen im Hidden Layer = keine Daumenregel, aber Zahl zwischen Anzahl Input Neuronen und Output-Neuronen
classifier.add(Dense(output_dim = ?, init = 'uniform', activation = 'relu'))

### Schritt 10. Hinzufügen des Ouput Layers für binäre Probleme
# Wahl der Sigmoid Aktivierungsfunktion da ein rein binäres Problem besteht = Output-Dimension = 1
# Sigmoid gibt anders als Relu eine Wahrscheinlichkeit zwischen 0 und 1 aus
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

### Schritt 11. Kompilieren des NNs
# Fehlerfunktion = binäre Crossentropy, da binäres Problem
# Adam Optimizer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Schritt 12. Eigentliches Training des NNs, Gewichtsanpassung mit Gradientenabstiegsverfahren
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

### Schritt 13. Vorhersagen des NN auf das Test Set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# ### Schritt 15. Konfusions-Matrix um das trainierte Netz zu bewerten
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
