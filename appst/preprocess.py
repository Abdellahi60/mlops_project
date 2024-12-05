
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

data_url = "./Data/bac.csv"
bac = pd.read_csv(data_url, sep=',')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Define categorical and numeric features
categorical_features = ['Etablissement', 'Serie,x', 'Centre', 'Willaya', 'moughataa']
numeric_features = ['Age']
target = 'Decision'

# Load your data (replace this with your actual data loading code)
#data_url = "C:/Users/amema/OneDrive/Desktop/data/bac.csv"
#bac = pd.read_csv(data_url)
 #data_url = "/app/bac.csv"
#bac = pd.read_csv(data_url)

# Preprocessing
# Prétraitement des variables catégorielles
cat_transformer = OneHotEncoder(handle_unknown='ignore')
cat_transformer.fit(bac[categorical_features])

# Prétraitement de la variable numérique
num_transformer = StandardScaler()
num_transformer.fit(bac[numeric_features])

# Création du transformateur de colonnes
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, categorical_features),
        ('num', num_transformer, numeric_features)
    ])

# Encodage des variables catégorielles de l'ensemble d'entraînement
X_encoded = preprocessor.fit_transform(bac[categorical_features + numeric_features])
y = pd.DataFrame({'Decision': bac['Decision']})

# Split dataset into training set and test set after encoding
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


#print(preprocessor)