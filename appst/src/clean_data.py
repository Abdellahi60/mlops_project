import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
def clean_data(data):
    # Vérification si le fichier est vide
    if data.empty:
        return pd.DataFrame()

    # Colonnes catégoriques et numériques
    categorical_features = ['Etablissement', 'Serie,x', 'Centre', 'Willaya', 'moughataa']
    numeric_features = ['Age']

    # Définir les transformations
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    num_transformer = StandardScaler()

    # Créer le préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )

    # Appliquer les transformations
    X_preprocessed = preprocessor.fit_transform(data)
    
    # Convertir en DataFrame
    feature_names = preprocessor.transformers_[0][1].get_feature_names_out(categorical_features).tolist() + numeric_features
    X_preprocessed = pd.DataFrame(X_preprocessed.toarray(), columns=feature_names)
    
    return X_preprocessed

def clean_data1(data):
    # Vérification si le fichier est vide
    if data.empty:
        return pd.DataFrame()

    # Colonnes catégoriques et numériques
    categorical_features = ['Etablissement', 'Serie,x', 'Centre', 'Willaya', 'moughataa']
    numeric_features = ['Age']

    # Définir les transformations
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    num_transformer = StandardScaler()

    # Créer le préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )

    # Appliquer les transformations
    X_preprocessed = preprocessor.fit_transform(data)
    
    # Convertir en DataFrame
    feature_names = preprocessor.transformers_[0][1].get_feature_names_out(categorical_features).tolist() + numeric_features
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
    
    return X_preprocessed