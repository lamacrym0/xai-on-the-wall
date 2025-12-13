import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_linnerud

def generate_sklearn_csv(dataset_loader, filename):
    print(f"--- Traitement pour {filename} ---")
    
    # 1. Charger les données
    data = dataset_loader()
    
    # 2. Créer le DataFrame des Features
    # Certains datasets n'ont pas de feature_names, on gère ce cas
    if hasattr(data, 'feature_names'):
        columns = data.feature_names
    else:
        columns = [f"feature_{i}" for i in range(data.data.shape[1])]
        
    df = pd.DataFrame(data.data, columns=columns)
    
    # 3. Gérer les Targets (C'est ici que ça se joue)
    # On vérifie la forme (shape) de la target
    targets = data.target
    
    if targets.ndim > 1 and targets.shape[1] > 1:
        # CAS MULTI-TARGET : On crée plusieurs colonnes
        print(f"Multi-target détecté ({targets.shape[1]} colonnes)")
        if hasattr(data, 'target_names') and len(data.target_names) == targets.shape[1]:
            # Si on a des noms officiels (ex: 'Weight', 'Waist', 'Pulse')
            target_cols = data.target_names
        else:
            # Sinon on génère target_0, target_1...
            target_cols = [f"target_{i}" for i in range(targets.shape[1])]
            
        # On ajoute les colonnes au DataFrame
        df_targets = pd.DataFrame(targets, columns=target_cols)
        df = pd.concat([df, df_targets], axis=1)
        
    else:
        # CAS SIMPLE (1 Target)
        print("Single-target détecté")
        target_name = 'target'
        # Parfois target_names existe mais contient juste les labels des classes (ex: 'malignant', 'benign')
        # On garde 'target' comme nom de colonne standard pour le CSV
        df[target_name] = targets

    # 4. Sauvegarde
    df.to_csv(filename, index=False)
    print(f"✅ Fichier '{filename}' créé : {df.shape}")
    print("-" * 30)

if __name__ == "__main__":
    # Exemple 1 : Breast Cancer (Classique, 1 target)
    generate_sklearn_csv(load_breast_cancer, 'breast_cancer.csv')
    
    # Exemple 2 : Iris (Classique, 1 target)
    generate_sklearn_csv(load_iris, 'iris.csv')
    
    # Exemple 3 : Linnerud (Multi-target regression : 3 sorties)
    # Cela générera un CSV avec 'Weight', 'Waist', 'Pulse' à la fin
    generate_sklearn_csv(load_linnerud, 'linnerud_multi.csv')