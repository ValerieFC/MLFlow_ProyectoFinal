import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
import yml
import os

def load_config():
    """Cargar configuración desde YML"""
    with open('src/config.yml', 'r') as file:
        return yml.safe_load(file)

def load_data(file_path):
    """Cargar y limpiar datos"""
    df = pd.read_csv(file_path, delimiter=';')
    
    # Limpieza básica
    print(f"Datos originales: {df.shape}")
    
    # Manejar valores nulos
    if df.isnull().sum().any():
        df = df.dropna()
        print("Valores nulos eliminados")
    
    # Codificación: la calidad ya está en formato numérico (3-9)
    # Crear variable binaria para clasificación (calidad buena/mala)
    df['quality_binary'] = (df['quality'] >= 7).astype(int)
    
    print(f"Datos después de limpieza: {df.shape}")
    return df

def preprocess_data(df, target_col='quality_binary'):
    """Preprocesamiento de datos"""
    # Separar características y target
    X = df.drop(['quality', target_col], axis=1)
    y = df[target_col]
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X_train, y_train, config):
    """Entrenar modelo con los parámetros de configuración"""
    model_params = config['model']['params']
    
    model = RandomForestClassifier(
        n_estimators=model_params['n_estimators'],
        max_depth=model_params['max_depth'],
        random_state=model_params['random_state']
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluar el modelo y retornar métricas"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, f1, y_pred

def main():
    """Pipeline principal de ML"""
    # Cargar configuración
    config = load_config()
    
    # Configurar MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Cargar datos
    df = load_data(config['data']['path'])
    
    # Preprocesamiento
    X, y, scaler = preprocess_data(df)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Iniciar run de MLflow
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config['model']['params'])
        mlflow.log_param("test_size", config['data']['test_size'])
        
        # Entrenar modelo
        model = train_model(X_train, y_train, config)
        
        # Evaluar modelo
        accuracy, f1, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1
        })
        
        # Log model with signature and input example
        from mlflow.models.signature import infer_signature
        
        # Crear ejemplo de entrada
        input_example = X_train[:1]
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model",
            signature=signature,
            input_example=input_example
        )
        
        # Log artifacts
        mlflow.log_artifact('src/config.yml')
        
        print(f"✅ Modelo entrenado y registrado!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")
        print(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    main()