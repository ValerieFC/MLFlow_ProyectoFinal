import pytest
import pandas as pd
import numpy as np
import sys
import os

# Asegurar que podemos importar desde src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Ahora podemos importar directamente
from src.train import load_data, preprocess_data


def test_data_loading():
    """Test para verificar la carga de datos"""
    try:
        # Esto probará la función load_data
        df = load_data("data/winequality-red.csv")
        assert df.shape[1] >= 11  # Debe tener al menos 11 columnas
        assert "quality" in df.columns
        print("✅ Test de carga de datos pasado")
    except Exception as e:
        pytest.fail(f"Error en carga de datos: {e}")

def test_data_preprocessing():
    """Test para verificar preprocesamiento"""
    # Crear datos de prueba
    df = pd.DataFrame({
        'fixed acidity': [7.4, 7.8],
        'volatile acidity': [0.7, 0.88],
        'citric acid': [0, 0],
        'residual sugar': [1.9, 2.6],
        'chlorides': [0.076, 0.098],
        'free sulfur dioxide': [11, 25],
        'total sulfur dioxide': [34, 67],
        'density': [0.9978, 0.9968],
        'pH': [3.51, 3.2],
        'sulphates': [0.56, 0.68],
        'alcohol': [9.4, 9.8],
        'quality': [5, 5]
    })
    
    # Probar preprocesamiento
    X, y, scaler = preprocess_data(df)
    
    assert X.shape[1] == 11  # 11 características
    assert len(y) == 2  # 2 muestras
    print("✅ Test de preprocesamiento pasado")

def test_model_imports():
    """Test que verifica que los imports funcionan"""
    try:
        from src.train import train_model, evaluate_model
        print("✅ Test de imports pasado")
    except ImportError as e:
        pytest.fail(f"Error de importación: {e}")

if __name__ == "__main__":
    test_data_loading()
    test_data_preprocessing() 
    test_model_imports()
    print("🎉 Todos los tests pasaron!")