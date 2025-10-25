"""
ML Pipeline Module
Funciones para entrenamiento y evaluación de modelos
"""

__version__ = "1.0.0"

from .train import (
    load_config,
    load_data, 
    preprocess_data, 
    train_model, 
    evaluate_model
)

__all__ = [
    'load_config',
    'load_data',
    'preprocess_data', 
    'train_model', 
    'evaluate_model'
]