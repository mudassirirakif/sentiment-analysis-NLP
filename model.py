# this model is implemented using scikit learn's `MLPClassifier`

from sklearn.neural_network import MLPClassifier


def prepare_model(random_state: int = 42) -> MLPClassifier:
    
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=50,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    return classifier


__all__ = ["prepare_model"]