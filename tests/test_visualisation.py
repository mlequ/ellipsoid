import numpy as np
import matplotlib.pyplot as plt
import pytest
from ellipsoid.projec_ellipse import generate_ellipsoid, generate_orbit, plot_ellipsoid


# Test pour la génération de l'ellipsoïde
def test_generate_ellipsoid():
    # Cas avec un nombre plus élevé de points pour un test plus robuste
    X, Y, Z = generate_ellipsoid(3, 2, 1, num_points=100)

    # Vérification des formes
    assert X.shape == (100, 100), f"Shape X: {X.shape}"
    assert Y.shape == (100, 100), f"Shape Y: {Y.shape}"
    assert Z.shape == (100, 100), f"Shape Z: {Z.shape}"

    # Vérification des bornes des axes, avec tolérance
    assert np.allclose(X.max(), 3, atol=0.1), f"X.max() = {X.max()}"
    assert np.allclose(Y.max(), 2, atol=0.1), f"Y.max() = {Y.max()}"
    assert np.allclose(Z.max(), 1, atol=0.1), f"Z.max() = {Z.max()}"
    assert np.allclose(X.min(), -3, atol=0.1), f"X.min() = {X.min()}"
    assert np.allclose(Y.min(), -2, atol=0.1), f"Y.min() = {Y.min()}"
    assert np.allclose(Z.min(), -1, atol=0.1), f"Z.min() = {Z.min()}"

    # Vérification des valeurs NaN ou infinies
    assert not np.any(np.isnan(X)), "X contains NaN values"
    assert not np.any(np.isnan(Y)), "Y contains NaN values"
    assert not np.any(np.isnan(Z)), "Z contains NaN values"
    assert not np.any(np.isinf(X)), "X contains infinite values"
    assert not np.any(np.isinf(Y)), "Y contains infinite values"
    assert not np.any(np.isinf(Z)), "Z contains infinite values"


# Test de la génération de l'orbite
def test_generate_orbit():
    center = (0, 0, 0)
    radius = 5
    num_points = 100
    X, Y, Z = generate_orbit(center, radius, num_points)

    # Vérifie que la forme des sorties est correcte
    assert len(X) == num_points, f"Length of X: {len(X)}"
    assert len(Y) == num_points, f"Length of Y: {len(Y)}"
    assert len(Z) == num_points, f"Length of Z: {len(Z)}"

    # Vérifie que la distance au centre est constante et proche du rayon
    distances = np.sqrt(X**2 + Y**2 + (Z - center[2]) ** 2)
    assert np.allclose(
        distances, radius, atol=1e-2
    ), f"Distance des points: {distances}"

    # Vérification des valeurs NaN ou infinies dans les résultats
    assert not np.any(np.isnan(X)), "X contains NaN values"
    assert not np.any(np.isnan(Y)), "Y contains NaN values"
    assert not np.any(np.isnan(Z)), "Z contains NaN values"
    assert not np.any(np.isinf(X)), "X contains infinite values"
    assert not np.any(np.isinf(Y)), "Y contains infinite values"
    assert not np.any(np.isinf(Z)), "Z contains infinite values"
    return X, Y, Z


# Test pour la fonction de plot de l'ellipsoïde
def test_plot_ellipsoid():
    # Test de la fonction de visualisation de l'ellipsoïde
    X, Y, Z = generate_ellipsoid(3, 2, 1, num_points=100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        plot_ellipsoid(X, Y, Z, ax)
    except Exception as e:
        pytest.fail(f"plot_ellipsoid raised an exception: {e}")

    # Assurer que le plot ne génère pas d'erreur, mais cela ne valide pas visuellement.
    assert ax.has_data(), "Le graphique est vide"


# Test pour la gestion des entrées invalides dans la génération d'orbite
def test_generate_orbit_with_invalid_input():
    # Test avec des entrées invalides
    with pytest.raises(ValueError):
        generate_orbit((0, 0, 0), -5, 100)  # Rayon négatif
    with pytest.raises(ValueError):
        generate_orbit((0, 0, 0), 5, -100)  # Nombre de points négatif
    with pytest.raises(ValueError):
        generate_orbit((0, 0, 0), 5, 0)  # Nombre de points égal à zéro
