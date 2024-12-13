import numpy as np
import matplotlib.pyplot as plt


def generate_ellipsoid(
    a: float, b: float, c: float, num_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère les coordonnées paramétriques d'un ellipsoïde.

    Arguments :
        a (float): Rayon le long de l'axe X.
        b (float): Rayon le long de l'axe Y.
        c (float): Rayon le long de l'axe Z.
        num_points (int): Nombre de points pour la grille paramétrique (résolution).

    Retourne :
        tuple[np.ndarray, np.ndarray, np.ndarray]: Coordonnées X, Y, Z de l'ellipsoïde.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle azimutal (0 à 2π)
    phi = np.linspace(0, np.pi, num_points)  # Angle polaire (0 à π)
    Theta, Phi = np.meshgrid(theta, phi)  # Grille 2D pour les angles

    X = a * np.sin(Phi) * np.cos(Theta)
    Y = b * np.sin(Phi) * np.sin(Theta)
    Z = c * np.cos(Phi)

    return X, Y, Z


def plot_ellipsoid(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax: plt.Axes, title: str = "Ellipsoid"
) -> None:
    """
    Trace un ellipsoïde à partir des coordonnées paramétriques.

    Arguments :
        X (np.ndarray): Coordonnées X de l'ellipsoïde.
        Y (np.ndarray): Coordonnées Y de l'ellipsoïde.
        Z (np.ndarray): Coordonnées Z de l'ellipsoïde.
        ax (plt.Axes): Objet Axes 3D pour le tracé.
        title (str): Titre du graphique.
    """
    ax.plot_surface(
        X, Y, Z, rstride=5, cstride=5, cmap="jet", alpha=0.8, edgecolor="none"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_zlabel("Z-axis", fontsize=12)
    ax.grid(True)


def generate_orbit(
    center: tuple[float, float, float], radius: float, num_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère une orbite circulaire autour d'un point donné.

    Arguments :
        center (tuple[float, float, float]): Coordonnées (x, y, z) du centre de l'orbite.
        radius (float): Rayon de l'orbite.
        num_points (int): Nombre de points pour la trajectoire orbitale.

    Retourne :
        tuple[np.ndarray, np.ndarray, np.ndarray]: Coordonnées X, Y, Z de l'orbite.
    """
    if radius <= 0:
        raise ValueError("Incorrect radius")
    if num_points <= 0:
        raise ValueError("Incorrect number if points")
    t = np.linspace(0, 2 * np.pi, num_points)  # Angle pour l'orbite
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = center[2] * np.ones_like(t)  # L'orbite est dans un plan constant en Z
    return x, y, z


if __name__ == "__main__":  # pragma: no cover
    # Configuration de l'ellipsoïde principal
    a_main, b_main, c_main = 4, 3, 1
    X_main, Y_main, Z_main = generate_ellipsoid(a_main, b_main, c_main)

    # Configuration de l'ellipsoïde secondaire
    a_secondary, b_secondary, c_secondary = 0.5, 0.3, 0.2
    orbit_radius = 6  # Rayon de l'orbite autour de l'ellipsoïde principal
    center = (0, 0, 0)  # Centre de l'ellipsoïde principal
    X_orbit, Y_orbit, Z_orbit = generate_orbit(center, orbit_radius)

    # Création de la figure et de l'axe 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Tracé de l'ellipsoïde principal
    plot_ellipsoid(X_main, Y_main, Z_main, ax, title="Ellipsoid with Orbit")

    # Tracé de l'orbite
    ax.plot(
        X_orbit,
        Y_orbit,
        Z_orbit,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Orbit",
    )

    # Tracé de l'ellipsoïde secondaire sur l'orbite
    for i in range(
        0, len(X_orbit), 20
    ):  # Positionner plusieurs ellipsoïdes le long de l'orbite
        X_secondary, Y_secondary, Z_secondary = generate_ellipsoid(
            a_secondary, b_secondary, c_secondary
        )
        ax.plot_surface(
            X_secondary + X_orbit[i],
            Y_secondary + Y_orbit[i],
            Z_secondary + Z_orbit[i],
            color="red",
            alpha=0.6,
            edgecolor="none",
        )

    # Afficher la légende et le graphique
    ax.legend()
    plt.show()
