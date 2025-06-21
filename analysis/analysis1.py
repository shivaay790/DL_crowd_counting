import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns

density_map = np.load(r"crowd_wala_dataset\val_data\density_map\IMG_211.npy")
density_map = np.squeeze(density_map)


CROWD_THRESHOLDS = {'small': 50, 'medium': 150}
STAMPEDE_DENSITY_THRESHOLD = 5.0  # Intense density in one patch
STAMPEDE_GRADIENT_THRESHOLD = 1.0  # Strong flow (movement)
GRID_SIZE = (4, 4)

def classify_crowd_total(density):
    total_count = np.sum(density)
    if total_count < CROWD_THRESHOLDS['small']:
        return "Small"
    elif total_count < CROWD_THRESHOLDS['medium']:
        return "Medium"
    else:
        return "Large"

def detect_stampede_zones(density):
    gx, gy = np.gradient(density)
    grad_magnitude = np.sqrt(gx**2 + gy**2)

    # Stampede = high density + strong gradient
    stampede_mask = (density > STAMPEDE_DENSITY_THRESHOLD) & (grad_magnitude > STAMPEDE_GRADIENT_THRESHOLD)

    plt.figure(figsize=(6, 5))
    plt.imshow(density, cmap='hot', alpha=0.6)
    plt.imshow(stampede_mask, cmap='cool', alpha=0.4)
    plt.title("Stampede-Prone Zones (Blue Overlay)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return np.sum(stampede_mask), stampede_mask

def plot_flow_gradients(density, step=10):
    gx, gy = np.gradient(density)
    x = np.arange(0, density.shape[1], step)
    y = np.arange(0, density.shape[0], step)
    X, Y = np.meshgrid(x, y)
    U = gx[::step, ::step]
    V = gy[::step, ::step]

    plt.figure(figsize=(8, 6))
    plt.imshow(density, cmap='hot', alpha=0.6)
    plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1)
    plt.title("Gradient Field of Crowd Density (Flow Directions)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_pci_heatmap(density, grid_size=GRID_SIZE):
    h, w = density.shape
    ph, pw = h // grid_size[0], w // grid_size[1]
    pci_map = np.zeros(grid_size)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            patch = density[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            pci_map[i, j] = patch.sum()

    plt.figure(figsize=(6, 5))
    sns.heatmap(pci_map, annot=True, cmap="YlOrRd")
    plt.title("Population Concentration Index (PCI) Heatmap")
    plt.xlabel("Grid Column")
    plt.ylabel("Grid Row")
    plt.show()

def plot_adaptive_risk(density):
    mean = np.mean(density)
    std = np.std(density)
    adaptive_threshold = mean + 1.5 * std

    risk_mask = density > adaptive_threshold

    plt.figure(figsize=(6, 5))
    plt.imshow(density, cmap='gray')
    plt.imshow(risk_mask, cmap='Reds', alpha=0.5)
    plt.title("Adaptive Risk Zones (Red Overlay)")
    plt.axis('off')
    plt.show()

def plot_dbscan_hotspots(density, density_threshold=1.0):
    points = np.column_stack(np.where(density > density_threshold))
    if len(points) == 0:
        print("No dense areas found.")
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(points)
    db = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
    labels = db.labels_

    plt.figure(figsize=(6, 6))
    plt.imshow(density, cmap='gray')
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], s=5, label=f'Cluster {label}')
    plt.title("DBSCAN Hotspot Clustering")
    plt.legend()
    plt.axis('off')
    plt.show()

def plot_fingerprints(density):
    vertical_profile = np.sum(density, axis=1)
    horizontal_profile = np.sum(density, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(vertical_profile)
    plt.title("Vertical Crowd Profile")
    plt.xlabel("Y-axis (rows)")
    plt.ylabel("Density Sum")

    plt.subplot(1, 2, 2)
    plt.plot(horizontal_profile)
    plt.title("Horizontal Crowd Profile")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("Density Sum")
    plt.tight_layout()
    plt.show()

crowd_size = classify_crowd_total(density_map)
print(f" Crowd classification: {crowd_size}")

stampede_count, stampede_mask = detect_stampede_zones(density_map)
print(f" Stampede-risky areas: {stampede_count}")

plot_flow_gradients(density_map)
plot_pci_heatmap(density_map)
plot_adaptive_risk(density_map)
plot_dbscan_hotspots(density_map)
plot_fingerprints(density_map)
