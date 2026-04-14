"""
Point Cloud Border Extraction Tool
Extracts land edge / water boundary from 3D lidar scan point clouds.
Works with Open3D + NumPy + SciPy.
"""

import open3d as o3d
import numpy as np
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# === CONFIG ===
DATA_DIR = "/Volumes/ssd/Downloads/"
P1_FILE = DATA_DIR + "allP1.txt"


def load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load JSON-encoded point array into Open3D point cloud."""
    with open(filepath, "r") as f:
        points = json.loads(f.readline())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    print(f"Loaded {len(points)} points from {filepath}")
    return pcd


def clean_outliers(pcd, nb_neighbors=20, std_ratio=1.0):
    """Remove statistical outliers from point cloud."""
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"After outlier removal: {len(pcd_clean.points)} points")
    return pcd_clean


def color_by_height(pcd):
    """Color point cloud by Z height using jet colormap (blue=low, red=high)."""
    z_vals = np.asarray(pcd.points)[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    print(f"Z range: {z_min:.4f} to {z_max:.4f}")
    z_norm = (z_vals - z_min) / (z_max - z_min + 1e-9)
    colors = cm.jet(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def extract_hull_border(pcd, voxel_size=0.1):
    """
    Extract land-edge border via convex hull.
    Returns downsampled hull points.
    """
    hull_mesh, _ = pcd.compute_convex_hull()
    hull_pts = np.asarray(hull_mesh.vertices)
    print(f"Convex hull: {len(hull_pts)} vertices")

    # Downsample
    hull_pcd = o3d.geometry.PointCloud()
    hull_pcd.points = o3d.utility.Vector3dVector(hull_pts)
    hull_pcd = hull_pcd.voxel_down_sample(voxel_size=voxel_size)
    hull_pts = np.asarray(hull_pcd.points)
    print(f"Downsampled hull: {len(hull_pts)} points")
    return hull_pts


def extract_alpha_border(pcd, alpha=0.3):
    """
    Extract detailed boundary via alpha shape (concave hull).
    Returns boundary vertices.
    """
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    vertices = np.asarray(mesh.vertices)
    print(f"Alpha mesh: {len(mesh.triangles)} triangles, {len(vertices)} vertices")

    # Find edges that appear only once (boundary edges)
    triangles = np.asarray(mesh.triangles)
    edges = {}
    for tri in triangles:
        for i in range(3):
            a, b = sorted([tri[i], tri[(i+1) % 3]])
            edges[(a, b)] = edges.get((a, b), 0) + 1

    boundary_keys = [k for k, v in edges.items() if v == 1]
    boundary_verts = set()
    for a, b in boundary_keys:
        boundary_verts.add(a)
        boundary_verts.add(b)

    border = vertices[list(boundary_verts)]
    print(f"Alpha boundary: {len(border)} vertices")
    return border


def filter_by_density(pts, radius=0.2, min_k=2, max_k=15):
    """Remove noise: keep points with reasonable local neighbor density."""
    if len(pts) == 0:
        return pts
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(pts)
    tree = o3d.geometry.KDTreeFlann(pcd_b)
    valid = []
    for i in range(len(pts)):
        k, _, _ = tree.search_radius_vector_3d(pts[i], radius)
        if min_k <= k <= max_k:
            valid.append(i)
    return pts[valid]


def sort_polygon(pts):
    """Sort points by angle from centroid to form proper polygon order."""
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


def visualize_3d(pcd, border_pts=None, title="Point Cloud"):
    """Interactive 3D visualization with optional border overlay."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)

    if border_pts is not None:
        hull_pcd = o3d.geometry.PointCloud()
        hull_pcd.points = o3d.utility.Vector3dVector(border_pts)
        hull_pcd.paint_uniform_color([0.0, 1.0, 0.3])  # bright green
        vis.add_geometry(hull_pcd)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([0.05, 0.05, 0.1])

    vis.run()
    vis.destroy_window()


def plot_2d_border(hull_pts, alpha_pts=None, save_path=None):
    """Plot top-down 2D view of land border."""
    hull_sorted = sort_polygon(hull_pts.copy())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill(hull_sorted[:, 0], hull_sorted[:, 1], alpha=0.3, color='steelblue', label='Land area')
    ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'o-', color='navy', markersize=5, label='Land edge')

    if alpha_pts is not None and len(alpha_pts) > 0:
        alpha_sorted = sort_polygon(alpha_pts.copy())
        ax.plot(alpha_sorted[:, 0], alpha_sorted[:, 1], 's-', color='darkgreen',
                markersize=3, alpha=0.6, label='Alpha border')

    cx, cy = hull_sorted[:, 0].mean(), hull_sorted[:, 1].mean()
    ax.plot(cx, cy, 'x', color='red', markersize=15, label='Centroid')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Land Border - 2D Top-Down View')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved 2D plot to {save_path}")

    plt.show()


def plot_2d_heatmap(pcd, hull_pts=None, save_path=None):
    """Plot top-down 2D heatmap colored by height."""
    pts = np.asarray(pcd.points)
    z_vals = pts[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.ptp() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=z_norm, cmap='jet', s=1, alpha=0.8)

    if hull_pts is not None:
        hull_sorted = sort_polygon(hull_pts.copy())
        ax.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'w-', linewidth=1.5,
                label='Land edge', alpha=0.9)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Height Heatmap (Blue=Low, Red=High)')
    ax.set_aspect('equal')
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Height', rotation=270, label_size=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved heatmap to {save_path}")

    plt.show()


# === MAIN ===
if __name__ == "__main__":
    # Load
    pcd = load_point_cloud(P1_FILE)

    # Clean
    pcd_clean = clean_outliers(pcd)

    # Estimate normals
    pcd_clean.estimate_normals()

    # Extract borders
    hull_pts = extract_hull_border(pcd_clean, voxel_size=0.1)
    hull_pts = filter_by_density(hull_pts)
    print(f"Hull border: {len(hull_pts)} points after filter")

    alpha_pts = extract_alpha_border(pcd_clean, alpha=0.3)
    alpha_pts = filter_by_density(alpha_pts)
    print(f"Alpha border: {len(alpha_pts)} points after filter")

    # Save
    np.savetxt(DATA_DIR + "border_hull.txt", hull_pts, delimiter=",")
    np.savetxt(DATA_DIR + "border_alpha.txt", alpha_pts, delimiter=",")
    print(f"Saved borders to {DATA_DIR}border_hull.txt and border_alpha.txt")

    # Color point cloud by height
    pcd_colored = color_by_height(pcd_clean)

    # 3D visualization (height colormap + green border)
    visualize_3d(pcd_colored, hull_pts, title="Height Map - Blue=Low, Red=High")

    # 2D plot
    plot_2d_border(hull_pts, alpha_pts, save_path=DATA_DIR + "border_2d.png")

    # 2D heatmap
    plot_2d_heatmap(pcd_clean, hull_pts, save_path=DATA_DIR + "border_heatmap.png")

    print("\nDone!")
    print(f"X range: {np.asarray(pcd_clean.points)[:,0].min():.2f} to {np.asarray(pcd_clean.points)[:,0].max():.2f}")
    print(f"Y range: {np.asarray(pcd_clean.points)[:,1].min():.2f} to {np.asarray(pcd_clean.points)[:,1].max():.2f}")
    print(f"Z range: {np.asarray(pcd_clean.points)[:,2].min():.4f} to {np.asarray(pcd_clean.points)[:,2].max():.4f}")