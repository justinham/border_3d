# border_3d

> 🗺️ Extract land-water borders from 3D lidar point cloud scans

A Python tool that loads a JSON-encoded 3D point cloud, cleans outliers, and extracts the land edge boundary using convex hull and alpha shape methods. Visualize results in 3D with height-based colormap or as a 2D top-down plot.

## What it does

1. **Load** JSON point cloud from file
2. **Clean** statistical outliers (noise from water scan gaps)
3. **Extract border** via convex hull (outer perimeter) + alpha shape (concave detail)
4. **Filter** by neighbor density to remove noise
5. **Visualize** in 3D (height colormap, blue=low, red=high) + 2D plots
6. **Save** border coordinates as CSV for downstream use (CNC, laser cutting, etc.)

## Quick start

```bash
# Edit DATA_DIR and P1_FILE in border_3d.py, then run:
python3.11 border_3d.py
```

## Input format

Point cloud file (`allP1.txt`) contains a single-line JSON array:
```json
[[x1, y1, z1], [x2, y2, z2], ...]
```

## Output files

| File | Description |
|------|-------------|
| `border_hull.txt` | Convex hull border points (CSV: x,y,z) |
| `border_alpha.txt` | Alpha shape boundary points |
| `border_2d.png` | Top-down 2D polygon view |
| `border_heatmap.png` | Height heatmap (blue=low, red=high) |

## Color guide (3D view)

- **Blue** → lowest elevation (water/land edge)
- **Cyan/Green** → mid-low
- **Yellow/Orange** → higher ground
- **Red** → highest points
- **Bright green dots** → extracted border

## Dependencies

```
open3d
numpy
matplotlib
scipy
```

Install: `pip install open3d numpy matplotlib scipy`

## Data

- `allP1.txt` — sample lidar scan (6,259 points, land scan where water gave no return)
- `border_hull.txt` — 32 extracted border points

## License

MIT