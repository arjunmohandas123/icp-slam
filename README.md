# 2D ICP (Iterative Closest Point)

A small, readable implementation of **point-to-point ICP** in 2D using only NumPy. It estimates the rotation and translation that align a LiDAR-style point cloud (the *scan*) to a reference (the *map*). This is the scan-matching step at the heart of LiDAR odometry and SLAM.

## How it works

Each iteration does four things:

1. **Match:** for every scan point, find the closest point in the reference.
2. **Estimate:** compute the single rigid transform (rotation `R`, translation `t`) that minimizes the total squared distance over all pairs. This uses the Kabsch algorithm (SVD).
3. **Apply:** move the whole scan by that transform.
4. **Repeat** until the mean matching error stops improving.

The accumulated transform is the pose change of the sensor, i.e. how the robot moved.

```
scan ──► match nearest points ──► Kabsch (R, t) ──► move scan ──┐
  ▲                                                             │
  └───────────────────── repeat until converged ◄───────────────┘
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (optional, only for the plot)

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

## Run the demo

```bash
python icp.py
```

The demo builds a 10 x 6 m room with a box obstacle, creates a noisy scan of it from a pose rotated by 8° and shifted by (0.4, -0.3) m, and lets ICP recover that pose.

Expected output:

```
true  angle [deg]: 8.0
est.  angle [deg]: 8.0069...
true  t: [ 0.4 -0.3]
est.  t: [ 0.4002 -0.2995]
mean point error [m]: 0.0128
```

If Matplotlib is installed, a plot shows the map (gray), the scan before ICP (red) and the scan after ICP (green).

## Code overview

| Function | Purpose |
|---|---|
| `best_fit_transform(A, B)` | Least-squares `R, t` for paired points (Kabsch / SVD) |
| `nearest_neighbors(src, dst)` | Distance and index of the closest `dst` point for each `src` point |
| `icp(scan, ref)` | The main loop; returns total `R, t`, the aligned scan and the final error |
| `make_room()` | Builds the demo reference map |

Points are stored as `N x 2` NumPy arrays, one row per point.

## Extending it

The algorithm is split into two replaceable parts, **matching** (`nearest_neighbors`) and **transform estimation** (`best_fit_transform`). Other scan-alignment methods usually change one or both of these, and the loop in `icp()` stays the same.

### Example 1: faster matching with a k-d tree

Brute-force matching compares every pair of points. For real scans with thousands of points, swap in a k-d tree. It has the same inputs and outputs, so nothing else changes:

```python
from scipy.spatial import cKDTree

def nearest_neighbors(src, dst):
    dist, idx = cKDTree(dst).query(src)
    return dist, idx
```

### Example 2: reject outlier matches

Moving objects and noise create bad pairs. Ignore pairs that are too far apart by changing two lines in `icp()`:

```python
dist, idx = nearest_neighbors(src, ref)
keep = dist < 0.5                                    # metres
R, t = best_fit_transform(src[keep], ref[idx[keep]])
```

### Example 3: point-to-line (point-to-plane) ICP, i.e. line fitting

Point-to-point ICP matches a scan point to one map point. On walls this is wasteful: a scan point can slide along the wall without the error changing. **Point-to-line** ICP instead measures the distance from the scan point to the *line through the matched map points*, so only the distance perpendicular to the wall counts. This usually converges faster and more accurately in rooms and corridors.

To implement it:

1. **Matching:** for each matched map point, fit a local line to its nearest map neighbours (for example with PCA) and store the line's unit normal `n`.
2. **Transform estimation:** replace `best_fit_transform` with a solver that minimizes the perpendicular distance
   `sum( (n_i . (R p_i + t - q_i))^2 )`.
   For small angles this is a linear least-squares problem in `(theta, tx, ty)`, solved with `np.linalg.lstsq`.
3. Keep the `icp()` loop as it is.

### Other methods that fit the same structure

| Method | What changes |
|---|---|
| **Generalized ICP (GICP)** | Both clouds get local covariances; the error is weighted by them |
| **NDT** | The map is a grid of Gaussians; matching is replaced by a likelihood score |
| **Feature-based matching (LOAM-style)** | Match only edge and plane features instead of all points |
| **Trimmed / robust ICP** | Keep only the best X% of pairs, or down-weight large errors |
| **Downsampling** | Voxel-filter the scan before matching to reduce cost |

## Limitations

- ICP converges to a *local* minimum, so it needs a reasonable initial guess (for example from wheel odometry or an IMU).
- It struggles in featureless environments (a long straight corridor gives no constraint along its axis).
- Dynamic objects and heavy noise degrade the result without outlier rejection.
- The brute-force matcher is only suitable for small point counts.

## License

Add a license of your choice (for example MIT).
