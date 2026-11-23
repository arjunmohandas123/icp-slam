"""Simple 2D point-to-point ICP (numpy only).

Idea: a LiDAR scan is a set of 2D points. ICP finds the rotation R and
translation t that move the scan onto a reference (map / previous scan).
That transformation is the robot's pose change.
"""
import numpy as np


def best_fit_transform(A, B):
    """Least-squares R, t such that R @ A_i + t ~= B_i (SVD / Kabsch method)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    H = (A - ca).T @ (B - cb)          # covariance of the centered point sets
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:           # avoid a reflection
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def nearest_neighbors(src, dst):
    """For every src point, find the closest dst point (brute force)."""
    d = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return d[np.arange(len(src)), idx], idx


def icp(scan, ref, max_iter=50, tol=1e-6):
    """Align `scan` to `ref`. Returns total R, t, aligned scan, final error."""
    src = scan.copy()
    R_total, t_total = np.eye(2), np.zeros(2)
    prev_err = np.inf

    for i in range(max_iter):
        # 1) match each scan point to its closest reference point
        dist, idx = nearest_neighbors(src, ref)
        # 2) best transform for these pairs
        R, t = best_fit_transform(src, ref[idx])
        # 3) apply it and accumulate the total transform
        src = src @ R.T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        # 4) stop when the error no longer improves
        err = dist.mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return R_total, t_total, src, err


def make_room():
    """Reference map: a 10 x 6 m room with a box obstacle (points along walls)."""
    def line(p, q, n):
        return np.linspace(p, q, n)
    pts = [
        line([0, 0], [10, 0], 100), line([10, 0], [10, 6], 60),
        line([10, 6], [0, 6], 100), line([0, 6], [0, 0], 60),
        line([4, 0], [4, 2], 20), line([4, 2], [6, 2], 20),  # obstacle
    ]
    return np.vstack(pts)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = make_room()

    # Ground truth: the scan is the map seen from a pose that is rotated 8 deg
    # and shifted by (0.4, -0.3). ICP should recover this transform.
    theta = np.deg2rad(8)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    t_true = np.array([0.4, -0.3])

    sample = ref[rng.choice(len(ref), 150, replace=False)]
    scan = (sample - t_true) @ R_true            # = R_true.T @ (p - t_true)
    scan += rng.normal(0, 0.01, scan.shape)      # sensor noise

    R, t, aligned, err = icp(scan, ref)

    print("true  angle [deg]:", np.rad2deg(theta))
    print("est.  angle [deg]:", np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
    print("true  t:", t_true)
    print("est.  t:", t)
    print("mean point error [m]:", err)

    try:
        import matplotlib.pyplot as plt
        plt.scatter(*ref.T, s=6, c="gray", label="reference (map)")
        plt.scatter(*scan.T, s=10, c="red", label="scan before ICP")
        plt.scatter(*aligned.T, s=10, c="green", label="scan after ICP")
        plt.axis("equal"); plt.legend(); plt.show()
    except ImportError:
        pass
