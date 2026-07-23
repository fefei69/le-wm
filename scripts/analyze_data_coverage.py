#!/usr/bin/env python3
"""Data-coverage analysis for the PushBox pilot dataset.

Produces the state/action coverage diagnostics described in "Hallucination in
World Models is Predictable and Preventable" (Fig. 4/6: state density of key
agent/object positions) and the QA coverage checklist in DATASET_SPEC.md par.8:

- box-position density (the object).  The merged pilot file has all-NaN
  ``state[:, 2:6]`` (AprilTag pose was never recorded), so the box is tracked
  directly from ``pixels`` by segmenting the red patch on its top face and
  taking the largest connected component.  Positions are therefore in *image*
  coordinates (px), which is sufficient for density/coverage maps.
- end-effector position density (the agent), in the table frame from proprio.
- visual end-effector/pusher-tip density in image pixels.  The pusher has a
  small red tip and a fixed orientation.  Candidate red components are
  measured from pixels, then associated with the smooth proprioceptive motion
  to reject the box patch and fixed red hardware in the background.  Missing
  visual detections remain NaN; the proprioceptive prior is never substituted
  for a missing image measurement.
- action-space density, push-direction and patch-orientation histograms,
  per-episode box travel, contact fraction, and bin-occupancy statistics.

Tracking is cached to ``box_track.npz`` in the output directory, so reruns
only redo the plots.  Usage::

    python scripts/analyze_data_coverage.py [--dataset data/pushbox_pilot_train.h5]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-matplotlib")

import h5py
import matplotlib
import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.linear_model import LinearRegression, RANSACRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]

# Red-patch segmentation on int16 RGB; the patch is 150-400 px, robot speckles
# are far smaller, so the largest connected component above MIN_BLOB_PX wins.
# The strict tier rejects the robot's red parts; the loose tier only runs when
# the strict one finds nothing, recovering the washed-out patch in the far
# (pale) region of the table without letting arm pixels win elsewhere.
TIERS = (  # (r_min, margin, min_blob_px)
    (100, 50, 60),
    (90, 35, 40),
)
CONTACT_SPEED_PX = 1.0  # per-step centroid motion counted as "box moving"

# Visual EE proxy: centroid of the small red pusher tip.  The permissive red
# threshold is needed when the tip is shadowed.  Shape and dark-neighbourhood
# gates distinguish it from the much larger box patch and red table speckles.
EE_RED_MIN = 20
EE_RED_GREEN_MARGIN = 8
EE_RED_BLUE_MARGIN = 6
EE_MIN_BLOB_PX = 3
EE_MAX_BLOB_PX = 100
EE_MAX_BLOB_WIDTH = 15
EE_MAX_BLOB_HEIGHT = 25
EE_DARK_MAX = 70
EE_MIN_DARK_CONTEXT = 0.18
EE_ASSOCIATION_RADIUS_PX = 10.0
EE_TRACKER_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/pushbox_pilot_train.h5")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: outputs/coverage/<dataset stem>")
    parser.add_argument("--chunk", type=int, default=1024, help="frames per HDF5 read while tracking")
    parser.add_argument("--bins", type=int, default=32, help="bins per axis for occupancy statistics")
    parser.add_argument("--retrack-ee", action="store_true", help="ignore a cached ee_track.npz")
    return parser.parse_args()


def track_box(pixels: h5py.Dataset, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame patch centroid (u, v), blob area, and orientation (deg, mod 180)."""
    n = pixels.shape[0]
    centroids = np.full((n, 2), np.nan, np.float32)
    areas = np.zeros(n, np.int32)
    angles = np.full(n, np.nan, np.float32)
    for start in range(0, n, chunk):
        imgs = pixels[start : start + chunk].astype(np.int16)
        r, g, b = imgs[..., 0], imgs[..., 1], imgs[..., 2]
        for j in range(len(imgs)):
            for r_min, margin, min_blob in TIERS:
                mask = (r[j] > r_min) & (r[j] - g[j] > margin) & (r[j] - b[j] > margin)
                labels, count = ndimage.label(mask)
                if count == 0:
                    continue
                sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, count + 1))
                best = int(np.argmax(sizes)) + 1
                if sizes[best - 1] < min_blob:
                    continue
                ys, xs = np.nonzero(labels == best)
                centroids[start + j] = (xs.mean(), ys.mean())
                areas[start + j] = len(xs)
                cov = np.cov(np.stack([xs, ys]).astype(np.float64))
                evals, evecs = np.linalg.eigh(cov)
                major = evecs[:, np.argmax(evals)]
                angles[start + j] = np.degrees(np.arctan2(major[1], major[0])) % 180.0
                break
        if (start // chunk) % 10 == 0:
            print(f"  tracked {min(start + chunk, n)}/{n} frames", flush=True)
    return centroids, areas, angles


def gripper_tip_candidates(image: np.ndarray) -> np.ndarray:
    """Return candidate red pusher-tip blobs as rows ``(u, v, area)``.

    The pusher tip is red, narrow, and surrounded by the dark gripper.  This
    function is deliberately image-only.  Candidate association happens in
    :func:`track_gripper_tip`, where measured EE motion rejects otherwise
    plausible red fixtures and reflections.
    """
    image_i16 = image.astype(np.int16, copy=False)
    r, g, b = image_i16[..., 0], image_i16[..., 1], image_i16[..., 2]
    red = (
        (r >= EE_RED_MIN)
        & (r - g >= EE_RED_GREEN_MARGIN)
        & (r - b >= EE_RED_BLUE_MARGIN)
    ).astype(np.uint8)
    dark = image_i16.mean(axis=2) < EE_DARK_MAX
    count, _, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    candidates: list[tuple[float, float, float]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        u, v = centroids[label]
        if not (
            EE_MIN_BLOB_PX <= area <= EE_MAX_BLOB_PX
            and width <= EE_MAX_BLOB_WIDTH
            and height <= EE_MAX_BLOB_HEIGHT
            and 20 <= u <= 205
            and 48 <= v <= 130
        ):
            continue
        x0, x1 = max(0, x - 10), min(image.shape[1], x + width + 10)
        y0, y1 = max(0, y - 10), min(image.shape[0], y + height + 10)
        if dark[y0:y1, x0:x1].mean() < EE_MIN_DARK_CONTEXT:
            continue
        candidates.append((float(u), float(v), float(area)))
    return np.asarray(candidates, dtype=np.float32).reshape(-1, 3)


def _camera_features(xy: np.ndarray) -> np.ndarray:
    """Low-order planar camera features for associating a visual candidate."""
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack((x, y, x * y, x * x, y * y))


def track_gripper_tip(
    pixels: h5py.Dataset,
    proprio_xy: np.ndarray,
    slices: list[slice],
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track the red pusher-tip centroid in pixels.

    Pixel segmentation supplies every measurement.  Proprioception supplies a
    smooth camera-space association prior only: it selects among red candidate
    components, but is not emitted as a detection when no image candidate is
    close enough.  A per-episode translation absorbs small camera shifts among
    the source recordings.
    """
    n = pixels.shape[0]
    candidates: list[np.ndarray] = []
    raw = np.full((n, 2), np.nan, np.float32)
    for start in range(0, n, chunk):
        images = pixels[start : start + chunk]
        for j, image in enumerate(images):
            frame_candidates = gripper_tip_candidates(image)
            candidates.append(frame_candidates)
            if len(frame_candidates):
                # The tip is normally the lowest plausible red component.  A
                # robust camera fit below removes occasional false choices.
                raw[start + j] = frame_candidates[np.argmax(frame_candidates[:, 1]), :2]
        if (start // chunk) % 10 == 0:
            print(f"  found EE candidates in {min(start + chunk, n)}/{n} frames", flush=True)

    valid_raw = ~np.isnan(raw[:, 0])
    if valid_raw.sum() < 20:
        raise RuntimeError("too few visual gripper-tip candidates to calibrate association")
    features = _camera_features(proprio_xy)
    regressor = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.10,
        residual_threshold=5.0,
        max_trials=1000,
        random_state=0,
    )
    regressor.fit(features[valid_raw], raw[valid_raw])
    base_prior = regressor.predict(features).astype(np.float32)

    centroids = np.full((n, 2), np.nan, np.float32)
    areas = np.zeros(n, np.int32)
    priors = np.full((n, 2), np.nan, np.float32)
    residuals = np.full(n, np.nan, np.float32)

    # Two association passes let the first selected track refine each
    # episode's camera offset without ever filling a missing visual point.
    offset_source = raw
    for _ in range(2):
        centroids.fill(np.nan)
        areas.fill(0)
        priors.fill(np.nan)
        residuals.fill(np.nan)
        for episode_slice in slices:
            source = offset_source[episode_slice]
            source_ok = ~np.isnan(source[:, 0])
            if source_ok.any():
                offset = np.median(
                    source[source_ok] - base_prior[episode_slice][source_ok], axis=0
                )
            else:
                offset = np.zeros(2, dtype=np.float32)
            for index in range(episode_slice.start, episode_slice.stop):
                prior = base_prior[index] + offset
                priors[index] = prior
                frame_candidates = candidates[index]
                if not len(frame_candidates):
                    continue
                distances = np.linalg.norm(frame_candidates[:, :2] - prior, axis=1)
                best = int(np.argmin(distances))
                if distances[best] > EE_ASSOCIATION_RADIUS_PX:
                    continue
                centroids[index] = frame_candidates[best, :2]
                areas[index] = int(frame_candidates[best, 2])
                residuals[index] = float(distances[best])
        offset_source = centroids.copy()
    return centroids, areas, priors, residuals


def save_ee_track_check(
    pixels: h5py.Dataset,
    centroids: np.ndarray,
    areas: np.ndarray,
    priors: np.ndarray,
    residuals: np.ndarray,
    path: Path,
) -> None:
    """Save a deterministic montage of visual EE detections and hard cases."""
    detected = ~np.isnan(centroids[:, 0])
    rng = np.random.default_rng(0)
    groups: list[tuple[str, np.ndarray]] = []
    valid_indices = np.flatnonzero(detected)
    if len(valid_indices):
        groups.append(("random", rng.choice(valid_indices, min(8, len(valid_indices)), replace=False)))
        groups.append(("small blobs", valid_indices[np.argsort(areas[valid_indices])[:8]]))
        extreme_order = np.unique(np.concatenate([
            valid_indices[np.argsort(centroids[valid_indices, 0])[:2]],
            valid_indices[np.argsort(centroids[valid_indices, 0])[-2:]],
            valid_indices[np.argsort(centroids[valid_indices, 1])[:2]],
            valid_indices[np.argsort(centroids[valid_indices, 1])[-2:]],
        ]))
        groups.append(("workspace extremes", extreme_order[:8]))
        groups.append(("largest association residual", valid_indices[np.argsort(residuals[valid_indices])[-8:]]))
    missed_indices = np.flatnonzero(~detected)
    if len(missed_indices):
        groups.append(("misses (cyan = prior only)", rng.choice(missed_indices, min(8, len(missed_indices)), replace=False)))

    cols = 8
    fig, axes = plt.subplots(len(groups), cols, figsize=(16, 2.35 * len(groups)), squeeze=False)
    for row, (label, indices) in enumerate(groups):
        for col, ax in enumerate(axes[row]):
            ax.set_axis_off()
            if col >= len(indices):
                continue
            index = int(indices[col])
            ax.imshow(pixels[index])
            pu, pv = priors[index]
            ax.plot(pu, pv, "o", ms=6, mfc="none", mec="#28d7e5", mew=1.2)
            if detected[index]:
                u, v = centroids[index]
                ax.plot(u, v, "+", ms=10, mew=1.6, color="#fff000")
                title = f"{index} a={areas[index]} r={residuals[index]:.1f}"
            else:
                title = f"{index} MISS"
            ax.set_title(title, fontsize=7)
        axes[row, 0].text(
            -0.04, 0.5, label, transform=axes[row, 0].transAxes,
            ha="right", va="center", rotation=90, fontsize=8,
        )
    fig.suptitle("Visual EE / red pusher-tip tracking (yellow = measured centroid)", fontsize=12)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def episode_slices(ep_offset: np.ndarray, ep_len: np.ndarray) -> list[slice]:
    return [slice(int(o), int(o + l)) for o, l in zip(ep_offset, ep_len)]


def occupancy(points: np.ndarray, bins: int, extent: tuple[float, float, float, float]) -> tuple[np.ndarray, float]:
    """2D histogram over a fixed extent and the fraction of bins visited."""
    hist, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=bins, range=[extent[:2], extent[2:]]
    )
    return hist, float((hist > 0).mean())


def sector_histogram(angles_rad: np.ndarray, sectors: int = 8) -> np.ndarray:
    edges = np.linspace(-np.pi, np.pi, sectors + 1)
    hist, _ = np.histogram(angles_rad, bins=edges)
    return hist


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or REPO_ROOT / "outputs/coverage" / args.dataset.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.dataset, "r")
    ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
    action, proprio, state = f["action"][:], f["proprio"][:], f["state"][:]
    n_steps, n_eps = len(action), len(ep_len)
    slices = episode_slices(ep_offset, ep_len)

    cache = out_dir / "box_track.npz"
    if cache.exists():
        data = np.load(cache)
        centroids, areas, angles = data["centroids"], data["areas"], data["angles"]
        print(f"loaded cached box track from {cache}")
    else:
        print("tracking box patch across all frames...")
        centroids, areas, angles = track_box(f["pixels"], args.chunk)
        np.savez_compressed(cache, centroids=centroids, areas=areas, angles=angles)

    detected = ~np.isnan(centroids[:, 0])
    ee_m = proprio[:, :2]

    ee_cache = out_dir / "ee_track.npz"
    use_ee_cache = False
    if ee_cache.exists() and not args.retrack_ee:
        ee_data = np.load(ee_cache)
        version = int(ee_data["tracker_version"]) if "tracker_version" in ee_data else 0
        use_ee_cache = version == EE_TRACKER_VERSION
    if use_ee_cache:
        ee_centroids = ee_data["centroids"]
        ee_areas = ee_data["areas"]
        ee_priors = ee_data["priors"]
        ee_residuals = ee_data["residuals"]
        print(f"loaded cached visual EE track from {ee_cache}")
    else:
        print("tracking visual EE / red pusher tip across all frames...")
        ee_centroids, ee_areas, ee_priors, ee_residuals = track_gripper_tip(
            f["pixels"], ee_m, slices, args.chunk
        )
        np.savez_compressed(
            ee_cache,
            tracker_version=np.asarray(EE_TRACKER_VERSION),
            centroids=ee_centroids,
            areas=ee_areas,
            priors=ee_priors,
            residuals=ee_residuals,
        )
    ee_detected = ~np.isnan(ee_centroids[:, 0])
    save_ee_track_check(
        f["pixels"], ee_centroids, ee_areas, ee_priors, ee_residuals,
        out_dir / "ee_track_check.png",
    )

    # --- per-episode box statistics (in image px) ---
    ep_travel, ep_start, ep_detect_rate, moving = [], [], [], np.zeros(n_steps, bool)
    for s in slices:
        c = centroids[s]
        ok = ~np.isnan(c[:, 0])
        ep_detect_rate.append(ok.mean())
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            ep_travel.append(0.0)
            continue
        ep_start.append(c[idx[0]])
        steps = np.linalg.norm(np.diff(c[idx], axis=0), axis=1)
        ep_travel.append(steps.sum())
        moving_local = np.zeros(s.stop - s.start, bool)
        moving_local[idx[1:]] = steps > CONTACT_SPEED_PX
        moving[s] = moving_local
    ep_travel = np.asarray(ep_travel)
    ep_start = np.asarray(ep_start)
    ep_detect_rate = np.asarray(ep_detect_rate)

    # push directions: image-frame direction of box motion while moving
    diffs = np.diff(centroids, axis=0)
    valid_diff = ~np.isnan(diffs[:, 0]) & moving[1:]
    push_dirs = np.arctan2(-diffs[valid_diff, 1], diffs[valid_diff, 0])  # image y points down

    # --- occupancy stats ---
    box_pts = centroids[detected]
    box_extent = (0.0, 224.0, 0.0, 224.0)
    box_roi = (
        float(box_pts[:, 0].min()), float(box_pts[:, 0].max()),
        float(box_pts[:, 1].min()), float(box_pts[:, 1].max()),
    )
    _, box_occ_roi = occupancy(box_pts, args.bins, box_roi)
    ee_m_roi = (
        float(ee_m[:, 0].min()), float(ee_m[:, 0].max()),
        float(ee_m[:, 1].min()), float(ee_m[:, 1].max()),
    )
    _, ee_m_occ = occupancy(ee_m, args.bins, ee_m_roi)
    ee_px = ee_centroids[ee_detected]
    ee_px_roi = (
        float(ee_px[:, 0].min()), float(ee_px[:, 0].max()),
        float(ee_px[:, 1].min()), float(ee_px[:, 1].max()),
    )
    _, ee_px_occ = occupancy(ee_px, args.bins, ee_px_roi)
    act_norm = np.linalg.norm(action, axis=1)
    act_cap = float(act_norm.max())
    _, act_occ = occupancy(action, args.bins, (-act_cap, act_cap, -act_cap, act_cap))
    uniq_actions, uniq_counts = np.unique(np.round(action, 5), axis=0, return_counts=True)
    top15 = float(np.sort(uniq_counts)[-15:].sum() / len(action))

    push_hist = sector_histogram(push_dirs)
    angle_hist, _ = np.histogram(angles[detected], bins=12, range=(0, 180))

    # alignment QA (DATASET_SPEC par.8.1): proprio delta vs action, contact-free steps
    d_ee = np.diff(ee_m, axis=0)
    same_ep = np.ones(n_steps - 1, bool)
    same_ep[ep_offset[1:] - 1] = False
    free = same_ep & ~moving[1:]
    align = [float(np.corrcoef(d_ee[free][:, i], action[:-1][free][:, i])[0, 1]) for i in range(2)]

    d_ee_px = np.diff(ee_centroids, axis=0)
    ee_valid_diff = same_ep & ee_detected[:-1] & ee_detected[1:]
    ee_stationary = ee_valid_diff & (np.linalg.norm(d_ee, axis=1) < 0.0005)
    ee_stationary_jitter = np.linalg.norm(d_ee_px[ee_stationary], axis=1)
    ee_moving = ee_valid_diff & (np.linalg.norm(d_ee, axis=1) >= 0.0005)
    ee_motion_align = (
        float(np.corrcoef(d_ee_px[ee_moving, 0], d_ee[ee_moving, 1])[0, 1]),
        float(np.corrcoef(d_ee_px[ee_moving, 1], d_ee[ee_moving, 0])[0, 1]),
    )

    dt = np.diff(f["image_timestamp_ns"][:]) / 1e9
    dt = dt[same_ep]

    report = [
        f"dataset: {args.dataset}",
        f"episodes: {n_eps}   steps: {n_steps}   hours @5Hz: {n_steps / 5 / 3600:.2f}",
        f"episode length min/median/max: {ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}",
        "",
        "state columns: box_x/box_y/cos/sin are 100% NaN in this file -> box tracked from pixels",
        f"box patch detection rate: {detected.mean():.1%} of frames "
        f"({(ep_detect_rate < 0.95).sum()} episodes below the 95% QA bar)",
        "",
        f"box occupancy: {box_occ_roi:.1%} of {args.bins}x{args.bins} bins over its visited "
        f"bounding box u[{box_roi[0]:.0f},{box_roi[1]:.0f}] v[{box_roi[2]:.0f},{box_roi[3]:.0f}] px "
        f"(image is 224x224)",
        f"box travel per episode (px): median {np.median(ep_travel):.0f}, "
        f"p10 {np.percentile(ep_travel, 10):.0f}, p90 {np.percentile(ep_travel, 90):.0f}; "
        f"{(ep_travel < 20).sum()} episodes moved the box < 20 px total",
        f"box-moving (contact) steps: {moving.mean():.1%} of all steps "
        "(spec wants 80-90% engagement, 10-20% contact-free)",
        f"push directions (8 sectors, counts): {push_hist.tolist()}  "
        f"min/max sector ratio: {push_hist.min() / max(push_hist.max(), 1):.2f}",
        "  (image-frame caveat: the oblique camera compresses toward/away motion, so",
        "   vertical sectors are undercounted relative to lateral ones)",
        f"patch orientation mod 180 deg (12 bins): min/max bin ratio "
        f"{angle_hist.min() / max(angle_hist.max(), 1):.2f} (1.0 = uniform yaw proxy)",
        "",
        f"visual EE (red pusher-tip) detection rate: {ee_detected.mean():.1%} of frames",
        f"visual EE occupancy: {ee_px_occ:.1%} of {args.bins}x{args.bins} bins over its visited "
        f"bounding box u[{ee_px_roi[0]:.0f},{ee_px_roi[1]:.0f}] "
        f"v[{ee_px_roi[2]:.0f},{ee_px_roi[3]:.0f}] px",
        f"visual EE association residual (px): median {np.nanmedian(ee_residuals):.2f}, "
        f"p95 {np.nanpercentile(ee_residuals, 95):.2f}; "
        f"stationary jitter median {np.median(ee_stationary_jitter):.2f}, "
        f"p95 {np.percentile(ee_stationary_jitter, 95):.2f}",
        f"visual EE motion alignment corr: du vs dy {ee_motion_align[0]:.3f}, "
        f"dv vs dx {ee_motion_align[1]:.3f}",
        "  (centroids are image measurements; proprio is used only to associate candidate blobs)",
        f"proprio EE occupancy: {ee_m_occ:.1%} of bins over x[{ee_m_roi[0]:.3f},{ee_m_roi[1]:.3f}] "
        f"y[{ee_m_roi[2]:.3f},{ee_m_roi[3]:.3f}] m",
        f"action occupancy: {act_occ:.1%} of bins over [-{act_cap:.3f},{act_cap:.3f}] m square; "
        f"|a| mean {act_norm.mean():.4f} max {act_cap:.4f}; zero-action steps {np.mean(act_norm < 1e-6):.1%}",
        f"action discreteness: {len(uniq_actions)} unique vectors; top-15 cover {top15:.1%} of steps "
        "(keyboard teleop -> 8-direction star, not a continuous distribution)",
        f"alignment corr(d_proprio, action) on contact-free steps: x {align[0]:.3f}, y {align[1]:.3f} "
        "(QA bar: > 0.9)",
        f"tick timing within episodes: mean {dt.mean():.3f}s std {dt.std():.3f}s, "
        f"{(dt > 0.3).sum()} gaps > 0.3s (QA bar: 0.20 +/- 0.02, no gaps)",
    ]
    text = "\n".join(report)
    (out_dir / "coverage_report.txt").write_text(text + "\n")
    print("\n" + text + "\n")

    # --- figure -------------------------------------------------------------
    ink, muted, accent = "#39485e", "#7a8699", "#3d6bb0"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.edgecolor": muted, "axes.labelcolor": ink,
        "text.color": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    fig.suptitle(f"Data coverage: {args.dataset.name} ({n_eps} eps / {n_steps} steps)", color=ink)

    ax = axes[0, 0]
    ax.imshow(f["pixels"][0])
    ax.plot(box_pts[::25, 0], box_pts[::25, 1], ".", color="#79a4dd", ms=1, alpha=0.25)
    ax.set_title("sample frame + box centroid trace")
    ax.set_axis_off()

    ax = axes[0, 1]
    h = ax.hist2d(box_pts[:, 0], box_pts[:, 1], bins=64, range=[box_extent[:2], box_extent[2:]],
                  cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax.plot(ep_start[:, 0], ep_start[:, 1], "o", mfc="none", mec="#c56a4c", ms=4, mew=0.8,
            label="episode starts")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("box position density (image px, log)")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.colorbar(h[3], ax=ax, fraction=0.046)

    ax = axes[0, 2]
    h = ax.hist2d(ee_px[:, 0], ee_px[:, 1], bins=64, range=[(0, 224), (0, 224)],
                  cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("visual EE / pusher-tip density (image px, log)")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    fig.colorbar(h[3], ax=ax, fraction=0.046)

    ax = axes[1, 0]
    h = ax.hist2d(action[:, 0], action[:, 1], bins=48,
                  range=[(-act_cap, act_cap), (-act_cap, act_cap)],
                  cmap="Blues", norm=matplotlib.colors.LogNorm())
    ax.set_aspect("equal")
    ax.set_title("action density (m/step, log)")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    fig.colorbar(h[3], ax=ax, fraction=0.046)

    fig.delaxes(axes[1, 1])
    ax = fig.add_subplot(2, 3, 5, projection="polar")
    centers = np.linspace(-np.pi, np.pi, len(push_hist), endpoint=False) + np.pi / len(push_hist)
    ax.bar(centers, push_hist, width=2 * np.pi / len(push_hist) * 0.92, color=accent, alpha=0.85)
    ax.set_title("box push directions (image frame)", pad=14)
    ax.tick_params(labelsize=7)

    ax = axes[1, 2]
    ax.hist(ep_travel, bins=30, color=accent, alpha=0.85)
    ax.axvline(20, color="#c56a4c", lw=1, ls="--")
    ax.set_title("box travel per episode (px)")
    ax.set_xlabel("total centroid path length")
    ax.set_ylabel("episodes")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = out_dir / "coverage.png"
    fig.savefig(fig_path, dpi=160)
    print(f"figure: {fig_path}\nreport: {out_dir / 'coverage_report.txt'}")


if __name__ == "__main__":
    main()
