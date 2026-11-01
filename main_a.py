import numpy as np
import pr3_utils as ut
from scipy.linalg import expm
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path

def circle_dot(s):

    x, y, z = s[0], s[1], s[2]
    
    # Construct -s_hat (skew-symmetric form of s)
    neg_s_hat = np.array([
        [ 0,  z, -y],
        [-z,  0,  x],
        [ y, -x,  0]
    ])
    
    # Assemble the 4x6 matrix
    op = np.zeros((4, 6))
    op[:3, :3] = np.eye(3)
    op[:3, 3:] = neg_s_hat
    
    return op

def _extract_dataset_index(path_obj):
    """Extract dataset index from filename stem (e.g., dataset02 -> 2)."""
    match = re.search(r"(\d+)$", path_obj.stem)
    return int(match.group(1)) if match else -1


def _discover_dataset_files(data_root):
    """Recursively find all dataset*.npy files under data_root."""
    dataset_files = list(data_root.rglob("dataset*.npy"))
    return sorted(dataset_files, key=lambda p: (_extract_dataset_index(p), str(p)))


def resolve_data_file(filename_arg=None, dataset_arg=None):
    """
    Resolve the input data file:
    1) Direct filename (positional arg) or --filename option
    2) --dataset by index number
    3) If neither, auto-select highest-numbered available dataset
    """
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"

    if filename_arg:
        candidate = Path(filename_arg).expanduser()
        if not candidate.is_absolute():
            cwd_candidate = (Path.cwd() / candidate).resolve()
            project_candidate = (project_root / candidate).resolve()
            if cwd_candidate.exists():
                return cwd_candidate
            if project_candidate.exists():
                return project_candidate
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Specified file does not exist: {filename_arg}")

    if not data_root.exists():
        raise FileNotFoundError(f"Could not find data directory: {data_root}")

    dataset_files = _discover_dataset_files(data_root)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset*.npy files found in {data_root}")

    if dataset_arg is not None:
        target = [p for p in dataset_files if _extract_dataset_index(p) == dataset_arg]
        if not target:
            raise FileNotFoundError(
                f"Could not find dataset {dataset_arg}; available files:\n"
                + "\n".join(str(p) for p in dataset_files)
            )
        return target[-1]

    # Default: auto-select highest-numbered dataset
    return dataset_files[-1]


def build_figure_paths(project_root, data_file):
    """Build output figure paths for IMU-only and landmark mapping visualizations."""
    data_path = Path(data_file)
    dataset_idx = _extract_dataset_index(data_path)
    dataset_var = f"dataset{dataset_idx:02d}" if dataset_idx >= 0 else data_path.stem

    figure_dir = project_root / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    imu_only_path = figure_dir / f"imu_ekf_only_{dataset_var}.png"
    mapping_path = figure_dir / f"Landmark_mapping_via_EKF_update_dataset_{dataset_var}.png"
    return imu_only_path, mapping_path, dataset_var

def mapping_update(landmarks, lm_cov, pose, features_t, K_S, o_T_i, V):
    """
    Decoupled update for Part (b): update landmarks only, keep pose fixed.
    landmarks: (3, M)
    lm_cov: (M, 3, 3) landmark covariance matrices
    """
    visible_mask = features_t[0, :] != -1
    indices = np.where(visible_mask)[0]
    
    # Performance optimization: subsample if too many observations
    if len(indices) > 50:
        indices = np.random.choice(indices, 50, replace=False)
    
    T_iw = ut.inversePose(pose.reshape(1,4,4))[0]
    T_cw = o_T_i @ T_iw
    
    # Key fix: auxiliary matrix P_matrix (4x3) to extract jacobian w.r.t. (x, y, z)
    P_matrix = np.vstack((np.eye(3), np.zeros((1, 3)))) 
    
    for idx in indices:
        m_w = np.append(landmarks[:, idx], 1)
        q_c = T_cw @ m_w
        
        if q_c[2] < 0.1: 
            continue 
        
        # Predicted observation z_hat = K @ q_c/q_c[2]
        z_hat = K_S @ (q_c / q_c[2])
        innovation = features_t[:, idx] - z_hat
        
        # Jacobian H (4x3)
        dz_dq = ut.projectionJacobian(q_c) # (4x4) matrix
        # Chain rule: K_S(4x4) @ dz_dq(4x4) @ T_cw(4x4) @ P_matrix(4x3) -> (4x3)
        H = K_S @ dz_dq @ T_cw @ P_matrix 
        
        # Standard EKF Update
        S = H @ lm_cov[idx] @ H.T + V
        K = lm_cov[idx] @ H.T @ np.linalg.inv(S)
        
        landmarks[:, idx] += K @ innovation
        lm_cov[idx] = (np.eye(3) - K @ H) @ lm_cov[idx]
        
    return landmarks, lm_cov

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECE276A Visual-Inertial-SLAM")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Data file path (positional argument).",
    )
    parser.add_argument(
        "--filename",
        dest="filename_option",
        type=str,
        default=None,
        help="Data file path (legacy, kept for compatibility).",
    )
    parser.add_argument(
        "--dataset",
        type=int,
        default=1,
        help="Dataset index (e.g., 0, 1, 2).",
    )
    args = parser.parse_args()

    # 1) Load the measurements
    selected_filename = args.input_file if args.input_file is not None else args.filename_option
    data_file = resolve_data_file(filename_arg=selected_filename, dataset_arg=args.dataset)
    print(f"Using dataset file: {data_file}")
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = ut.load_data(str(data_file))

    # --- (a) IMU Localization initialization ---
    num_steps = v_t.shape[0]
    mu = np.eye(4) 
    sigma = np.eye(6) * 0.1 
    W = np.eye(6) * 0.1 
    
    all_poses = np.zeros((num_steps, 4, 4))
    all_poses[0] = mu
    
    # --- (b) Landmark Mapping initialization ---
    num_landmarks = features.shape[1]
    landmarks = np.zeros((3, num_landmarks))
    lm_cov = np.tile(np.eye(3) * 1.0, (num_landmarks, 1, 1))
    observed = np.zeros(num_landmarks, dtype=bool) 

    # Prepare camera parameters
    baseline = np.linalg.norm(extL_T_imu[:3, 3] - extR_T_imu[:3, 3])
    f_su, f_sv, cu, cv = K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2]
    K_S = np.array([[f_su, 0, cu, 0],
                    [0, f_sv, cv, 0],
                    [f_su, 0, cu, -f_su * baseline],
                    [0, f_sv, cv, 0]])
    
    # Optical coordinate system transformation
    c_T_i = np.linalg.inv(extL_T_imu) 
    o_T_c = np.array([[ 0, -1,  0, 0], 
                  [ 0,  0, -1, 0], 
                  [ 1,  0,  0, 0], 
                  [ 0,  0,  0, 1]])
    o_T_i = o_T_c @ c_T_i  
    i_T_o = np.linalg.inv(o_T_i)
    
    V = np.eye(4) * 1.0 # Observation noise

    print("Starting Decoupled Mapping...")

    for i in range(num_steps - 1):
        tau = timestamps[i+1] - timestamps[i]
        v, w = v_t[i], w_t[i]
        
        # --- 1) State prediction (IMU update only) ---
        zeta_vec = np.hstack([v, w])
        zeta_hat = ut.axangle2twist(zeta_vec)
        
        mu = mu @ expm(tau * zeta_hat)
        
        ad_zeta = ut.axangle2adtwist(zeta_vec)
        Phi = expm(-tau * ad_zeta)
        sigma = Phi @ sigma @ Phi.T + W
        
        all_poses[i+1] = mu
        features_t = features[:, :, i+1]
        
        # --- 2) Process new landmarks (initialization) ---
        new_idx = np.where((features_t[0, :] != -1) & (~observed))[0]
        for idx in new_idx:
            z = features_t[:, idx]
            disparity = z[0] - z[2] # Avoid division by near-zero
            if disparity < 0.0001: 
                continue
            depth = (f_su * baseline) / disparity
            p_c = np.array([(z[0]-cu)*depth/f_su, (z[1]-cv)*depth/f_sv, depth, 1])
            p_w = mu @ i_T_o @ p_c # mu = T_world_imu
            landmarks[:, idx] = p_w[:3]
            observed[idx] = True

        # --- 3) Landmark update (call mapping_update function) ---
        landmarks, lm_cov = mapping_update(landmarks, lm_cov, mu, features_t, K_S, o_T_i, V)

        if i % 500 == 0:
            print(f"Processed {i}/{num_steps} steps...")

    # --- Visualization (save two separate figures) ---
    imu_only_path, mapping_path, dataset_var = build_figure_paths(Path(__file__).resolve().parent, data_file)

    # Figure 1: IMU EKF only trajectory
    fig_imu, ax_imu = ut.visualize_trajectory_2d(all_poses, show_ori=False, path_name="IMU EKF only")
    ax_imu.set_title(f"IMU EKF only dataset: {dataset_var}")
    ax_imu.legend()
    fig_imu.savefig(imu_only_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {imu_only_path}")

    # Figure 2: Landmark mapping + trajectory
    fig_map, ax_map = ut.visualize_trajectory_2d(all_poses, show_ori=False, path_name="Trajectory")
    valid = (observed) & (np.linalg.norm(landmarks, axis=0) < 200)
    ax_map.scatter(landmarks[0, valid][::2], landmarks[1, valid][::2], s=1, c='g', alpha=0.3, label='Landmarks')
    ax_map.set_title(f"Landmark mapping via EKF update dataset: {dataset_var}")
    ax_map.legend()
    fig_map.savefig(mapping_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {mapping_path}")
    plt.show()
