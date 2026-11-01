import numpy as np
import pr3_utils as ut
from scipy.linalg import expm
import scipy.linalg as la
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path
MIN_APPEARANCES = 50       # Basic threshold: feature must survive at least 50 frames
MAX_GLOBAL_FEATURES = 1000  # Global cap: keep at most 1000 features
INNOVATION_THRESHOLD = 25

def _extract_dataset_index(path_obj):
    match = re.search(r"(\d+)$", path_obj.stem)
    return int(match.group(1)) if match else -1

def _discover_dataset_files(data_root):
    dataset_files = list(data_root.rglob("dataset*.npy"))
    return sorted(dataset_files, key=lambda p: (_extract_dataset_index(p), str(p)))

def resolve_data_file(filename_arg=None, dataset_arg=None):
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data"

    if filename_arg:
        candidate = Path(filename_arg).expanduser()
        if not candidate.is_absolute():
            cwd_candidate = (Path.cwd() / candidate).resolve()
            project_candidate = (project_root / candidate).resolve()
            if cwd_candidate.exists(): return cwd_candidate
            if project_candidate.exists(): return project_candidate
        if candidate.exists(): return candidate.resolve()
        raise FileNotFoundError(f"Specified file does not exist: {filename_arg}")

    if not data_root.exists():
        raise FileNotFoundError(f"Could not find data directory: {data_root}")

    dataset_files = _discover_dataset_files(data_root)
    if not dataset_files:
        raise FileNotFoundError(f"No dataset*.npy files found in {data_root}")

    if dataset_arg is not None:
        target = [p for p in dataset_files if _extract_dataset_index(p) == dataset_arg]
        if not target:
            raise FileNotFoundError(f"Could not find dataset {dataset_arg}")
        return target[-1]

    return dataset_files[-1]

def _format_float_for_filename(value):
    raw = np.format_float_positional(float(value), trim='-')
    return raw.replace('-', 'm').replace('.', 'p')

def build_figure_path(project_root, data_file, v_noise_scale, w_noise_scale):
    data_path = Path(data_file)
    dataset_idx = _extract_dataset_index(data_path)
    dataset_tag = f"dataset{dataset_idx:02d}" if dataset_idx >= 0 else data_path.stem
    v_tag = _format_float_for_filename(v_noise_scale)
    w_tag = _format_float_for_filename(w_noise_scale)

    figure_dir = project_root / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / f"visual_inertial_slam_{dataset_tag}_V{v_tag}_W{w_tag}.png"

def circle_dot(s):
    """Compute the circle-dot operator for homogeneous coordinate s (SE(3) perturbation model)."""
    x, y, z = s[0], s[1], s[2]
    neg_s_hat = np.array([
        [ 0,  z, -y],
        [-z,  0,  x],
        [ y, -x,  0]
    ])
    op = np.zeros((4, 6))
    op[:3, :3] = np.eye(3)
    op[:3, 3:] = neg_s_hat
    return op
def joint_update(pose, landmarks, Sigma, features_t, K_S, o_T_i, V, initialized_mask):
    M = landmarks.shape[1]
    
    # 1) Find valid observations: visible (!= -1) and already initialized landmarks.
    visible_mask = features_t[0, :] != -1
    valid_mask = visible_mask & initialized_mask
    indices = np.where(valid_mask)[0]
    Nt = len(indices)
    
    if Nt == 0:
        return pose, landmarks, Sigma
    T_iw = ut.inversePose(pose.reshape(1,4,4))[0] 
    T_cw = o_T_i @ T_iw
    P_matrix = np.vstack((np.eye(3), np.zeros((1, 3))))
    
    # 2) Pre-allocate the largest possible matrix blocks.
    H_joint_full = np.zeros((4 * Nt, 3 * M + 6))
    H_Sigma_full = np.zeros((4 * Nt, 3 * M + 6)) # For block-matrix acceleration
    innovation_full = np.zeros((4 * Nt, 1))
    
    valid_count = 0 # Number of observations that pass all guards
    
    # 3) Build Jacobians for each observed landmark.
    for j in indices:
        m_w = np.append(landmarks[:, j], 1)
        q_c = T_cw @ m_w
        
        # Depth guard
        if q_c[2] <= 0.5: 
            continue
        z_hat = K_S @ (q_c / q_c[2])
        # Subtract first: (4,) - (4,), then reshape to (4,1) to avoid broadcast issues.
        innovation = (features_t[:, j] - z_hat).reshape(4, 1)
        # Innovation outlier rejection threshold
        if np.linalg.norm(innovation) > INNOVATION_THRESHOLD: 
            # Reject as outlier (noise / tracking failure)
            continue
        # Landmark Jacobian (H_m)
        dz_dq = ut.projectionJacobian(q_c) 
        H_m = K_S @ dz_dq @ T_cw @ P_matrix 
        s = T_iw @ m_w 
        H_p = - K_S @ dz_dq @ o_T_i @ circle_dot(s) 
        # Fill stacked matrices using valid_count for row alignment.
        idx_start = 4 * valid_count
        idx_end = 4 * valid_count + 4
        innovation_full[idx_start : idx_end, :] = innovation
        H_joint_full[idx_start : idx_end, :6] = H_p
        H_joint_full[idx_start : idx_end, 6 + 3*j : 6 + 3*j + 3] = H_m
        H_Sigma_full[idx_start : idx_end, :] = (
            H_p @ Sigma[:6, :] + 
            H_m @ Sigma[6 + 3*j : 6 + 3*j + 3, :]
        ) 
        valid_count += 1
    if valid_count == 0:
        return pose, landmarks, Sigma
    H_joint = H_joint_full[:4 * valid_count, :]
    H_Sigma = H_Sigma_full[:4 * valid_count, :]
    innovation_stack = innovation_full[:4 * valid_count, :]
    V_stack = np.kron(np.eye(valid_count), V) 
    S = H_Sigma @ H_joint.T + V_stack
    K = la.solve(S, H_Sigma).T 
    Delta = K @ innovation_stack 
    delta_pose = Delta[:6, 0] 
    twist_hat = ut.axangle2twist(delta_pose.reshape(1,6))[0]
    pose_new = pose @ expm(twist_hat)

    delta_landmarks = Delta[6:, 0].reshape((M, 3)).T 
    landmarks_new = landmarks + delta_landmarks
    
    Sigma -= K @ H_Sigma
    Sigma = (Sigma + Sigma.T) / 2.0 
    
    return pose_new, landmarks_new, Sigma
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ECE276A Joint Visual-Inertial-SLAM")
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--dataset", type=int, default=2, help="For example, 10 means dataset10")
    parser.add_argument("--v-noise", type=float, default=350.0, help="Diagonal scale of observation noise V")
    parser.add_argument("--w-noise", type=float, default=1e-5, help="Diagonal scale of motion noise W")
    args = parser.parse_args()

    data_file = resolve_data_file(filename_arg=args.filename, dataset_arg=args.dataset)
    print(f"Using dataset: {data_file}")
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = ut.load_data(str(data_file))
    
    print(f"Original feature matrix shape: {features.shape}")
 
    # 1) Check feature visibility over time (!= -1).
    is_visible = (features[0, :, :] != -1)
    
    # 2) Count total appearances (feature lifetime).
    appearance_counts = np.sum(is_visible, axis=1)

    valid_indices = np.where(appearance_counts >= MIN_APPEARANCES)[0]
    
    # 4) If too many features remain, keep the longest-lived subset.
    if len(valid_indices) > MAX_GLOBAL_FEATURES:
        # Lifetimes of threshold-passing features
        valid_counts = appearance_counts[valid_indices]
        
        # Sort by lifetime descending
        sorted_args = np.argsort(valid_counts)[::-1]
        
        # Keep top-K features
        top_indices = valid_indices[sorted_args][:MAX_GLOBAL_FEATURES]
        
        # Optional: reorder by index to keep deterministic ordering
        valid_indices = np.sort(top_indices)
        
    # 5) Slice the feature matrix by the final selected indices.
    features = features[:, valid_indices, :]
    
    print(f"Filtered feature matrix shape: {features.shape} (longest-lived features kept)")
    
    num_steps = v_t.shape[0]
    M = features.shape[1] 
    print(f"Final downsampled steps: {num_steps}, Landmarks (M): {M}")
    num_steps = v_t.shape[0]
    M = features.shape[1] 
    print(f"Downsampled Steps: {num_steps}, Landmarks: {M}")
    baseline = np.linalg.norm(extL_T_imu[:3, 3] - extR_T_imu[:3, 3])
    f_su, f_sv, cu, cv = K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2]
    print(f"cu: {cu}, cv: {cv}")
    K_S = np.array([[f_su, 0, cu, 0],
                    [0, f_sv, cv, 0],
                    [f_su, 0, cu, -f_su * baseline],
                    [0, f_sv, cv, 0]])
    c_T_i = np.linalg.inv(extL_T_imu) 
    o_T_c = np.array([[ 0, -1,  0,  0], 
                      [ 0,  0, -1,  0], 
                      [ 1,  0,  0,  0], 
                      [ 0,  0,  0,  1]])
    o_T_i = o_T_c @ c_T_i  
    i_T_o = np.linalg.inv(o_T_i)
    V = np.eye(4) * args.v_noise 
    W = np.eye(6) * args.w_noise 
    mu = np.eye(4) 
    all_poses = np.zeros((num_steps, 4, 4))
    all_poses[0] = mu
    
    landmarks = np.zeros((3, M))
    initialized_mask = np.zeros(M, dtype=bool)
    Sigma = np.zeros((3 * M + 6, 3 * M + 6))
    Sigma[:6, :6] = np.eye(6) * 1e-3 
    np.fill_diagonal(Sigma[6:, 6:], 1e5) 

    print("Starting Joint VI-SLAM (Part D)...")

    time_step = 4 # Run camera update once every 4 steps for speed.

    for i in range(num_steps - 1):
        tau = timestamps[i+1] - timestamps[i]
        v, w = v_t[i], w_t[i] 
        features_t = features[:, :, i+1]
        
        # --- 1) IMU state prediction (every frame for integration accuracy) ---
        zeta_vec = np.hstack([v, w])
        zeta_hat = ut.axangle2twist(zeta_vec)
        mu = mu @ expm(tau * zeta_hat)
        
        ad_zeta = ut.axangle2adtwist(zeta_vec)
        Phi = expm(-tau * ad_zeta)
        Sigma[:6, :6] = Phi @ Sigma[:6, :6] @ Phi.T + W 
        
        # --- Control camera update frequency ---
        if i % time_step == 0:
            
            # --- 2) Joint measurement update (use map to refine pose first) ---
            mu, landmarks, Sigma = joint_update(
                mu, landmarks, Sigma, features_t, K_S, o_T_i, V, initialized_mask
            )
            
            # --- 3) Initialize new landmarks (using corrected pose) ---
            new_idx = np.where((features_t[0, :] != -1) & (~initialized_mask))[0]
            for idx in new_idx:
                z = features_t[:, idx]
                disparity = z[0] - z[2]
                
                # Guard on disparity/depth: reject too far or invalid triangulation.
                if disparity < 3.0 or disparity <= 0: 
                    continue
                depth = (f_su * baseline) / disparity
                if depth < 1.5: 
                    continue
                    
                p_c = np.array([(z[0]-cu)*depth/f_su, (z[1]-cv)*depth/f_sv, depth, 1])
                p_w = mu @ i_T_o @ p_c 
                landmarks[:, idx] = p_w[:3]
                
                # Mark as initialized once it becomes part of the 3D map.
                initialized_mask[idx] = True 
                
                # Initialize landmark covariance
                idx_start = 6 + 3*idx
                np.fill_diagonal(Sigma[idx_start:idx_start+3, idx_start:idx_start+3], 1.0)
                
            # Record trajectory
        all_poses[i+1] = mu
        
        if i % 100 == 0:
            print(f"Processed {i}/{num_steps} steps...")

    print("Done! Visualizing...")
    
    # --- Visualization ---
    fig, ax = ut.visualize_trajectory_2d(all_poses, show_ori=False, path_name="Joint SLAM Trajectory")
    
    valid = (initialized_mask) & (np.linalg.norm(landmarks, axis=0) < 200) # Display only landmarks within a reasonable range
    ax.scatter(landmarks[0, valid][::3], landmarks[1, valid][::3], s=1, c='g', alpha=0.3, label='Landmarks')
    
    plt.title("Part D: Joint Visual-Inertial SLAM")
    plt.legend()
    figure_path = build_figure_path(Path(__file__).resolve().parent, data_file, args.v_noise, args.w_noise)
    fig.savefig(figure_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure: {figure_path}")
    plt.show()
