import glob
import os
import pickle
import shutil

import numpy as np
import zarr


LEAP_HAND_PHYS_MIN = np.array(
    [
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.09459,
        2.82759,
        2.63559,
        2.77559,
        2.79259,
        2.67159,
        1.94159,
        1.80159,
    ],
    dtype=np.float32,
)
LEAP_HAND_PHYS_MAX = np.array(
    [
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        4.18859,
        5.37159,
        5.02659,
        5.18359,
        5.23559,
        5.58459,
        5.04159,
        5.02159,
    ],
    dtype=np.float32,
)
LEAP_HAND_SPAN = LEAP_HAND_PHYS_MAX - LEAP_HAND_PHYS_MIN



# leap_pose(16), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6)

STATE_SLICES = [
    ("leap_pose", slice(0, 16)),
    ("tcp_force", slice(16, 19)),
    ("tcp_pose", slice(19, 25)),
    ("tcp_torque", slice(25, 28)),
    ("tcp_vel", slice(28, 34)),
]


ACTION_SLICES = [
    ("action_arm", slice(0, 6)),
    ("action_hand", slice(6, 22)),
]


IMAGE_KEYS = [
    ("side_policy_256", "side_policy_256"),
]


def load_pkl_payload(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "transitions" not in payload:
        raise ValueError(f"{pkl_path} must be a dict containing 'transitions'")
    transitions = payload["transitions"]
    if not isinstance(transitions, list):
        raise ValueError(f"{pkl_path} payload['transitions'] must be a list")
    return payload, transitions


def ensure_2d_float16(x, name):
    x = np.asarray(x, dtype=np.float16)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D [T, D], got shape={x.shape}")
    return x


def ensure_4d_uint8(x, name):
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim != 4:
        raise ValueError(f"{name} must be 4D [T, H, W, C], got shape={x.shape}")
    return x


def nan_to_zero(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype == np.uint8 or not np.issubdtype(arr.dtype, np.floating):
        return arr
    bad = np.isnan(arr) | np.isinf(arr)
    if bad.any():
        print(f"[Warn] {name}: replacing {bad.sum()} nan/inf with 0")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(arr.dtype, copy=False)


def normalize_leap_hand_physical(hand16):
    x = np.asarray(hand16, dtype=np.float32).reshape(16).copy()
    x = np.clip(x, LEAP_HAND_PHYS_MIN, LEAP_HAND_PHYS_MAX)
    return (2.0 * (x - LEAP_HAND_PHYS_MIN) / LEAP_HAND_SPAN - 1.0).astype(np.float16)


def _validate_image_keys(image_keys):
    if not isinstance(image_keys, (list, tuple)):
        raise TypeError(f"IMAGE_KEYS must be list/tuple, got {type(image_keys)}")
    seen = set()
    for item in image_keys:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each IMAGE_KEYS item must be (src_key, dst_key), got {item!r}")
        src_key, dst_key = item
        if not isinstance(src_key, str) or not isinstance(dst_key, str):
            raise ValueError(f"image key names must be strings, got {item!r}")
        if dst_key in seen:
            raise ValueError(f"duplicate zarr image key: {dst_key}")
        seen.add(dst_key)


def _validate_slices(slice_items, item_name):
    if not isinstance(slice_items, (list, tuple)):
        raise TypeError(f"{item_name} must be list/tuple, got {type(slice_items)}")
    seen = set()
    for item in slice_items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each {item_name} item must be (name, slice), got {item!r}")
        name, s = item
        if not isinstance(name, str):
            raise ValueError(f"{item_name} name must be str, got {name!r}")
        if not isinstance(s, slice):
            raise ValueError(f"{item_name} slice must be slice(...), got {s!r}")
        if name in seen:
            raise ValueError(f"duplicate {item_name} key: {name!r}")
        seen.add(name)


def _stack_state_field(transitions, obs_key, state_slice, name):
    xs = []
    for i, tr in enumerate(transitions):
        obs = tr.get("observations", {})
        state = obs.get(obs_key)
        if state is None:
            raise ValueError(f"missing observations[{obs_key!r}] at step {i}")
        state = np.asarray(state, dtype=np.float16)
        if state.ndim >= 2:
            
            part = state[:, state_slice].reshape(-1)
        else:
            state = state.reshape(-1)
            part = state[state_slice]
        if part.size == 0:
            raise ValueError(
                f"{name} slice {state_slice} is empty at step {i}, state shape={state.shape}"
            )
        xs.append(part)
    return ensure_2d_float16(np.stack(xs, axis=0), name)


def _get_obs_horizon(transitions, obs_key="state"):
    if not transitions:
        return 1
    obs = transitions[0].get("observations", {})
    state = np.asarray(obs.get(obs_key))
    if state.ndim >= 2:
        return int(state.shape[0])
    return 1


def _stack_leap_pose_from_hw(transitions, name="leap_pose"):
    obs_horizon = _get_obs_horizon(transitions, obs_key="state")
    leap_seq = []
    for i, tr in enumerate(transitions):
        leap_hw_pos = tr.get("leap_hw_pos")
        if leap_hw_pos is None:
            raise ValueError(f"missing leap_hw_pos at step {i}")
        leap_seq.append(normalize_leap_hand_physical(leap_hw_pos))

    xs = []
    for i in range(len(leap_seq)):
        start = max(0, i - obs_horizon + 1)
        hist = leap_seq[start : i + 1]
        if len(hist) < obs_horizon:
            hist = [leap_seq[0]] * (obs_horizon - len(hist)) + hist
        xs.append(np.stack(hist, axis=0).reshape(-1))
    return ensure_2d_float16(np.stack(xs, axis=0), name)


def _stack_action_field(transitions, action_slice, name):
    xs = []
    for i, tr in enumerate(transitions):
        action = tr.get("actions")
        if action is None:
            raise ValueError(f"missing actions at step {i}")
        action = np.asarray(action, dtype=np.float16).reshape(-1)
        part = action[action_slice]
        if part.size == 0:
            raise ValueError(
                f"{name} slice {action_slice} is empty at step {i}, action shape={action.shape}"
            )
        xs.append(part)
    return ensure_2d_float16(np.stack(xs, axis=0), name)


def _stack_image_field(transitions, src_key, dst_key):
    imgs_by_stack = None
    for i, tr in enumerate(transitions):
        obs = tr.get("observations", {})
        img = obs.get(src_key)
        if img is None:
            raise ValueError(f"missing observations[{src_key!r}] at step {i}")
        img = np.asarray(img)
        if img.ndim == 3:
            img = img[None, ...]
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(
                f"{src_key} at step {i} must be HWC RGB or stacked THWC, got shape={img.shape}"
            )
        if imgs_by_stack is None:
            imgs_by_stack = [[] for _ in range(img.shape[0])]
        elif len(imgs_by_stack) != img.shape[0]:
            raise ValueError(
                f"{src_key} stack size mismatch at step {i}: expected {len(imgs_by_stack)}, got {img.shape[0]}"
            )
        for stack_idx in range(img.shape[0]):
            imgs_by_stack[stack_idx].append(img[stack_idx])

    out = {}
    assert imgs_by_stack is not None
    for stack_idx, imgs in enumerate(imgs_by_stack):
        key = dst_key if len(imgs_by_stack) == 1 else f"{dst_key}_{stack_idx}"
        out[key] = ensure_4d_uint8(np.stack(imgs, axis=0), key)
    return out


def get_episode_arrays(pkl_path):
    payload, transitions = load_pkl_payload(pkl_path)
    del payload

    out = {}

    for name, s in STATE_SLICES:
        if name == "leap_pose":
            continue
        out[name] = _stack_state_field(transitions, "state", s, name)
    out["leap_pose"] = _stack_leap_pose_from_hw(transitions, "leap_pose")

    out["action"] = _stack_action_field(transitions, slice(None), "action")
    for name, s in ACTION_SLICES:
        out[name] = _stack_action_field(transitions, s, name)

    for src_key, dst_key in IMAGE_KEYS:
        out.update(_stack_image_field(transitions, src_key, dst_key))

    lengths = {k: len(v) for k, v in out.items()}
    T = min(lengths.values())

    if len(set(lengths.values())) != 1:
        print(f"[Warn] {os.path.basename(pkl_path)} length mismatch: {lengths}, truncate to {T}")

    for k in out:
        out[k] = out[k][:T]
        out[k] = nan_to_zero(out[k], k)

    return out, T


def create_appendable_array(group, name, sample_array, chunk_len=64):
    shape = (0,) + sample_array.shape[1:]
    chunks = (min(chunk_len, max(1, len(sample_array))),) + sample_array.shape[1:]
    return group.create_dataset(
        name=name,
        shape=shape,
        chunks=chunks,
        dtype=sample_array.dtype,
    )


def append_to_zarr(arr, data):
    old_n = arr.shape[0]
    new_n = old_n + data.shape[0]
    arr.resize(new_n, *arr.shape[1:])
    arr[old_n:new_n] = data


def _normalize_pkl_roots(pkl_root):
    if isinstance(pkl_root, str):
        return [pkl_root]
    if isinstance(pkl_root, (list, tuple)):
        roots = [os.path.expanduser(os.path.normpath(r)) for r in pkl_root]
        if not roots:
            raise ValueError("pkl_root list/tuple must be non-empty")
        for r in roots:
            if not isinstance(r, str):
                raise TypeError(f"each pkl_root entry must be str, got {type(r)}")
        return roots
    raise TypeError(f"pkl_root must be str or list/tuple of str, got {type(pkl_root)}")


def _collect_episode_pkl_paths(pkl_roots):
    seen = set()
    paths = []
    for root in pkl_roots:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"pkl root is not a directory: {root}")
        for p in glob.glob(os.path.join(root, "**", "*.pkl"), recursive=True):
            if "_ep" not in os.path.basename(p):
                continue
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            paths.append(p)
    return sorted(paths)


def convert_pkls_to_zarr(
    pkl_root,
    output_zarr,
    overwrite=False,
    max_episodes=None,
    random_sample=False,
    random_seed=42,
):
    _validate_image_keys(IMAGE_KEYS)
    _validate_slices(STATE_SLICES, "STATE_SLICES")
    _validate_slices(ACTION_SLICES, "ACTION_SLICES")

    if max_episodes is not None:
        if not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError("max_episodes must be None or a positive int")

    if os.path.exists(output_zarr):
        if overwrite:
            shutil.rmtree(output_zarr)
        else:
            raise FileExistsError(output_zarr)

    pkl_roots = _normalize_pkl_roots(pkl_root)
    pkl_paths = _collect_episode_pkl_paths(pkl_roots)

    if len(pkl_paths) == 0:
        raise RuntimeError(f"No episode pkl found under {pkl_roots}")

    print(f"Found {len(pkl_paths)} episode pkls")
    if max_episodes is not None:
        if random_sample:
            sample_n = min(max_episodes, len(pkl_paths))
            rng = np.random.default_rng(random_seed)
            pick_idx = rng.choice(len(pkl_paths), size=sample_n, replace=False)
            pkl_paths = [pkl_paths[i] for i in pick_idx]
            pkl_paths = sorted(pkl_paths)
            print(
                f"Using random {len(pkl_paths)} (max_episodes={max_episodes}, seed={random_seed})"
            )
        else:
            pkl_paths = pkl_paths[:max_episodes]
            print(f"Using first {len(pkl_paths)} (max_episodes={max_episodes})")

    root = zarr.open(output_zarr, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    arrays = None
    episode_ends = []
    total_steps = 0

    for i, pkl_path in enumerate(pkl_paths):
        print(f"[{i+1}/{len(pkl_paths)}] processing {os.path.basename(pkl_path)}")
        try:
            ep_arrays, T = get_episode_arrays(pkl_path)
        except Exception as e:
            print(f"[Skip] failed on {pkl_path}: {e}")
            continue

        if T == 0:
            print(f"[Skip] empty episode: {pkl_path}")
            continue

        if arrays is None:
            print("Per-observation shapes (from first successful episode):")
            for k, v in sorted(ep_arrays.items()):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            arrays = {}
            for k, v in ep_arrays.items():
                arrays[k] = create_appendable_array(data_group, k, v)

        for k, v in ep_arrays.items():
            append_to_zarr(arrays[k], v)

        total_steps += T
        episode_ends.append(total_steps)

    meta_group.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
    )

    print("\nDone.")
    print(f"output: {output_zarr}")
    print(f"episodes: {len(episode_ends)}")
    print(f"total steps: {total_steps}")

    if arrays is not None:
        print("\nFinal arrays:")
        for k, arr in arrays.items():
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  meta/episode_ends: shape={meta_group['episode_ends'].shape}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert recorded episode pickle files to Zarr")
    parser.add_argument("--input", nargs="+", required=True, help="Episode directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert_pkls_to_zarr(pkl_root=args.input, output_zarr=args.output,
                        overwrite=args.overwrite, max_episodes=args.max_episodes,
                        random_sample=args.random_sample, random_seed=args.seed)


if __name__ == "__main__":
    main()
