# Modified for the ResMerge source release: replaced a local machine path in the usage example.
"""
analyze_correction_vector_field.py
分析两个rlinf LoRA residual policies的correction vector field相似性，判断是否可以merge

使用方法:
cd examples/embodiment/eval/correction_field_analysis
python analyze_correction_vector_field.py --config configs/eval_lora_config.yaml --task_i 0 --task_j 1
"""

import os
import warnings
import argparse
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore")

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入rlinf模块
from rlinf.models import get_model
from rlinf.models.embodiment.residual_policy.lora_residual_policy import LoRAResidualPolicy
from rlinf.models.embodiment.openvla_oft import get_model as get_base_model
from rlinf.envs.libero.utils import get_benchmark_overridden
from libero.libero.benchmark import Benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import h5py


def load_config(config_path):
    """加载Hydra配置文件"""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # 相对路径，从脚本目录开始
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path
    
    # 使用Hydra加载配置
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # 清除已有的hydra实例
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    config_dir = config_path.parent
    config_name = config_path.stem
    
    # 设置EMBODIED_PATH环境变量（如果未设置）
    if "EMBODIED_PATH" not in os.environ:
        # 尝试从项目根目录推断
        project_root = Path(__file__).parent.parent.parent.parent.parent
        embodied_path = project_root / "examples" / "embodiment"
        os.environ["EMBODIED_PATH"] = str(embodied_path)
    
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=config_name)
    
    return cfg


class SimpleLiberoWrapper:
    """
    简化的LIBERO环境包装器，复用rlinf的观察提取逻辑
    确保接口与rlinf的LiberoEnv一致

    obs_mode:
      - "privileged": rl_flatten_obs = proprio + object_to_robot_relations
      - "rgb":        rl_flatten_obs = [f(I_{t-1}), f(I_t), p_{t-1}, p_t]，
                      与 LiberoEnv 的 RGB 模式一致。这里是单环境顺序 rollout，
                      所以帧堆叠直接在 wrapper 内维护即可。
    """
    def __init__(self, raw_env, task_description, visual_encoder=None):
        self.raw_env = raw_env
        self.task_description = task_description
        self.visual_encoder = visual_encoder
        self._prev_frame = None

        # 复用rlinf的观察提取方法
        from rlinf.envs.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle

        self.get_libero_image = get_libero_image
        self.get_libero_wrist_image = get_libero_wrist_image
        self.quat2axisangle = quat2axisangle
    
    def _extract_image_and_state(self, obs):
        """提取图像和状态（复用rlinf的逻辑）"""
        return {
            "full_image": self.get_libero_image(obs),
            "wrist_image": self.get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    self.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }
    
    def _extract_rl_observations(self, obs):
        """
        提取RL观察：robot_proprio_state和object_to_robot_relations
        复用rlinf的逻辑
        """
        # Extract robot proprioceptive state
        if 'robot0_proprio-state' in obs:
            robot_proprio_state = obs['robot0_proprio-state'].flatten()
        else:
            robot_proprio_state = np.concatenate(
                [
                    obs['robot0_joint_pos'],
                    obs['robot0_joint_vel'] if 'robot0_joint_vel' in obs else np.zeros(7),
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_quat'] if 'robot0_eef_quat' in obs else np.zeros(4),
                    obs['robot0_gripper_qpos'],
                    obs['robot0_gripper_qvel'] if 'robot0_gripper_qvel' in obs else np.zeros(2),
                ]
            )
        
        # Extract object-to-robot relations
        relations = []
        for key, value in obs.items():
            if key.endswith('_to_robot0_eef_pos') or key.endswith('_to_robot0_eef_quat'):
                relations.append(value.flatten())
        
        if relations:
            object_to_robot_relations = np.concatenate(relations)
        else:
            object_to_robot_relations = np.array([])
        
        return {
            "robot_proprio_state": robot_proprio_state,
            "object_to_robot_relations": object_to_robot_relations,
        }
    
    def _wrap_obs(self, raw_obs):
        """包装观察，确保接口一致"""
        # 提取图像和状态
        images_and_states = self._extract_image_and_state(raw_obs)
        
        # 提取RL观察
        rl_obs = self._extract_rl_observations(raw_obs)
        
        # 构建rl_flatten_obs (numpy array for consistency)
        # 使用copy()确保数组是连续的（避免负stride问题）
        main_image = images_and_states["full_image"].copy()
        proprio = rl_obs["robot_proprio_state"].copy()

        if self.visual_encoder is None:
            rl_flatten_obs = np.concatenate([
                proprio,
                rl_obs["object_to_robot_relations"]
            ]).copy()  # [74]
        else:
            # 每个 episode 的第一帧没有前一帧，用当前帧复制填充，与 LiberoEnv 一致
            if self._prev_frame is None:
                self._prev_frame = {"main_image": main_image.copy(), "proprio": proprio.copy()}
            visual_features = self.visual_encoder.encode_frames(
                self._prev_frame["main_image"][None], main_image[None]
            )  # [1, 2 * feature_dim]
            rl_flatten_obs = np.concatenate([
                visual_features[0].cpu().numpy(),
                self._prev_frame["proprio"],
                proprio,
            ]).astype(np.float32)
            self._prev_frame = {"main_image": main_image.copy(), "proprio": proprio.copy()}


        wrapped_obs = {
            "main_images": images_and_states["full_image"].copy(),  # [H, W, C] numpy (for base model) - copy()确保连续
            "wrist_images": images_and_states["wrist_image"].copy(),  # [H, W, C] numpy (for base model) - copy()确保连续
            "agentview_rgb": images_and_states["full_image"],  # [H, W, C] numpy (for backward compatibility)
            "eye_in_hand_rgb": images_and_states["wrist_image"],  # [H, W, C] numpy (for backward compatibility)
            "states": images_and_states["state"].copy(),  # [state_dim] numpy - copy()确保连续
            "task_descriptions": [self.task_description],
            "robot_proprio_state": rl_obs["robot_proprio_state"].copy(),  # [robot_proprio_dim] numpy
            "object_to_robot_relations": rl_obs["object_to_robot_relations"].copy(),  # [relations_dim] numpy
            "rl_flatten_obs": rl_flatten_obs,  # [74] numpy
        }
        
        return wrapped_obs
    
    def reset(self):
        """重置环境并返回包装后的观察"""
        raw_obs = self.raw_env.reset()
        # 清空帧堆叠，避免跨 episode 泄漏
        self._prev_frame = None
        wrapped_obs = self._wrap_obs(raw_obs)
        return wrapped_obs
    
    def step(self, action):
        """执行动作并返回包装后的观察"""
        raw_obs, reward, done, info = self.raw_env.step(action)
        wrapped_obs = self._wrap_obs(raw_obs)
        return wrapped_obs, reward, done, info
    
    def close(self):
        """关闭环境"""
        self.raw_env.close()


def create_simple_libero_env(task_suite_name, task_id, seed=42, visual_encoder=None):
    """
    创建简化的LIBERO环境用于状态收集

    Args:
        task_suite_name: 任务suite名称
        task_id: 任务ID
        seed: 随机种子
        visual_encoder: FrozenVisualEncoder实例（RGB observation模式），None表示privileged模式

    Returns:
        env: SimpleLiberoWrapper实例
        task_description: 任务描述
    """
    benchmark: Benchmark = get_benchmark_overridden(task_suite_name)()
    task = benchmark.get_task(task_id)
    
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    
    # 创建原始环境
    raw_env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
    )
    raw_env.seed(seed)
    
    # 包装环境
    env = SimpleLiberoWrapper(raw_env, task.language, visual_encoder=visual_encoder)

    return env, task.language


def collect_state_samples_from_demo(task_suite_name, task_id, env, max_demos=5):
    """
    从演示数据中rollout收集state samples
    
    Args:
        task_suite_name: 任务suite名称
        task_id: 任务ID
        env: 环境实例
        max_demos: 最多使用多少个演示
    
    Returns:
        states: np.ndarray, [N, obs_dim] - N个state samples
    """
    print(f"\n=== Collecting State Samples from Demo (Task {task_id}) ===")
    
    benchmark: Benchmark = get_benchmark_overridden(task_suite_name)()
    
    # 获取演示数据路径
    demo_path_rel = benchmark.get_task_demonstration(task_id)
    demo_path = os.path.join(get_libero_path("datasets"), demo_path_rel)
    
    print(f"Demo path: {demo_path}")
    
    if not os.path.exists(demo_path):
        print(f"Warning: Demo file not found: {demo_path}")
        return None
    
    states_list = []
    
    # 加载HDF5文件
    with h5py.File(demo_path, 'r') as f:
        demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
        
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        
        print(f"Loading {len(demo_keys)} demos...")
        
        for demo_idx, demo_key in enumerate(demo_keys):
            traj_data = f['data'][demo_key]
            
            # 提取actions
            actions = traj_data['actions'][:]  # [T, 7]
            
            # 使用演示actions在环境中rollout，收集真实的states
            try:
                # 重置环境
                obs = env.reset()
                
                episode_states = []
                
                # 使用演示actions执行，收集states
                for t in range(len(actions)):
                    action = actions[t:t+1]  # [1, 7]
                    
                    # 提取RL观察（flatten后的状态）
                    rl_obs = obs['rl_flatten_obs']
                    episode_states.append(rl_obs.copy())
                    
                    # 执行action
                    obs, reward, done, info = env.step(action)
                    
                    if done:
                        break
                
                if len(episode_states) > 0:
                    states_array = np.array(episode_states)  # [T', 74]
                    states_list.append(states_array)
                    print(f"  Demo {demo_idx+1}: collected {len(episode_states)} states")
                
            except Exception as e:
                print(f"Warning: Failed to collect states from {demo_key}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(states_list) == 0:
        print("Error: No valid states collected!")
        return None
    
    # Flatten所有states
    flattened_states = np.concatenate(states_list, axis=0)  # [N, 74]
    
    print(f"Collected {len(states_list)} trajectories")
    print(f"Total states: {len(flattened_states)}")
    print(f"State shape: {flattened_states.shape}")
    
    return flattened_states


def collect_state_samples_from_base_rollout(base_model, task_suite_name, task_id, env, 
                                            cfg, device, num_episodes=5, max_steps=240):
    """
    使用Base Model实际rollout收集state samples
    
    Args:
        base_model: Base Model实例 (OpenVLA-OFT)
        task_suite_name: 任务suite名称
        task_id: 任务ID
        env: 环境实例
        cfg: 配置
        device: 设备
        num_episodes: 收集多少个episode
        max_steps: 每个episode最多多少步
    
    Returns:
        states: np.ndarray, [N, obs_dim] - N个state samples
    """
    print(f"\n=== Collecting State Samples from Base Model Rollout (Task {task_id}) ===")
    print(f"Using Base Model to rollout and collect states that RL actually sees")
    
    states_list = []
    
    # 获取任务描述
    benchmark: Benchmark = get_benchmark_overridden(task_suite_name)()
    task = benchmark.get_task(task_id)
    task_description = task.language
    
    for episode_id in range(num_episodes):
        # 重置环境
        obs = env.reset()
        
        episode_states = []
        step_count = 0
        
        while step_count < max_steps:
            # 保存当前state（flatten后的RL state）
            rl_obs = obs['rl_flatten_obs']
            episode_states.append(rl_obs.copy())
            
            # 准备base model输入
            # obs已经是包装后的格式，包含main_images, wrist_images, states等（numpy格式）
            import torch
            
            # 从包装后的观察中提取所需字段（numpy格式）
            main_images_np = obs['main_images']  # [H, W, C] numpy
            wrist_images_np = obs['wrist_images']  # [H, W, C] numpy
            states_np = obs['states']  # [state_dim] numpy
            
            # 转换为tensor并移动到设备
            # 注意：需要先copy()确保数组是连续的（避免负stride问题）
            main_images = torch.from_numpy(main_images_np.copy()).unsqueeze(0).to(device=device, dtype=torch.uint8)  # [1, H, W, C]
            wrist_images = torch.from_numpy(wrist_images_np.copy()).unsqueeze(0).to(device=device, dtype=torch.uint8)  # [1, H, W, C]
            states = torch.from_numpy(states_np.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, state_dim]
            
            # 构建base model输入（符合 predict_action_batch 期望的格式）
            env_obs = {
                'main_images': main_images,  # [1, H, W, C]
                'wrist_images': wrist_images,  # [1, H, W, C]
                'states': states,  # [1, state_dim]
                'task_descriptions': [task_description],
            }
            
            # 获取Base Model的action
            with torch.no_grad():
                # 预测action（base_model.predict_action_batch 会处理 env_obs）
                base_actions, _ = base_model.predict_action_batch(
                    env_obs=env_obs,
                    mode="eval",
                    calulate_logprobs=False,
                    calulate_values=False,
                    return_obs=False,
                    do_sample=False,  # eval模式使用确定性策略
                )
            
            # 使用Base Model的action执行（不添加residual）
            # base_actions 已经是 numpy 数组，shape: [B, num_action_chunks, action_dim]
            action = base_actions[0]  # [num_action_chunks, action_dim]
            # 取第一个chunk的action
            action = action[0]  # [action_dim] - 已经是 numpy array
            
            next_obs, reward, done, info = env.step(action)
            
            if done:
                break
            
            obs = next_obs
            step_count += 1
        
        if len(episode_states) > 0:
            states_array = np.array(episode_states)  # [T', 74]
            states_list.append(states_array)
            print(f"  Episode {episode_id+1}: collected {len(episode_states)} states")
    
    if len(states_list) == 0:
        print("Error: No valid states collected!")
        return None
    
    # Flatten所有states
    flattened_states = np.concatenate(states_list, axis=0)  # [N, 74]
    
    print(f"Collected {len(states_list)} episodes")
    print(f"Total states: {len(flattened_states)}")
    print(f"State shape: {flattened_states.shape}")
    
    return flattened_states


def load_residual_policy(checkpoint_path, cfg, device):
    """
    加载rlinf LoRA residual policy
    
    Args:
        checkpoint_path: checkpoint路径（huggingface model目录）
        cfg: 配置
        device: 设备
    
    Returns:
        residual_model: LoRAResidualPolicy实例
    """
    print(f"\n=== Loading Residual Policy from {checkpoint_path} ===")
    
    checkpoint_path = Path(checkpoint_path)
    
    # rlinf的checkpoint是huggingface格式，需要加载state_dict
    if checkpoint_path.is_dir():
        # RLinf checkpoint dir: model{-000xx-of-000yy}.safetensors (+ index json).
        #
        # get_model() only *builds* a randomly initialised module from the config;
        # it never reads model_path. Without the explicit load below the probe runs
        # on random weights, which silently turns DC/MC into noise around zero.
        model_cfg = cfg.actor.model.copy()
        model_cfg.model_path = str(checkpoint_path)

        model = get_model(model_cfg)

        shards = sorted(checkpoint_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"No .safetensors found in {checkpoint_path}; cannot load residual weights"
            )

        from safetensors.torch import load_file

        state_dict = {}
        for shard in shards:
            state_dict.update(load_file(str(shard)))

        target_dtype = next(model.parameters()).dtype
        state_dict = {
            k: (v.to(target_dtype) if v.is_floating_point() else v)
            for k, v in state_dict.items()
        }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Residual checkpoint does not match the model built from config.\n"
                f"  missing (left at random init): {sorted(missing)}\n"
                f"  unexpected (ignored): {sorted(unexpected)}"
            )

        model.eval()
        model.to(device)

        print(f"✅ Residual Policy loaded: {len(state_dict)} tensors from "
              f"{len(shards)} shard(s)")
    else:
        # 可能是直接的checkpoint文件
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型
        model_cfg = cfg.actor.model.copy()
        model = get_model(model_cfg)
        
        # 加载权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        model.to(device)
        
        print(f"✅ Residual Policy loaded from checkpoint file")
    
    return model


def get_eval_action(model, states, device):
    """
    获取模型的deterministic action（eval模式）
    
    Args:
        model: LoRAResidualPolicy实例
        states: np.ndarray, [B, obs_dim] - state batch
        device: 设备
    
    Returns:
        actions: np.ndarray, [B, num_action_chunks, action_dim] - actions
    """
    with torch.no_grad():
        # 转换为tensor
        states_tensor = torch.Tensor(states).to(device)
        
        # 构建env_obs格式
        env_obs = {
            'rl_flatten_obs': states_tensor,
        }
        
        # 使用predict_action_batch获取eval action
        actions, _ = model.predict_action_batch(
            env_obs=env_obs,
            mode="eval",
            calulate_logprobs=False,
            calulate_values=False,
            return_obs=False,
        )
        
        # actions shape: [B, num_action_chunks, action_dim]
        # 我们需要flatten到 [B, action_dim]（取第一个chunk或平均）
        # 这里取第一个chunk
        actions = actions[:, 0, :]  # [B, action_dim]
    
    return actions


def compute_direction_consistency(a1, a2, epsilon=1e-8):
    """
    计算方向一致性（cosine similarity）
    
    Args:
        a1: np.ndarray, [action_dim] - residual action 1
        a2: np.ndarray, [action_dim] - residual action 2
        epsilon: 防止除零的小值
    
    Returns:
        dir_sim: float, cosine similarity [-1, 1]
    """
    # Step 1: 计算两个向量的L2范数
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    
    # Step 2: 检查是否为零向量
    if norm1 < epsilon and norm2 < epsilon:
        # 两个都是零向量，认为方向一致
        return 1.0
    
    if norm1 < epsilon or norm2 < epsilon:
        # 一个是零向量，认为方向不一致
        return 0.0
    
    dot_product = np.dot(a1, a2)
    dir_sim = dot_product / (norm1 * norm2)
    
    # 确保在[-1, 1]范围内（数值误差）
    dir_sim = np.clip(dir_sim, -1.0, 1.0)
    
    return dir_sim


def compute_scale_consistency(a1, a2, epsilon=1e-8):
    """
    计算尺度一致性
    
    Args:
        a1: np.ndarray, [action_dim] - residual action 1
        a2: np.ndarray, [action_dim] - residual action 2
        epsilon: 防止除零的小值
    
    Returns:
        log_scale: float, |log(||a1|| / ||a2||)|
    """
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)
    
    if norm2 < epsilon:
        # a2是零向量，使用a1的norm作为分母
        if norm1 < epsilon:
            return 0.0  # 两个都是零
        else:
            return np.log(norm1 / epsilon)  # 很大的scale差异
    
    scale = norm1 / norm2
    log_scale = np.abs(np.log(scale + epsilon))
    
    return log_scale


def compute_magnitude_consistency(a1, a2, epsilon=1e-8):
    """
    Magnitude Consistency, paper Eq. (4): bounded harmonic-mean ratio

        MC_i = 2 * min(||a1||, ||a2||) / (||a1|| + ||a2|| + eps)

    Bounded in [0, 1], symmetric, well-defined when one residual is near zero,
    and equal to 1 only when the two pointwise magnitudes match.

    Note this is a *similarity* (higher is better). It is not the same quantity
    as compute_scale_consistency, which returns |log(||a1||/||a2||)|, an
    unbounded dissimilarity kept here only for backward compatibility with the
    older summary files.
    """
    norm1 = np.linalg.norm(a1)
    norm2 = np.linalg.norm(a2)

    return float(2.0 * min(norm1, norm2) / (norm1 + norm2 + epsilon))


def pointwise_probe(residual1, residual2, states, device):
    """
    在states上pointwise probing两个residual policies
    
    Args:
        residual1: LoRAResidualPolicy实例 - 任务i的residual
        residual2: LoRAResidualPolicy实例 - 任务j的residual
        states: np.ndarray, [N, obs_dim] - state samples
        device: 设备
    
    Returns:
        results: List[Dict], 每个元素包含一个state的分析结果
    """
    print(f"\n=== Pointwise Probing on {len(states)} States ===")
    
    results = []
    
    # 批量处理
    batch_size = 32
    
    for i in range(0, len(states), batch_size):
        end_idx = min(i + batch_size, len(states))
        batch_states = states[i:end_idx]  # [B, obs_dim]
        
        # 获取两个residual policies的actions
        a1 = get_eval_action(residual1, batch_states, device)  # [B, action_dim]
        a2 = get_eval_action(residual2, batch_states, device)  # [B, action_dim]
        
        # 对每个state计算相似性
        for j in range(len(batch_states)):
            a1_j = a1[j]  # [action_dim]
            a2_j = a2[j]  # [action_dim]
            
            # 计算direction consistency
            dir_sim = compute_direction_consistency(a1_j, a2_j)
            
            # 计算scale consistency
            log_scale = compute_scale_consistency(a1_j, a2_j)

            # Magnitude Consistency, paper Eq. (4)
            mc = compute_magnitude_consistency(a1_j, a2_j)

            results.append({
                'state_idx': i + j,
                'state': batch_states[j].copy(),
                'a1': a1_j.copy(),
                'a2': a2_j.copy(),
                'dir': dir_sim,
                'mc': mc,
                'log_scale': log_scale,
                'a1_norm': np.linalg.norm(a1_j),
                'a2_norm': np.linalg.norm(a2_j),
            })
    
    print(f"Completed probing on {len(results)} states")
    
    return results


def safety_aggregation(results, delta_dir=0.5):
    """
    安全聚合分析
    
    Args:
        results: List[Dict], pointwise probing的结果
        delta_dir: float, 危险state的方向阈值
    
    Returns:
        aggregation: Dict, 包含聚合统计信息
    """
    print(f"\n=== Safety Aggregation Analysis ===")
    
    dirs = np.array([r['dir'] for r in results])
    log_scales = np.array([r['log_scale'] for r in results])
    mcs = np.array([r['mc'] for r in results])

    # ① Overall alignment check
    mean_dir = np.mean(dirs)
    std_dir = np.std(dirs)
    median_dir = np.median(dirs)

    # Paper Eq. (3)-(5): DC and MC are each averaged over the probe set first,
    # and RFC is the product of the two averages (not the average of products).
    mean_mc = np.mean(mcs)
    rfc = mean_dir * mean_mc
    
    # ② Scale safety check
    mean_log_scale = np.mean(log_scales)
    std_log_scale = np.std(log_scales)
    median_log_scale = np.median(log_scales)
    
    # ③ Worst-case check
    dangerous_mask = dirs < delta_dir
    dangerous_states = np.sum(dangerous_mask)
    p_bad = dangerous_states / len(results)
    
    # 统计危险state的详细信息
    dangerous_dirs = dirs[dangerous_mask]
    dangerous_log_scales = log_scales[dangerous_mask]
    
    aggregation = {
        'num_states': len(results),
        'delta_dir': delta_dir,
        
        # Direction statistics
        'mean_dir': mean_dir,
        'std_dir': std_dir,
        'median_dir': median_dir,
        'min_dir': np.min(dirs),
        'max_dir': np.max(dirs),

        # Paper Eq. (4)-(5): magnitude consistency and the composite RFC score
        'mean_mc': mean_mc,
        'std_mc': np.std(mcs),
        'rfc': rfc,
        
        # Scale statistics
        'mean_log_scale': mean_log_scale,
        'std_log_scale': std_log_scale,
        'median_log_scale': median_log_scale,
        'min_log_scale': np.min(log_scales),
        'max_log_scale': np.max(log_scales),
        
        # Worst-case statistics
        'dangerous_count': dangerous_states,
        'p_bad': p_bad,
        'dangerous_mean_dir': np.mean(dangerous_dirs) if len(dangerous_dirs) > 0 else 0.0,
        'dangerous_mean_log_scale': np.mean(dangerous_log_scales) if len(dangerous_log_scales) > 0 else 0.0,
        
        # Raw data for visualization
        'dirs': dirs,
        'mcs': mcs,
        'log_scales': log_scales,
        'dangerous_mask': dangerous_mask,
    }

    print(f"Overall Direction Consistency: {mean_dir:.4f} ± {std_dir:.4f}")
    print(f"Magnitude Consistency (MC): {mean_mc:.4f} ± {np.std(mcs):.4f}")
    print(f"RFC (DC * MC): {rfc:.4f}")
    print(f"Overall Scale Consistency (log): {mean_log_scale:.4f} ± {std_log_scale:.4f}")
    print(f"Dangerous States: {dangerous_states}/{len(results)} ({p_bad*100:.2f}%)")
    
    return aggregation


def merge_decision(aggregation, tau=0.5):
    """
    Merge decision, paper Eq. (5)-(6): merge iff RFC = DC * MC >= tau.

    Args:
        aggregation: Dict, safety aggregation的结果
        tau: float, the single safety threshold on the composite RFC score

    Returns:
        decision: Dict, 包含决策结果和原因

    The legacy three-threshold rule (mean_dir >= 0.7 AND mean_log_scale <= 1.0
    AND p_bad <= 0.05) is not the paper's criterion; its per-term verdicts are
    still reported under 'legacy' for continuity with older summary files, but
    they do not affect can_merge.
    """
    rfc = aggregation['rfc']
    mean_dir = aggregation['mean_dir']
    mean_mc = aggregation['mean_mc']

    can_merge = bool(rfc >= tau)

    reasons = []
    if not can_merge:
        reasons.append(
            f"RFC={rfc:.4f} < tau={tau} (DC={mean_dir:.4f}, MC={mean_mc:.4f})"
        )

    legacy_thresholds = {
        'threshold_dir': 0.7,
        'threshold_scale': 1.0,
        'threshold_bad': 0.05,
    }
    legacy_reasons = []
    if mean_dir < legacy_thresholds['threshold_dir']:
        legacy_reasons.append(f"mean_dir={mean_dir:.4f} < {legacy_thresholds['threshold_dir']}")
    if aggregation['mean_log_scale'] > legacy_thresholds['threshold_scale']:
        legacy_reasons.append(f"mean_log_scale={aggregation['mean_log_scale']:.4f} > {legacy_thresholds['threshold_scale']}")
    if aggregation['p_bad'] > legacy_thresholds['threshold_bad']:
        legacy_reasons.append(f"p_bad={aggregation['p_bad']*100:.2f}% > {legacy_thresholds['threshold_bad']*100}%")

    decision = {
        'can_merge': can_merge,
        'reasons': reasons,
        'rfc': rfc,
        'tau': tau,
        'mean_dir': mean_dir,
        'mean_mc': mean_mc,
        'mean_log_scale': aggregation['mean_log_scale'],
        'p_bad': aggregation['p_bad'],
        'legacy': {
            'thresholds': legacy_thresholds,
            'reasons': legacy_reasons,
            'can_merge': len(legacy_reasons) == 0,
        },
    }

    return decision


def visualize_correction_field_analysis(results, aggregation, decision, output_dir, task_i, task_j):
    """可视化correction vector field分析结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    dirs = aggregation['dirs']
    log_scales = aggregation['log_scales']
    dangerous_mask = aggregation['dangerous_mask']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Direction consistency分布
    ax1 = axes[0, 0]
    ax1.hist(dirs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(aggregation['mean_dir'], color='red', linestyle='--', linewidth=2, label=f'Mean: {aggregation["mean_dir"]:.4f}')
    ax1.axvline(aggregation['delta_dir'], color='orange', linestyle='--', linewidth=2, label=f'Danger Threshold: {aggregation["delta_dir"]}')
    ax1.set_xlabel('Direction Consistency (Cosine Similarity)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Direction Consistency Distribution\n(Task {task_i} vs Task {task_j})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Log scale分布
    ax2 = axes[0, 1]
    ax2.hist(log_scales, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(aggregation['mean_log_scale'], color='red', linestyle='--', linewidth=2, label=f'Mean: {aggregation["mean_log_scale"]:.4f}')
    ax2.set_xlabel('Log Scale Consistency', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Scale Consistency Distribution\n(Task {task_i} vs Task {task_j})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 散点图：dir vs log_scale
    ax3 = axes[1, 0]
    safe_mask = ~dangerous_mask
    ax3.scatter(dirs[safe_mask], log_scales[safe_mask], alpha=0.5, color='green', label='Safe States', s=20)
    ax3.scatter(dirs[dangerous_mask], log_scales[dangerous_mask], alpha=0.7, color='red', label='Dangerous States', s=30)
    ax3.axvline(aggregation['delta_dir'], color='orange', linestyle='--', linewidth=2, label=f'Danger Threshold')
    ax3.set_xlabel('Direction Consistency', fontsize=12)
    ax3.set_ylabel('Log Scale Consistency', fontsize=12)
    ax3.set_title(f'Direction vs Scale Consistency\n(Task {task_i} vs Task {task_j})', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 决策结果总结
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    decision_text = f"""
    Merge Decision: {'✅ CAN MERGE' if decision['can_merge'] else '❌ CANNOT MERGE'}
    RFC = DC x MC = {decision['rfc']:.4f}   (tau = {decision['tau']})

    Statistics:
    • DC (direction consistency): {aggregation['mean_dir']:.4f}
    • MC (magnitude consistency): {aggregation['mean_mc']:.4f}
    • Mean Log Scale (legacy):    {aggregation['mean_log_scale']:.4f}
    • Dangerous States: {aggregation['dangerous_count']}/{aggregation['num_states']} ({aggregation['p_bad']*100:.2f}%)
    """
    
    if decision['reasons']:
        decision_text += "\nReasons:\n"
        for reason in decision['reasons']:
            decision_text += f"• {reason}\n"
    
    ax4.text(0.1, 0.5, decision_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"correction_field_analysis_task{task_i}_task{task_j}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Correction Vector Field Similarity for rlinf LoRA Residual Policy')
    parser.add_argument('--config', type=str, help='Evaluation config file path',
                       default='configs/eval_lora_config.yaml')
    parser.add_argument('--task_i', type=int, required=True, help='Task i ID')
    parser.add_argument('--task_j', type=int, required=True, help='Task j ID')
    parser.add_argument('--checkpoint_i', type=str, help='Checkpoint path for task i (huggingface model dir)')
    parser.add_argument('--checkpoint_j', type=str, help='Checkpoint path for task j (huggingface model dir)')
    parser.add_argument('--reference_task', type=int, default=None,
                       help='Reference task for collecting states (used when state_method=demo)')
    parser.add_argument('--max_demos', type=int, default=5, 
                       help='Number of episodes/demos to use for state collection')
    parser.add_argument('--delta_dir', type=float, default=0.5, help='Danger threshold for direction')
    parser.add_argument('--state_method', type=str, default='base_rollout',
                       choices=['demo', 'both_demo', 'base_rollout'],
                       help='State collection method: demo (single reference task), both_demo (both tasks from demo), base_rollout (both tasks from base model rollout)')
    parser.add_argument('--obs_mode', type=str, default='privileged',
                       choices=['privileged', 'rgb'],
                       help='Residual observation space: privileged (proprio + object-to-eef relations) '
                            'or rgb (frozen-ViT features + proprio, 2 stacked frames). Must match the '
                            'observation space the residual checkpoints were trained with.')
    parser.add_argument('--visual_encoder_path', type=str, default=None,
                       help='Frozen ViT to use in rgb obs_mode (default: facebook/dinov2-small)')
    parser.add_argument('--probe_tasks', type=int, nargs='+', default=None,
                       help='Tasks whose base rollouts form the probe set, overriding '
                            '(task_i, task_j). Needed when one side is a merged expert: '
                            'the probe set should cover every task that expert serves, '
                            'e.g. --task_i 7 --task_j 1 --probe_tasks 1 6 7 for '
                            'RFC(task7, merged[1-6]). Each task contributes '
                            '--num_probe_states states.')
    parser.add_argument('--tau', type=float, default=0.5,
                       help='Merge threshold on the composite RFC score (paper Eq. 6). '
                            'The real-robot experiments use 0.5.')
    parser.add_argument('--num_probe_states', type=int, default=None,
                       help='Trim each task to exactly this many probe states (base_rollout only). '
                            'Keeps N identical across every pair of a matrix; --max_demos must be '
                            'large enough to produce at least this many.')
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base_path = os.path.join(
        cfg.runner.logger.log_path if hasattr(cfg.runner, 'logger') else "./results",
        f"correction_field_analysis_task{args.task_i}_task{args.task_j}_{timestamp}"
    )
    os.makedirs(log_base_path, exist_ok=True)
    
    # 加载base model
    print(f"\n=== Loading Base Model ===")
    base_model_path = cfg.residual_policy.base_model_path
    base_model_cfg = cfg.actor.base_model.copy()
    base_model_cfg.model_path = base_model_path
    base_model = get_base_model(base_model_cfg)
    base_model.eval()
    base_model.to(device)
    print(f"✅ Base Model loaded from {base_model_path}")
    
    # 获取任务suite名称
    task_suite_name = cfg.env.train.task_suite_name if hasattr(cfg.env, 'train') else cfg.env.task_suite_name

    # RGB observation模式：加载与训练时相同的冻结视觉编码器
    visual_encoder = None
    if args.obs_mode == 'rgb':
        from rlinf.models.embodiment.residual_policy.frozen_visual_encoder import (
            DEFAULT_VISUAL_ENCODER,
            FrozenVisualEncoder,
        )

        visual_encoder = FrozenVisualEncoder(
            model_path=args.visual_encoder_path or DEFAULT_VISUAL_ENCODER
        )
        visual_encoder.eval().to(device)
        print(f"✅ Frozen visual encoder loaded (feature_dim={visual_encoder.feature_dim})")


    # Step 1: 根据state_method收集state samples
    print(f"\n=== Collecting States (Method: {args.state_method}) ===")
    
    if args.state_method == 'demo':
        # 方法1: 从单个reference task的演示数据收集
        if args.reference_task is None:
            args.reference_task = args.task_i
            print(f"Using task_i ({args.task_i}) as reference_task")
        
        env_ref, _ = create_simple_libero_env(task_suite_name, args.reference_task, visual_encoder=visual_encoder)
        states = collect_state_samples_from_demo(
            task_suite_name,
            args.reference_task,
            env_ref,
            max_demos=args.max_demos
        )
        env_ref.close()
        
        if states is None:
            print(f"Error: Failed to collect state samples from task {args.reference_task}!")
            return
        
        states_i = None
        states_j = None
        print(f"Collected {len(states)} states from reference task {args.reference_task}")
        
    elif args.state_method == 'both_demo':
        # 方法2: 从两个任务的演示数据收集并合并
        env_i, _ = create_simple_libero_env(task_suite_name, args.task_i, visual_encoder=visual_encoder)
        states_i = collect_state_samples_from_demo(
            task_suite_name,
            args.task_i,
            env_i,
            max_demos=args.max_demos
        )
        env_i.close()
        
        if states_i is None:
            print(f"Error: Failed to collect state samples from task {args.task_i}!")
            return
        
        env_j, _ = create_simple_libero_env(task_suite_name, args.task_j, visual_encoder=visual_encoder)
        states_j = collect_state_samples_from_demo(
            task_suite_name,
            args.task_j,
            env_j,
            max_demos=args.max_demos
        )
        env_j.close()
        
        if states_j is None:
            print(f"Error: Failed to collect state samples from task {args.task_j}!")
            return
        
        # 合并两个任务的states
        states = np.concatenate([states_i, states_j], axis=0)
        print(f"\n=== Combined States ===")
        print(f"Task {args.task_i} states: {len(states_i)}")
        print(f"Task {args.task_j} states: {len(states_j)}")
        print(f"Total combined states: {len(states)}")
        
    elif args.state_method == 'base_rollout':
        # 方法3: 从Base Model rollout收集
        print(f"Key insight: RL sees states determined by Base Model (IL), not demo trajectories")

        # Which tasks contribute probe states. Defaults to the compared pair; an
        # explicit list is needed when a side is a merged expert serving more
        # than one task, so that the probe set covers all of them.
        probe_tasks = args.probe_tasks if args.probe_tasks else [args.task_i, args.task_j]
        print(f"Probe tasks: {probe_tasks}")

        per_task = []
        for t in probe_tasks:
            env_t, _ = create_simple_libero_env(task_suite_name, t, visual_encoder=visual_encoder)
            s_t = collect_state_samples_from_base_rollout(
                base_model,
                task_suite_name,
                t,
                env_t,
                cfg,
                device,
                num_episodes=args.max_demos,
                max_steps=cfg.env.train.max_episode_steps if hasattr(cfg.env.train, 'max_episode_steps') else 240
            )
            env_t.close()

            if s_t is None:
                print(f"Error: Failed to collect state samples from task {t}!")
                return

            # Trim every task to the same probe count so that N is identical across
            # pairs. Episodes that terminate early otherwise make N depend on how
            # well the base policy happens to do on that task, which would confound
            # RFC comparisons.
            if args.num_probe_states is not None:
                if len(s_t) < args.num_probe_states:
                    print(f"Warning: task {t} yielded only {len(s_t)} states "
                          f"(< requested {args.num_probe_states}); raise --max_demos")
                s_t = s_t[:args.num_probe_states]

            per_task.append(s_t)

        states_i, states_j = per_task[0], per_task[-1]
        states = np.concatenate(per_task, axis=0)
        print(f"\n=== Combined States ===")
        for t, s_t in zip(probe_tasks, per_task):
            print(f"Task {t} states: {len(s_t)}")
        print(f"Total combined states: {len(states)}")

    # Step 2: 加载两个residual policies
    # 确定checkpoint路径
    if args.checkpoint_i:
        checkpoint_i_path = args.checkpoint_i
    else:
        checkpoint_i_path = cfg.runner.eval_policy_path if hasattr(cfg.runner, 'eval_policy_path') else None
        if checkpoint_i_path is None:
            raise ValueError("checkpoint_i must be specified or provided in config.runner.eval_policy_path")
    
    if args.checkpoint_j:
        checkpoint_j_path = args.checkpoint_j
    else:
        checkpoint_j_path = cfg.runner.eval_policy_path if hasattr(cfg.runner, 'eval_policy_path') else None
        if checkpoint_j_path is None:
            raise ValueError("checkpoint_j must be specified or provided in config.runner.eval_policy_path")
        print(f"Warning: checkpoint_j not specified, using same as checkpoint_i")
    
    # The residual policy is built from cfg.actor.model, whose obs_dim is written
    # for the privileged observation space (88 for libero_object, 74 for spatial).
    # In rgb mode the probe states are [2*feature_dim + 2*proprio_dim] wide, so
    # take the width from the states actually collected -- that is by construction
    # the width the checkpoint's fc1_A expects, and it avoids hard-coding 846.
    if args.obs_mode == 'rgb':
        rgb_obs_dim = int(states.shape[-1])
        if cfg.actor.model.obs_dim != rgb_obs_dim:
            print(f"obs_mode=rgb: overriding actor.model.obs_dim "
                  f"{cfg.actor.model.obs_dim} -> {rgb_obs_dim}")
            cfg.actor.model.obs_dim = rgb_obs_dim

    residual_i = load_residual_policy(checkpoint_i_path, cfg, device)
    residual_j = load_residual_policy(checkpoint_j_path, cfg, device)
    
    # Step 3: Pointwise probing
    results = pointwise_probe(residual_i, residual_j, states, device)
    
    # Step 4: Safety aggregation
    aggregation = safety_aggregation(results, delta_dir=args.delta_dir)
    
    # Step 5: Merge decision
    decision = merge_decision(aggregation, tau=args.tau)

    # Persist the per-state quantities. Re-running a pair costs a full base-policy
    # rollout, so keeping the raw dirs/mcs/norms makes re-thresholding, bootstrap
    # CIs and cross-repeat variance analysis possible without touching the GPU.
    raw_path = os.path.join(log_base_path, "pointwise_raw.npz")
    np.savez_compressed(
        raw_path,
        dirs=aggregation['dirs'],
        mcs=aggregation['mcs'],
        log_scales=aggregation['log_scales'],
        a1_norms=np.array([r['a1_norm'] for r in results]),
        a2_norms=np.array([r['a2_norm'] for r in results]),
        task_i=args.task_i,
        task_j=args.task_j,
        num_probe_states=len(aggregation['dirs']),
    )
    print(f"Saved per-state raw data to: {raw_path}")


    # Step 6: 可视化
    visualize_correction_field_analysis(results, aggregation, decision, 
                                       log_base_path, args.task_i, args.task_j)
    
    # 保存结果
    summary_file = os.path.join(log_base_path, "summary_analysis.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Correction Vector Field Analysis Summary\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Task Pair: Task {args.task_i} vs Task {args.task_j}\n")
        f.write(f"Checkpoint i: {checkpoint_i_path}\n")
        f.write(f"Checkpoint j: {checkpoint_j_path}\n\n")
        
        f.write(f"State Collection:\n")
        f.write(f"  Method: {args.state_method}\n")
        if args.state_method == 'demo':
            f.write(f"  Reference Task: {args.reference_task}\n")
            f.write(f"  Total States: {aggregation['num_states']}\n")
            f.write(f"  Max Demos: {args.max_demos}\n")
        else:
            if states_i is not None and states_j is not None:
                f.write(f"  Task {args.task_i} states: {len(states_i)}\n")
                f.write(f"  Task {args.task_j} states: {len(states_j)}\n")
            f.write(f"  Total Combined States: {aggregation['num_states']}\n")
            f.write(f"  Episodes/Demos per task: {args.max_demos}\n")
        if args.state_method == 'base_rollout':
            f.write(f"  Key Insight: RL sees states determined by Base Model (IL), not demo trajectories\n")
        f.write(f"\n")
        
        f.write(f"Direction Consistency:\n")
        f.write(f"  Mean: {aggregation['mean_dir']:.4f}\n")
        f.write(f"  Std: {aggregation['std_dir']:.4f}\n")
        f.write(f"  Median: {aggregation['median_dir']:.4f}\n")
        f.write(f"  Min: {aggregation['min_dir']:.4f}\n")
        f.write(f"  Max: {aggregation['max_dir']:.4f}\n\n")
        
        f.write(f"Scale Consistency:\n")
        f.write(f"  Mean Log Scale: {aggregation['mean_log_scale']:.4f}\n")
        f.write(f"  Std Log Scale: {aggregation['std_log_scale']:.4f}\n")
        f.write(f"  Median Log Scale: {aggregation['median_log_scale']:.4f}\n")
        f.write(f"  Min Log Scale: {aggregation['min_log_scale']:.4f}\n")
        f.write(f"  Max Log Scale: {aggregation['max_log_scale']:.4f}\n\n")
        
        f.write(f"Worst-Case Analysis:\n")
        f.write(f"  Dangerous States: {aggregation['dangerous_count']}/{aggregation['num_states']} ({aggregation['p_bad']*100:.2f}%)\n")
        f.write(f"  Delta Dir Threshold: {aggregation['delta_dir']}\n")
        if aggregation['dangerous_count'] > 0:
            f.write(f"  Dangerous States Mean Dir: {aggregation['dangerous_mean_dir']:.4f}\n")
            f.write(f"  Dangerous States Mean Log Scale: {aggregation['dangerous_mean_log_scale']:.4f}\n")
        f.write(f"\n")
        
        f.write(f"Magnitude Consistency (paper Eq. 4):\n")
        f.write(f"  Mean MC: {aggregation['mean_mc']:.4f}\n")
        f.write(f"  Std MC: {aggregation['std_mc']:.4f}\n\n")

        f.write(f"RFC Score (paper Eq. 5, DC * MC):\n")
        f.write(f"  RFC: {aggregation['rfc']:.4f}\n\n")

        f.write(f"Merge Decision (paper Eq. 6, RFC >= tau):\n")
        f.write(f"  {'✅ CAN MERGE' if decision['can_merge'] else '❌ CANNOT MERGE'}\n")
        f.write(f"  tau: {decision['tau']}\n")
        if decision['reasons']:
            f.write(f"  Reasons:\n")
            for reason in decision['reasons']:
                f.write(f"    - {reason}\n")
        f.write(f"  [legacy three-threshold rule, not the paper's criterion]: "
                f"{'pass' if decision['legacy']['can_merge'] else 'fail'}\n")
        for reason in decision['legacy']['reasons']:
            f.write(f"    - {reason}\n")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"Merge Decision: {'✅ CAN MERGE' if decision['can_merge'] else '❌ CANNOT MERGE'}")
    print(f"Results saved to: {log_base_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

