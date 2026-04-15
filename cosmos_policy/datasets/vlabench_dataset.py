# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLABench Dataset loader for Cosmos Policy training.

HDF5 structure:
  data/
    {timestamp}/
      instruction: (1,) bytes
      observation/
        rgb: (T, 4, 480, 480, 3) uint8  [camera_0, 1, 2(primary), 3(wrist)]
        ee_state: (T, 8) float32  [proprio]
        q_state: (T, 7) float32
        depth, point_cloud_*: ignored
      trajectory: (T, 8) float32  [actions]
      meta_info/: task metadata

Usage:
    dataset = VLABenchDataset(
        data_dir='/path/to/VLABench_release/primitive',
        t5_text_embeddings_path='/path/t5_embeddings.pkl',
    )
"""

import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    get_action_chunk_with_padding,
    compute_monte_carlo_returns,
)
from cosmos_policy.datasets.dataset_utils import (
    preprocess_image,
)
from cosmos_policy.utils.utils import duplicate_array


class VLABenchDataset(Dataset):
    """
    VLABench dataset for visuomotor task learning.
    
    Camera mapping:
      - camera_2: primary (table view)
      - camera_3: wrist (end-effector view)
    
    Proprioception: ee_state (8D)
    Actions: trajectory (8D)
    """

    HDF5_FILE_MANIFEST_NAME = "vlabench_hdf5_files.pkl"
    EPISODE_MANIFEST_NAME = "vlabench_episode_manifest.pkl"

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_wrist_images: bool = True,         # Always True for VLABench; accepted for config compatibility
        use_third_person_images: bool = True,  # Accepted for config compatibility
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",            # Accepted for config compatibility; VLABench has no rollouts
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,  # Accepted for config compatibility
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        is_train: bool = True,
        train_val_split: float = 0.9,
    ):
        """
        Initialize VLABench dataset.

        Args:
            data_dir: Root directory containing task subdirectories (select_drink, select_mahjong, select_poker)
            chunk_size: Action chunk size for training
            final_image_size: Target image resolution (square)
            t5_text_embeddings_path: Path to precomputed T5 embeddings pickle file
            normalize_images: Whether to normalize images to float32
            normalize_actions: Whether to normalize actions
            normalize_proprio: Whether to normalize proprioception
            use_image_aug: Whether to apply image augmentation
            use_stronger_image_aug: Whether to use stronger augmentation
            use_proprio: Whether to include proprioception
            num_duplicates_per_image: Image duplication for tokenizer
            demonstration_sampling_prob: Probability of sampling demonstrations
            success_rollout_sampling_prob: Probability of sampling success rollouts
            return_value_function_returns: Whether to compute value returns
            gamma: Discount factor for value returns
            is_train: Whether loading train or val set
            train_val_split: Fraction of episodes for training (rest goes to val)
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        # use_wrist_images is always True for VLABench (camera_3 is wrist)
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.is_train = is_train
        self.train_val_split = train_val_split
        self.hdf5_file_handles = {}
        self._cache_pid = os.getpid()
        self.t5_text_embeddings = {}

        # Discover files without recursive os.walk on NAS, then reuse a cached manifest.
        all_hdf5_files = self._load_or_build_hdf5_file_manifest()

        # Split into train/val based on file index
        num_files = len(all_hdf5_files)
        split_idx = int(num_files * train_val_split)
        if is_train:
            hdf5_files = all_hdf5_files[:split_idx]
        else:
            hdf5_files = all_hdf5_files[split_idx:]

        # Build episode metadata (lazy loading - only store file paths and indices)
        self.episode_metadata = {}  # {episode_idx: {hdf5_path, timestamp_key, instruction, num_steps, ...}}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands = set()

        self._populate_episode_metadata(hdf5_files)

        if self.num_episodes == 0 or self.num_steps == 0:
            split_name = "train" if is_train else "val"
            raise ValueError(
                f"No VLABench episodes found for the {split_name} split in '{data_dir}'. "
                "Check data_dir contents and train_val_split."
            )

        # Build step index mapping (using metadata only)
        self._build_step_index_mapping()

        # Load T5 embeddings if provided
        if t5_text_embeddings_path:
            if not os.path.exists(t5_text_embeddings_path):
                raise FileNotFoundError(
                    f"T5 embeddings file not found: {t5_text_embeddings_path}"
                )
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)

        # Compute dataset statistics from sampled proprio/actions only.
        self.dataset_stats = self._compute_dataset_statistics_lazy()

        # Calculate epoch structure
        self._calculate_epoch_structure()

    def _hdf5_manifest_path(self):
        return os.path.join(self.data_dir, self.HDF5_FILE_MANIFEST_NAME)

    def _episode_manifest_path(self):
        return os.path.join(self.data_dir, self.EPISODE_MANIFEST_NAME)

    def _list_hdf5_files_in_directory(self, directory_path: str):
        """List HDF5 files in one directory without recursive traversal."""
        hdf5_files = []
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".h5", ".hdf5", ".he5")):
                    hdf5_files.append(entry.path)
        return sorted(hdf5_files)

    def _discover_hdf5_files(self):
        """Discover HDF5 files assuming data_dir is a task folder or contains one-level task folders."""
        direct_files = self._list_hdf5_files_in_directory(self.data_dir)
        if direct_files:
            return direct_files

        hdf5_files = []
        with os.scandir(self.data_dir) as entries:
            subdirs = sorted((entry.path for entry in entries if entry.is_dir()))

        for subdir in subdirs:
            hdf5_files.extend(self._list_hdf5_files_in_directory(subdir))

        return hdf5_files

    def _load_or_build_hdf5_file_manifest(self):
        """Load cached file discovery results to avoid repeated NAS scans."""
        manifest_path = self._hdf5_manifest_path()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "rb") as file:
                    manifest = pickle.load(file)
                if isinstance(manifest, list):
                    return manifest
            except Exception as exc:
                print(f"Failed to load HDF5 file manifest: {exc}, rebuilding...")

        hdf5_files = self._discover_hdf5_files()

        try:
            with open(manifest_path, "wb") as file:
                pickle.dump(hdf5_files, file)
        except Exception as exc:
            print(f"Failed to cache HDF5 file manifest: {exc}")

        return hdf5_files

    def _build_episode_manifest(self, hdf5_files):
        """Build cached episode metadata for all files once."""
        episode_manifest = []
        for file_path in tqdm(hdf5_files, desc="Building VLABench episode manifest"):
            with h5py.File(file_path, "r") as file_handle:
                data_group = file_handle["data"]
                for timestamp_key in data_group.keys():
                    ep_group = data_group[timestamp_key]
                    instruction_bytes = ep_group["instruction"][0]
                    if isinstance(instruction_bytes, bytes):
                        instruction = instruction_bytes.decode("utf-8")
                    else:
                        instruction = str(instruction_bytes)

                    episode_manifest.append(
                        {
                            "hdf5_path": file_path,
                            "timestamp_key": timestamp_key,
                            "command": instruction,
                            "num_steps": len(ep_group["trajectory"]),
                        }
                    )
        return episode_manifest

    def _load_or_build_episode_manifest(self, hdf5_files):
        """Load cached episode metadata to avoid reopening every file at startup."""
        manifest_path = self._episode_manifest_path()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "rb") as file:
                    manifest = pickle.load(file)
                if isinstance(manifest, list):
                    return manifest
            except Exception as exc:
                print(f"Failed to load episode manifest: {exc}, rebuilding...")

        manifest = self._build_episode_manifest(hdf5_files)

        try:
            with open(manifest_path, "wb") as file:
                pickle.dump(manifest, file)
        except Exception as exc:
            print(f"Failed to cache episode manifest: {exc}")

        return manifest

    def _populate_episode_metadata(self, selected_hdf5_files):
        """Populate split-specific metadata from the cached all-file episode manifest."""
        selected_hdf5_files = set(selected_hdf5_files)
        all_episode_metadata = self._load_or_build_episode_manifest(self._load_or_build_hdf5_file_manifest())

        for metadata in all_episode_metadata:
            if metadata["hdf5_path"] not in selected_hdf5_files:
                continue

            self.episode_metadata[self.num_episodes] = metadata
            self.unique_commands.add(metadata["command"])
            self.num_steps += metadata["num_steps"]
            self.num_episodes += 1

    def _build_step_index_mapping(self):
        """Build mapping from global step index to (episode_idx, relative_idx)."""
        _step_to_episode_map = {}
        current_step = 0
        for episode_idx in range(self.num_episodes):
            num_steps = self.episode_metadata[episode_idx]["num_steps"]
            for relative_step_idx in range(num_steps):
                _step_to_episode_map[current_step] = (episode_idx, relative_step_idx)
                current_step += 1
        self._step_to_episode_map = _step_to_episode_map
        self._total_steps = current_step

    def _calculate_epoch_structure(self):
        """Calculate epoch layout."""
        # For now, simple structure with just demos (no rollouts)
        self.adjusted_demo_count = self.num_steps
        self.adjusted_success_rollout_count = 0
        self.adjusted_failure_rollout_count = 0
        self.epoch_length = self.num_steps

    def _close_hdf5_files(self):
        """Close cached HDF5 file handles."""
        for file_handle in self.hdf5_file_handles.values():
            try:
                file_handle.close()
            except Exception:
                pass
        self.hdf5_file_handles = {}

    def _reset_hdf5_cache_if_needed(self):
        """Reset HDF5 cache when the dataset object is used from a forked worker."""
        current_pid = os.getpid()
        if self._cache_pid != current_pid:
            self._close_hdf5_files()
            self._cache_pid = current_pid

    def _get_hdf5_file(self, hdf5_path):
        """Return a cached HDF5 file handle for the current process."""
        self._reset_hdf5_cache_if_needed()
        file_handle = self.hdf5_file_handles.get(hdf5_path)
        if file_handle is None or not file_handle.id.valid:
            file_handle = h5py.File(hdf5_path, "r")
            self.hdf5_file_handles[hdf5_path] = file_handle
        return file_handle

    def _load_episode_stats_data(self, episode_idx):
        """Load proprio and actions for a sampled episode when computing statistics."""
        metadata = self.episode_metadata[episode_idx]
        file_handle = self._get_hdf5_file(metadata["hdf5_path"])
        ep_group = file_handle["data"][metadata["timestamp_key"]]
        return {
            "actions": ep_group["trajectory"][:].astype(np.float32),
            "proprio": ep_group["observation"]["ee_state"][:].astype(np.float32),
        }

    def _load_episode_sample(self, episode_idx, relative_step_idx, future_frame_idx):
        """Load only the frames and arrays needed for a single training sample."""
        metadata = self.episode_metadata[episode_idx]
        file_handle = self._get_hdf5_file(metadata["hdf5_path"])
        ep_group = file_handle["data"][metadata["timestamp_key"]]
        obs = ep_group["observation"]

        frame_indices = np.array(sorted({relative_step_idx, future_frame_idx}), dtype=np.int64)
        rgb_frames = obs["rgb"][frame_indices]
        proprio_frames = obs["ee_state"][frame_indices].astype(np.float32)
        actions = ep_group["trajectory"][:].astype(np.float32)

        index_lookup = {frame_idx: offset for offset, frame_idx in enumerate(frame_indices.tolist())}
        current_offset = index_lookup[relative_step_idx]
        future_offset = index_lookup[future_frame_idx]

        return {
            "current_primary": rgb_frames[current_offset, 2],
            "future_primary": rgb_frames[future_offset, 2],
            "current_wrist": rgb_frames[current_offset, 3],
            "future_wrist": rgb_frames[future_offset, 3],
            "current_proprio": proprio_frames[current_offset],
            "future_proprio": proprio_frames[future_offset],
            "actions": actions,
            "command": metadata["command"],
            "num_steps": metadata["num_steps"],
        }

    def _normalize_array(self, arr: np.ndarray, data_key: str) -> np.ndarray:
        """Match the codebase's min-max normalization to [-1, +1]."""
        curr_min = self.dataset_stats[f"{data_key}_min"]
        curr_max = self.dataset_stats[f"{data_key}_max"]
        denom = np.where((curr_max - curr_min) < 1e-8, 1.0, curr_max - curr_min)
        return 2 * ((arr - curr_min) / denom) - 1

    def _get_text_embedding(self, command: str) -> torch.Tensor:
        """Fetch the precomputed text embedding or fail with a clear error."""
        if not self.t5_text_embeddings_path:
            raise RuntimeError(
                "t5_text_embeddings_path must be provided for VLABenchDataset."
            )
        if command not in self.t5_text_embeddings:
            raise KeyError(
                f"Missing T5 embedding for instruction: {command}"
            )
        return torch.squeeze(self.t5_text_embeddings[command])

    def _compute_dataset_statistics_lazy(self):
        """Compute dataset statistics by sampling proprio and actions only."""
        stats_path = os.path.join(self.data_dir, "vlabench_dataset_statistics.pkl")

        if os.path.exists(stats_path):
            try:
                with open(stats_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cached stats: {e}, recomputing...")

        print("Computing dataset statistics...")
        action_values = []
        proprio_values = []

        sample_size = min(self.num_episodes, max(1, self.num_episodes // 10))
        if self.num_episodes <= 10:
            episode_indices = np.arange(self.num_episodes, dtype=int)
        else:
            episode_indices = np.linspace(0, self.num_episodes - 1, sample_size, dtype=int)

        for episode_idx in tqdm(episode_indices, desc="Sampling for statistics"):
            episode_data = self._load_episode_stats_data(int(episode_idx))
            action_values.append(episode_data["actions"])
            proprio_values.append(episode_data["proprio"])

        all_actions = np.concatenate(action_values, axis=0)
        all_proprio = np.concatenate(proprio_values, axis=0)

        stats = {
            "actions_min": np.min(all_actions, axis=0),
            "actions_max": np.max(all_actions, axis=0),
            "actions_mean": np.mean(all_actions, axis=0),
            "actions_std": np.std(all_actions, axis=0),
            "actions_median": np.median(all_actions, axis=0),
            "proprio_min": np.min(all_proprio, axis=0),
            "proprio_max": np.max(all_proprio, axis=0),
            "proprio_mean": np.mean(all_proprio, axis=0),
            "proprio_std": np.std(all_proprio, axis=0),
            "proprio_median": np.median(all_proprio, axis=0),
        }

        try:
            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)
        except Exception as e:
            print(f"Failed to cache statistics: {e}")

        return stats

    def __len__(self):
        return self.epoch_length

    def __del__(self):
        """Clean up HDF5 file cache."""
        if hasattr(self, "hdf5_file_handles"):
            self._close_hdf5_files()

    def __getitem__(self, idx):
        """
        Fetch sample by index (with lazy loading).
        
        Returns dict with keys:
          video, actions, t5_text_embeddings, t5_text_mask, fps, padding_mask, image_size,
          proprio, __key__, action_latent_idx, current_wrist_image_latent_idx, 
          current_image_latent_idx, value_latent_idx, rollout_data_mask, etc.
        """
        # Get episode from step index
        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]

        # Calculate future frame index
        future_frame_idx = min(
            relative_step_idx + self.chunk_size,
            self.episode_metadata[episode_idx]["num_steps"] - 1
        )

        episode_data = self._load_episode_sample(
            episode_idx=episode_idx,
            relative_step_idx=relative_step_idx,
            future_frame_idx=future_frame_idx,
        )

        if self.normalize_actions:
            episode_data["actions"] = self._normalize_array(episode_data["actions"], "actions")

        if self.normalize_proprio:
            episode_data["current_proprio"] = self._normalize_array(
                episode_data["current_proprio"], "proprio"
            )
            episode_data["future_proprio"] = self._normalize_array(
                episode_data["future_proprio"], "proprio"
            )

        # Get frames
        current_wrist = episode_data["current_wrist"]
        current_primary = episode_data["current_primary"]
        future_wrist = episode_data["future_wrist"]
        future_primary = episode_data["future_primary"]
        current_proprio = episode_data["current_proprio"]
        future_proprio = episode_data["future_proprio"]

        # Build image sequence (matching LIBERO/ROBOCASA pattern)
        # Structure: [blank, (proprio), wrist, primary, action, (future_proprio), future_wrist, future_primary, (value)]
        image_list = []
        current_sequence_idx = 0

        # Blank first input frame
        first_input = np.expand_dims(np.zeros_like(current_primary), axis=0)
        image_list.append(first_input)
        current_sequence_idx += 1

        # Current proprio (if used)
        if self.use_proprio:
            blank_img = np.zeros_like(current_primary)
            blank_img_dup = duplicate_array(blank_img, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_img_dup)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            current_proprio_latent_idx = -1

        # Current wrist image
        if self.use_wrist_images:
            wrist_dup = duplicate_array(current_wrist, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_dup)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            current_wrist_image_latent_idx = -1

        # Current primary image
        if self.use_third_person_images:
            primary_dup = duplicate_array(current_primary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(primary_dup)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            current_image_latent_idx = -1

        # Action chunk (blank image as placeholder)
        action_blank = duplicate_array(np.zeros_like(current_primary), total_num_copies=self.num_duplicates_per_image)
        image_list.append(action_blank)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Future proprio
        if self.use_proprio:
            blank_img = np.zeros_like(current_primary)
            blank_img_dup = duplicate_array(blank_img, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_img_dup)
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            future_proprio_latent_idx = -1

        # Future wrist image
        if self.use_wrist_images:
            future_wrist_dup = duplicate_array(future_wrist, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_dup)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            future_wrist_image_latent_idx = -1

        # Future primary image
        if self.use_third_person_images:
            future_primary_dup = duplicate_array(future_primary, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_primary_dup)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        else:
            future_image_latent_idx = -1

        # Always include the value slot in the video sequence when using the state_t=9 robot setup.
        value_blank = duplicate_array(np.zeros_like(current_primary), total_num_copies=self.num_duplicates_per_image)
        image_list.append(value_blank)
        value_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # Concatenate and preprocess
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Get action chunk
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get next action chunk
        next_relative_step_idx = min(
            relative_step_idx + self.chunk_size,
            episode_data["num_steps"] - 1
        )
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Value function returns (compute on-the-fly if needed)
        if self.return_value_function_returns:
            returns = compute_monte_carlo_returns(
                episode_data["num_steps"], terminal_reward=1.0, gamma=self.gamma
            )
            value_function_return = returns[future_frame_idx]
            next_future_frame_idx = min(next_relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
            next_value_function_return = returns[next_future_frame_idx]
        else:
            value_function_return = float("-100")
            next_value_function_return = float("-100")

        # Prepare return dict
        return_dict = {
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": self._get_text_embedding(episode_data["command"]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": current_proprio if self.use_proprio else np.zeros_like(current_proprio),
            "future_proprio": future_proprio if self.use_proprio else np.zeros_like(future_proprio),
            "__key__": idx,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 1,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }

        return return_dict
