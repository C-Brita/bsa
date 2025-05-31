# Modified for PyTorch compatibility from https://github.com/smsharma/eqnn-jax/tree/main/benchmarks/galaxies.

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_cluster import knn_graph

from functools import partial
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


MEAN_HALOS_DICT = {
    "x": 499.91877075908684,
    "y": 500.0947802559321,
    "z": 499.964508664328,
    "Jx": 212560050888254.06,
    "Jy": 349712732356652.25,
    "Jz": -100259775332585.12,
    "v_x": -0.0512854365234889,
    "v_y": -0.01263126442198149,
    "v_z": -0.06458034372345466,
    "M200c": 321308383763206.9,
    "Rvir": 1424.4071655758826,
}
STD_HALOS_DICT = {
    "x": 288.71092533309235,
    "y": 288.7525818573022,
    "z": 288.70234893905575,
    "Jx": 2.4294356933448945e18,
    "Jy": 2.3490019110577966e18,
    "Jz": 2.406422979830857e18,
    "v_x": 344.0231468131901,
    "v_y": 343.9333673335964,
    "v_z": 344.071876710777,
    "M200c": 405180433634974.75,
    "Rvir": 298.14502916425675,
}
MEAN_PARAMS_DICT = {
    "Omega_m": 0.29994175,
    "Omega_b": 0.049990308,
    "h": 0.69996387,
    "n_s": 0.9999161,
    "sigma_8": 0.7999111,
}
STD_PARAMS_DICT = {
    "Omega_m": 0.11547888,
    "Omega_b": 0.017312417,
    "h": 0.11543678,
    "n_s": 0.115482554,
    "sigma_8": 0.11545073,
}
MEAN_TPCF_VEC = [1.47385902e+01, 4.52754450e+00, 1.89688166e+00, 1.00795493e+00,
                6.09400184e-01, 3.98518764e-01, 2.79545049e-01, 2.01358601e-01,
                1.53487009e-01, 1.18745081e-01, 9.51346027e-02, 7.83494908e-02,
                6.92183650e-02, 6.41181254e-02, 6.05992822e-02, 5.77399258e-02,
                5.27855615e-02, 4.64777462e-02, 3.97492901e-02, 3.17941626e-02,
                2.49663476e-02, 1.92553030e-02, 1.28971533e-02, 9.48586955e-03]
STD_TPCF_VEC = [8.37356624, 2.36190046, 1.15493691, 0.73567994, 0.52609708, 0.40239359,
                0.32893873, 0.27772011, 0.24173466, 0.21431925, 0.19276616, 0.17816693,
                0.16773013, 0.15968612, 0.15186733, 0.14234885, 0.13153203, 0.11954234,
                0.10549666, 0.09024256, 0.07655078, 0.06350282, 0.05210615, 0.0426435]


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


def _parse_function(
    proto,
    features=["x", "y", "z", "J_x", "J_y", "J_z", "v_x", "v_y", "v_z", "M200c"],
    params=["Omega_m", "Omega_b", "h", "n_s", "sigma_8"],
    include_tpcf=False,
):
    keys_to_features = {k: tf.io.FixedLenFeature([], tf.string) for k in features}
    keys_to_params = {k: tf.io.FixedLenFeature([], tf.string) for k in params}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_params = tf.io.parse_single_example(proto, keys_to_params)

    # Convert each feature from a serialized string to a tensor and store in a list
    feature_tensors = [tf.io.parse_tensor(parsed_features[k], out_type=tf.float32) for k in features]
    param_tensors = [tf.io.parse_tensor(parsed_params[k], out_type=tf.float32) for k in params]

    # Stack the feature tensors to create a single tensor
    # Each tensor must have the same shape
    stacked_features = tf.stack(feature_tensors, axis=1)  # Creates a [num_points, num_features] tensor
    stacked_params = tf.stack(param_tensors, axis=0)  # Creates a [num_params] tensor

    if include_tpcf:
        key_to_tpcf = {"tpcf": tf.io.FixedLenFeature([], tf.string)}
        parsed_tpcf = tf.io.parse_single_example(proto, key_to_tpcf)
        tpcf_tensor = tf.io.parse_tensor(parsed_tpcf["tpcf"], out_type=tf.float32)
        return stacked_features, stacked_params, tpcf_tensor

    return stacked_features, stacked_params


def get_halo_dataset(
    batch_size=64,
    num_samples=None,  # If not None, only return this many samples
    split="train",
    features=["x", "y", "z", "J_x", "J_y", "J_z", "v_x", "v_y", "v_z", "M200c"],
    params=["Omega_m", "sigma_8"],
    return_mean_std=False,
    standardize=True,
    tfrecords_path="quijote_records",
    include_tpcf=False,
):

    files = tf.io.gfile.glob(f"{tfrecords_path}/halos*{split}*.tfrecord")
    dataset = tf.data.TFRecordDataset(files)

    if num_samples is not None:
        dataset = dataset.take(num_samples)
        num_total = num_samples  # Adjust num_total if num_samples is specified
    else:
        num_total = dataset.reduce(0, lambda x, _: x + 1).numpy()
        dataset = dataset.take(num_total)

    dataset = dataset.map(partial(_parse_function, features=features, params=params, include_tpcf=include_tpcf))

    # Get mean and std as tf arrays
    mean = tf.constant([MEAN_HALOS_DICT[f] for f in features], dtype=tf.float32)
    std = tf.constant([STD_HALOS_DICT[f] for f in features], dtype=tf.float32)
    mean_params = tf.constant([MEAN_PARAMS_DICT[f] for f in params], dtype=tf.float32)
    std_params = tf.constant([STD_PARAMS_DICT[f] for f in params], dtype=tf.float32)
    mean_tpcf = tf.constant(MEAN_TPCF_VEC)
    std_tpcf = tf.constant(STD_TPCF_VEC)

    if standardize:
        if include_tpcf:
            dataset = dataset.map(lambda x, p, t: ((x - mean) / std, (p - mean_params) / std_params, (t - mean_tpcf) / std_tpcf))
        else:
            dataset = dataset.map(lambda x, p: ((x - mean) / std, (p - mean_params) / std_params))

    if batch_size is None:
        batch_size = num_total

    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_total))
    dataset = dataset.batch(batch_size)

    if return_mean_std:
        return dataset, num_total, mean, std, mean_params, std_params
    else:
        return dataset, num_total


class CosmologyDataset(Dataset):
    def __init__(self, task, split, num_samples, tfrecords_path, knn):
        self.task = task

        if task == "graph":
            features = ["x", "y", "z"]

        elif task == "node":
            features = ["x", "y", "z", "v_x", "v_y", "v_z"]

        dataset, _ = get_halo_dataset(
            batch_size=1,
            num_samples=num_samples,
            split=split,
            standardize=True,
            return_mean_std=False,
            features=features,
            params=["Omega_m", "sigma_8"],
            include_tpcf=False,
            tfrecords_path=tfrecords_path,
        )

        self.split = split
        self.num_samples = num_samples
        self.pos, self.target = self.convert_to_torch(dataset, num_samples)
        self.knn = knn
        self.dataset = [self.getitem(i) for i in tqdm(range(num_samples))]
        self.pos, self.target = None, None

    def convert_to_torch(self, dataset, num_data):
        if self.task == "graph":
            pos, target = [], []
            for i, (pos_, target_) in enumerate(tqdm(dataset)):
                pos.append(pos_.numpy())
                target.append(target_.numpy())
                if i == num_data:
                    break
        else:
            pos, target = [], []
            for i, (x, _) in enumerate(tqdm(dataset)):
                pos.append(x[..., :3].numpy())
                target.append(x[..., 3:].numpy())
                if i == num_data:
                    break
        return [torch.from_numpy(x).squeeze() for x in pos], [torch.from_numpy(x).squeeze() for x in target]

    def __len__(self):
        return self.num_samples

    def getitem(self, idx):
        output_dict = {"pos": self.pos[idx], "target": self.target[idx]}
        return output_dict

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch: list):
        """
        Collate function for the dataset. Mainly handles batch offsets.
        """
        num_nodes = torch.tensor([d["pos"].shape[0] for d in batch])
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(num_nodes)
        batch["edge_index"] = knn_graph(batch["pos"], k=self.knn, batch=batch["batch_idx"])
        return batch
