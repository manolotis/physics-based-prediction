import random
import os
from torch.utils.data import Dataset
import torch
import numpy as np


class CVDataset(Dataset):

    def __init__(self, config):
        self._data_path = config["data_path"]
        self._config = config
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]
        self._files = sorted(self._files)

        assert config["n_shards"] > 0

        if "max_length" in config:
            self._files = self._files[:config["max_length"]]

            if config["n_shards"] > 1:
                raise ValueError("Limiting training data with limit>0 and n_shards>1. Choose one or the other.")

        if config["n_shards"] > 1:
            self._files = [file for (i, file) in enumerate(self._files) if i % config["n_shards"] == 0]

        random.shuffle(self._files)

        assert len(self._files) > 0

    def __len__(self):
        return len(self._files)

    def _compute_agent_type_and_is_sdc_ohe(self, data, subject):
        I = np.eye(5)
        agent_type_ohe = I[np.array(data[f"{subject}/agent_type"])]
        is_sdc = np.array(data[f"{subject}/is_sdc"]).reshape(-1, 1)
        ohe_data = np.concatenate([agent_type_ohe, is_sdc], axis=-1)[:, None, :]
        ohe_data = np.repeat(ohe_data, data["target/history/xy"].shape[1], axis=1)
        return ohe_data

    def _compute_input_data(self, data):
        keys_to_stack = self._config["input_data"]
        data[f"target/history/lstm_data"] = np.concatenate([data[f"target/history/{k}"] for k in keys_to_stack],
                                                           axis=-1)
        data[f"target/history/lstm_data"] *= data[f"target/history/valid"]

        return data

    def __getitem__(self, idx):
        try:
            np_data = dict(np.load(self._files[idx], allow_pickle=True))
        except:
            print("Error reading", self._files[idx])
            idx = 0
            np_data = dict(np.load(self._files[0], allow_pickle=True))

        np_data["scenario_id"] = np_data["scenario_id"].item()
        np_data["filename"] = self._files[idx]
        np_data = self._compute_input_data(np_data)

        return np_data

    @staticmethod
    def collate_fn(batch):
        batch_keys = batch[0].keys()
        result_dict = {k: [] for k in batch_keys}

        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])

        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))

        result_dict["batch_size"] = len(batch)
        return result_dict
