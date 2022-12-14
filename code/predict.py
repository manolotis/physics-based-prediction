import sys

sys.path.append("/home/manolotis/sandbox/robustness_benchmark/")

import torch

torch.multiprocessing.set_sharing_strategy('file_system')

# from model.data import get_dataloader, dict_to_cuda, normalize
from physicsBased.code.model.data import CVDataset
from torch.utils.data import Dataset, DataLoader
from multipathPP.code.utils.predict_utils import get_config, parse_arguments
from physicsBased.code.models import CV
import os
import glob
import random
import numpy as np
from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_dataloader(config):
    dataset = CVDataset(config["dataset_config"])
    dataloader = DataLoader(dataset, collate_fn=CVDataset.collate_fn, **config["dataloader_config"])
    return dataloader


def generate_filename(scene_data, agent_index):
    scenario_id = scene_data["scenario_id"][agent_index]
    agent_id = scene_data["agent_id"][agent_index]
    agent_type = scene_data["target/agent_type"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"


args = parse_arguments()
config = get_config(args)

test_dataloader = get_dataloader(config["test"]["data_config"])

savefolder = os.path.join(config["test"]["output_config"]["out_path"], config["model"]["name"])
if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)

for data in tqdm(test_dataloader):

    xy = data["target/history/xy"].numpy()
    v_xy = data["target/history/v_xy"].numpy()
    valid = data["target/history/valid"].numpy()
    is_valid = data["target/history/valid"].numpy() > 0
    t = np.arange(0.1, 8.01, 0.1)

    x = xy[:, -1:, 0]
    y = xy[:, -1:, 1]
    vx = v_xy[:, -1:, 0]
    vy = v_xy[:, -1:, 1]

    # ToDo: acocunt for possibly invalid current timesteps

    x_pred, y_pred = CV.predict(x, y, vx, vy, t)

    coordinates = np.stack([x_pred, y_pred], axis=-1)

    for agent_index, agent_id in enumerate(data["agent_id"]):

        if not valid[agent_index, -1]:
            continue
        filename = generate_filename(data, agent_index)
        savedata = {
            "scenario_id": data["scenario_id"][agent_index],
            "agent_id": data["agent_id"][agent_index],
            "agent_type": data["target/agent_type"][agent_index].flatten(),
            "coordinates": coordinates[agent_index:agent_index + 1],
            "probabilities": None,
            "target/history/xy": data["target/history/xy"][agent_index],
            "target/future/xy": data["target/future/xy"][agent_index],
            "target/history/valid": data["target/history/valid"][agent_index],
            "target/future/valid": data["target/future/valid"][agent_index]
        }

        np.savez_compressed(os.path.join(savefolder, filename), **savedata)
