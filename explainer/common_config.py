import torch
import numpy as np
from model import ResNet18
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import pandas as pd
import pathlib
from explainer.path import Path


def get_model(p):
    if p["model"] == "ResNet18":
        return ResNet18(**p["model_kwargs"])
    else:
        raise NotImplementedError


def get_test_dataloader(p, dataset):
    train_sampler = DistributedSampler(dataset) if p['use_ddp'] else None

    return torch.utils.data.DataLoader(
                            dataset,
                            num_workers=p["num_workers"],
                            batch_size=p["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            # sampler=train_sampler
                            ), train_sampler


def get_customized_dataset(filename=None, return_image_only=False):
    file = np.load(filename)
    return DictDataset(file['x'], file['y'], return_image_only=return_image_only)


class DictDataset(Dataset):
    def __init__(self,
                 data,
                 label,
                 transform=None,
                 return_image_only=False
                 ):
        self.data_dict = {
            'image': torch.from_numpy(np.abs(data)).float(),
            'target': torch.from_numpy(label).long()
        }
        self._dataset_folder = pathlib.Path(Path.db_root_dir("BCICIV2a"))
        self._transform = transform
        self._return_image_only = return_image_only

        self._load_meta()

    @property
    def parts_name_index(self):
        return self._parts_name_index

    def __len__(self):
        return len(self.data_dict['image'])

    # def __getitem__(self, idx):
    #     return {'image': self.data_dict['image'][idx], 'target': self.data_dict['target'][idx]}
    def __getitem__(self, idx):

        image = self.data_dict['image'][idx]
        width, height = image.shape[1], image.shape[2]

        # return image only
        if self._return_image_only:
            if self._transform is None:
                return image

            else:
                if "albumentations" in str(type(self._transform)):
                    return self._transform(image=np.array(image, dtype=np.uint8))[
                        "image"
                    ]
                else:
                    return self._transform(image)

        target = self.data_dict['target'][idx]

        # load parts
        part_ids = np.array(self._meta.iloc[idx].part_id, dtype=np.int32) - 1
        part_locs = np.stack(
            (
                np.array(self._meta.iloc[idx].x, dtype=np.float32),
                np.array(self._meta.iloc[idx].y, dtype=np.float32),
            ),
            axis=1,
        )

        valid_x_coords = np.logical_and(part_locs[:, 0] > 0, part_locs[:, 0] < width)
        valid_y_coords = np.logical_and(part_locs[:, 1] > 0, part_locs[:, 1] < height)
        valid_coords = np.logical_and(valid_x_coords, valid_y_coords)
        part_ids = part_ids[valid_coords]
        part_locs = part_locs[valid_coords]

        # transform
        if self._transform is not None:
            sample = self._transform(
                image=np.array(image, dtype=np.uint8),
                keypoints=part_locs,
                keypoints_ids=part_ids,
            )
            sample = {
                "image": sample["image"],
                "part_locs": np.array(sample["keypoints"]),
                "part_ids": np.array(sample["keypoints_ids"]),
            }

        else:
            sample = {
                "image": image,
                "part_locs": part_locs,
                "part_ids": part_ids,
            }

        # return parts as binary mask on 7 x 7 grid to ease evaluation
        part_locs = sample["part_locs"]
        part_ids = sample["part_ids"]
        n_pix_per_cell_h = sample["image"].shape[1] // 7
        n_pix_per_cell_w = sample["image"].shape[2] // 7
        parts = np.zeros((len(self.parts_name_index), 7, 7), dtype=np.uint8)
        for part_loc, part_id in zip(part_locs, part_ids):
            x_coord = int(part_loc[0] // n_pix_per_cell_w)
            y_coord = int(part_loc[1] // n_pix_per_cell_h)
            # new_part_id = self._parts_index_remap[part_id]
            parts[part_id, y_coord, x_coord] = 1

        output = {
            "image": sample["image"],
            "target": target,
            "parts": parts,
        }

        return output

    def get_target(self, target):
        return (
            np.argwhere(np.array(self.data_dict["target"].tolist()) == target).reshape(-1).tolist()
        )

    def _load_meta(self):
        # create a dataframe to store meta information from self.data_dict
        self._meta = pd.DataFrame(
            {
                "img_id": np.arange(1, len(self.data_dict["image"])+1),
                "target": self.data_dict["target"].tolist(),
            }
        )

        self._original_parts_name_index = {}

        with open(self._dataset_folder.joinpath("parts", "parts_EEG_10kp.txt")) as f:
            for line in f:
                cols = line.strip().split(" ", 1)
                assert len(cols) == 2
                part_id = int(cols[0]) - 1
                part_name = cols[1]
                self._original_parts_name_index[part_id] = part_name

        self._inverse_original_parts_name_index = {
            value: key for key, value in self._original_parts_name_index.items()
        }

        image_parts = pd.read_csv(
            self._dataset_folder.joinpath("parts", "part_locs_EEG_10kp.txt"),
            sep=" ",
            names=["img_id", "part_id", "x", "y", "visible"],
        )
        image_parts = image_parts[image_parts["visible"] == 1]
        image_parts = image_parts.groupby("img_id")[["part_id", "x", "y"]].agg(
            lambda x: list(x)
        )
        self._parts_name_index = {i: f'kp{i + 1}' for i in range(10)}
        self._meta = self._meta.join(image_parts, on="img_id")

