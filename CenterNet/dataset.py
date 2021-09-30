import csv
import os
from typing import List

from PIL import Image
from torchvision.datasets import Kitti


class CustomKittiDataset(Kitti):
    """<root>
    └── training
    |   ├── image_2
    |   └── label_2
    └── testing
        └── image_2
    """

    @property
    def _raw_folder(self) -> str:
        return self.root

    def _parse_target(self, index: int) -> List:
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "type": line[0],
                        "bbox": [float(x) for x in line[4:8]],
                        "distance": line[13],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                    }
                )
        return target
