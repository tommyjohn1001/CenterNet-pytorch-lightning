import math

import torch
import numpy as np

from utils.gaussian import draw_umich_gaussian, draw_msra_gaussian, gaussian_radius


class CenterDetectionSample:
    def __init__(
        self,
        down_ratio=4,
        num_classes=80,
        max_objects=128,
        gaussian_type="msra",
    ):

        self.down_ratio = down_ratio

        self.num_classes = num_classes
        self.max_objects = max_objects
        self.gaussian_type = gaussian_type

    @staticmethod
    def _coco_box_to_bbox(box):
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]],
            dtype=np.float32)

    def scale_point(self, point, output_size):
        x, y = point / self.down_ratio
        output_h, output_w = output_size

        x = np.clip(x, 0, output_w - 1)
        y = np.clip(y, 0, output_h - 1)

        return [x, y]

    def __call__(self, img, target):
        _, input_w, input_h = img.shape

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        hm = torch.zeros((self.num_classes, output_w, output_h), dtype=torch.float32)
        wh = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        reg = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        ind = torch.zeros(self.max_objects, dtype=torch.int64)
        reg_mask = torch.zeros(self.max_objects, dtype=torch.bool)

        draw_gaussian = (
            draw_msra_gaussian if self.gaussian_type == "msra" else draw_umich_gaussian
        )

        num_objects = min(len(target), self.max_objects)
        for k in range(num_objects):
            ann = target[k]
            bbox = self._coco_box_to_bbox(ann["bbox"])
            cls_id = ann["class_id"]

            # Scale to output size
            bbox[:2] = self.scale_point(bbox[:2], (output_h, output_w))
            bbox[2:] = self.scale_point(bbox[2:], (output_h, output_w))

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(1e-5, int(radius))
                ct = torch.FloatTensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                ct_int = ct.to(torch.int32)

                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = torch.tensor([1.0 * w, 1.0 * h])
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

        ret = {"hm": hm, "reg_mask": reg_mask, "ind": ind, "wh": wh, "reg": reg}

        return img, ret
