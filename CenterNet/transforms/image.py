import copy

import cv2
import numpy as np
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage, Keypoint
from imgaug.augmenters import Augmenter, Identity


class ImageAugmentation:
    def __init__(
        self, imgaug_augmenter: Augmenter = Identity(), img_transforms=None, num_joints=17
    ):
        self.ia_sequence = imgaug_augmenter
        self.img_transforms = img_transforms
        self.num_joints = num_joints

    def __call__(self, img, target):
        # PIL to array BGR
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        target = copy.deepcopy(target)

        # Prepare augmentables for imgaug
        bounding_boxes = []
        for idx in range(len(target)):
            ann = target[idx]

            # Bounding Box
            box = ann["bbox"]
            bounding_boxes.append(
                BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=idx)
            )

        # Augmentation
        image_aug, bbs_aug = self.ia_sequence(
            image=img,
            bounding_boxes=BoundingBoxesOnImage(bounding_boxes, shape=img.shape),
        )

        # Write augmentation back to annotations
        for bb in bbs_aug:
            target[bb.label]["bbox"] = [bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1]

        # torchvision transforms
        if self.img_transforms:
            image_aug = self.img_transforms(image_aug)

        return image_aug, target
