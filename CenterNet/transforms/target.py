class CategoryIdToClass:
    def __init__(self, object_types):
        self.object_types = object_types

    def __call__(self, img, target):
        for ann in target:
            ann["class_id"] = self.object_types[ann["type"]]

        return img, target
