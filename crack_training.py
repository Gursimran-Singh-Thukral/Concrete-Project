"""Full-patch transforms: cropping/erasing can remove the only labeled crack."""
from copy import copy
import torchvision.transforms as T
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class FullPatchDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix="", **kwargs):
        super().__init__(root, args, augment, prefix, **kwargs)
        transforms = [T.Resize((args.imgsz, args.imgsz))]
        if augment:
            transforms += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            ]
        self.torch_transforms = T.Compose(transforms + [T.ToTensor()])


class FullPatchValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return FullPatchDataset(img_path, self.args, augment=False, prefix=self.args.split)


class FullPatchTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return FullPatchDataset(img_path, self.args, augment=mode == "train", prefix=mode, names=self.data["names"])

    def get_validator(self):
        return FullPatchValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
