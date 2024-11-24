import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

# Data Augmentation Version for Object Detection Task
class COCODataset(Dataset):
    def __init__(self, root, train, transform=None):
        super().__init__()
        self.directory = "train2017" if train else "val2017"
        annotations = os.path.join(root, "annotations", f"instances_{self.directory}.json")

        self.coco = COCO(annotations)
        self.root = root
        self.transform = transform
        self.categories = self._get_categories()

        # category idx 중 없는거 제외하고 순서대로 remapping
        self.new_categories = {}
        for i in range(len(self.categories)):
            self.new_categories[list(self.categories.values())[i]] = i
            
        # id_list 재작성
        self.id_list = []
        for img_id in list(self.coco.imgs.keys()):
            ids = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            if len(ids) != 0:
                self.id_list.append(img_id)

        # explicit exception of error of bbox index
        if self.directory == 'train':
            del self.id_list[self.id_list.index(200365)]
            del self.id_list[self.id_list.index(550395)]

    def _get_categories(self):
        categories = {0: "background"}
        for category in self.coco.cats.values():
            categories[category["id"]] = category["name"]
        return categories

    def __getitem__(self, index):
        se_id = self.id_list[index]
        file_name = self.coco.loadImgs(se_id)[0]["file_name"]
        image_path = os.path.join(self.root + self.directory + '/' + file_name)
        
        # load img
        image = Image.open(image_path).convert("RGB")

        # load ann
        boxes = []
        labels = []
        anns = self.coco.loadAnns(self.coco.getAnnIds(se_id))
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            category_id = self.new_categories[self.categories[ann["category_id"]]] # category remapping
            labels.append(category_id)

        target = {"image_id": torch.LongTensor([se_id]),
                  "boxes": torch.FloatTensor(boxes),
                  "labels": torch.LongTensor(labels)
                 }
        # img, box augmentation (val의 경우 transform 다르게 설정)
        image, target = self.transform(image, target)
        image_id = target['image_id']
        
        return image, target, image_id

    def __len__(self):
        return len(self.id_list)
