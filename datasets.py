from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

classes = ['Lion', 'Tiger', 'Elephant', 'Giraffe', 'Zebra', 'Bear', 'Fox', 'Leopard', 'Hedgehog', 'Lynx']


class AnimalsDataset(Dataset):
    def __init__(self, data_path, classes, augmentations):

        class_paths = []
        self.X = []
        self.y = []

        self.augmentations = augmentations

        labels_name = {class_i: index for index, class_i in enumerate(classes)}

        if classes:
            for i in classes:
                class_paths.append(f'{data_path}/{i}')
        else:
            glob_path = f'{data_path}/*'
            class_paths = glob(glob_path)

        for (class_curr, path) in (zip(classes, class_paths)):
            glob_path = f'{path}/*.jpg'
            for img_path in (glob(glob_path)):
                self.X.append(img_path)
                self.y.append(labels_name[class_curr])

    def __getitem__(self, idx):
        img, label = self.X[idx], self.y[idx]
        img = Image.open(img).convert('RGB')
        img = self.augmentations(img)

        return img, label

    def __len__(self):
        return len(self.X)


def get_loaders(batch_size=64):
    augmentations = T.Compose([
        T.Resize((224, 224)),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize([0.4861202538013458, 0.48226398229599, 0.4337790906429291],
                    [0.2418711632490158, 0.23503021895885468, 0.25225475430488586])
    ])

    preprocess_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.4861202538013458, 0.48226398229599, 0.4337790906429291],
                    [0.2418711632490158, 0.23503021895885468, 0.25225475430488586])
    ])

    path_data_train = 'D:/animals/dataset/train'
    train_dataset = AnimalsDataset(data_path=path_data_train, classes=classes, augmentations=augmentations)

    path_data_test = 'D:/animals/dataset/test'
    test_dataset = AnimalsDataset(data_path=path_data_test, classes=classes, augmentations=preprocess_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
