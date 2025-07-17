import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)

    # Оригинальное изображение
    orig_np = np.array(orig_resized)  # Если PIL.Image -> (H, W, C) для RGB, (H, W) для Grayscale
    if orig_np.ndim == 3:
        if orig_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            orig_np = np.transpose(orig_np, (1, 2, 0))
        elif orig_np.shape[1] == 3:  # (H, C, W) -> (H, W, C)
            orig_np = np.transpose(orig_np, (0, 2, 1))
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_resized)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
