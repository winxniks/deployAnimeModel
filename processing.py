import torch
import torchvision.transforms as transforms
from PIL import Image
import os
#from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str = "checkpoints/final_netG.pth") -> torch.nn.Module:
    """
    Загружает обученную модель генератора (Generator) для преобразования фото в аниме.

    Args:
        model_path (str): Путь к файлу .pth с весами модели.

    Returns:
        torch.nn.Module: Готовая модель в режиме eval на нужном устройстве.
    """
    from models.generator import Generator
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    return netG


def get_no_aug_transform(image: Image.Image, size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Предобработка изображения: resize, to tensor, normalize.

    Args:
        image: Путь к изображению.
        size: Целевой размер (по умолчанию 256).

    Returns:
        tuple: (тензор [1,3,H,W] на device, оригинальный размер (w, h))
    """
    orig_size = image.size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, orig_size


def tensor_to_pil_imagenet(pred_tensor: torch.Tensor, orig_size: tuple[int, int] | None = None) -> Image.Image:
    """
    Денормализация по ImageNet + конвертация в PIL.

    Args:
        pred_tensor: Выход модели [1, C, H, W]
        orig_size: Оригинальный размер (w, h), если нужно восстановить.

    Returns:
        PIL.Image.Image: Обработанное изображение.
    """
    pred_tensor = pred_tensor.detach().cpu().squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    pred_tensor = pred_tensor * std + mean
    pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    img = transforms.ToPILImage()(pred_tensor)
    if orig_size is not None:
        img = img.resize(orig_size, Image.BICUBIC)
    return img


def tensor_to_pil_autonorm(pred_tensor: torch.Tensor, orig_size: tuple[int, int] | None = None) -> Image.Image:
    """
    Автонормализация: min-max нормализация тензора + конвертация в PIL.

    Args:
        pred_tensor: Выход модели.
        orig_size: Оригинальный размер.

    Returns:
        PIL.Image.Image: Обработанное изображение.
    """
    pred_tensor = pred_tensor.detach().cpu().squeeze(0)
    min_val, max_val = pred_tensor.min(), pred_tensor.max()
    if max_val > min_val:
        pred_tensor = (pred_tensor - min_val) / (max_val - min_val)
    else:
        pred_tensor = torch.zeros_like(pred_tensor)
    pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    img = transforms.ToPILImage()(pred_tensor)
    if orig_size is not None:
        img = img.resize(orig_size, Image.BICUBIC)
    return img


def generate(model: torch.nn.Module, input_image: Image.Image, use_imagenet: bool = False) -> Image.Image:
    """
    Обрабатывает изображение моделью и возвращает стилизованное изображение.

    Args:
        model: Загруженная модель-генератор.
        input_image: Входное изображение в режиме RGB.
        use_imagenet: Использовать ли ImageNet-денормализацию при постобработке.

    Returns:
        PIL.Image.Image: Преобразованное изображение в аниме-стиле.
    """
    # Предобработка: PIL → тензор [1, 3, 256, 256]
    image_tensor, orig_size = get_no_aug_transform(input_image)

    # Генерация
    with torch.no_grad():
        output_tensor = model(image_tensor)

    # Постобработка: тензор → PIL
    if use_imagenet:
        output_pil = tensor_to_pil_imagenet(output_tensor, orig_size)
    else:
        output_pil = tensor_to_pil_autonorm(output_tensor, orig_size)

    return output_pil