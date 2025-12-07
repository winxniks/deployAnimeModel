import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="checkpoints/final_netG.pth"):
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


def get_no_aug_transform(input_path, size=256):
    """
    Предобработка изображения: resize, to tensor, normalize.

    Args:
        input_path (str): Путь к изображению.
        size (int): Целевой размер (по умолчанию 256).

    Returns:
        tuple: (тензор [1,3,H,W] на device, оригинальный размер (w, h))
    """
    image = Image.open(input_path).convert('RGB')
    orig_size = image.size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, orig_size


def tensor_to_pil_imagenet(pred_tensor, orig_size=None):
    """
    Денормализация по ImageNet + конвертация в PIL.

    Args:
        pred_tensor (torch.Tensor): Выход модели [1, C, H, W]
        orig_size (tuple): Оригинальный размер (w, h), если нужно восстановить.

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


def tensor_to_pil_autonorm(pred_tensor, orig_size=None):
    """
    Автонормализация: min-max нормализация тензора.

    Args:
        pred_tensor (torch.Tensor): Выход модели.
        orig_size (tuple): Оригинальный размер.

    Returns:
        PIL.Image.Image: Результат.
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


def generate(model, input_path, use_imagenet=False):
    """
    Генерирует аниме-стилизацию изображения.

    Args:
        model (torch.nn.Module): Загруженная модель.
        input_path (str): Путь к входному изображению.

    Returns:
        tuple: (предсказанный тензор, оригинальный размер)
    """
    with torch.no_grad():
        image_tensor, orig_size = get_no_aug_transform(input_path)
        pred_image = model(image_tensor)
        
    if use_imagenet:
        pil_img = tensor_to_pil_imagenet(pred_image, orig_size)
    else:
        pil_img = tensor_to_pil_autonorm(pred_image, orig_size)

    return pil_img


def save_image(pil_img, output_path):
    """
    Сохраняет изображение.

    Args:
        pil_img (torch.Tensor): Выход модели.
        output_path (str): Путь для сохранения.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_img.save(output_path)
    print(f"✅ Сохранено: {output_path}")


def process_single_image(input_path, output_path, model, use_imagenet=False):
    """
    Обрабатывает один файл.

    Args:
        input_path (str): Путь к входному изображению.
        output_path (str): Путь для сохранения.
        model (torch.nn.Module): Загруженная модель (чтобы не грузить каждый раз).
        use_imagenet (bool): Использовать ImageNet-денормализацию.

    Returns:
        bool: Успешно ли обработано.
    """
    try:
        pred_image = generate(model, input_path)
        save_image(pred_image, output_path)
        return True
    except Exception as e:
        print(f"❌ Ошибка при обработке {input_path}: {e}")
        return False


def process_folder(input_folder, output_folder, model_path="deployAnimeModel/checkpoints/final_netG.pth", use_imagenet=False):
    """
    Обрабатывает все изображения в папке.

    Args:
        input_folder (str): Путь к папке с входными изображениями.
        output_folder (str): Путь к папке для сохранения результатов.
        model_path (str): Путь к весам модели.
        use_imagenet (bool): Использовать ImageNet-денормализацию.
    """
    # Поддерживаемые форматы
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"❌ Папка не найдена: {input_folder}")
        return

    # Создаём выходную папку
    output_path.mkdir(parents=True, exist_ok=True)

    # Загружаем модель один раз
    print("🔄 Загрузка модели...")
    model = load_model(model_path)
    print("✅ Модель загружена.")

    # Список изображений
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"⚠️ Нет изображений в папке: {input_folder}")
        return

    print(f"🚀 Найдено {len(image_files)} изображений. Начинаем обработку...")

    success_count = 0
    for img_file in image_files:
        out_file = output_path / img_file.name
        if process_single_image(str(img_file), str(out_file), model, use_imagenet):
            success_count += 1

    print(f"✅ Готово: обработано {success_count}/{len(image_files)} изображений.")
    print(f"📁 Результаты сохранены в: {output_folder}")


if __name__ == "__main__":
    input_dir = "data/input"
    output_dir = "data/output"

    process_folder(
        input_folder=input_dir,
        output_folder=output_dir,
        model_path="checkpoints/final_netG.pth",
        use_imagenet=False
    )