import os
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Takes in a training and testing directory and turns them into 
    Datasets and then DataLoaders.

    Args:
        train_dir: Path to directory with training images.
        test_dir: Path to directory with testing images.
        transform: torchvison transform to perform on training and testing data.
        batch_size: Batch size for DataLoader.
        num_workers: Number of cpu workers per DataLoader.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of target classes.
    
    Examples:
        >>> train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir='PATH/TO/TRAIN_DIR',
                test_dir='PATH/TO/TEST_DIR',
                transform=some_transform,
                batch_size=32,
                num_workers=4
            )
    """
    train_data = ImageFolder(
        root=train_dir,
        transform=transform
    )
    test_data = ImageFolder(
        root=test_dir,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader, train_data.classes