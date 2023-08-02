import os
import torch
import torch.nn as nn
import data_setup, model_builder, engine, utils

from torchvision import transforms


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Setup data directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    This function is the main entry point of the program. It performs the following steps:
    
    1. Creates an image transform using the `transforms.Compose` function.
    2. Creates dataloaders using the `data_setup.create_dataloaders` function, passing in the `train_dir`, `test_dir`, `transform`, and `batch_size` parameters.
    3. Creates a model using the `model_builder.TinyVGG` class, passing in the `in_channels` and `num_classes` parameters.
    4. Creates a loss function using the `nn.CrossEntropyLoss` class.
    5. Creates an optimizer using the `torch.optim.Adam` class, passing in the `model.parameters()` and `lr` parameters.
    6. Trains the model using the `engine.train` function, passing in the `model`, `loss_fn`, `optimizer`, `train_data_loader`, `test_data_loader`, `device`, and `epochs` parameters.
    7. Saves the model using the `utils.save_model` function, passing in the `model`, `path`, and `name` parameters.
    """
    # Create image transform
    data_trasform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    # Create dataloaders
    train_loader, test_loader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_trasform,
        batch_size=BATCH_SIZE
    )

    # Create model
    model = model_builder.TinyVGG(
        in_channels=3,
        num_classes=len(class_names) 
    ).to(device)

    # Create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    # Train model
    engine.train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        device=device,
        epochs=NUM_EPOCHS
    )

    # Save model
    utils.save_model(
        model=model,
        path='models',
        model_name='tiny_vgg_food.pt'
    )


if __name__ == "__main__":
    main()