import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from going_modular import data_setup
from going_modular import engine
from going_modular.utils import save_model


# Hyperparameters
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

# Data directories
train_dir = Path('./data/pizza_steak_sushi/train')
test_dir = Path('./data/pizza_steak_sushi/test')

# Model directory
MODEL_PATH = Path('./models')
MODEL_NAME = 'food_vision.pth'

# Setup device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def freeze_model_features(model) -> None:
    """
    Freezes the features of the given model.

    Args:
        model (nn.Module): The model to freeze the features of
    """
    for param in model.features.parameters():
        param.requires_grad = False
    

def create_model(class_size: int) -> nn.Module:
    """
    Creates a new efficientnet_b0 model 
    for food vison classification.
    
    Returns:
        nn.Module: The new efficientnet_b0 model
    """
    
    efficientnet = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    ).to(device)

    # Set classifier layer for our problem
    efficientnet.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(
            in_features=1280,
            out_features=class_size
        )
    ).to(device)
    
    return efficientnet


def train_model(epochs: int = 15) -> None:
    # Create transforms
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    
    # Create dataloaders
    train_loader, test_loader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )


    # Create model
    efficientnet = create_model(len(class_names))
    freeze_model_features(efficientnet)
    
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=efficientnet.parameters(),
        lr=0.001
    )
    
    # Train the model
    result = engine.train(
        model=efficientnet,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_data_loader=train_loader,
        test_data_loader=test_loader,
        device=device,
        epochs=epochs
    )

    # Save the model
    save_model(
        model=efficientnet,
        path=MODEL_PATH,
        model_name=MODEL_NAME
    )

    # Plot the train and test losses
    plt.plot(result['train_loss'], label='train')
    plt.plot(result['test_loss'], label='test')
    plt.legend()
    plt.show()


def predict(image_path: Path, class_names: List[str]) -> None:
    """
    Predicts the class of an image.
    Args:
        image_path (Path): The path to the image.
        class_names (List[str]): The list of class names.
    """
    if not image_path.exists():
        print('Image does not exist!')
        return
    
    # Load the image
    img = Image.open(image_path)
    transform = EfficientNet_B0_Weights.DEFAULT.transforms()
    img_tensor = transform(img).to(device).unsqueeze(0)


    # Check if the model exists
    if not (MODEL_PATH / MODEL_NAME).exists():
        print('Model does not exist!')
        print('Training the model...')
        train_model()
    
    # Load the model
    model = create_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH / MODEL_NAME))
    model.eval()
    
    # Predict
    with torch.inference_mode():
        model_pred = model(img_tensor).argmax(dim=1)
        print(f'Predicted class: {class_names[model_pred.item()]}')
        print('_______________________________\n')


def clear_screen():
    """
    Clears the screen
    """
    if os.name == 'nt':  # for Windows
        _ = os.system('cls')
    else:  # for other platforms (Unix/Linux/MacOS)
        _ = os.system('clear')



def main():
    class_names = ['pizza', 'steak', 'sushi']
    MESSAGE = """
    Welcome to food vision.
    What would you like to do?
    1. Train the model
    2. Predict on a new image
    3. Exit
    """
    while True:
        input('Press Enter to continue...: ')
        clear_screen()
        print(MESSAGE)
        option = input('Enter your option: ')
        
        if option == '1':
            epochs = int(input('Enter the number of epochs: '))
            train_model(epochs)
        elif option == '2':
            path = input('Enter the path to the image: ')
            predict(Path(path), class_names)
        elif option == '3':
            break

if __name__ == '__main__':
    main()