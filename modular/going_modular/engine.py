import torch
import torch.nn as nn

from typing import List, Tuple, Dict
from tqdm.auto import tqdm


def train_step(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs a single training step.
    Args:
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function for model.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device (cpu or gpu).
    
    Returns:
        Tuple[float, float]: The loss and accuracy.
    """
    model.train()
    total_accuracy, total_loss = 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred_logit = model(x)
        loss = loss_fn(y_pred_logit, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.softmax(y_pred_logit, dim=1).argmax(dim=1)
        total_loss += loss.item()
        total_accuracy += (y_pred_class == y).sum().item() / len(y)
    

    return total_accuracy / len(data_loader), total_loss / len(data_loader)


def test_step(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Performs a single test step.
    Args:
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function for model.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device (cpu or gpu).
    
    Returns:
        Tuple[float, float]: The loss and accuracy.
    """
    model.eval()
    total_accuracy, total_loss = 0, 0
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred_logit = model(x)
            loss = loss_fn(y_pred_logit, y)
            y_pred_class = torch.softmax(y_pred_logit, dim=1).argmax(dim=1)
            total_loss += loss.item()
            total_accuracy += (y_pred_class == y).sum().item() / len(y)
    
    return total_accuracy / len(data_loader), total_loss / len(data_loader)

def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int
) -> Dict[str, List[float]]:
    """
    Trains the neural network model.
    Args:
        model (nn.Module): The neural network model.
        loss_fn (nn.Module): The loss function for model.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device (cpu or gpu).
    
    Returns:
        Dict[str, List[float]]: A dictionary containing the train and test
        losses and accuracies with keys 'train_loss', 'train_acc', 'test_loss', 'test_acc'.
    """
    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    for e in tqdm(range(epochs)):
        train_acc, train_loss = train_step(
            model,
            loss_fn,
            optimizer,
            train_data_loader,
            device
        )

        test_acc, test_loss = test_step(
            model,
            loss_fn,
            test_data_loader,
            device
        )

        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc)

        if e % 5 == 0:
            print(f'Epoch {e}:')
            print(f'Test accuracy : {test_acc} | Test loss : {test_loss}')
            print(f'Train accuracy : {train_acc} | Train loss : {train_loss}')
            print('_____________________________________________')
    
    return result