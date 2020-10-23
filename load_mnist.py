from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_test_from_MNIST(root, batch_size):
    train_set = datasets.MNIST(
        root=root, train=True, transform=transforms.ToTensor(), download=True
    )
    test_set = datasets.MNIST(
        root=root, train=False, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_set, test_set, train_loader, test_loader

