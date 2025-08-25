import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# Define dataset paths
train_data_path = r"C:\Users\moosa\PycharmProjects\IFS353 indv\data_set\train"
val_data_path = r"C:\Users\moosa\PycharmProjects\IFS353 indv\data_set\validate"
test_data_path = r"C:\Users\moosa\PycharmProjects\IFS353 indv\data_set\test"

# Define the model
class SensorModel(nn.Module):
    def __init__(self, num_classes):
        super(SensorModel, self).__init__()

        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)

# Get class names from the folders
class_names = train_dataset.classes

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Initialize model and set the optimizer
num_classes = len(class_names)
model = SensorModel(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
def train_model(model, train_loader, val_loader, num_epochs):
    model.train()
    overall_loss = 0.0
    overall_accuracy = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        overall_loss += epoch_loss
        overall_accuracy += epoch_accuracy

        # Validate the model
        val_loss, val_accuracy = validate_model(model, val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Print overall loss and accuracy after all epochs
    print(f'Overall Loss: {overall_loss / num_epochs:.4f}, Overall Accuracy: {overall_accuracy / num_epochs:.2f}%')

def validate_model(model, val_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient calculation only for training
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Testing
def test_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient calculation only needed for training
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

# run the code with 20 epochs
if __name__ == "__main__":
    num_epochs = 20
    train_model(model, train_loader, val_loader, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "sensor_type_model.pth")

    # Test the model
    test_model(model, test_loader)
