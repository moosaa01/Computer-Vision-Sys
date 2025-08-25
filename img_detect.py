import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from torchvision import models
from PIL import Image
import os

class SensorModel(nn.Module):
    def __init__(self, num_classes):
        super(SensorModel, self).__init__()
        self.resnet = models.resnet18(weights='DEFAULT')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_classes = 3
model = SensorModel(num_classes)

model.load_state_dict(torch.load("sensor_type_model.pth", weights_only=True))
model.eval()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  load class names from the dataset folder
train_data_path = r"C:\Users\moosa\PycharmProjects\IFS353 indv\data_set\train"
class_names = os.listdir(train_data_path)

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert the OpenCV image (NumPy array) to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare the image for the model
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Get the predicted class and confidence percentage
    predicted_class = predicted_class.item()
    confidence_percentage = confidence.item() * 100

    # Label the sensor if percentage is greater than 75
    if confidence_percentage > 75:
        label = f'{class_names[predicted_class]}: {confidence_percentage:.2f}%'
    else:
        label = 'No object detected'

    # Font for label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):  # Press 'x' to quit
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
