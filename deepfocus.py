import torch
import torch.nn as nn
from torch.nn import functional as F
from PCamDataset import PCamDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pcam_directory = '/content/'
label_mapping = {}
with open(os.path.join(pcam_directory, 'train_labels.csv'), 'r') as f:
    reader = csv.reader(f)
    next(reader)  # To skip the header
    label_mapping = {slide_id: int(label) for [slide_id, label] in reader}
all_fps = [fp for fp in os.listdir(os.path.join(pcam_directory, 'train'))]
for fp in all_fps: assert fp[-4:] == '.tif', fp[-4:]
permutation = np.random.permutation(range(len(all_fps)))
dataset = PCamDataset([
        (os.path.join(pcam_directory, 'train', all_fps[index]), label_mapping[all_fps[index][:-4]])
        for index in permutation[:int(len(permutation) * .8)]
    ], transform= transforms.Compose([
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet statistics
    ]))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class DeepFocusV3(nn.Module):
    def __init__(self, filters=(32, 32, 64, 128, 128), kernel_sizes=(5, 3, 3, 3, 3), fc=(128, 64)):
        super(DeepFocusV3, self).__init__()

        # Ensuring each convolution layer correctly matches the output/input channels
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=kernel_sizes[0], padding=(kernel_sizes[0]//2))
        self.bn1 = nn.BatchNorm2d(filters[0])

        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=kernel_sizes[1], padding=(kernel_sizes[1]//2))
        self.bn2 = nn.BatchNorm2d(filters[1])

        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=kernel_sizes[2], padding=(kernel_sizes[2]//2))
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=kernel_sizes[3], padding=(kernel_sizes[3]//2))
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(filters[3], filters[4], kernel_size=kernel_sizes[4], padding=(kernel_sizes[4]//2))
        self.bn5 = nn.BatchNorm2d(filters[4])
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # Adjust the input size here based on the output size from the last pooling layer
        self.fc1 = nn.Linear(filters[4] * 8 * 8, fc[0])  # Adjust based on your actual output size
        self.bn6 = nn.BatchNorm1d(fc[0])
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(fc[0], fc[1])
        self.bn7 = nn.BatchNorm1d(fc[1])
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(fc[1], 2)  # Output layer for binary classification

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout2(x)

        x = F.softmax(self.fc3(x), dim=1)
        return x

# Instantiate the model and transfer it to the device
model = DeepFocusV3()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode

# Function to predict clarity
exclude_indices = []
def predict_clarity(dataloader, model, threshold = 0.75):
    with torch.no_grad():  # No need to track gradients
        for i,data in enumerate(dataloader):
            images = data[0]
            images = images.to(device)
            #to get image into the input shape DeepFocus expects:
            resize = transforms.Resize((64, 64))
            images = resize(images)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            clear_probs = probabilities[:, 1]  # Index 1 for 'clear'

            # Decide which images to exclude based on the threshold
            for j, prob in enumerate(clear_probs):
                if prob.item() < threshold:  # Less than 75% probability of being 'clear'
                    exclude_indices.append(i * dataloader.batch_size + j)
    dataset.update_exclusion_list(set(exclude_indices))
    print(len(exclude_indices))
# Predict clarity of PCAM images
predict_clarity(dataloader, model)
