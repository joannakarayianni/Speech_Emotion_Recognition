import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Checking device, I have a mac so I have cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print()device

# Setting seed for reproducibility of the code
torch.manual_seed(42)

# Paths to our datasets
training = "datasets/train.json"
validation = "datasets/dev.json"
testing = "datasets/test.json"

# Loading training data from JSON
train_data = json.load(open(training, 'r'))
train_df = pd.DataFrame(train_data).T

# Loading validation data from JSON
validation_data = json.load(open(validation, 'r'))
validation_df = pd.DataFrame(validation_data).T

# Loading testing data from JSON
testing_data = json.load(open(testing, 'r'))
testing_df = pd.DataFrame(testing_data).T

# Defining a custom Dataset class
class DLDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        valence = torch.tensor(sample['valence'], dtype=torch.long)  
        activation = torch.tensor(sample['activation'], dtype=torch.long)  
        features = torch.tensor(sample['features'], dtype=torch.float32)
        
        return valence, activation, features
    
    @property
    def features(self):
        all_features = []
        for idx in range(len(self)):
            _, _, features = self[idx]
            all_features.append(features)
        return all_features

# Initializing training & validation datasets
train_dataset = DLDataset(train_df)
validation_dataset = DLDataset(validation_df)

# Accessing and printing features of the entire train dataset
""" print("Features of the train dataset:")
for idx, features in enumerate(train_dataset.features):
    print(f"Sample {idx}: Features: {features}") """

# collating function to use with dataloader
def collate_fn_pad_sequences(batch):
    # Sorting batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x[-1].size(0), reverse=True)
    
    # Padding the sequences together allow them to be batched together
    features_padded = torch.nn.utils.rnn.pad_sequence([item[-1] for item in batch], batch_first=True)
    
    # Stacking activations and valences into tensors
    activations = torch.stack([item[1].clone().detach() for item in batch])
    valences = torch.stack([item[0].clone().detach() for item in batch])
    
    return activations, valences, features_padded

# Initializing DataLoader with custom collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_pad_sequences)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_pad_sequences)

# Defining the Bi-LSTM classifier
class ClassifierBiLSTM(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        dropout = 0.5 if num_layers > 1 else 0  # Adjust dropout based on num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # Forward LSTM
        output, (hn, cn) = self.lstm(x)
        # For bidirectional, concatenating the last hidden states from both directions
        if self.lstm.bidirectional:
            hn = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            hn = hn[-1]
        # Forward linear layer
        outputs = self.fc(hn)
        return outputs

# Initializing the model, loss function, and optimizer (Cross Entropy & Adam)
input_size = train_dataset[0][-1].shape[1]  # Assuming features have shape [sequence_length, input_size]
hidden_size = 128
num_layers = 2  
num_classes = 4 # we have 4 classes
model = ClassifierBiLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers, bidirectional=True).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5) # learning rate and weight decay

# Now we create the training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for activations, valences, features_padded in train_loader:  # Each iteration here processes a mini-batch
        activations, valences, features_padded = activations.to(device), valences.to(device), features_padded.to(device)
        
        optimizer.zero_grad()
        outputs = model(features_padded)
        
        # For CrossEntropyLoss, we need to combine valences and activations as targets
        targets = activations * 2 + valences  # Assuming valence and activation are binary (0 or 1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and true labels
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

# Validation function
def validate_model(model, validation_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for activations, valences, features_padded in validation_loader:
            activations, valences, features_padded = activations.to(device), valences.to(device), features_padded.to(device)
            
            outputs = model(features_padded)
            
            targets = activations * 2 + valences  # Assuming valence and activation are binary (0 or 1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Store predictions and true labels
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(validation_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1


# Initializing lists to store loss and accuracy for each epoch
train_losses = []
train_accuracies = []
train_f1s = []
validation_losses = []
validation_accuracies = []
validation_f1s = []
#*************************************************************** Training ***************************************************************
# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    train_loss, train_accuracy, train_f1 = train_model(model, train_loader, criterion, optimizer, device)
    validation_loss, validation_accuracy, validation_f1 = validate_model(model, validation_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_f1s.append(train_f1)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)
    validation_f1s.append(validation_f1)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, 
          Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Validation F1: {validation_f1:.4f}")

# ***************************************************** Plotting the results *****************************************************
# Plotting the learning curve
plt.figure(figsize=(12, 4))

# Plotting loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.show()

# Plotting F1 score
plt.figure(figsize=(6, 4))
plt.plot(train_f1s, label='Train F1 Score')
plt.plot(validation_f1s, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()

plt.show()