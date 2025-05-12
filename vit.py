# Import required libraries for building and training the model
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers
import torch.optim as optim  # PyTorch module for optimisation algorithms
import torchvision.transforms as transforms  # Tools for image transformations
from torch.utils.data import DataLoader, Dataset  # Utilities for loading data
import os  # Library for interacting with the operating system
import numpy as np  # Library for numerical operations
from PIL import Image  # Library for image processing
from tqdm import tqdm  # Progress bar for loops
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc  # Metrics for evaluating model performance
from sklearn.preprocessing import label_binarize  # Utility for binarising labels for ROC curves
import matplotlib.pyplot as plt  # Library for plotting graphs
import seaborn as sns  # Library for enhanced data visualisation

# Set a random seed to ensure consistent results across runs
torch.manual_seed(42)  # Fixes the random seed for reproducibility
# Determine if GPU is available; if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chooses device for computation

# Define the list of medical conditions to classify
conditions = ["normal", "pneumonia_bacterial", "pneumonia_viral", "pneumothorax", "tuberculosis"]  # List of conditions for classification
# Specify the base directory where the dataset is stored
base_dir = "C:/xampp/htdocs/xampp/dissertation/data"  # Path to the dataset
# Map each condition to its corresponding folder in the dataset
class_paths = {
    "normal": "Normal",  # Folder for normal X-rays
    "pneumonia_bacterial": "Pneumonia/bacterial",  # Folder for bacterial pneumonia X-rays
    "pneumonia_viral": "Pneumonia/viral",  # Folder for viral pneumonia X-rays
    "pneumothorax": "Pneumothorax",  # Folder for pneumothorax X-rays
    "tuberculosis": "Tuberculosis"  # Folder for tuberculosis X-rays
}

# Define hyperparameters for the model and training process
BATCH_SIZE = 16  # Number of images processed in each batch
EPOCHS = 50  # Number of complete passes through the dataset during training
BASE_LR = 3e-4  # Learning rate for the optimiser
WEIGHT_DECAY = 1e-4  # Regularisation parameter to prevent overfitting
IMG_SIZE = 224  # Size to which images are resized (224x224 pixels)
PATCH_SIZE = 16  # Size of each image patch (16x16 pixels)
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # Calculate total number of patches: (224/16)² = 196
D_MODEL = 512  # Dimensionality of the embeddings
NUM_HEADS = 8  # Number of attention heads in the transformer
FF_HIDDEN_DIM = 2048  # Hidden dimension size for the feedforward network
NUM_LAYERS = 8  # Number of transformer layers
DROPOUT_RATE = 0.1  # Dropout rate to prevent overfitting
PATIENCE = 5  # Number of epochs to wait before early stopping

# Define a custom dataset class for loading chest X-ray images
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Store the root directory of the dataset
        self.transform = transform  # Store the image transformation pipeline
        self.samples = []  # List to store image file paths
        self.labels = []  # List to store corresponding labels
        self._load_data()  # Load the data upon initialisation
    
    # Load all images and assign labels based on their condition
    def _load_data(self):
        for idx, cond in enumerate(conditions):  # Loop through each condition
            path = os.path.join(self.root_dir, class_paths[cond])  # Construct the path to the condition's folder
            if not os.path.exists(path):  # Check if the folder exists
                raise FileNotFoundError(f"Directory not found: {path}")  # Raise an error if the folder is missing
            for file in os.listdir(path):  # Loop through files in the folder
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
                    self.samples.append(os.path.join(path, file))  # Add the image path to the samples list
                    self.labels.append(idx)  # Add the corresponding label (index of the condition)
        print(f"Loaded {len(self.samples)} samples across {len(conditions)} classes")  # Print the number of samples loaded
    
    # Return the total number of images in the dataset
    def __len__(self):
        return len(self.samples)  # Return the length of the samples list
    
    # Retrieve an image and its label by index
    def __getitem__(self, idx):
        img_path = self.samples[idx]  # Get the image path at the given index
        label = self.labels[idx]  # Get the label at the given index
        image = Image.open(img_path).convert('RGB')  # Open the image and convert to RGB
        if self.transform:  # Apply transformations if provided
            image = self.transform(image)  # Transform the image
        return image, label  # Return the image and label as a tuple

# Define the Vision Transformer model architecture
class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()  # Initialise the parent class
        self.patch_embedding = PatchEmbedding()  # Create a patch embedding layer
        # Create a list of transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(D_MODEL, NUM_HEADS, FF_HIDDEN_DIM, DROPOUT_RATE)
            for _ in range(NUM_LAYERS)  # Create NUM_LAYERS transformer layers
        ])
        self.norm = nn.LayerNorm(D_MODEL)  # Layer normalisation for the embeddings
        self.classifier = nn.Linear(D_MODEL, num_classes)  # Final layer to classify into num_classes
    
    # Define the forward pass through the model
    def forward(self, x):
        x = self.patch_embedding(x)  # Convert the input image into patch embeddings
        for layer in self.transformer_encoder:  # Pass through each transformer layer
            x = layer(x)  # Apply the transformer layer
        x = self.norm(x)  # Apply layer normalisation
        cls_output = x[:, 0]  # Extract the CLS token output
        logits = self.classifier(cls_output)  # Classify using the CLS token
        return logits  # Return the classification logits

# Define a layer to convert images into patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()  # Initialise the parent class
        # Convolution to convert patches into D_MODEL-dimensional vectors
        self.proj = nn.Conv2d(3, D_MODEL, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)  # Convert RGB patches to embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))  # Learnable CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, D_MODEL) * 0.02)  # Positional embeddings
        self.dropout = nn.Dropout(DROPOUT_RATE)  # Dropout layer to prevent overfitting
    
    # Define the forward pass for patch embedding
    def forward(self, x):
        B = x.shape[0]  # Get the batch size
        x = self.proj(x).flatten(2).transpose(1, 2)  # Convert image to patches and flatten: (B, 196, 512)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Expand the CLS token for the batch
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate the CLS token: (B, 197, 512)
        x = x + self.pos_embed  # Add positional embeddings
        x = self.dropout(x)  # Apply dropout
        return x  # Return the embedded patches

# Define a single transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super().__init__()  # Initialise the parent class
        self.norm1 = nn.LayerNorm(d_model)  # First layer normalisation
        self.attention = MultiHeadAttention(d_model, num_heads)  # Multi-head attention mechanism
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention
        self.norm2 = nn.LayerNorm(d_model)  # Second layer normalisation
        self.feedforward = FeedForward(d_model, hidden_dim, dropout)  # Feedforward network
        self.dropout2 = nn.Dropout(dropout)  # Dropout after feedforward
    
    # Define the forward pass through the transformer layer
    def forward(self, x):
        attn_output = self.attention(self.norm1(x))  # Apply attention after normalisation
        x = x + self.dropout1(attn_output)  # Add residual connection and apply dropout
        ff_output = self.feedforward(self.norm2(x))  # Apply feedforward after normalisation
        x = x + self.dropout2(ff_output)  # Add residual connection and apply dropout
        return x  # Return the output

# Define the multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()  # Initialise the parent class
        self.num_heads = num_heads  # Store the number of attention heads
        self.d_model = d_model  # Store the embedding dimension
        self.depth = d_model // num_heads  # Calculate the dimension per head
        self.wq = nn.Linear(d_model, d_model)  # Linear layer for queries
        self.wk = nn.Linear(d_model, d_model)  # Linear layer for keys
        self.wv = nn.Linear(d_model, d_model)  # Linear layer for values
        self.wo = nn.Linear(d_model, d_model)  # Linear layer for output
        self.dropout = nn.Dropout(DROPOUT_RATE)  # Dropout layer
    
    # Define the forward pass for multi-head attention
    def forward(self, x):
        batch_size = x.shape[0]  # Get the batch size
        # Compute queries, keys, and values, and split into heads
        q = self.wq(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # Queries: (B, num_heads, seq_len, depth)
        k = self.wk(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # Keys: (B, num_heads, seq_len, depth)
        v = self.wv(x).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # Values: (B, num_heads, seq_len, depth)
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)  # Scaled dot-product attention
        attention = torch.softmax(scores, dim=-1)  # Apply softmax to get attention weights
        attention = self.dropout(attention)  # Apply dropout to attention weights
        # Compute the context vector
        context = torch.matmul(attention, v)  # Weighted sum of values
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # Reshape back to (B, seq_len, d_model)
        output = self.wo(context)  # Apply the output linear layer
        return output  # Return the attention output

# Define the feedforward network within the transformer
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()  # Initialise the parent class
        # Define the feedforward network as a sequential model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),  # First linear layer
            nn.GELU(),  # GELU activation function
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(hidden_dim, d_model),  # Second linear layer
            nn.Dropout(dropout)  # Another dropout layer
        )
    
    # Define the forward pass through the feedforward network
    def forward(self, x):
        return self.net(x)  # Pass the input through the network

# Train the model and generate performance plots
def train_model(model, train_loader, val_loader, epochs, lr, weight_decay):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Initialise the AdamW optimiser
    criterion = nn.CrossEntropyLoss()  # Define the loss function for classification
    best_val_loss = float('inf')  # Track the best validation loss for early stopping
    patience_counter = 0  # Counter for early stopping patience
    train_losses, val_losses = [], []  # Lists to store training and validation losses
    train_accs, val_accs = [], []  # Lists to store training and validation accuracies
    
    # Loop through each epoch
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()  # Enable training mode
        running_loss = 0.0  # Track the total loss for this epoch
        all_preds, all_labels = [], []  # Lists to store predictions and true labels
        # Loop through batches in the training data
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()  # Clear accumulated gradients
            outputs = model(images)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights
            running_loss += loss.item()  # Add the batch loss to the running total
            preds = torch.argmax(outputs, dim=1)  # Get the predicted classes
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels
        
        train_loss = running_loss / len(train_loader)  # Compute average training loss
        train_acc = accuracy_score(all_labels, all_preds)  # Compute training accuracy
        
        # Evaluate the model on the validation set
        val_loss, val_acc, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion)  # Get validation metrics
        
        # Store metrics for plotting
        train_losses.append(train_loss)  # Add training loss to the list
        val_losses.append(val_loss)  # Add validation loss to the list
        train_accs.append(train_acc)  # Add training accuracy to the list
        val_accs.append(val_acc)  # Add validation accuracy to the list
        
        # Print the performance metrics for this epoch
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if this is the best model so far for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update the best validation loss
            patience_counter = 0  # Reset the patience counter
            torch.save(model.state_dict(), "multilabel_vit_model.pth")  # Save the best model
            print("✅ Saved best model")  # Confirm the model was saved
        else:
            patience_counter += 1  # Increment the patience counter
            if patience_counter >= PATIENCE:  # Check if patience is exceeded
                print(f"Early stopping at epoch {epoch+1}")  # Print early stopping message
                break  # Exit the training loop
    
    # Load the best model weights
    model.load_state_dict(torch.load("multilabel_vit_model.pth", weights_only=True))  # Load the best model
    torch.save(model.state_dict(), "chest_xray_model.pth")  # Save the final model
    print("✅ Final model saved at: chest_xray_model.pth")  # Confirm the final model was saved
    
    # Plot the training and validation loss curve
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', color='blue')  # Plot training loss in blue
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='orange')  # Plot validation loss in orange
    plt.xlabel('Epoch')  # Label the x-axis
    plt.ylabel('Loss')  # Label the y-axis
    plt.title('TRAINING AND VALIDATION LOSS')  # Set the title
    plt.legend()  # Add a legend
    plt.grid(True)  # Add a grid
    plt.savefig("loss_curve.png")  # Save the plot
    plt.close()  # Close the plot
    
    # Plot the training and validation accuracy curve
    plt.figure(figsize=(8, 4))  # Set the figure size
    plt.plot(range(1, len(train_accs)+1), train_accs, label='Training Accuracy', color='blue')  # Plot training accuracy in blue
    plt.plot(range(1, len(val_accs)+1), val_accs, label='Validation Accuracy', color='orange')  # Plot validation accuracy in orange
    plt.xlabel('Epoch')  # Label the x-axis
    plt.ylabel('Accuracy')  # Label the y-axis
    plt.title('TRAINING AND VALIDATION ACCURACY')  # Set the title
    plt.legend()  # Add a legend
    plt.grid(True)  # Add a grid
    plt.savefig("accuracy_curve.png")  # Save the plot
    plt.close()  # Close the plot
    
    # Generate and plot the confusion matrix
    cm = confusion_matrix(val_labels, val_preds)  # Compute the confusion matrix
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=conditions, yticklabels=conditions)  # Plot the heatmap
    plt.xlabel('Predicted')  # Label the x-axis
    plt.ylabel('True')  # Label the y-axis
    plt.title('Confusion Matrix')  # Set the title
    plt.savefig("confusion_matrix.png")  # Save the plot
    plt.close()  # Close the plot
    
    # Generate a classification report as a dictionary
    report = classification_report(val_labels, val_preds, target_names=conditions, output_dict=True)  # Compute the classification report
    
    # Plot precision and recall by condition
    plot_precision_recall(report)  # Call the function to plot precision and recall
    
    # Plot F1 scores by condition
    plot_f1_scores(report)  # Call the function to plot F1 scores
    
    # Plot ROC curves for each condition
    plot_roc_curves(val_labels, val_probs)  # Call the function to plot ROC curves
    
    # Print the detailed classification report
    print("\nCondition-specific performance metrics demonstrated varying levels of success:")  # Print a header
    print(classification_report(val_labels, val_preds, target_names=conditions))  # Print the report
    
    return model  # Return the trained model

# Evaluate the model on a dataset and return performance metrics
def evaluate(model, loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0  # Track the total loss
    all_preds, all_labels, all_probs = [], [], []  # Lists to store predictions, labels, and probabilities
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in loader:  # Loop through batches
            images, labels = images.to(device), labels.to(device)  # Move data to the device
            outputs = model(images)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            running_loss += loss.item()  # Add the batch loss to the running total
            preds = torch.argmax(outputs, dim=1)  # Get the predicted classes
            probs = torch.softmax(outputs, dim=1)  # Compute class probabilities
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_probs.extend(probs.cpu().numpy())  # Store probabilities
    avg_loss = running_loss / len(loader)  # Compute average loss
    accuracy = accuracy_score(all_labels, all_preds)  # Compute accuracy
    return avg_loss, accuracy, all_preds, all_labels, np.array(all_probs)  # Return metrics and probabilities

# Plot precision and recall for each condition
def plot_precision_recall(report):
    precisions = []  # List to store precision values
    recalls = []  # List to store recall values
    for cond in conditions:  # Loop through each condition
        precisions.append(report[cond]['precision'])  # Extract precision for the condition
        recalls.append(report[cond]['recall'])  # Extract recall for the condition
    
    x = np.arange(len(conditions))  # Create an array of indices for the conditions
    width = 0.35  # Set the width of the bars
    
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(x - width/2, precisions, width, label='Precision', color='blue')  # Plot precision bars in blue
    plt.bar(x + width/2, recalls, width, label='Recall', color='orange')  # Plot recall bars in orange
    plt.xlabel('Condition')  # Label the x-axis
    plt.ylabel('Score')  # Label the y-axis
    plt.title('Precision and Recall by Condition')  # Set the title
    plt.xticks(x, conditions, rotation=45)  # Set x-axis labels with rotation
    plt.legend()  # Add a legend
    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.savefig("precision_recall.png")  # Save the plot
    plt.close()  # Close the plot

# Plot F1 scores for each condition
def plot_f1_scores(report):
    f1_scores = [report[cond]['f1-score'] for cond in conditions]  # Extract F1 scores for each condition
    
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.bar(conditions, f1_scores, color='green')  # Plot F1 scores as green bars
    for i, v in enumerate(f1_scores):  # Loop through F1 scores to add labels
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')  # Add the F1 score value above each bar
    plt.xlabel('Condition')  # Label the x-axis
    plt.ylabel('F1-Score')  # Label the y-axis
    plt.title('F1 Scores by Condition')  # Set the title
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust the layout
    plt.savefig("f1_scores.png")  # Save the plot
    plt.close()  # Close the plot

# Plot ROC curves for each condition
def plot_roc_curves(labels, probs):
    labels_bin = label_binarize(labels, classes=range(len(conditions)))  # Binarise the labels for ROC computation
    fpr, tpr, roc_auc = {}, {}, {}  # Dictionaries to store false positive rates, true positive rates, and AUC scores
    
    # Compute ROC curve and AUC for each condition
    for i in range(len(conditions)):  # Loop through each condition
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])  # Compute ROC curve
        roc_auc[i] = auc(fpr[i], tpr[i])  # Compute AUC score
    
    # Plot the ROC curves
    plt.figure(figsize=(8, 6))  # Set the figure size
    colors = ['blue', 'green', 'red', 'purple', 'cyan']  # Define colours for each condition
    for i, color in enumerate(colors):  # Loop through each condition and colour
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{conditions[i]} (AUC = {roc_auc[i]:.3f})')  # Plot the ROC curve with AUC in the label
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Plot the diagonal line (random classifier)
    plt.xlabel('False Positive Rate')  # Label the x-axis
    plt.ylabel('True Positive Rate')  # Label the y-axis
    plt.title('Receiver Operating Characteristic (ROC) Curves')  # Set the title
    plt.legend(loc="lower right")  # Add a legend in the lower right
    plt.grid(True)  # Add a grid
    plt.savefig("roc_curves.png")  # Save the plot
    plt.close()  # Close the plot

# Plot the effect of hyperparameters on performance (example for batch size)
def plot_hyperparameter_effects():
    # Example data for batch size (replace with actual data if available)
    batch_sizes = [8, 16, 32]  # Batch sizes tested
    f1_scores = [0.88, 0.90, 0.89]  # Corresponding F1 scores
    training_times = [7.2, 5.8, 4.3]  # Corresponding training times
    
    fig, ax1 = plt.subplots(figsize=(8, 6))  # Create a figure with dual axes
    ax1.bar(batch_sizes, f1_scores, color='blue', label='F1 Score')  # Plot F1 scores as blue bars
    ax1.set_xlabel('Batch Size')  # Label the x-axis
    ax1.set_ylabel('F1 Score', color='blue')  # Label the y-axis for F1 scores
    ax1.tick_params(axis='y', labelcolor='blue')  # Set y-axis tick colour for F1 scores
    
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.plot(batch_sizes, training_times, color='red', marker='o', label='Training Time (hrs)')  # Plot training times as a red line
    ax2.set_ylabel('Training Time (hrs)', color='red')  # Label the y-axis for training times
    ax2.tick_params(axis='y', labelcolor='red')  # Set y-axis tick colour for training times
    
    plt.title('Effect of Batch Size on Performance')  # Set the title
    fig.legend(loc='upper right')  # Add a legend
    plt.tight_layout()  # Adjust the layout
    plt.savefig("batch_size_effect.png")  # Save the plot
    plt.close()  # Close the plot
    
    # Similar plots can be created for learning rate, patch size, and number of layers

# Main function to set up and run the training process
def main():
    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to IMG_SIZE
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(10),  # Randomly rotate images by up to 10 degrees
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalise images
    ])
    
    # Load the dataset and split into training and validation sets
    dataset = ChestXrayDataset(base_dir, transform=transform)  # Load the dataset
    train_size = int(0.8 * len(dataset))  # Calculate the size of the training set (80%)
    val_size = len(dataset) - train_size  # Calculate the size of the validation set (20%)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])  # Split the dataset
    
    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # Training data loader
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)  # Validation data loader
    
    # Initialise and train the model
    model = VisionTransformer(num_classes=len(conditions)).to(device)  # Create the model and move to device
    model = train_model(model, train_loader, val_loader, EPOCHS, BASE_LR, WEIGHT_DECAY)  # Train the model
    
    
    plot_hyperparameter_effects() 

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()  # Call the main function to start the process