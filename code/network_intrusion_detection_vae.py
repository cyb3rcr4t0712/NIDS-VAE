# Downloading the KDD99 dataset
get_ipython().system('wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz')
get_ipython().system('wget http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz')
get_ipython().system('wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names')
# unzipping the files
get_ipython().system('gzip -d /content/kddcup.data.gz')
get_ipython().system('gzip -d /content/corrected.gz')




# In[ ]:


"""
Network Intrusion Detection using Variational Autoencoder (VAE)
Based on KDD Cup 1999 dataset
"""

# Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, precision_recall_curve, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")


# In[ ]:


class Config:
    """Configuration parameters for the project."""

    # Data parameters
    TRAIN_DATA_PATH = "/content/kddcup.data"
    TEST_DATA_PATH = "/content/corrected"
    NAMES_PATH = "/content/kddcup.names"

    RANDOM_SEED = 42
    SERVICE_TYPE = "http"  # Focus on HTTP traffic

    # Model parameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3
    HIDDEN_DIM = [28, 14, 7]  # Dimensions of hidden layers
    LATENT_DIM = 3  # Latent space dimension

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
print(f"Using device: {config.DEVICE}")


# In[ ]:


def load_and_preprocess_data(config):
    """Load and preprocess the KDD99 dataset."""
    # Read column names
    with open(config.NAMES_PATH, 'r') as txt_file:
        col_names = txt_file.readlines()

    # Clean column names
    col_names_cleaned = [i.split(':')[0] for i in col_names[1:]]
    col_names_cleaned.append('result')

    # Read training data
    df_train = pd.read_csv(config.TRAIN_DATA_PATH, header=None, names=col_names_cleaned)

    # Read test data
    df_test = pd.read_csv(config.TEST_DATA_PATH, header=None, names=col_names_cleaned)

    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")

    return df_train, df_test, col_names_cleaned


# In[ ]:


def explore_data(df_train, df_test, service_type=None):
    """Explore and visualize the dataset."""
    plt.figure(figsize=(12, 6))

    # Plot distribution of full dataset
    plt.subplot(1, 2, 1)
    train_counts = df_train['result'].value_counts()
    sns.barplot(x=train_counts.index, y=train_counts.values)
    plt.title('Distribution in Training Data')
    plt.xticks(rotation=45)

    # Plot distribution of test dataset
    plt.subplot(1, 2, 2)
    test_counts = df_test['result'].value_counts()
    sns.barplot(x=test_counts.index, y=test_counts.values)
    plt.title('Distribution in Test Data')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # If service type is specified, filter and show those distributions
    if service_type:
        plt.figure(figsize=(12, 6))

        # Filter by service type
        df_train_service = df_train[df_train.service == service_type]
        df_test_service = df_test[df_test.service == service_type]

        # Plot distribution of service in training data
        plt.subplot(1, 2, 1)
        train_service_counts = df_train_service['result'].value_counts()
        sns.barplot(x=train_service_counts.index, y=train_service_counts.values)
        plt.title(f'Distribution in Training Data ({service_type} service)')
        plt.xticks(rotation=45)

        # Plot distribution of service in test data
        plt.subplot(1, 2, 2)
        test_service_counts = df_test_service['result'].value_counts()
        sns.barplot(x=test_service_counts.index, y=test_service_counts.values)
        plt.title(f'Distribution in Test Data ({service_type} service)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return df_train_service, df_test_service

    return df_train, df_test


# In[ ]:


def prepare_data_for_model(df_train, df_test, config):
    """Prepare data for the VAE model."""
    # Filter normal samples for training (we only train on normal data)
    df_train_normal = df_train[df_train.result == "normal."].copy()
    print(f"Training on {len(df_train_normal)} normal samples")

    # Create a copy of the test data to avoid SettingWithCopyWarning
    df_test = df_test.copy()

    # Add target column to test data (1 for anomaly, 0 for normal)
    df_test.loc[:, "target"] = df_test["result"].apply(lambda x: 1 if x != "normal." else 0)
    print(f"Test data distribution: {df_test['target'].value_counts().to_dict()}")

    # Store target values before dropping
    y_test = df_test["target"].values

    # Identify categorical and continuous columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    bool_cols = ['land', 'logged_in', 'is_host_login', 'is_guest_login']

    # Define columns to drop - note that 'target' is only in the test data, not train data
    train_drop_cols = ['result', 'wrong_fragment', 'urgent', 'num_failed_logins',
                       'su_attempted', 'num_file_creations', 'num_outbound_cmds']
    test_drop_cols = train_drop_cols + ['target']  # Include 'target' for test data

    # Drop unnecessary columns from train and test data
    X_train = df_train_normal.drop(categorical_cols + bool_cols + train_drop_cols, axis=1).values
    X_test = df_test.drop(categorical_cols + bool_cols + test_drop_cols, axis=1).values

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    # Create dataset and dataloader for training
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # input = target for autoencoder
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    return train_loader, X_train_tensor, X_test_tensor, y_test, X_train.shape[1]


# In[ ]:


class VAE(nn.Module):
    """Variational Autoencoder for Anomaly Detection."""

    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.2):
        """
        Initialize the VAE.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden dimensions for encoder
            latent_dim: Dimension of latent space
            dropout_rate: Dropout rate for regularization
        """
        super(VAE, self).__init__()

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space parameters
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent space to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# In[ ]:


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function.

    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight of KL divergence term
    """
    # Reconstruction loss (MSE)
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL-Divergence
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return MSE + beta * KLD, MSE, KLD


def train_model(model, train_loader, config):
    """Train the VAE model."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }

    # Early stopping parameters
    best_loss = float('inf')
    early_stopping_counter = 0

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        for inputs, _ in progress_bar:
            inputs = inputs.to(config.DEVICE)

            # Forward pass
            recon, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = vae_loss_function(recon, inputs, mu, logvar)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon_loss': f"{recon_loss.item():.4f}",
                'kl_loss': f"{kl_loss.item():.4f}"
            })

        # Calculate average epoch loss
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)

        # Update history
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
              f"Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}")

        # Learning rate scheduler step
        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stopping_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_vae_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model
    model.load_state_dict(torch.load('best_vae_model.pth'))

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.plot(history['kl_loss'], label='KL Divergence')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model, history


# In[ ]:


def compute_reconstruction_errors(model, data_tensor, config):
    """Compute reconstruction errors for given data."""
    model.eval()
    data_tensor = data_tensor.to(config.DEVICE)

    with torch.no_grad():
        recon, _, _ = model(data_tensor)

    # Calculate MSE for each sample
    errors = torch.nn.functional.mse_loss(
        recon, data_tensor, reduction='none'
    ).sum(dim=1).cpu().numpy()

    # Normalize errors to [0,1] range for easier threshold selection
    errors = errors - errors.min()
    errors = errors / (errors.max() + 1e-10)  # Add small epsilon to avoid division by zero

    return errors


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics including FDR for anomaly detection.

    Args:
        y_true: Ground truth labels (0=normal, 1=anomaly)
        y_pred: Predicted labels (0=normal, 1=anomaly)

    Returns:
        Dictionary of metrics
    """
    # Calculate basic counts from confusion matrix
    TP = np.sum((y_pred == 1) & (y_true == 1))  # True positives
    TN = np.sum((y_pred == 0) & (y_true == 0))  # True negatives
    FP = np.sum((y_pred == 1) & (y_true == 0))  # False positives
    FN = np.sum((y_pred == 0) & (y_true == 1))  # False negatives

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # False Discovery Rate
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0

    # False Positive Rate
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    # True Positive Rate (same as Recall)
    tpr = recall

    # Additional metrics
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative Predictive Value

    return {
        'accuracy': (TP + TN) / (TP + TN + FP + FN),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fdr': fdr,
        'fpr': fpr,
        'tpr': tpr,
        'specificity': specificity,
        'npv': npv,
        'total_samples': len(y_true),
        'total_anomalies': np.sum(y_true == 1),
        'predicted_anomalies': np.sum(y_pred == 1),
        'true_positives': TP,
        'false_positives': FP,
        'true_negatives': TN,
        'false_negatives': FN
    }


def find_optimal_threshold(recon_errors, y_true):
    """Find the optimal threshold for anomaly detection."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, recon_errors)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find the threshold that gives the best F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

    print(f"Optimal threshold: {optimal_threshold:.6f} with F1: {f1_scores[optimal_idx]:.4f}")

    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='.', label='Precision-Recall Curve')
    plt.scatter(recalls[optimal_idx], precisions[optimal_idx], c='red', marker='o',
                label=f'Optimal (F1={f1_scores[optimal_idx]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Threshold Selection')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, recon_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold


# In[ ]:


def evaluate_model(recon_errors, y_true, threshold):
    """Evaluate model performance based on reconstruction errors and threshold."""
    # Make predictions using the threshold
    y_pred = (recon_errors >= threshold).astype(int)

    # Calculate comprehensive metrics
    metrics = calculate_metrics(y_true, y_pred)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Display summary metrics
    print(f"\nSummary Metrics:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Total actual anomalies: {metrics['total_anomalies']}")
    print(f"Total predicted anomalies: {metrics['predicted_anomalies']}")
    print(f"Correctly identified anomalies: {metrics['true_positives']}")
    print(f"Missed anomalies: {metrics['false_negatives']}")
    print(f"False alarms: {metrics['false_positives']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"False Discovery Rate (FDR): {metrics['fdr']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

    # Plot histogram of reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.hist(recon_errors[y_true == 0], bins=50, alpha=0.5, label='Normal')
    plt.hist(recon_errors[y_true == 1], bins=50, alpha=0.5, label='Anomaly')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.6f}')
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    return metrics


# In[ ]:


def visualize_latent_space(model, data_tensor, labels, config):
    """Visualize data points in the latent space."""
    # For 2D visualization, we need latent dimension to be at least 2
    if config.LATENT_DIM < 2:
        print("Cannot visualize latent space: dimension must be at least 2")
        return

    model.eval()
    data_tensor = data_tensor.to(config.DEVICE)

    with torch.no_grad():
        mu, _ = model.encode(data_tensor)

    # Convert to numpy for plotting
    latent_points = mu.cpu().numpy()

    # Subsample if too many points
    if len(latent_points) > 5000:
        indices = np.random.choice(len(latent_points), 5000, replace=False)
        latent_points = latent_points[indices]
        labels = labels[indices]

    # For 2D visualization
    if config.LATENT_DIM == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_points[:, 0], latent_points[:, 1],
                            c=labels, cmap='viridis', alpha=0.6, s=5)
        plt.colorbar(scatter, label='Class')
        plt.title('VAE Latent Space Visualization')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.show()

    # For 3D visualization
    elif config.LATENT_DIM >= 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            latent_points[:, 0],
            latent_points[:, 1],
            latent_points[:, 2],
            c=labels, cmap='viridis', alpha=0.6, s=5
        )

        plt.colorbar(scatter, label='Class')
        ax.set_title('VAE 3D Latent Space Visualization')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        plt.show()


# In[ ]:


def main():
    """Main function to run the network intrusion detection."""
    # Load and preprocess data
    df_train, df_test, col_names = load_and_preprocess_data(config)

    # Explore data
    df_train_http, df_test_http = explore_data(df_train, df_test, config.SERVICE_TYPE)

    # Prepare data for model
    train_loader, X_train_tensor, X_test_tensor, y_test, input_dim = prepare_data_for_model(
        df_train_http, df_test_http, config
    )

    # Initialize model
    model = VAE(
        input_dim=input_dim,
        hidden_dims=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)

    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    model, history = train_model(model, train_loader, config)

    # Compute reconstruction errors on training and test data
    train_recon_errors = compute_reconstruction_errors(model, X_train_tensor, config)
    test_recon_errors = compute_reconstruction_errors(model, X_test_tensor, config)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(test_recon_errors, y_test)

    # Evaluate model
    metrics = evaluate_model(test_recon_errors, y_test, optimal_threshold)

    # Visualize latent space
    visualize_latent_space(model, X_test_tensor, y_test, config)

    print("Network intrusion detection analysis complete!")
    return model, metrics

# Execute main function
if __name__ == "__main__":
    model, metrics = main()
