import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. Physics Engine
# ---------------------------------------------------------
def run_simulation_fast(absorption_np, threshold=0.5):
    """
    Returns the 4 physics fields as 64x64 maps.
    This enables both Feature Extraction AND Physics-CNN input.
    """
    res = 64
    phi = np.zeros((res, res), dtype=np.float32)
    phi[:, 0] = 1.0
    phi[:, -1] = 0.0

    obstacle = zoom(absorption_np, (res/28, res/28), order=1)
    obstacle = (obstacle > threshold)

    for _ in range(100):
        phi_new = 0.25 * (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
                          np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1))
        phi_new[:, 0] = 1.0
        phi_new[:, -1] = 0.0
        phi_new[obstacle] = 0.5
        phi = phi_new

    u = np.gradient(phi, axis=1)
    v = -np.gradient(phi, axis=0)

    u[obstacle] = 0
    v[obstacle] = 0

    speed = np.sqrt(u**2 + v**2)
    vorticity = np.gradient(v, axis=1) - np.gradient(u, axis=0)

    # Return fields as a (4, 64, 64) array for Physics-CNN
    # Channels: [Vorticity, U, V, Speed]
    fields = np.stack([vorticity, u, v, speed], axis=0)

    return fields, obstacle

# ---------------------------------------------------------
# 2. Data Processing
# ---------------------------------------------------------
def extract_features_from_fields(fields):
    """Converts 4x64x64 fields into a compact feature vector."""
    vort, u, v, speed = fields

    feats = []
    # Global Stats
    for field in [vort, u, v, speed]:
        feats.extend([np.mean(field), np.std(field), np.max(field), np.min(field)])

    # Spatial Histograms (4x4 Grid)
    bins = 4
    _, h, w = fields.shape
    step_h, step_w = h // bins, w // bins

    for field in [vort, u, v, speed]:
        for i in range(bins):
            for j in range(bins):
                patch = field[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
                feats.append(np.mean(patch))
    return np.array(feats)

def process_batch(imgs, threshold=0.5):
    """Returns both Feature Vectors and Field Maps for a batch."""
    feat_vectors = []
    field_maps = []

    for img in imgs:
        mask = (img.squeeze() > 0.1).astype(np.float32)
        fields, _ = run_simulation_fast(mask, threshold)

        # 1. Create Feature Vector (for MLP)
        feat_vectors.append(extract_features_from_fields(fields))

        # 2. Create Field Map (for Physics-CNN)
        field_maps.append(fields)

    return np.array(feat_vectors), np.array(field_maps)

# ---------------------------------------------------------
# 3. Models
# ---------------------------------------------------------

# Model A: Standard CNN on Raw Images (Baseline)
class RawCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 28->14
        x = self.pool(F.relu(self.conv2(x))) # 14->7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Model B: MLP on Physics Features (Replaces SVM)
class PhysicsMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

# Model C: CNN on Physics Fields (Hybrid)
class PhysicsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 channels (Vort, U, V, Speed)
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Input image is 64x64 -> Pool(32) -> Pool(16)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 64 -> 32
        x = self.pool(F.relu(self.conv2(x))) # 32 -> 16
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Generic Trainer
def train_model(model, X_train, y_train, epochs=50, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.long).to(device)

    bs = min(32, len(X_train))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, y_t), batch_size=bs, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test, preds)

# ---------------------------------------------------------
# 4. Experiment Logic
# ---------------------------------------------------------
def add_noise(imgs, level):
    noise = np.random.randn(*imgs.shape) * level
    return np.clip(imgs + noise, 0, 1)

def get_data_subset(dataset, samples_per_class):
    imgs, labels = [], []
    counts = {i:0 for i in range(10)}
    for img, label in dataset:
        if counts[label] < samples_per_class:
            imgs.append(img.numpy())
            labels.append(label)
            counts[label] += 1
        if all(v >= samples_per_class for v in counts.values()):
            break
    return np.array(imgs), np.array(labels)

def run_scarcity_experiment(train_dataset, test_data, test_labels):
    print("\n" + "="*50)
    print("EXPERIMENT: Extreme Data Scarcity (1-20 samples)")
    print("="*50)

    # Pre-process Test Data ONCE
    print("Processing Test Data...")
    test_feats, test_fields = process_batch(test_data)
    # Scale features for MLP
    scaler = StandardScaler().fit(test_feats) # Fit on test just for structure, train will refit

    sizes = [1, 5, 20]
    results = {'raw_cnn': [], 'phys_mlp': [], 'phys_cnn': []}

    for spc in sizes:
        print(f"\n--- Samples/Class: {spc} ---")
        X_train_img, y_train = get_data_subset(train_dataset, spc)

        # Process Train Data
        train_feats, train_fields = process_batch(X_train_img)

        # Scale features for MLP
        scaler = StandardScaler().fit(train_feats)
        train_feats_s = scaler.transform(train_feats)
        test_feats_s = scaler.transform(test_feats)

        # 1. Raw CNN
        print(" Train Raw CNN...")
        model = RawCNN()
        trained = train_model(model, X_train_img, y_train, epochs=100)
        acc = evaluate_model(trained, test_data, test_labels)
        results['raw_cnn'].append(acc)
        print(f"  Acc: {acc:.4f}")

        # 2. Physics MLP
        print(" Train Physics MLP...")
        model = PhysicsMLP(input_dim=train_feats.shape[1])
        trained = train_model(model, train_feats_s, y_train, epochs=100)
        acc = evaluate_model(trained, test_feats_s, test_labels)
        results['phys_mlp'].append(acc)
        print(f"  Acc: {acc:.4f}")

        # 3. Physics CNN
        print(" Train Physics CNN...")
        model = PhysicsCNN()
        trained = train_model(model, train_fields, y_train, epochs=100)
        acc = evaluate_model(trained, test_fields, test_labels)
        results['phys_cnn'].append(acc)
        print(f"  Acc: {acc:.4f}")

    return sizes, results

def run_noise_experiment(train_dataset, test_data, test_labels):
    print("\n" + "="*50)
    print("EXPERIMENT: Noise Robustness")
    print("="*50)

    # Train on Clean (200 samples)
    print("Training on Clean Data (200 samples)...")
    X_train_img, y_train = get_data_subset(train_dataset, 200)
    train_feats, train_fields = process_batch(X_train_img)

    scaler = StandardScaler().fit(train_feats)
    train_feats_s = scaler.transform(train_feats)

    # Train all 3 models
    m_raw = train_model(RawCNN(), X_train_img, y_train)
    m_mlp = train_model(PhysicsMLP(input_dim=train_feats.shape[1]), train_feats_s, y_train)
    m_cnn = train_model(PhysicsCNN(), train_fields, y_train)

    noise_levels = [0.0, 0.3, 0.6, 0.8]
    results = {'raw_cnn': [], 'phys_mlp': [], 'phys_cnn': []}

    print("\nTesting against noise...")
    for level in noise_levels:
        print(f"--- Noise: {level} ---")
        X_test_noisy = add_noise(test_data, level)

        # Process Noisy Data
        # Threshold for physics: needs to be higher than noise level
        thresh = 0.5 if level < 0.5 else 0.7
        test_feats, test_fields = process_batch(X_test_noisy, threshold=thresh)
        test_feats_s = scaler.transform(test_feats)

        # Eval
        results['raw_cnn'].append(evaluate_model(m_raw, X_test_noisy, test_labels))
        results['phys_mlp'].append(evaluate_model(m_mlp, test_feats_s, test_labels))
        results['phys_cnn'].append(evaluate_model(m_cnn, test_fields, test_labels))

        print(f"  RawCNN: {results['raw_cnn'][-1]:.4f} | PhysMLP: {results['phys_mlp'][-1]:.4f} | PhysCNN: {results['phys_cnn'][-1]:.4f}")

    return noise_levels, results

def plot_all(sizes, scarcity_res, noise_levels, noise_res):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, scarcity_res['raw_cnn'], 'o-', label='Raw CNN (Baseline)', color='red')
    plt.plot(sizes, scarcity_res['phys_mlp'], 's-', label='Physics MLP (Vector)', color='blue')
    plt.plot(sizes, scarcity_res['phys_cnn'], '^-', label='Physics CNN (Hybrid)', color='green')
    plt.xscale('log')
    plt.title('Accuracy vs. Data Scarcity')
    plt.xlabel('Samples/Class')
    plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(noise_levels, noise_res['raw_cnn'], 'o-', label='Raw CNN', color='red')
    plt.plot(noise_levels, noise_res['phys_mlp'], 's-', label='Physics MLP', color='blue')
    plt.plot(noise_levels, noise_res['phys_cnn'], '^-', label='Physics CNN', color='green')
    plt.title('Accuracy vs. Input Noise')
    plt.xlabel('Noise Std Dev')
    plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig('final_comparison.png')
    plt.show()

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
print("Loading MNIST...")
transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_imgs = np.array([img.numpy() for img, _ in test_set])
test_labels = test_set.targets.numpy()

sizes, scarcity_res = run_scarcity_experiment(train_set, test_imgs, test_labels)
noise_levels, noise_res = run_noise_experiment(train_set, test_imgs, test_labels)

plot_all(sizes, scarcity_res, noise_levels, noise_res)

print("\nDone. Check 'final_comparison.png'.")
print("Hypothesis: Physics-CNN (Green) should win on Scarcity & Noise.")
