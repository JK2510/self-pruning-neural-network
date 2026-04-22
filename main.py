# Install (only needed in Colab)
!pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Prunable Linear Layer
# -------------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias


# -------------------------------
# Model
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 128)
        self.fc2 = PrunableLinear(128, 64)
        self.fc3 = PrunableLinear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -------------------------------
# Data
# -------------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False
)


# -------------------------------
# Sparsity Loss
# -------------------------------
def sparsity_loss(model):
    loss = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            loss += torch.sum(gates)
    return loss


# -------------------------------
# Training Function
# -------------------------------
def train(model, lambda_val, epochs=2):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            cls_loss = criterion(output, target)
            sp_loss = sparsity_loss(model)

            loss = cls_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

    return model


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            correct += (pred == target).sum().item()
            total += target.size(0)

    return 100 * correct / total


# -------------------------------
# Sparsity Calculation
# -------------------------------
def calculate_sparsity(model):
    total = 0
    zero = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores)
            total += gates.numel()
            zero += (gates < 0.01).sum().item()

    return 100 * zero / total


# -------------------------------
# Run for different lambda values
# -------------------------------
lambdas = [0.0001, 0.001, 0.01]
results = []

for l in lambdas:
    print(f"\nTraining with lambda = {l}")
    model = Net().to(device)
    model = train(model, l)

    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    print(f"Lambda: {l}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")
    results.append((l, acc, sparsity))


# -------------------------------
# Plot gate distribution (best model)
# -------------------------------
best_model = model
all_gates = []

for m in best_model.modules():
    if isinstance(m, PrunableLinear):
        gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
        all_gates.extend(gates)

plt.hist(all_gates, bins=50)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()