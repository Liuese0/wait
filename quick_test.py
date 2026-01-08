"""
Quick test script for optimized quantum phase classifier
This is a simplified version for faster testing
"""
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import json

# Config
n_qubits = 8
n_layers = 6  # Fewer layers for quick test
device = "cpu"
dev = qml.device("default.qubit", wires=n_qubits)

# Optimized circuit
def optimized_ansatz(params):
    wires = list(range(n_qubits))
    params = params.reshape(n_layers, 3, n_qubits)

    for layer in range(n_layers):
        for i in wires:
            qml.Rot(params[layer, 0, i], params[layer, 1, i], params[layer, 2, i], wires=i)

        if layer < n_layers - 1:
            if layer % 2 == 0:
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
            else:
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])

measurement_qubits = [3, 4]

@qml.qnode(dev, interface='torch')
def circuit(state, params):
    qml.StatePrep(state, wires=list(range(n_qubits)))
    optimized_ansatz(params)
    return qml.probs(wires=measurement_qubits)

# Model
class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_params = n_layers * 3 * n_qubits
        torch.manual_seed(42)
        bound = np.sqrt(6.0 / (n_qubits + 4))
        self.params = nn.Parameter(
            torch.FloatTensor(self.total_params).uniform_(-bound, bound)
        )

    def forward(self, x):
        return circuit(x, self.params)

# Loss
def loss_fn(probs, labels):
    eps = 1e-8
    probs = probs + eps
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    label_one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.shape[1])
    loss = -torch.sum(label_one_hot * torch.log(probs), dim=1)
    return torch.mean(loss)

# Load data
print("Loading data...")
train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
print(f"Data shape: {train_X.shape}, Labels: {train_y.shape}")

# Prepare
t_train_X = torch.tensor(train_X, dtype=torch.complex64)
t_train_y = torch.tensor(train_y, dtype=torch.long)
train_dataset = TensorDataset(t_train_X, t_train_y)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Train
model = QNN()
optimizer = optim.Adam(model.parameters(), lr=0.05)

print(f"\nTraining with {n_layers} layers, {model.total_params} params")
print(f"Measurement qubits: {measurement_qubits}\n")

best_acc = 0
for epoch in range(200):
    total_loss = 0
    correct = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_classes = torch.argmax(preds, dim=1)
        correct += (pred_classes == batch_y).sum().item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / len(train_dataset)

    if avg_acc > best_acc:
        best_acc = avg_acc

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

print(f"\n✅ Best Accuracy: {best_acc:.4f}")

# Export
params = model.params.detach().cpu().numpy()

@qml.qnode(dev)
def export_circuit(params):
    optimized_ansatz(params)

qasm = qml.to_openqasm(export_circuit, measure_all=False)(params)

with open("quick_test_submission.json", "w") as f:
    json.dump({"qasm": qasm, "measurements": measurement_qubits}, f)

print("✅ Created quick_test_submission.json")
print(f"QASM length: {len(qasm)} chars")
