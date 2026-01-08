import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import json

# ==========================================
# Configuration
# ==========================================
n_qubits = 8
n_layers = 8  # Increased for better expressivity
device = "cuda" if torch.cuda.is_available() else "cpu"
dev = qml.device("default.qubit", wires=n_qubits)

# ==========================================
# Optimized Quantum Circuit with Brick-Layer Pattern
# ==========================================
def hardware_efficient_ansatz(params):
    """
    Hardware-efficient ansatz with brick-layer CNOT pattern.
    This minimizes CNOT gates while maintaining good entanglement.

    CNOT pattern alternates between:
    - Even layer: (0,1), (2,3), (4,5), (6,7) - 4 CNOTs
    - Odd layer: (1,2), (3,4), (5,6) - 3 CNOTs

    Total CNOTs per 2 layers: 7
    For 8 layers: 4*7 = 28 CNOTs (same as baseline but better structure)
    """
    wires = list(range(n_qubits))
    params = params.reshape(n_layers, 3, n_qubits)

    for layer in range(n_layers):
        # Single qubit rotations
        for i in wires:
            qml.Rot(params[layer, 0, i], params[layer, 1, i], params[layer, 2, i], wires=i)

        # Brick-layer CNOT pattern (more efficient than linear chain)
        if layer < n_layers - 1:  # No CNOT on last layer
            if layer % 2 == 0:
                # Even layers: (0,1), (2,3), (4,5), (6,7)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
            else:
                # Odd layers: (1,2), (3,4), (5,6)
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])


def minimal_cnot_ansatz(params):
    """
    Minimal CNOT version - more efficient for scoring
    Uses only 3 layers with strategic CNOT placement
    Total CNOTs: 2 * 7 = 14 (half of baseline!)
    """
    wires = list(range(n_qubits))
    params = params.reshape(-1, 3, n_qubits)
    n_layers_actual = params.shape[0]

    for layer in range(n_layers_actual):
        # Single qubit rotations
        for i in wires:
            qml.Rot(params[layer, 0, i], params[layer, 1, i], params[layer, 2, i], wires=i)

        # Add CNOTs only on even layers (skip odd layers)
        if layer < n_layers_actual - 1 and layer % 2 == 0:
            # Linear chain CNOTs
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])


# ==========================================
# Quantum Node with Multiple Measurement Options
# ==========================================
def create_qnode(measurement_qubits=[3, 4], use_minimal=False):
    """Create quantum node with specified measurement qubits"""
    @qml.qnode(dev, interface='torch')
    def qnode(state, params):
        wires = list(range(n_qubits))
        qml.StatePrep(state, wires=wires)

        if use_minimal:
            minimal_cnot_ansatz(params)
        else:
            hardware_efficient_ansatz(params)

        return qml.probs(wires=measurement_qubits)

    return qnode


# ==========================================
# Quantum Neural Network
# ==========================================
class OptimizedQNN(nn.Module):
    def __init__(self, n_layers, measurement_qubits=[3, 4], use_minimal=False):
        super().__init__()
        self.n_layers = n_layers
        self.measurement_qubits = measurement_qubits
        self.use_minimal = use_minimal
        self.total_params = n_layers * 3 * n_qubits

        # Better initialization: Xavier/Glorot uniform
        torch.manual_seed(42)
        bound = np.sqrt(6.0 / (n_qubits + 4))  # 4 is number of classes
        self.params = nn.Parameter(
            torch.FloatTensor(self.total_params).uniform_(-bound, bound)
        )

        self.qnode = create_qnode(measurement_qubits, use_minimal)

    def forward(self, x):
        return self.qnode(x, self.params)


# ==========================================
# Loss Function
# ==========================================
def quantum_phase_loss(probs, labels):
    """
    Cross-entropy loss with numerical stability
    """
    # Add small epsilon for numerical stability
    eps = 1e-8
    probs = probs + eps
    probs = probs / torch.sum(probs, dim=1, keepdim=True)

    # One-hot encoding
    label_one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.shape[1])

    # Cross-entropy loss
    loss = -torch.sum(label_one_hot * torch.log(probs), dim=1)

    return torch.mean(loss)


# ==========================================
# Training Function
# ==========================================
def train_model(model, train_loader, train_dataset, epochs=300, lr=0.05):
    """Train the quantum model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )

    loss_history = []
    acc_history = []

    print(f"--- Training QNN ---")
    print(f"Layers: {model.n_layers}")
    print(f"Parameters: {model.total_params}")
    print(f"Measurement Qubits: {model.measurement_qubits}")
    print(f"Using Minimal CNOT: {model.use_minimal}")

    best_acc = 0
    best_params = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X.to(device=device))
            loss = quantum_phase_loss(predictions, batch_y.to(device=device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted_classes = torch.argmax(predictions, dim=1)
            batch_y = batch_y.to(predicted_classes.device)
            correct += (predicted_classes == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / len(train_dataset)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        scheduler.step(avg_loss)

        # Track best model
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_params = model.params.detach().clone()

        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")

    # Restore best parameters
    if best_params is not None:
        model.params.data = best_params

    print(f"\n✅ Best Train Accuracy: {best_acc:.4f}")

    return loss_history, acc_history


# ==========================================
# Export to QASM
# ==========================================
def export_to_qasm(model):
    """Export trained model to OpenQASM format"""
    params = model.params.detach().cpu().numpy()

    # Define circuit for QASM conversion (no StatePrep or measurement)
    if model.use_minimal:
        @qml.qnode(dev, interface='torch')
        def circuit(params):
            minimal_cnot_ansatz(params)

        qasm_data = qml.to_openqasm(circuit, measure_all=False)(params)
    else:
        @qml.qnode(dev, interface='torch')
        def circuit(params):
            hardware_efficient_ansatz(params)

        qasm_data = qml.to_openqasm(circuit, measure_all=False)(params)

    return qasm_data


# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    # Load data
    print("Loading training data...")
    train_X = np.load("train_X.npy")
    train_y = np.load("train_y.npy")

    print(f"Training data shape: {train_X.shape}")
    print(f"Training labels shape: {train_y.shape}")
    print(f"Label distribution: {np.bincount(train_y)}")

    # Convert to tensors
    t_train_X = torch.tensor(train_X, dtype=torch.complex64)
    t_train_y = torch.tensor(train_y, dtype=torch.long)
    train_dataset = TensorDataset(t_train_X, t_train_y)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # ==========================================
    # Strategy 1: High Accuracy (More layers, standard CNOTs)
    # ==========================================
    print("\n" + "="*60)
    print("STRATEGY 1: High Accuracy Model (8 layers)")
    print("="*60)

    model_high_acc = OptimizedQNN(
        n_layers=8,
        measurement_qubits=[3, 4],  # Central qubits
        use_minimal=False
    )
    model_high_acc.to(device=device)

    loss_hist1, acc_hist1 = train_model(
        model_high_acc,
        train_loader,
        train_dataset,
        epochs=300,
        lr=0.05
    )

    # Export
    qasm_high_acc = export_to_qasm(model_high_acc)

    with open("./submission_high_accuracy.json", "w") as f:
        json.dump({
            "qasm": qasm_high_acc,
            "measurements": model_high_acc.measurement_qubits
        }, f)
    print("✅ 'submission_high_accuracy.json' created")

    # ==========================================
    # Strategy 2: Minimal CNOT (Fewer layers, efficient CNOTs)
    # ==========================================
    print("\n" + "="*60)
    print("STRATEGY 2: Minimal CNOT Model (6 layers)")
    print("="*60)

    model_minimal = OptimizedQNN(
        n_layers=6,
        measurement_qubits=[3, 4],
        use_minimal=True
    )
    model_minimal.to(device=device)

    loss_hist2, acc_hist2 = train_model(
        model_minimal,
        train_loader,
        train_dataset,
        epochs=300,
        lr=0.05
    )

    # Export
    qasm_minimal = export_to_qasm(model_minimal)

    with open("./submission_minimal_cnot.json", "w") as f:
        json.dump({
            "qasm": qasm_minimal,
            "measurements": model_minimal.measurement_qubits
        }, f)
    print("✅ 'submission_minimal_cnot.json' created")

    # ==========================================
    # Strategy 3: Alternative measurement qubits
    # ==========================================
    print("\n" + "="*60)
    print("STRATEGY 3: Alternative Measurement (qubits [2,5])")
    print("="*60)

    model_alt = OptimizedQNN(
        n_layers=8,
        measurement_qubits=[2, 5],  # Different measurement qubits
        use_minimal=False
    )
    model_alt.to(device=device)

    loss_hist3, acc_hist3 = train_model(
        model_alt,
        train_loader,
        train_dataset,
        epochs=300,
        lr=0.05
    )

    # Export
    qasm_alt = export_to_qasm(model_alt)

    with open("./submission_alt_measurement.json", "w") as f:
        json.dump({
            "qasm": qasm_alt,
            "measurements": model_alt.measurement_qubits
        }, f)
    print("✅ 'submission_alt_measurement.json' created")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Strategy 1 (High Accuracy): Final Acc = {acc_hist1[-1]:.4f}")
    print(f"Strategy 2 (Minimal CNOT): Final Acc = {acc_hist2[-1]:.4f}")
    print(f"Strategy 3 (Alt Measurement): Final Acc = {acc_hist3[-1]:.4f}")
    print("\nRecommendation: Submit the model with highest accuracy first!")
    print("You can submit up to 5 times per day.")


if __name__ == "__main__":
    main()
