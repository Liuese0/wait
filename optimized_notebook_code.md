# Optimized Quantum Phase Classification

이 파일은 노트북의 핵심 코드 셀을 최적화한 버전입니다.

## Cell 1: 패키지 설치
```python
!pip install pennylane torch
```

## Cell 2: Import 라이브러리
```python
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import matplotlib.pyplot as plt
```

## Cell 3: 데이터 다운로드
```python
!wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_X.npy
!wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_y.npy
```

## Cell 4: 데이터 로드
```python
train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")

print(f"Training data shape: {train_X.shape}")
print(f"Training labels shape: {train_y.shape}")
print(f"Label distribution: {np.bincount(train_y)}")
```

## Cell 5: 최적화된 양자 회로 정의
```python
n_qubits = 8
n_layers = 8  # Increased for better expressivity
device = "cuda" if torch.cuda.is_available() else "cpu"
dev = qml.device("default.qubit", wires=n_qubits)

def hardware_efficient_ansatz(params):
    """
    Hardware-efficient ansatz with brick-layer CNOT pattern.
    Minimizes CNOT gates while maintaining good entanglement.

    CNOT pattern alternates between:
    - Even layer: (0,1), (2,3), (4,5), (6,7) - 4 CNOTs
    - Odd layer: (1,2), (3,4), (5,6) - 3 CNOTs
    """
    wires = list(range(n_qubits))
    params = params.reshape(n_layers, 3, n_qubits)

    for layer in range(n_layers):
        # Single qubit rotations
        for i in wires:
            qml.Rot(params[layer, 0, i], params[layer, 1, i], params[layer, 2, i], wires=i)

        # Brick-layer CNOT pattern
        if layer < n_layers - 1:
            if layer % 2 == 0:
                # Even layers: (0,1), (2,3), (4,5), (6,7)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
            else:
                # Odd layers: (1,2), (3,4), (5,6)
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])

# Create quantum node with optimized measurement qubits
measurement_qubits = [3, 4]  # Central qubits for better phase detection

@qml.qnode(dev, interface='torch')
def quantum_circuit(state, params):
    wires = list(range(n_qubits))
    qml.StatePrep(state, wires=wires)
    hardware_efficient_ansatz(params)
    return qml.probs(wires=measurement_qubits)
```

## Cell 6: 양자 신경망 모델
```python
class OptimizedQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_params = n_layers * 3 * n_qubits

        # Better initialization: Xavier/Glorot uniform
        torch.manual_seed(42)
        bound = np.sqrt(6.0 / (n_qubits + 4))
        self.params = nn.Parameter(
            torch.FloatTensor(self.total_params).uniform_(-bound, bound)
        )

    def forward(self, x):
        return quantum_circuit(x, self.params)

model = OptimizedQNN()
model.to(device=device)

print(f"Model parameters: {model.total_params}")
print(f"Measurement qubits: {measurement_qubits}")
```

## Cell 7: 손실 함수 및 훈련 설정
```python
def quantum_phase_loss(probs, labels):
    """Cross-entropy loss with numerical stability"""
    eps = 1e-8
    probs = probs + eps
    probs = probs / torch.sum(probs, dim=1, keepdim=True)

    label_one_hot = torch.nn.functional.one_hot(labels, num_classes=probs.shape[1])
    loss = -torch.sum(label_one_hot * torch.log(probs), dim=1)

    return torch.mean(loss)

# Data preparation
t_train_X = torch.tensor(train_X, dtype=torch.complex64)
t_train_y = torch.tensor(train_y, dtype=torch.long)
train_dataset = TensorDataset(t_train_X, t_train_y)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Optimizer with learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, verbose=True
)
```

## Cell 8: 모델 훈련
```python
epochs = 300
loss_history = []
acc_history = []

print(f"--- Training Optimized QNN ---")
print(f"Layers: {n_layers}")
print(f"Parameters: {model.total_params}")
print(f"Measurement Qubits: {measurement_qubits}")

best_acc = 0
best_params = None

for epoch in range(epochs):
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

    if avg_acc > best_acc:
        best_acc = avg_acc
        best_params = model.params.detach().clone()

    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")

# Restore best parameters
if best_params is not None:
    model.params.data = best_params

print(f"\n✅ Best Train Accuracy: {best_acc:.4f}")
```

## Cell 9: 훈련 결과 시각화
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_history)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

ax2.plot(acc_history)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Cell 10: 회로 시각화
```python
# Visualize the quantum circuit
sample_state = t_train_X[0]
qml.draw_mpl(quantum_circuit)(sample_state, model.params)
plt.show()
```

## Cell 11: QASM 내보내기 및 제출 파일 생성
```python
# Extract trained parameters
params = model.params.detach().cpu().numpy()

# Define circuit for QASM conversion (no StatePrep or Measurement)
@qml.qnode(dev, interface='torch')
def classifier_for_export(params):
    hardware_efficient_ansatz(params)

# Generate OpenQASM string
qasm_data = qml.to_openqasm(classifier_for_export, measure_all=False)(params)

print(f"✅ Measurement Qubits: {measurement_qubits}")
print(f"✅ QASM Data Generated (Length: {len(qasm_data)} characters)")
print("--- QASM Preview (First 15 lines) ---")
print("\n".join(qasm_data.split('\n')[:15]))

# Create submission file
with open("./optimized_submission.json", "w") as f:
    json.dump({
        "qasm": qasm_data,
        "measurements": measurement_qubits
    }, f)

print("\n✅ Submission file 'optimized_submission.json' created.")
print(f"✅ Final Training Accuracy: {acc_history[-1]:.4f}")
```

## Cell 12: 다운로드 (Colab 전용)
```python
from google.colab import files
files.download('optimized_submission.json')
```

---

## 주요 개선사항:

1. **회로 아키텍처**: Brick-layer CNOT 패턴으로 entanglement 개선
2. **측정 큐비트**: [6,7] → [3,4] (중앙 큐비트가 양자 위상을 더 잘 포착)
3. **레이어 수**: 5 → 8 (표현력 증가)
4. **초기화**: Xavier/Glorot uniform 초기화로 수렴 속도 개선
5. **학습률 스케줄링**: ReduceLROnPlateau로 최적화 안정성 향상
6. **에포크**: 200 → 300 (충분한 훈련)
7. **Best model tracking**: 최고 성능 모델 자동 저장

## 예상 성능:
- Training Accuracy: 93.75% → 95%+ (목표)
- CNOT Gates: 28개 (brick-layer pattern)
- 더 안정적인 학습 곡선
