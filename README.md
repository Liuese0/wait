# 양자 위상 분류 챌린지 - 최적화 솔루션

양자 머신러닝(QML)을 사용하여 8큐비트 양자 시스템의 바닥 상태를 4가지 양자 위상으로 분류하는 최적화된 솔루션입니다.

## 📊 챌린지 개요

- **시스템**: 8큐비트 1D 체인
- **훈련 데이터**: 16개 (레이블 포함)
- **테스트 데이터**: 2,000개
- **클래스**: 4가지 양자 위상 (클러스터, 강자성, 반강자성, 자명)
- **목표**: 최고의 정확도 + 최소 CNOT 게이트 수

## 🚀 주요 개선사항

### 1. **회로 아키텍처 최적화**
- ✅ **Brick-layer CNOT 패턴** 적용
  - Even layers: (0,1), (2,3), (4,5), (6,7) - 4 CNOTs
  - Odd layers: (1,2), (3,4), (5,6) - 3 CNOTs
  - 더 효율적인 entanglement 구조

### 2. **측정 큐비트 최적화**
- ✅ Baseline: [6, 7] → **Optimized: [3, 4]**
- 중앙 큐비트가 양자 위상을 더 잘 포착
- 대체 옵션: [2, 5]도 테스트 가능

### 3. **하이퍼파라미터 튜닝**
- ✅ 레이어 수: 5 → **8 layers**
- ✅ 에포크: 200 → **300 epochs**
- ✅ Learning rate: 0.05 (ReduceLROnPlateau 스케줄러)
- ✅ 초기화: Xavier/Glorot uniform
- ✅ Best model tracking

### 4. **성능 예상**
- Training Accuracy: **95%+** (baseline: 93.75%)
- CNOT Gates: **28개** (brick-layer) 또는 **14개** (minimal version)
- 더 안정적인 학습 곡선

## 📁 파일 구조

```
.
├── optimized_quantum_classifier.py   # 메인 Python 스크립트 (3가지 전략)
├── quick_test.py                     # 빠른 테스트용 간소화 버전
├── optimized_notebook_code.md        # Colab 노트북용 코드
├── baseline_251215_ipynb의_사본.ipynb # 원본 baseline 노트북
├── baseline_251215_ipynb의_사본_backup.ipynb # 백업
└── README.md                         # 이 파일
```

## 🔧 실행 방법

### 방법 1: Python 스크립트 실행 (권장)

```bash
# 패키지 설치
pip install pennylane torch numpy

# 훈련 데이터 다운로드
wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_X.npy
wget https://raw.githubusercontent.com/aifactory-team/AFCompetition/main/9245/train_y.npy

# 메인 스크립트 실행 (3가지 전략 모두 훈련)
python optimized_quantum_classifier.py

# 또는 빠른 테스트
python quick_test.py
```

**생성되는 파일:**
- `submission_high_accuracy.json` - 높은 정확도 모델 (8 layers)
- `submission_minimal_cnot.json` - 최소 CNOT 모델 (6 layers)
- `submission_alt_measurement.json` - 대체 측정 큐비트 ([2,5])

### 방법 2: Google Colab에서 실행

1. `baseline_251215_ipynb의_사본.ipynb` 파일을 Colab에서 열기
2. `optimized_notebook_code.md`의 코드를 복사하여 셀 교체
3. 순서대로 실행
4. `optimized_submission.json` 다운로드

## 📈 3가지 최적화 전략

### Strategy 1: High Accuracy (높은 정확도 우선)
- **레이어**: 8
- **CNOT 패턴**: Brick-layer (28 CNOTs)
- **측정 큐비트**: [3, 4]
- **목표**: 최고 정확도 달성

### Strategy 2: Minimal CNOT (게이트 수 최소화)
- **레이어**: 6
- **CNOT 패턴**: Sparse (14 CNOTs - 절반!)
- **측정 큐비트**: [3, 4]
- **목표**: 정확도 유지하면서 CNOT 최소화

### Strategy 3: Alternative Measurement (다른 측정 위치)
- **레이어**: 8
- **CNOT 패턴**: Brick-layer (28 CNOTs)
- **측정 큐비트**: [2, 5]
- **목표**: 다른 측정 위치 탐색

## 💡 제출 전략

1. **첫 번째 제출**: `submission_high_accuracy.json`
   - 가장 높은 정확도 기대

2. **두 번째 제출**: `submission_minimal_cnot.json`
   - 정확도가 비슷하면 CNOT 수로 승부

3. **세 번째 제출**: `submission_alt_measurement.json`
   - 대체 전략

4. **추가 튜닝**: 필요시 하이퍼파라미터 조정
   - learning rate
   - epochs
   - n_layers

## 🎯 핵심 알고리즘

### Brick-layer CNOT 패턴
```python
for layer in range(n_layers):
    # Rotation gates
    for i in range(n_qubits):
        qml.Rot(φ, θ, ω, wires=i)

    # Alternating CNOT pattern
    if layer % 2 == 0:
        # Even: (0,1), (2,3), (4,5), (6,7)
        for i in range(0, n_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
    else:
        # Odd: (1,2), (3,4), (5,6)
        for i in range(1, n_qubits-1, 2):
            qml.CNOT(wires=[i, i+1])
```

### 손실 함수
```python
def quantum_phase_loss(probs, labels):
    """Cross-entropy with numerical stability"""
    eps = 1e-8
    probs = (probs + eps) / sum(probs + eps)
    return -mean(log(probs[correct_class]))
```

## 📊 예상 결과

| 전략 | Layers | CNOTs | Train Acc | 예상 Test Acc |
|------|--------|-------|-----------|--------------|
| Baseline | 5 | 28 | 93.75% | ~90% |
| Strategy 1 | 8 | 28 | **95%+** | **92%+** |
| Strategy 2 | 6 | **14** | 94%+ | 91%+ |
| Strategy 3 | 8 | 28 | 95%+ | 92%+ |

## 🔍 이론적 배경

### 양자 위상이란?
- **클러스터 위상**: 높은 entanglement
- **강자성**: 모든 스핀이 같은 방향
- **반강자성**: 이웃 스핀이 반대 방향
- **자명(Trivial)**: entanglement 없음

### 왜 중앙 큐비트를 측정하나?
- 경계 효과 최소화
- 위상 전이는 bulk(중앙)에서 더 명확
- [3,4] 또는 [2,5]가 이론적으로 최적

### Brick-layer의 장점
- Hardware-efficient: 실제 양자 컴퓨터에서 구현 용이
- 충분한 entanglement 제공
- CNOT 수 최소화

## 🐛 문제 해결

### 정확도가 낮을 때
- [ ] Epochs 증가 (300 → 500)
- [ ] Learning rate 조정 (0.05 → 0.03)
- [ ] Layers 증가 (8 → 10)
- [ ] 다른 측정 큐비트 시도

### 과적합 발생 시
- [ ] Layers 감소 (8 → 6)
- [ ] 정규화 추가
- [ ] Early stopping

### CNOT 수를 더 줄이려면
- [ ] `use_minimal=True` 옵션 사용
- [ ] Layers 감소
- [ ] Sparse connectivity 패턴 시도

## 📚 참고 자료

- PennyLane 문서: https://pennylane.ai
- 양자 위상 전이: arXiv:2103.xxxxx
- Hardware-efficient ansatz: arXiv:1704.05018

## 🏆 성공을 위한 팁

1. ✅ **정확도가 최우선**: CNOT는 동점일 때만 영향
2. ✅ **하루 5회 제출**: 신중하게 전략적으로
3. ✅ **다양한 측정 큐비트 실험**: [3,4], [2,5], [4,5] 등
4. ✅ **충분한 훈련**: 최소 300 epochs
5. ✅ **Best model 저장**: 과적합 방지

---

**Good Luck!** 🎉

문의사항이 있으면 이슈를 열어주세요.
