# Full Benchmark Results — D3QN Agent vs SABRE

**Generated**: 2026-04-01 14:48:50

## Overall Summary

| Metric | Value |
|--------|-------|
| Total circuits | 150 |
| Completed (with SWAPs) | 113 |
| Mean ratio | **0.9245** |
| Median ratio | 1.0000 |
| Std dev | 0.2516 |
| Best ratio | 0.0000 |
| Worst ratio | 1.5000 |
| Agent wins | 48 (42%) |
| Ties | 44 (39%) |
| SABRE wins | 21 (19%) |

### Per-Suite Summary

| Suite | Circuits | Completed | Mean Ratio | Median | Wins | Ties | Losses |
|-------|----------|-----------|------------|--------|------|------|--------|
| Generated | 110 | 89 | **0.9885** | 1.0000 | 37 | 34 | 18 |
| QASMBench | 40 | 24 | **0.6869** | 1.0000 | 11 | 10 | 3 |

---

## Suite: Generated

**110 circuits**, 89 completed with SWAPs, mean ratio 0.9885

### Category Summary

| Category | Circuits | Completed | Mean Ratio | Median | Min | Max | Wins | Losses |
|----------|----------|-----------|------------|--------|-----|-----|------|--------|
| Bernstein-Vazirani | 5 | 4 | **1.0218** | 1.0000 | 0.944 | 1.143 | 1 | 1 |
| GHZ | 5 | 1 | **1.5000** | 1.5000 | 1.500 | 1.500 | 0 | 1 |
| QFT | 5 | 5 | **0.8895** | 0.9545 | 0.600 | 1.022 | 4 | 1 |
| Quantum Volume | 20 | 19 | **1.0213** | 1.0000 | 1.000 | 1.200 | 0 | 3 |
| Random | 30 | 30 | **0.9882** | 0.9723 | 0.862 | 1.170 | 19 | 10 |
| Structured | 3 | 3 | **1.0000** | 1.0000 | 1.000 | 1.000 | 0 | 0 |
| VQE | 42 | 27 | **0.9590** | 1.0000 | 0.600 | 1.118 | 13 | 2 |

### Bernstein-Vazirani

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| bernstein_vazirani_4 | 4 | 3 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| bernstein_vazirani_8 | 8 | 7 | 5 | 5 | 1.000 | 5 | 0.0s | OK |
| bernstein_vazirani_12 | 12 | 11 | 8 | 7 | 1.143 | 8 | 0.0s | OK |
| bernstein_vazirani_16 | 16 | 15 | 9 | 9 | 1.000 | 9 | 0.0s | OK |
| bernstein_vazirani_19 | 19 | 18 | 17 | 18 | **0.944** | 17 | 0.0s | OK |

### GHZ

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| ghz_4 | 4 | 3 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| ghz_8 | 8 | 7 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| ghz_12 | 12 | 11 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| ghz_16 | 16 | 15 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| ghz_19 | 19 | 18 | 3 | 2 | 1.500 | 3 | 0.0s | OK |

### QFT

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| qft_4 | 4 | 8 | 3 | 5 | **0.600** | 3 | 0.9s | OK |
| qft_8 | 8 | 32 | 30 | 33 | **0.909** | 30 | 0.0s | OK |
| qft_12 | 12 | 72 | 75 | 78 | **0.962** | 75 | 0.1s | OK |
| qft_16 | 16 | 128 | 147 | 154 | **0.955** | 147 | 0.2s | OK |
| qft_19 | 19 | 180 | 228 | 223 | 1.022 | 228 | 0.3s | OK |

### Quantum Volume

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| quantum_volume_4_s0 | 4 | 18 | 1 | 1 | 1.000 | 1 | 0.0s | OK |
| quantum_volume_4_s1 | 4 | 18 | 3 | 3 | 1.000 | 3 | 0.0s | OK |
| quantum_volume_4_s2 | 4 | 18 | 1 | 1 | 1.000 | 1 | 0.0s | OK |
| quantum_volume_4_s3 | 4 | 18 | 3 | 3 | 1.000 | 3 | 0.0s | OK |
| quantum_volume_4_s4 | 4 | 18 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| quantum_volume_8_s0 | 8 | 36 | 10 | 10 | 1.000 | 10 | 0.0s | OK |
| quantum_volume_8_s1 | 8 | 36 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| quantum_volume_8_s2 | 8 | 36 | 7 | 7 | 1.000 | 7 | 0.0s | OK |
| quantum_volume_8_s3 | 8 | 36 | 5 | 5 | 1.000 | 5 | 0.0s | OK |
| quantum_volume_8_s4 | 8 | 36 | 7 | 7 | 1.000 | 7 | 0.0s | OK |
| quantum_volume_12_s0 | 12 | 54 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| quantum_volume_12_s1 | 12 | 54 | 12 | 10 | 1.200 | 12 | 0.0s | OK |
| quantum_volume_12_s2 | 12 | 54 | 11 | 10 | 1.100 | 11 | 0.0s | OK |
| quantum_volume_12_s3 | 12 | 54 | 12 | 12 | 1.000 | 12 | 0.1s | OK |
| quantum_volume_12_s4 | 12 | 54 | 8 | 8 | 1.000 | 8 | 0.0s | OK |
| quantum_volume_19_s0 | 19 | 81 | 14 | 14 | 1.000 | 14 | 0.1s | OK |
| quantum_volume_19_s1 | 19 | 81 | 14 | 14 | 1.000 | 14 | 0.1s | OK |
| quantum_volume_19_s2 | 19 | 81 | 16 | 16 | 1.000 | 16 | 0.0s | OK |
| quantum_volume_19_s3 | 19 | 81 | 21 | 19 | 1.105 | 21 | 0.1s | OK |
| quantum_volume_19_s4 | 19 | 81 | 15 | 15 | 1.000 | 15 | 0.1s | OK |

### Random

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| random_d20_s0 | 19 | 135 | 194 | 180 | 1.078 | 194 | 0.2s | OK |
| random_d20_s1 | 19 | 119 | 176 | 182 | **0.967** | 176 | 0.2s | OK |
| random_d20_s2 | 19 | 140 | 189 | 196 | **0.964** | 189 | 0.2s | OK |
| random_d20_s3 | 19 | 124 | 198 | 180 | 1.100 | 198 | 0.2s | OK |
| random_d20_s4 | 19 | 137 | 212 | 201 | 1.055 | 212 | 0.3s | OK |
| random_d20_s5 | 19 | 131 | 186 | 193 | **0.964** | 186 | 0.2s | OK |
| random_d20_s6 | 19 | 121 | 163 | 175 | **0.931** | 163 | 0.2s | OK |
| random_d20_s7 | 19 | 125 | 170 | 171 | **0.994** | 170 | 0.2s | OK |
| random_d20_s8 | 19 | 136 | 195 | 210 | **0.929** | 195 | 0.2s | OK |
| random_d20_s9 | 19 | 131 | 188 | 194 | **0.969** | 188 | 0.2s | OK |
| random_d20_s10 | 19 | 121 | 159 | 163 | **0.975** | 159 | 0.2s | OK |
| random_d20_s11 | 19 | 124 | 166 | 176 | **0.943** | 166 | 0.2s | OK |
| random_d20_s12 | 19 | 127 | 172 | 186 | **0.925** | 172 | 0.2s | OK |
| random_d20_s13 | 19 | 129 | 178 | 180 | **0.989** | 178 | 0.2s | OK |
| random_d20_s14 | 19 | 132 | 177 | 177 | 1.000 | 177 | 0.2s | OK |
| random_d20_s15 | 19 | 120 | 200 | 171 | 1.170 | 200 | 0.2s | OK |
| random_d20_s16 | 19 | 128 | 163 | 189 | **0.862** | 163 | 0.2s | OK |
| random_d20_s17 | 19 | 124 | 174 | 180 | **0.967** | 174 | 0.2s | OK |
| random_d20_s18 | 19 | 130 | 163 | 186 | **0.876** | 163 | 0.2s | OK |
| random_d20_s19 | 19 | 131 | 221 | 194 | 1.139 | 221 | 0.2s | OK |
| random_d20_s20 | 19 | 120 | 149 | 161 | **0.925** | 149 | 0.2s | OK |
| random_d20_s21 | 19 | 128 | 182 | 194 | **0.938** | 182 | 0.2s | OK |
| random_d20_s22 | 19 | 131 | 179 | 176 | 1.017 | 179 | 0.2s | OK |
| random_d20_s23 | 19 | 128 | 195 | 186 | 1.048 | 195 | 0.2s | OK |
| random_d20_s24 | 19 | 136 | 187 | 200 | **0.935** | 187 | 0.2s | OK |
| random_d20_s25 | 19 | 130 | 187 | 195 | **0.959** | 187 | 0.2s | OK |
| random_d20_s26 | 19 | 139 | 219 | 224 | **0.978** | 219 | 0.3s | OK |
| random_d20_s27 | 19 | 128 | 178 | 175 | 1.017 | 178 | 0.2s | OK |
| random_d20_s28 | 19 | 135 | 198 | 197 | 1.005 | 198 | 0.2s | OK |
| random_d20_s29 | 19 | 132 | 205 | 200 | 1.025 | 205 | 0.2s | OK |

### Structured

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| cnot_ring_19_3rep | 19 | 57 | 15 | 15 | 1.000 | 15 | 0.0s | OK |
| cnot_ring_19_5rep | 19 | 95 | 45 | 45 | 1.000 | 45 | 0.1s | OK |
| cnot_ring_19_10rep | 19 | 190 | 57 | 57 | 1.000 | 57 | 0.1s | OK |

### VQE

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| vqe_linear_4q_1r | 4 | 3 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_4q_3r | 4 | 9 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_4q_5r | 4 | 15 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_8q_1r | 8 | 7 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_8q_3r | 8 | 21 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_8q_5r | 8 | 35 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_12q_1r | 12 | 11 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_12q_3r | 12 | 33 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_12q_5r | 12 | 55 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_16q_1r | 16 | 15 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_16q_3r | 16 | 45 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_16q_5r | 16 | 75 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_linear_19q_1r | 19 | 18 | 12 | 13 | **0.923** | 12 | 0.0s | OK |
| vqe_linear_19q_3r | 19 | 54 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| vqe_linear_19q_5r | 19 | 90 | 15 | 15 | 1.000 | 15 | 0.0s | OK |
| vqe_circular_4q_1r | 4 | 4 | 1 | 1 | 1.000 | 1 | 0.0s | OK |
| vqe_circular_4q_3r | 4 | 12 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| vqe_circular_4q_5r | 4 | 20 | 9 | 10 | **0.900** | 9 | 0.0s | OK |
| vqe_circular_8q_1r | 8 | 8 | 4 | 4 | 1.000 | 4 | 0.0s | OK |
| vqe_circular_8q_3r | 8 | 24 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| vqe_circular_8q_5r | 8 | 40 | 10 | 10 | 1.000 | 10 | 0.0s | OK |
| vqe_circular_12q_1r | 12 | 12 | 8 | 8 | 1.000 | 8 | 0.0s | OK |
| vqe_circular_12q_3r | 12 | 36 | 19 | 17 | 1.118 | 19 | 0.0s | OK |
| vqe_circular_12q_5r | 12 | 60 | 30 | 31 | **0.968** | 30 | 0.1s | OK |
| vqe_circular_16q_1r | 16 | 16 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_circular_16q_3r | 16 | 48 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_circular_16q_5r | 16 | 80 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_circular_19q_1r | 19 | 19 | 3 | 3 | 1.000 | 3 | 0.0s | OK |
| vqe_circular_19q_3r | 19 | 57 | 32 | 34 | **0.941** | 32 | 0.1s | OK |
| vqe_circular_19q_5r | 19 | 95 | 27 | 27 | 1.000 | 27 | 0.1s | OK |
| vqe_full_4q_1r | 4 | 6 | 2 | 2 | 1.000 | 2 | 0.0s | OK |
| vqe_full_4q_2r | 4 | 12 | 3 | 5 | **0.600** | 3 | 0.0s | OK |
| vqe_full_4q_3r | 4 | 18 | 6 | 6 | 1.000 | 6 | 0.0s | OK |
| vqe_full_8q_1r | 8 | 28 | 21 | 20 | 1.050 | 21 | 0.0s | OK |
| vqe_full_8q_2r | 8 | 56 | 44 | 47 | **0.936** | 44 | 0.1s | OK |
| vqe_full_8q_3r | 8 | 84 | 69 | 77 | **0.896** | 69 | 0.1s | OK |
| vqe_full_12q_1r | 12 | 66 | 55 | 64 | **0.859** | 55 | 0.1s | OK |
| vqe_full_12q_2r | 12 | 132 | 130 | 143 | **0.909** | 130 | 0.2s | OK |
| vqe_full_12q_3r | 12 | 198 | 220 | 224 | **0.982** | 220 | 0.3s | OK |
| vqe_full_19q_1r | 19 | 171 | 187 | 209 | **0.895** | 187 | 0.2s | OK |
| vqe_full_19q_2r | 19 | 342 | 395 | 414 | **0.954** | 395 | 0.7s | OK |
| vqe_full_19q_3r | 19 | 513 | 585 | 608 | **0.962** | 585 | 1.2s | OK |

---

## Suite: QASMBench

**40 circuits**, 24 completed with SWAPs, mean ratio 0.6869

### Category Summary

| Category | Circuits | Completed | Mean Ratio | Median | Min | Max | Wins | Losses |
|----------|----------|-----------|------------|--------|-----|-----|------|--------|
| QB:Error Correction | 3 | 3 | **1.0392** | 1.0000 | 1.000 | 1.118 | 0 | 1 |
| QB:Factoring/Arithmetic | 7 | 7 | **0.3929** | 0.3750 | 0.000 | 1.000 | 6 | 0 |
| QB:Linear Algebra | 2 | 1 | **1.1395** | 1.1395 | 1.140 | 1.140 | 0 | 1 |
| QB:ML/Classification | 2 | 1 | **1.0000** | 1.0000 | 1.000 | 1.000 | 0 | 0 |
| QB:Other | 8 | 4 | **0.6000** | 0.7000 | 0.000 | 1.000 | 2 | 0 |
| QB:QFT/QPE | 5 | 4 | **0.9945** | 1.0000 | 0.978 | 1.000 | 1 | 0 |
| QB:Search | 3 | 1 | **0.0000** | 0.0000 | 0.000 | 0.000 | 1 | 0 |
| QB:Simulation | 2 | 0 | — | — | — | — | 0 | 0 |
| QB:State Prep | 4 | 1 | **0.0000** | 0.0000 | 0.000 | 0.000 | 1 | 0 |
| QB:VQE/Variational | 4 | 2 | **1.0500** | 1.0500 | 1.000 | 1.100 | 0 | 1 |

### QB:Error Correction

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| error_correctiond3_n5 | 5 | 49 | 4 | 4 | 1.000 | 4 | 0.0s | OK |
| qec_en_n5 | 5 | 10 | 2 | 2 | 1.000 | 2 | 0.0s | OK |
| qec9xz_n17 | 17 | 32 | 38 | 34 | 1.118 | 38 | 0.1s | OK |

### QB:Factoring/Arithmetic

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| adder_n10 | 10 | 1 | 0 | 21 | **0.000** | 0 | 0.0s | OK |
| adder_n4 | 4 | 10 | 2 | 2 | 1.000 | 2 | 0.1s | OK |
| shor_n5 | 5 | 6 | 3 | 8 | **0.375** | 3 | 0.0s | OK |
| multiplier_n15 | 15 | 30 | 15 | 111 | **0.135** | 15 | 0.1s | OK |
| multiply_n13 | 13 | 4 | 8 | 19 | **0.421** | 8 | 0.0s | OK |
| qf21_n15 | 15 | 46 | 59 | 73 | **0.808** | 59 | 0.1s | OK |
| square_root_n18 | 18 | 118 | 6 | 543 | **0.011** | 6 | 0.2s | OK |

### QB:Linear Algebra

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| hhl_n7 | 7 | 196 | 49 | 43 | 1.140 | 49 | 0.1s | OK |
| linearsolver_n3 | 3 | 4 | 0 | 0 | 1.000 | 0 | 0.0s | OK |

### QB:ML/Classification

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| dnn_n2 | 2 | 42 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| dnn_n8 | 8 | 192 | 10 | 10 | 1.000 | 10 | 0.1s | OK |

### QB:Other

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| basis_change_n3 | 3 | 10 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| basis_test_n4 | 4 | 34 | 0 | 6 | **0.000** | 0 | 0.0s | OK |
| fredkin_n3 | 3 | 8 | 3 | 3 | 1.000 | 3 | 0.0s | OK |
| hs4_n4 | 4 | 4 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| iswap_n2 | 2 | 2 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| lpn_n5 | 5 | 2 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| toffoli_n3 | 3 | 6 | 1 | 1 | 1.000 | 1 | 0.0s | OK |
| seca_n11 | 11 | 36 | 10 | 25 | **0.400** | 10 | 0.0s | OK |

### QB:QFT/QPE

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| ipea_n2 | 2 | 15 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| pea_n5 | 5 | 21 | 5 | 5 | 1.000 | 5 | 0.0s | OK |
| qft_n4 | 4 | 6 | 3 | 3 | 1.000 | 3 | 0.0s | OK |
| qpe_n9 | 9 | 16 | 15 | 15 | 1.000 | 15 | 0.0s | OK |
| qft_n18 | 18 | 306 | 133 | 136 | **0.978** | 133 | 0.3s | OK |

### QB:Search

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| deutsch_n2 | 2 | 1 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| grover_n2 | 2 | 2 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| simon_n6 | 6 | 2 | 0 | 5 | **0.000** | 0 | 0.0s | OK |

### QB:Simulation

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| ising_n10 | 10 | 90 | 0 | 0 | 1.000 | 0 | 0.1s | OK |
| quantumwalks_n2 | 2 | 3 | 0 | 0 | 1.000 | 0 | 0.0s | OK |

### QB:State Prep

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| bell_n4 | 4 | 7 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| cat_state_n4 | 4 | 3 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| teleportation_n3 | 3 | 2 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| wstate_n3 | 3 | 2 | 0 | 3 | **0.000** | 0 | 0.0s | OK |

### QB:VQE/Variational

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Steps | Time | Status |
|---------|--------|----------|-------------|-------------|-------|-------|------|--------|
| qaoa_n3 | 3 | 6 | 1 | 1 | 1.000 | 1 | 0.0s | OK |
| qaoa_n6 | 6 | 54 | 11 | 10 | 1.100 | 11 | 0.0s | OK |
| variational_n4 | 4 | 16 | 0 | 0 | 1.000 | 0 | 0.0s | OK |
| vqe_n4 | 4 | 9 | 0 | 0 | 1.000 | 0 | 0.0s | OK |

---

## QASMBench — Skipped Circuits

80 circuits could not be used:

| Circuit | Reason |
|---------|--------|
| bb84_n8 | no_2q_gates |
| hhl_n10 | parse_crash |
| inverseqft_n4 | no_2q_gates |
| qec_sm_n5 | no_2q_gates |
| qrng_n4 | no_2q_gates |
| sat_n7 | no_2q_gates |
| vqe_uccsd_n4 | parse_crash |
| vqe_uccsd_n6 | parse_crash |
| vqe_uccsd_n8 | parse_crash |
| ising_n26 | too_large (26q) |
| knn_n25 | too_large (25q) |
| qram_n20 | too_large (20q) |
| sat_n11 | no_2q_gates |
| swap_test_n25 | too_large (25q) |
| vqe_n24 | too_large (24q) |
| wstate_n27 | too_large (27q) |
| random_QAOA_angles_k3_N10000_p1 | parse_crash |
| random_QAOA_angles_k3_N1000_p1 | parse_crash |
| random_QAOA_angles_k3_N100_p100 | parse_crash |
| 100 | too_large (100q) |
| 32 | too_large (32q) |
| adder_n118 | too_large (118q) |
| adder_n28 | too_large (28q) |
| adder_n433 | too_large (433q) |
| adder_n64 | too_large (64q) |
| bv_n140 | too_large (140q) |
| bv_n280 | too_large (280q) |
| bv_n30 | too_large (30q) |
| bv_n70 | too_large (70q) |
| bwt_n177 | too_large (177q) |
| bwt_n37 | too_large (37q) |
| bwt_n57 | too_large (57q) |
| bwt_n97 | too_large (97q) |
| cat_n130 | too_large (130q) |
| cat_n260 | too_large (260q) |
| cat_n35 | too_large (35q) |
| cat_n65 | too_large (65q) |
| cc_n151 | parse_crash |
| cc_n301 | parse_crash |
| cc_n32 | too_large (32q) |
| cc_n64 | too_large (64q) |
| dnn_n33 | too_large (33q) |
| dnn_n51 | too_large (51q) |
| ghz_n127 | too_large (127q) |
| ghz_state_n255 | too_large (255q) |
| ghz_n40 | too_large (40q) |
| ghz_n78 | too_large (78q) |
| ising_n34 | too_large (34q) |
| ising_n42 | too_large (42q) |
| ising_n420 | too_large (420q) |
| ising_n66 | too_large (66q) |
| ising_n98 | too_large (98q) |
| knn_129 | too_large (129q) |
| knn_n31 | too_large (31q) |
| knn_341 | too_large (341q) |
| knn_n41 | too_large (41q) |
| knn_n67 | too_large (67q) |
| multiplier_n350 | too_large (350q) |
| multiplier_n400 | too_large (400q) |
| multiplier_n45 | too_large (45q) |
| multiplier_n75 | too_large (75q) |
| qft_n160 | too_large (160q) |
| qft_n29 | too_large (29q) |
| qft_n320 | too_large (320q) |
| qft_n63 | too_large (63q) |
| qugan_n111 | too_large (111q) |
| qugan_n39 | too_large (39q) |
| qugan_n395 | too_large (395q) |
| qugan_n71 | too_large (71q) |
| square_root_n45 | too_large (45q) |
| square_root_n60 | too_large (60q) |
| swap_test_n115 | too_large (115q) |
| swap_test_n361 | too_large (361q) |
| swap_test_n41 | too_large (41q) |
| swap_test_n83 | too_large (83q) |
| vqe_uccsd_n28 | parse_crash |
| wstate_n118 | too_large (118q) |
| wstate_n36 | too_large (36q) |
| wstate_n380 | too_large (380q) |
| wstate_n76 | too_large (76q) |
