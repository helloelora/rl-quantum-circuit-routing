# Benchmark Test Results

**Overall**: 110 circuits, 89 completed, mean ratio 1.0027, median 1.0000


## Bernstein-Vazirani

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| bernstein_vazirani_4 | 4 | 3 | 0 | 0 | 1.000 | OK |
| bernstein_vazirani_8 | 8 | 7 | 4 | 4 | 1.000 | OK |
| bernstein_vazirani_12 | 12 | 11 | 9 | 9 | 1.000 | OK |
| bernstein_vazirani_16 | 16 | 15 | 14 | 13 | 1.077 | OK |
| bernstein_vazirani_19 | 19 | 18 | 15 | 15 | 1.000 | OK |

**Bernstein-Vazirani summary**: 4/5 completed, mean ratio 1.019, median 1.000


## GHZ

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| ghz_4 | 4 | 3 | 0 | 0 | 1.000 | OK |
| ghz_8 | 8 | 7 | 0 | 0 | 1.000 | OK |
| ghz_12 | 12 | 11 | 0 | 0 | 1.000 | OK |
| ghz_16 | 16 | 15 | 0 | 0 | 1.000 | OK |
| ghz_19 | 19 | 18 | 13 | 12 | 1.083 | OK |

**GHZ summary**: 1/5 completed, mean ratio 1.083, median 1.083


## QFT

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| qft_4 | 4 | 8 | 3 | 5 | 0.600 | OK |
| qft_8 | 8 | 32 | 29 | 30 | 0.967 | OK |
| qft_12 | 12 | 72 | 82 | 81 | 1.012 | OK |
| qft_16 | 16 | 128 | 148 | 154 | 0.961 | OK |
| qft_19 | 19 | 180 | 235 | 207 | 1.135 | OK |

**QFT summary**: 5/5 completed, mean ratio 0.935, median 0.967


## Quantum Volume

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| quantum_volume_4_s0 | 4 | 18 | 1 | 1 | 1.000 | OK |
| quantum_volume_4_s1 | 4 | 18 | 3 | 3 | 1.000 | OK |
| quantum_volume_4_s2 | 4 | 18 | 1 | 1 | 1.000 | OK |
| quantum_volume_4_s3 | 4 | 18 | 3 | 3 | 1.000 | OK |
| quantum_volume_4_s4 | 4 | 18 | 0 | 0 | 1.000 | OK |
| quantum_volume_8_s0 | 8 | 36 | 9 | 9 | 1.000 | OK |
| quantum_volume_8_s1 | 8 | 36 | 5 | 5 | 1.000 | OK |
| quantum_volume_8_s2 | 8 | 36 | 7 | 7 | 1.000 | OK |
| quantum_volume_8_s3 | 8 | 36 | 6 | 6 | 1.000 | OK |
| quantum_volume_8_s4 | 8 | 36 | 6 | 6 | 1.000 | OK |
| quantum_volume_12_s0 | 12 | 54 | 8 | 8 | 1.000 | OK |
| quantum_volume_12_s1 | 12 | 54 | 13 | 12 | 1.083 | OK |
| quantum_volume_12_s2 | 12 | 54 | 9 | 9 | 1.000 | OK |
| quantum_volume_12_s3 | 12 | 54 | 12 | 11 | 1.091 | OK |
| quantum_volume_12_s4 | 12 | 54 | 7 | 7 | 1.000 | OK |
| quantum_volume_19_s0 | 19 | 81 | 20 | 18 | 1.111 | OK |
| quantum_volume_19_s1 | 19 | 81 | 18 | 17 | 1.059 | OK |
| quantum_volume_19_s2 | 19 | 81 | 15 | 14 | 1.071 | OK |
| quantum_volume_19_s3 | 19 | 81 | 17 | 17 | 1.000 | OK |
| quantum_volume_19_s4 | 19 | 81 | 22 | 21 | 1.048 | OK |

**Quantum Volume summary**: 19/20 completed, mean ratio 1.024, median 1.000


## Random

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| random_d20_s0 | 19 | 135 | 171 | 188 | 0.910 | OK |
| random_d20_s1 | 19 | 119 | 173 | 178 | 0.972 | OK |
| random_d20_s2 | 19 | 140 | 171 | 181 | 0.945 | OK |
| random_d20_s3 | 19 | 124 | 196 | 191 | 1.026 | OK |
| random_d20_s4 | 19 | 137 | 192 | 212 | 0.906 | OK |
| random_d20_s5 | 19 | 131 | 184 | 184 | 1.000 | OK |
| random_d20_s6 | 19 | 121 | 156 | 174 | 0.897 | OK |
| random_d20_s7 | 19 | 125 | 184 | 183 | 1.005 | OK |
| random_d20_s8 | 19 | 136 | 208 | 193 | 1.078 | OK |
| random_d20_s9 | 19 | 131 | 177 | 193 | 0.917 | OK |
| random_d20_s10 | 19 | 121 | 156 | 163 | 0.957 | OK |
| random_d20_s11 | 19 | 124 | 180 | 178 | 1.011 | OK |
| random_d20_s12 | 19 | 127 | 185 | 194 | 0.954 | OK |
| random_d20_s13 | 19 | 129 | 181 | 186 | 0.973 | OK |
| random_d20_s14 | 19 | 132 | 161 | 175 | 0.920 | OK |
| random_d20_s15 | 19 | 120 | 178 | 176 | 1.011 | OK |
| random_d20_s16 | 19 | 128 | 165 | 182 | 0.907 | OK |
| random_d20_s17 | 19 | 124 | 173 | 178 | 0.972 | OK |
| random_d20_s18 | 19 | 130 | 177 | 188 | 0.941 | OK |
| random_d20_s19 | 19 | 131 | 191 | 201 | 0.950 | OK |
| random_d20_s20 | 19 | 120 | 166 | 163 | 1.018 | OK |
| random_d20_s21 | 19 | 128 | 189 | 182 | 1.038 | OK |
| random_d20_s22 | 19 | 131 | 154 | 178 | 0.865 | OK |
| random_d20_s23 | 19 | 128 | 212 | 187 | 1.134 | OK |
| random_d20_s24 | 19 | 136 | 195 | 197 | 0.990 | OK |
| random_d20_s25 | 19 | 130 | 177 | 182 | 0.973 | OK |
| random_d20_s26 | 19 | 139 | 206 | 212 | 0.972 | OK |
| random_d20_s27 | 19 | 128 | 162 | 160 | 1.012 | OK |
| random_d20_s28 | 19 | 135 | 215 | 196 | 1.097 | OK |
| random_d20_s29 | 19 | 132 | 173 | 194 | 0.892 | OK |

**Random summary**: 30/30 completed, mean ratio 0.975, median 0.972


## Structured

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| cnot_ring_19_3rep | 19 | 57 | 26 | 24 | 1.083 | OK |
| cnot_ring_19_5rep | 19 | 95 | 29 | 27 | 1.074 | OK |
| cnot_ring_19_10rep | 19 | 190 | 57 | 57 | 1.000 | OK |

**Structured summary**: 3/3 completed, mean ratio 1.052, median 1.074


## VQE

| Circuit | Qubits | 2Q Gates | Agent SWAPs | SABRE SWAPs | Ratio | Status |
|---------|--------|----------|-------------|-------------|-------|--------|
| vqe_linear_4q_1r | 4 | 3 | 0 | 0 | 1.000 | OK |
| vqe_linear_4q_3r | 4 | 9 | 0 | 0 | 1.000 | OK |
| vqe_linear_4q_5r | 4 | 15 | 0 | 0 | 1.000 | OK |
| vqe_linear_8q_1r | 8 | 7 | 0 | 0 | 1.000 | OK |
| vqe_linear_8q_3r | 8 | 21 | 0 | 0 | 1.000 | OK |
| vqe_linear_8q_5r | 8 | 35 | 0 | 0 | 1.000 | OK |
| vqe_linear_12q_1r | 12 | 11 | 0 | 0 | 1.000 | OK |
| vqe_linear_12q_3r | 12 | 33 | 0 | 0 | 1.000 | OK |
| vqe_linear_12q_5r | 12 | 55 | 0 | 0 | 1.000 | OK |
| vqe_linear_16q_1r | 16 | 15 | 0 | 0 | 1.000 | OK |
| vqe_linear_16q_3r | 16 | 45 | 0 | 0 | 1.000 | OK |
| vqe_linear_16q_5r | 16 | 75 | 0 | 0 | 1.000 | OK |
| vqe_linear_19q_1r | 19 | 18 | 5 | 5 | 1.000 | OK |
| vqe_linear_19q_3r | 19 | 54 | 9 | 9 | 1.000 | OK |
| vqe_linear_19q_5r | 19 | 90 | 22 | 18 | 1.222 | OK |
| vqe_circular_4q_1r | 4 | 4 | 2 | 2 | 1.000 | OK |
| vqe_circular_4q_3r | 4 | 12 | 5 | 6 | 0.833 | OK |
| vqe_circular_4q_5r | 4 | 20 | 10 | 11 | 0.909 | OK |
| vqe_circular_8q_1r | 8 | 8 | 2 | 2 | 1.000 | OK |
| vqe_circular_8q_3r | 8 | 24 | 6 | 6 | 1.000 | OK |
| vqe_circular_8q_5r | 8 | 40 | 10 | 10 | 1.000 | OK |
| vqe_circular_12q_1r | 12 | 12 | 6 | 6 | 1.000 | OK |
| vqe_circular_12q_3r | 12 | 36 | 9 | 9 | 1.000 | OK |
| vqe_circular_12q_5r | 12 | 60 | 18 | 17 | 1.059 | OK |
| vqe_circular_16q_1r | 16 | 16 | 0 | 0 | 1.000 | OK |
| vqe_circular_16q_3r | 16 | 48 | 0 | 0 | 1.000 | OK |
| vqe_circular_16q_5r | 16 | 80 | 0 | 0 | 1.000 | OK |
| vqe_circular_19q_1r | 19 | 19 | 13 | 13 | 1.000 | OK |
| vqe_circular_19q_3r | 19 | 57 | 34 | 34 | 1.000 | OK |
| vqe_circular_19q_5r | 19 | 95 | 27 | 27 | 1.000 | OK |
| vqe_full_4q_1r | 4 | 6 | 2 | 2 | 1.000 | OK |
| vqe_full_4q_2r | 4 | 12 | 9 | 8 | 1.125 | OK |
| vqe_full_4q_3r | 4 | 18 | 7 | 6 | 1.167 | OK |
| vqe_full_8q_1r | 8 | 28 | 22 | 21 | 1.048 | OK |
| vqe_full_8q_2r | 8 | 56 | 48 | 46 | 1.043 | OK |
| vqe_full_8q_3r | 8 | 84 | 80 | 83 | 0.964 | OK |
| vqe_full_12q_1r | 12 | 66 | 68 | 65 | 1.046 | OK |
| vqe_full_12q_2r | 12 | 132 | 167 | 131 | 1.275 | OK |
| vqe_full_12q_3r | 12 | 198 | 224 | 222 | 1.009 | OK |
| vqe_full_19q_1r | 19 | 171 | 180 | 184 | 0.978 | OK |
| vqe_full_19q_2r | 19 | 342 | 363 | 400 | 0.907 | OK |
| vqe_full_19q_3r | 19 | 513 | 572 | 600 | 0.953 | OK |

**VQE summary**: 27/42 completed, mean ratio 1.020, median 1.000

