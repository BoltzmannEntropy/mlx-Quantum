<div align="left">
  
  <img src="assets/mlx_logo.svg" alt="mlx-Quantum Logo"/>

<img src="assets/apple_mlx.jpg" width="20%">  

**High-Performance Quantum Computing Framework for Apple Silicon - WORK IN PROGRESS -** 

*MLX-Based Modular Architecture for Energy-Efficient Quantum Circuit Simulation*

  [![macOS](https://img.shields.io/badge/macOS-13.3+-blue.svg)](https://www.apple.com/macos/)
  [![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://developer.apple.com/silicon/)
  [![MLX](https://img.shields.io/badge/MLX-v0.26.3-blue.svg)](https://github.com/ml-explore/mlx)
  [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![ICLR 2025](https://img.shields.io/badge/ICLR-2025-purple.svg)](paper/iclr2025_conference.pdf)
</div>

**mlx-Quantum** is the first quantum computing simulation framework designed specifically for Apple Silicon using Apple's Metal Shaders and MLX framework. Unlike traditional quantum simulators requiring manual Metal shader programming, mlx-Quantum provides pure C++ interfaces with automatic GPU acceleration, making quantum computing accessible without specialized GPU programming knowledge.
Surprisingly, no framework had previously been created to run quantum-classical simulations natively on Apple Silicon. MLX-Quantum fills this gap as the first such framework, harnessing Apple's MLX and Shaders for optimal performance.


Our framework leverages Apple Silicon's unified memory architecture to deliver quantum simulations competitive with NVIDIA's cuQuantum while offering better energy efficiency and demonstrates that consumer Apple hardware can achieve research-quality quantum simulation performance.

## 🚀 Key Features

- 🚀 **MLX-Powered**: Automatic GPU acceleration without manual shader programming
- 🧩 **Modular Architecture**: Reusable algorithm classes (mlxQFT, mlxGrover, mlxVQE, etc.)
- 📚 **Educational Focus**: Three-part examples (Theory → Paper → Code) for learning
- 🍎 **Apple Silicon Optimized**: Unified memory architecture with M1/M2/M3/M4 support
- 📊 **Comprehensive Benchmarks**: cuQuantum comparison suite with performance analysis
- **Performance Benchmarking**: Direct comparison framework with cuQuantum metrics
- **Educational Focus**: Accessible interface for quantum algorithm research and education


## 📌 Current Status

- Core subsystems (`src/core`) implement states, operations, information theory, simulator, and device layers backed by MLX tensors.
- The consolidated regression suite in `src/test/mlxQuantumCoreTest.cpp` now covers 49 scenarios, including QuTiP-derived quantum information checks, circuit algorithms, and appendix examples.
- Build configurations are maintained for Xcode (`bin64/`) and Ninja (`build_ninja/`); the latter is used for continuous validation during development.
- Benchmark harnesses are scaffolded under `bench/`, with data generation pending verification before publication.


<h1 align="center">    
  <img src="https://github.com/BoltzmannEntropy/polarization/blob/main/HXH.png" width="50%"></a>  
</h1>
  
## 📊 Performance Benchmarks 

Benchmark numbers are being re-validated. The tables below are intentionally left blank until we have reproducible measurements.

| Metric | mlxQuantum | Reference | Notes |
|--------|------------|-----------|-------|
| Gate Throughput | TBD | TBD | Measurements pending |
| Memory Footprint | TBD | TBD | Measurements pending |
| Energy per Circuit | TBD | TBD | Measurements pending |
| Maximum Simulated Qubits | TBD | TBD | Measurements pending |

### cuQuantum vs. mlxQuantum Benchmark Matrix (to be populated)

| Algorithm | Platform | Hardware | Qubits | Time (ms) | Power (W) | Efficiency Notes |
|-----------|----------|----------|--------|-----------|-----------|-------------------|
| QFT | TBD | TBD | TBD | TBD | TBD | Pending profiling |
| QAOA | TBD | TBD | TBD | TBD | TBD | Pending profiling |
| Quantum Volume | TBD | TBD | TBD | TBD | TBD | Pending profiling |
| Time Evolution | TBD | TBD | TBD | TBD | TBD | Pending profiling |

## 📋 Requirements

- **Operating System**: macOS 15.5 or later
- **Hardware**: Apple Silicon (M1/M2/M3/M4 series)
- **Memory**: 8GB+ RAM (16GB+ recommended for >20 qubits)
- **Development Tools**: Xcode Command Line Tools, CMake 3.20+
- **C++ Standard**: C++20 or later

## 🔬 Technical Background

### GPU Computing Frameworks Comparison

**CUDA Dominance in Quantum Computing**: NVIDIA's cuQuantum SDK remains the gold standard for quantum computing acceleration, with academic papers reporting up to 900x speedup on quantum machine learning workloads and the ability to simulate hundreds of qubits on a single A100 GPU. The cuQuantum SDK provides state-of-the-art libraries optimized for GPU-accelerated quantum circuit simulations [[Bayraktar et al., 2023]](https://arxiv.org/abs/2308.01999).

**Performance Limitations of Alternative Frameworks**: Vulkan compute shaders show approximately 30x slower performance than CUDA on equivalent NVIDIA hardware for compute-intensive tasks, while offering better cross-platform compatibility [[NVIDIA Developer Forums, 2024]](https://forums.developer.nvidia.com/t/vulkan-compute-shaders-vs-cuda/194944).

**Metal's Unique Position**: While Metal provides lower raw performance than CUDA, its integration with Apple Silicon's unified memory architecture and energy efficiency make it compelling for quantum computing research and education, particularly given Apple's growing presence in high-performance computing.

### Quantum-Classical Hybrid Computing

### Framework Comparison (data pending)

| Framework | Platform | Relative Performance | Energy Profile | Memory Model | Developer Experience |
|-----------|----------|----------------------|----------------|--------------|---------------------|
| CUDA (cuQuantum) | TBD | TBD | TBD | TBD | TBD |
| Vulkan Compute | TBD | TBD | TBD | TBD | TBD |
| Metal (mlxQuantum) | TBD | TBD | TBD | TBD | TBD |

## 🏗️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlxQuantum.git
cd mlxQuantum

# Build the project
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)

# Run the main application
./mlxquantum

# Run with interactive UI
./mlxquantum --interactive

# Run test suites
./QuantumTests
./MetalTests
```

### Basic Usage

```cpp
#include "mlxquantum.h"
using namespace mlxQuantum;

// Create quantum simulator with Metal acceleration
mlxQuantumSimulator simulator;

// Create and initialize quantum state utilizing unified memory
QuantumState state(2);  // 2-qubit system
state.initializeZero(); // Start in |00⟩

// Create Bell state: (|00⟩ + |11⟩)/√2
simulator.applyHadamard(0, state);  // Executed on Metal compute shader
simulator.applyCNOT(0, 1, state);   // Two-qubit gate with optimized memory access

// Measure probabilities (CPU-GPU coordination via unified memory)
auto probabilities = state.getAllProbabilities();
std::cout << "|00⟩: " << probabilities[0] << std::endl; // 0.5
std::cout << "|11⟩: " << probabilities[3] << std::endl; // 0.5
```

### Advanced: Unified Memory Optimization

```cpp
// Demonstrate unified memory advantages
QuantumState large_state(20);  // 1M amplitude quantum state
large_state.initializeRandom();

// No CPU-GPU memory copying required thanks to unified memory
simulator.executeCircuit(complex_circuit, large_state);

// Direct CPU access to GPU-computed results
float expectation = simulator.computeExpectationValue(large_state, hamiltonian);
```

## 🏛️ Architecture

### Project Structure
```
mlxQuantum/
├── main.cpp              # Consolidated application with all implementations
├── mlxquantum.h          # Single unified header file
├── tests/
│   ├── quantum_tests.cpp  # Quantum algorithm test suite
│   └── metal_tests.cpp    # Metal GPU functionality tests
├── Shaders/
│   └── QuantumGates.metal # Metal compute shaders for quantum operations
├── paper/                 # LaTeX research paper
├── bkup/                  # Backup of legacy files
└── README.md
```

### Core Components


### Metal Shader Architecture

The quantum operations are implemented using Metal compute shaders for optimal performance:

```metal
kernel void apply_hadamard(device float2* state_buffer [[buffer(0)]],
                          constant uint& qubit_index [[buffer(1)]],
                          constant uint& num_qubits [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    // Hadamard gate implementation optimized for Apple Silicon
    // Leverages unified memory and parallel processing capabilities
}
```

## 🧪 Testing & Validation

### Running Tests

```bash
```

### Test Coverage


## 📖 Research & Documentation

### Academic Citations and References

- Bayraktar, H., et al. (2023). "cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science." arXiv:2308.01999
- Zhang, Y., et al. (2024). "Quantum-HPC Framework with multi-GPU-Enabled Hybrid Quantum-Classical Workflow." arXiv:2403.05828
- NVIDIA Developer Documentation (2024). "CUDA-Q Platform Performance Benchmarks"
- Apple Inc. (2024). "Apple Silicon Unified Memory Architecture Technical Overview"




## 📄 License

```
MIT License

Copyright (c) 2023 Shlomo Kashani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Contact & Support

- **Author**: Shlomo Kashani

## 🌟 Acknowledgments

mlxQuantum builds upon the theoretical foundations established by:
- **NVIDIA's cuQuantum Team**: For quantum simulation benchmarking standards and GPU acceleration techniques [[Bayraktar et al., 2023]](https://arxiv.org/abs/2308.01999)
- **Quantum++ (QPP) Library**: For C++ quantum computing design patterns and numerical algorithms [[Software Impacts, 2018]](https://doi.org/10.1016/j.simpa.2018.07.002)
- **Apple Metal Performance Shaders Team**: For GPU acceleration frameworks and unified memory optimization techniques
- **Academic Quantum Computing Community**: Including IBM Qiskit, Google Cirq, and PennyLane teams for algorithm implementations and validation
- **High-Performance Computing Research**: Leveraging advances in GPU computing, parallel algorithms, and energy-efficient computing architectures

### Recent Developments in Quantum-GPU Computing

The field has seen significant advances in 2024-2025:
- **NVIDIA CUDA-Q Platform**: Demonstrated up to 900x speedups on quantum machine learning workloads
- **Multi-GPU Quantum Simulation**: Scaling to thousands of qubits using supercomputing clusters
- **Apple Silicon AI Efficiency**: Breakthrough performance in LLM inference suggesting quantum computing potential
- **Hybrid Quantum-Classical Algorithms**: Growing importance of integrated classical-quantum workflows

*"The future of quantum computing lies not just in raw computational power, but in the intelligent integration of specialized hardware, unified memory architectures, and energy-efficient design—exactly what Apple Silicon brings to quantum research."*

**Version**: 1.0.0 | **License**: MIT | **Platform**: macOS 15.5+ with Apple Silicon
