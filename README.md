# Matrix Performance Lab：CPU 矩阵乘法性能分析

## 项目简介

本项目是一个 CPU 矩阵乘法性能测试实验，主要比较 Python 原生循环、NumPy 和 PyTorch 三种实现方式在不同矩阵规模下的运行表现，并进一步研究 PyTorch 线程数对计算性能的影响。

项目关注三个问题：

1. 为什么相同的矩阵乘法运算，不同实现方式的执行时间会有明显差异？
2. 随着矩阵规模增大，PyTorch 使用更多 CPU 线程是否总能获得更好的性能？
3. 在相同软硬件环境下，多次独立运行 Benchmark，测量结果是否稳定？

这是一个面向计算机系统与机器学习系统的入门实验，目的在于练习性能测量、数据分析和实验复现，而不是对 NumPy、PyTorch 的整体性能作出普遍排名。

## 实验设计

实验使用随机生成的 `float32` 方阵进行矩阵乘法：

$$
C=AB,\qquad A,B\in\mathbb{R}^{N\times N}
$$

测试矩阵规模：

`N = 16, 32, 128, 256, 512`

比较的实现方式如下：

| 实现方式        | 测试配置                  |
| ----------- | --------------------- |
| Python 原生循环 | 三重循环实现，仅测试 N=16、32    |
| NumPy       | 使用 `@` 运算符            |
| PyTorch     | CPU 计算，分别使用 1、2、4 个线程 |

输入矩阵使用固定随机种子生成。执行性能测试前，程序会进行数值正确性检查。

使用 `torch.utils.benchmark.Timer` 测量单次矩阵乘法的估计执行延迟，并根据运算量估算有效计算吞吐量：

$$
\text{GFLOP/s}=\frac{2N^3}{t\times10^9}
$$

其中，\(t\) 是以秒为单位的执行延迟。

完整 Benchmark 独立运行三次。对于相同的测试配置，计算三次运行的平均延迟、样本标准差和变异系数（CV），用于分析测量结果的稳定性。

需要说明的是，Python 原生循环使用 Python 浮点数运算及 `float64` 累加，而 NumPy、PyTorch 主要使用 `float32`，因此三者并不是完全相同的底层数值执行路径。

## 实验环境

| 项目            | 配置                         |
| ------------- | -------------------------- |
| CPU           | Intel Core Ultra 9 285H    |
| CPU 核心        | 16 核、16 逻辑处理器              |
| 操作系统          | Windows 11 家庭版，build 26200 |
| Python        | 3.11.17                    |
| NumPy         | 2.4.6                      |
| PyTorch       | 2.14.1+cpu                 |
| NumPy BLAS 后端 | OpenBLAS 0.3.31.188.0      |
| PyTorch 测试线程数 | 1、2、4                      |
| 计算设备          | CPU                        |

通过 `threadpoolctl` 独立检查发现，NumPy 使用的 OpenBLAS 线程池报告为 16 线程。但实验中没有显式固定 NumPy 的 BLAS 线程数，因此不能将 NumPy 结果理解为严格的单线程基准，也不能直接视为与 PyTorch 相同线程数下的公平比较。

## 项目结构

```text
Matrix_Performance_Lab/
├── matmul.py
├── test_matmul.py
├── benchmark.py
├── analyze_runs.py
├── plot_results.py
├── plot_summary.py
├── report.md
├── README.md
└── results/
    ├── benchmark_run1.csv
    ├── benchmark_run2.csv
    ├── benchmark_run3.csv
    ├── benchmark_summary.csv
    ├── latency_summary.png
    └── throughput_summary.png
```

主要文件说明：

| 文件                | 作用                     |
| ----------------- | ---------------------- |
| `matmul.py`       | 使用 Python 三重循环实现矩阵乘法   |
| `test_matmul.py`  | 测试矩阵乘法实现的数值正确性         |
| `benchmark.py`    | 执行性能测试并记录延迟、吞吐量        |
| `analyze_runs.py` | 汇总三次独立实验，计算平均值、标准差及 CV |
| `plot_results.py` | 绘制单次实验结果               |
| `plot_summary.py` | 绘制三次实验的汇总图             |
| `report.md`       | 详细记录实验方法、结果、讨论及局限性     |

以上是建议的项目结构，实际提交时应与本地文件保持一致。

## 环境安装

项目使用 Python 3.11 和 Miniconda。

创建并激活 Conda 环境：

```powershell
conda create -n matrix-lab python=3.11
conda activate matrix-lab
```

安装依赖：

```powershell
python -m pip install numpy torch matplotlib pytest threadpoolctl
```

如需尽量复现本项目的测试环境，应参考前文列出的具体软件版本。不同 CPU、操作系统或数值计算库配置下，性能结果可能存在明显差异。

## 运行方法

### 1. 检查矩阵乘法实现

在项目根目录运行：

```powershell
python -m pytest test_matmul.py -v
```

程序将检查基础矩阵乘法、不同矩阵形状以及异常输入等情况。

### 2. 执行性能测试

```powershell
python benchmark.py
```

程序会测试不同实现方式、矩阵规模和线程配置，并将结果保存到：

```text
results/benchmark.csv
```

**注意：每次运行都会覆盖 `benchmark.csv`。**

如果需要保存三次独立实验，可以在 PowerShell 中依次执行：

```powershell
python benchmark.py
Copy-Item results\benchmark.csv results\benchmark_run1.csv

python benchmark.py
Copy-Item results\benchmark.csv results\benchmark_run2.csv

python benchmark.py
Copy-Item results\benchmark.csv results\benchmark_run3.csv
```

项目已经保存了三次实验的原始结果，因此只查看或重新绘制现有数据时，不需要再次运行 Benchmark。

### 3. 汇总三次实验

```powershell
python analyze_runs.py
```

程序读取三份原始 CSV，按照实现方式、矩阵规模和线程数进行分组，计算：

* 平均延迟（Mean Latency）
* 样本标准差（Standard Deviation）
* 变异系数（CV）
* 根据平均延迟估算的有效吞吐量（GFLOP/s）

输出文件：

`results/benchmark_summary.csv`

### 4. 绘制汇总图

```powershell
python plot_summary.py
```

生成：

* `results/latency_summary.png`
* `results/throughput_summary.png`

其中，延迟图使用三次实验的平均值，并通过误差棒表示 ±1 个样本标准差；吞吐量图使用根据平均延迟计算的估算吞吐量。

## 实验结果

### 平均执行延迟

![CPU 矩阵乘法平均延迟](results/latency_summary.png)

*图 1：三次独立实验的平均延迟。误差棒表示 ±1 个样本标准差，横纵坐标均采用对数刻度。*

### 有效计算吞吐量

![CPU 矩阵乘法有效吞吐量](results/throughput_summary.png)

*图 2：根据三次实验的平均延迟计算的估算吞吐量，横坐标采用对数刻度。*

实验观察到：

**1. Python 原生循环与优化数值库存在明显性能差距。**

在 N=32 时，Python 三重循环的平均执行延迟约为 8.241 ms，而 NumPy 约为 0.001502 ms。这说明使用优化数值计算库可以显著降低该任务的执行时间，但实验不能单独量化 Python 解释器开销的贡献。

**2. 较大矩阵更容易从 PyTorch 多线程中受益。**

在 N=256 时，PyTorch 四线程相对于单线程的平均延迟加速比约为 3.16 倍；在 N=512 时约为 3.23 倍。

**3. 不同独立运行之间存在明显波动。**

部分配置的变异系数超过 40%。例如，N=256 时 NumPy 的 CV 为 48.50%，N=512 时 PyTorch 四线程的 CV 为 42.03%。

这说明单次性能测试结果未必能代表稳定的执行性能，需要通过重复实验和更严格的环境控制进一步验证。

更详细的实验数据、分析及讨论见 [完整实验报告](report.md)。

## 实验局限性

本项目仍存在以下局限：

* NumPy 的 OpenBLAS 线程数没有在测试过程中被显式固定。
* 仅进行了三次独立完整实验，部分配置的波动仍然较大。
* Python 原生循环与优化数值库使用的数值执行路径不同。
* 仅测试方阵乘法，尚未涉及非方阵、不同内存布局和 GPU。
* 未系统记录 CPU 频率、温度、后台负载和线程实际运行的核心位置。

因此，本项目结果应被理解为特定实验环境下的初步观察，而不是对不同数值库的普遍性能排名。

## 后续工作

下一阶段计划从实验可复现性入手，进一步控制 NumPy 的 BLAS 线程数，增加独立运行次数，并研究不同系统状态下的性能波动。

在此基础上，可以逐步扩展到矩阵内存布局、缓存行为、CPU 并行调度及 GPU 算子性能分析等问题。
