# 把一个 vLLM 内核搬进 GPTQModel：AllSpark W8A16 移植记

> 对应 PR：[#3490](https://github.com/ModelCloud/GPTQModel/pull/3490) W8A16 内核移植、[#3515](https://github.com/ModelCloud/GPTQModel/pull/3515) CUDA Graph 回归测试
> 涉及硬件：RTX 5090 Laptop（sm_120）、RTX 4090（sm_89）、Radeon R9700（gfx1201）

---

## 一、动机：解码阶段不是算力瓶颈

大模型推理分两个阶段：

- **prefill**：一次处理整段 prompt，矩阵乘的 M 很大（几百到几千），算力是瓶颈；
- **decode**：每步只生成一个 token，M=1，**每次都要把全部权重从显存读一遍**。

在 decode 阶段，计算量只有 2×参数量 FLOPs，而数据量是参数量本身。以 70 亿参数的 4bit 模型为例：权重约 3.5 GB，单步要读 3.5 GB，而算力只用了不到 1% 的峰值。

**所以 decode 的性能上限不是 FLOPS，是显存带宽。**

GPTQModel 当时的主力内核是 Marlin（w4a16）。它在 4bit 上很强，但 8bit 场景（W8A16）没有专门的解码内核——8bit 用户只能走通用路径。

vLLM 在 2025 年合并了 AllSpark：一个专为 W8A16 小 batch 解码设计的 GEMM 内核。维护者在这个仓库的 issue #1359 里提到，希望把 vLLM 的两个内核移植过来。AllSpark 是其中之一。

---

## 二、技术原理

### 2.1 W8A16 是什么

- 权重：uint8（8bit 整数），每列（输出通道）一个 fp16 scale；
- 激活：fp16 / bf16；
- 计算：`Y = (X @ (W_uint8 - zero) * scale)`，累加用 fp32。

和 4bit 相比，8bit 的权重精度更高、量化损失更小，代价是显存翻倍。适合显存充裕但要求精度的场景（比如 8bit 的 GPTQ checkpoint）。

### 2.2 为什么内核要专门写

朴素实现是"反量化成 fp16 再 GEMM"。问题是：

1. 反量化后权重变成 2 字节，显存读取量翻倍；
2. 反量化本身要额外一轮显存往返。

AllSpark 的做法是**在寄存器里反量化**：直接从 uint8 读，边读边转成 fp16，喂给 Tensor Core 的 `mma.sync.m16n8k16` 指令。这样显存里始终只有 1 字节/权重。

### 2.3 关键技术点

**（一）指令选择：`mma.sync.m16n8k16`**

Ampere 及以后的架构支持这条指令：16×8 的输出块，K 方向 16 个元素，输入 fp16，累加 fp32。

```cuda
// 一个 warp 处理 16x8 的输出块
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3};
```

**没有用 wgmma（Hopper 的异步指令）**——因为 wgmma 只在 sm_90 上可用，而消费级显卡（sm_86/89/120）不支持。用 `mma.sync` 可以覆盖所有 Ampere 之后的卡。

**（二）数据布局：N32K16**

权重在显存里按 32（N）× 16（K）的小块重排，让每个 warp 连续读取自己需要的部分，避免跨步访问。

重排发生在加载时（`allspark_repack.cu`），只做一次。

**（三）异步拷贝：`cp.async`**

```cuda
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;
```

把全局显存直接拷到共享内存，不经过寄存器，也不阻塞计算。配合双缓冲，可以实现"算第 i 块的同时读第 i+1 块"。

**（四）cuBLAS 回退**

当 M > 1024 时，AllSpark 的优势消失（变成算力瓶颈），直接回退到 cuBLAS：

```cuda
// allspark_qgemm_w8a16.cu
if (M > ALLSPARK_M_CUBLAS_THRESHOLD) {  // 1024
    return cublas_fallback(...);
}
```

**（五）限制：group_size 必须为 -1**

AllSpark 只支持 per-channel 量化（group_size = -1），因为它的 scale 读取是"每个输出列一个"：

```cuda
// allspark_qgemm_w8a16.cu:332
float scale = scales[col];
```

内核里有硬检查：

```cuda
TORCH_CHECK(group_size == -1, "AllSpark only supports group_size=-1");
```

---

## 三、代码讲解：内核怎么接进框架

移植一个内核不只是拷 `.cu` 文件，要打通四层：

```
gptqmodel_ext/allspark/        CUDA 源码 + pybind 绑定
        ↓
gptqmodel/utils/allspark.py    JIT 编译与符号加载
        ↓
gptqmodel/nn_modules/qlinear/allspark.py   模块封装（解包、重排、前向）
        ↓
gptqmodel/utils/backend.py + importer.py   后端注册与自动选择
```

### 3.1 第一层：C++ 绑定

```cpp
// allspark_torch.cpp
TORCH_LIBRARY(gptqmodel_allspark, m) {
    m.def("allspark_qgemm_w8a16(Tensor a, Tensor qweight, Tensor scales, ...) -> Tensor");
}
```

用 `TORCH_LIBRARY` 而不是 pybind11，因为它跟 PyTorch 的算子注册机制一致，能享受 dispatcher 和 CUDA Graph 支持。

### 3.2 第二层：JIT 编译

```python
# gptqmodel/utils/allspark.py
class AllSparkOps(TorchOpsJitExtension):
    def __init__(self):
        super().__init__(
            name="allspark",
            sources=[...],
            extra_cuda_cflags=[
                "-U__CUDA_NO_HALF_CONVERSIONS__",  # 关键：允许 fp16 隐式转换
                ...
            ],
            extra_ldflags=["-lcublas"],
        )
```

`-U__CUDA_NO_HALF_CONVERSIONS__` 这个宏必须去掉，否则 `__half` 和 `float` 之间不能隐式转换，内核里大量 `__half2float` 会编译失败。

**ROCm 上直接返回 False**：

```python
def allspark_supported() -> bool:
    if torch.version.hip is not None:
        return False
    return True
```

因为 AllSpark 用的是 `mma.sync`，HIP 的对应实现不同（维护者后来补了 `allspark_hip.py`）。

### 3.3 第三层：模块封装

AllSpark 的权重布局和 GPTQ 的 checkpoint 不一样，加载时要重排：

```python
# gptqmodel/nn_modules/qlinear/allspark.py
class AllSparkLinear(GPTQQuantLinear):
    # 关键：0 表示"注册但不参与 AUTO 选择"
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
```

**`SUPPORTS_FORMATS` 的值是优先级，不是布尔值。**

`importer.py` 里的逻辑：

```python
if priority > 0:
    # 参与 AUTO 选择，数字越大越优先
```

我一开始写了 95，意味着在所有 8bit 场景里 AllSpark 会抢在 Marlin（90）前面被选中——**这会影响所有用户，而不是只影响想用 AllSpark 的人**。后来改成 0。

权重解包：checkpoint 里是 int32 打包的 uint8（每 4 个字节一个 int32），要还原成 uint8：

```python
def post_init(self):
    # int32 → uint8
    unpacked = (self.qweight >> shift) & 0xFF
```

### 3.4 第四层：后端注册

```python
# gptqmodel/utils/backend.py
ALLSPARK = "allspark"
```

```python
# gptqmodel/utils/importer.py
BACKEND_MAP[BACKEND.ALLSPARK] = AllSparkLinear
```

---

## 四、验证

### 4.1 数值正确性

| 对比项 | 最大误差 |
|---|---|
| 内核 vs CPU fp32 参考 | 3.9e-4 |
| cuBLAS vs 同一参考 | 1.8e-4 |
| fp16 输出舍入（半 ULP） | 4.9e-4 |

**内核误差在 fp16 舍入范围内**，说明它算的是同一件事。

逐层对比（单层输出 vs torch 参考）：

```
marlin   1.67e-3
allspark 2.16e-3
```

### 4.2 端到端困惑度

3577 个 token、512 窗口：

```
base bf16      13.2209
torch 参考     13.2515   (+0.23%)
allspark       13.2710   (+0.38%)
```

量化本身损失 0.23%，换成 AllSpark 内核额外损失 0.15%。

### 4.3 性能：这里踩了一个大坑

**第一版结论**："980–1023 GB/s，贴着 1 TB/s 的 HBM 上限。"

**错了。**

5090 Laptop 的 L2 是 64 MiB，测试用的权重矩阵 43 MiB——**装得下**。我测的是 L2 带宽，不是 HBM。

加了 `--flush-l2`（每次调用前用 512 MiB 随机缓冲区刷 L2）之后：

| 场景 | 加速比 |
|---|---|
| hot（不刷 L2） | 1.5–1.76× |
| **flushed（刷 L2）** | **1.07–1.18×** |

差了一倍。维护者在 4090 上复现后也确认了这一点。

**为什么缓冲区要随机而不是全零**：GPU 有内存压缩，全零缓冲区可能被压缩而不真正驱逐权重。

### 4.4 CUDA Graph 测试（#3515）

decode 每步只有几十微秒，CUDA 的启动开销（约 8 µs）占比很高。用 CUDA Graph 捕获整段计算再重放，可以去掉这部分。

我加了测试，但**第一版又错了一次**：我写"这个测试能防止有人把显存分配挪进 capture 区域"。

实测结果：

```
capture 内 torch::empty   → OK（allocator 的 capture 池会兜住）
capture 内 .cpu()         → FAIL
capture 内 .item()        → FAIL
```

**分配不会失败，回主机才会。** 所以测试真正保护的是 host 同步和 M 相关的分派。

维护者收紧了测试：

| 我的版本 | 他的版本 |
|---|---|
| fp16 参考 | FP64 参考 |
| rtol=5e-2 | **rtol=atol=2e-3** |
| 无负向控制 | zero / stale 两种故障注入 |

我把容差改回 0.05 重跑：**4 个负向控制全部失效**——输出全零在 0.05 容差下也能通过。

---

## 五、复盘

**一、带宽数字必须写清楚测量条件。** "1.5 倍"和"1.1 倍"的差别不在内核，在于矩阵装不装得进 L2。

**二、优先级字段有全局影响。** `SUPPORTS_FORMATS` 写 95 会抢走所有用户的 AUTO 选择，不是"让我的内核更容易被选中"这么简单。

**三、测试要能证明自己会失败。** 没有负向控制的测试，容差放松到一定程度就只是在测"函数有没有崩"。

**四、签名要在第一个 commit 之前配好。** 前两个 commit 的 author 是 `fanxi@local`，没关联上 GitHub 账号，不计入贡献图。
