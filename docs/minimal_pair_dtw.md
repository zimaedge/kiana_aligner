# Minimal Pair DTW Method

## 概述 (Overview)

`get_pair_via_dtw_minimal` 是 KIANA 工具包中新增的高效 DTW 配对方法。当处理较长的时间序列时，该方法通过分块处理显著提升效率。

`get_pair_via_dtw_minimal` is a new efficient DTW pairing method in the KIANA toolkit. It significantly improves efficiency when processing long time series through chunk-based processing.

## 核心特性 (Key Features)

1. **分块处理 (Chunk-based Processing)**: 每次只处理序列的一小部分，降低计算复杂度
2. **自动验证 (Automatic Validation)**: 每个块配对后自动检查间隔一致性
3. **自适应重试 (Adaptive Retry)**: 当验证失败时，自动调整块大小并重新配对
4. **增量处理 (Incremental Processing)**: 逐步处理完整个序列，确保全局对齐

## 使用示例 (Usage Examples)

### 基础用法 (Basic Usage)

```python
from kiana import get_pair_via_dtw_minimal
import numpy as np

# 准备两个长时间序列
template = np.cumsum(np.random.uniform(0.9, 1.1, 1000))
query = template + np.random.normal(0, 0.1, 1000)

# 使用分块 DTW 配对
pairs = get_pair_via_dtw_minimal(
    template, 
    query, 
    chunk_size=50,      # 每个块的大小
    tolerance=0.01,     # 验证容差
    verbose=True        # 显示详细信息
)

print(f"配对数量: {len(pairs)}")
print(f"示例配对: {pairs[:5]}")
```

### 与常规 DTW 对比 (Comparison with Regular DTW)

```python
from kiana import get_pair_via_dtw, get_pair_via_dtw_minimal
import numpy as np
import time

# 准备测试数据
template = np.cumsum(np.random.uniform(0.9, 1.1, 500))
query = template + np.random.normal(0, 0.05, 500)

# 常规 DTW
start = time.time()
pairs_regular = get_pair_via_dtw(template, query)
time_regular = time.time() - start

# 分块 DTW
start = time.time()
pairs_minimal = get_pair_via_dtw_minimal(template, query, chunk_size=50)
time_minimal = time.time() - start

print(f"常规 DTW: {len(pairs_regular)} 对, 耗时 {time_regular:.3f}秒")
print(f"分块 DTW: {len(pairs_minimal)} 对, 耗时 {time_minimal:.3f}秒")
print(f"提速: {time_regular/time_minimal:.2f}x")
```

### 在 BehavioralProcessor 中使用 (Usage in BehavioralProcessor)

未来版本将支持在 `BehavioralProcessor` 中直接使用该方法来加速长序列的配对过程。

## 参数说明 (Parameters)

- **template** (array-like): 参考序列（模板）
- **query** (array-like): 待对齐序列
- **chunk_size** (int, 默认=50): 每个块的大小。较大的值提供更准确的对齐，但计算量更大
- **tolerance** (float, 默认=0.01): 间隔一致性检查的容差。较小的值要求更严格的对齐
- **step_pattern** (str, 默认="symmetric2"): DTW 步进模式
- **verbose** (bool, 默认=False): 是否显示详细处理信息
- **max_iterations** (int, 默认=3): 每个块的最大重试次数

## 返回值 (Return Value)

返回配对列表，格式为 `[(template_idx, query_idx), ...]`，表示模板序列和查询序列中对应点的索引。

## 性能建议 (Performance Tips)

1. **选择合适的 chunk_size**: 
   - 较小的值（20-50）: 更快，但可能牺牲准确性
   - 较大的值（100-200）: 更准确，但速度较慢
   
2. **调整 tolerance**: 
   - 数据质量好、时钟漂移小: 使用较小的 tolerance (0.001-0.01)
   - 数据噪声大、时钟漂移明显: 使用较大的 tolerance (0.05-0.1)

3. **序列长度建议**:
   - < 50 个点: 直接使用常规 DTW
   - 50-500 个点: chunk_size=30-50
   - > 500 个点: chunk_size=50-100

## 工作原理 (How It Works)

```
原始序列: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
         ↓
         分块 (chunk_size=4)
         ↓
块 1: [1, 2, 3, 4] → DTW 配对 → 验证 → 添加到结果
         ↓
块 2: [3, 4, 5, 6] → DTW 配对 → 验证失败 → 扩大块 → 重新配对 → 验证 → 添加到结果
         ↓
块 3: [5, 6, 7, 8] → DTW 配对 → 验证 → 添加到结果
         ↓
         ...
         ↓
最终配对结果
```

## 注意事项 (Notes)

1. 该方法适用于长序列的配对，短序列会自动回退到常规 DTW
2. 验证失败不会导致错误，而是会尝试调整块大小重新配对
3. 块与块之间有小范围重叠，以确保全局对齐的连续性

## 技术细节 (Technical Details)

### 验证逻辑 (Validation Logic)

验证通过检查配对点之间的间隔一致性来判断配对质量：

```python
# 计算间隔差异
template_diffs = np.diff([template[i] for i, j in pairs])
query_diffs = np.diff([query[j] for i, j in pairs])
interval_diffs = query_diffs - template_diffs

# 检查不一致的间隔
inconsistent = np.abs(interval_diffs) > tolerance
inconsistent_ratio = np.sum(inconsistent) / len(interval_diffs)

# 允许最多 10% 的间隔不一致
valid = inconsistent_ratio <= 0.1
```

### 重试策略 (Retry Strategy)

当验证失败时，算法会自动调整块大小：
- 第 1 次重试: chunk_size × 1.5
- 第 2 次重试: chunk_size × 1.5² = chunk_size × 2.25
- 第 3 次重试: chunk_size × 1.5³ = chunk_size × 3.375

## 相关函数 (Related Functions)

- `get_pair_via_dtw()`: 常规 DTW 配对方法
- `get_paired_ephys_event_index()`: 将配对转换为索引映射
- `_purify_pairs()`: 清理配对结果

## 贡献 (Contributing)

欢迎提交问题报告和改进建议！如果您发现任何 bug 或有新功能建议，请在 GitHub 上开启 issue。
