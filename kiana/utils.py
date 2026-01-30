import dtw
import numpy as np
from collections import defaultdict

def get_pair_via_dtw(template, query, step_pattern="symmetric2", verbose=False):
    template = np.asarray(template)
    query = np.asarray(query)
    
    # Need at least 2 elements to compute diff
    if len(template) < 2 or len(query) < 2:
        # Return trivial pairing for single elements
        if len(template) == 1 and len(query) == 1:
            return [(0, 0), (1, 1)]
        elif len(template) == 1:
            return [(0, 0)] + [(1, i+1) for i in range(len(query))]
        elif len(query) == 1:
            return [(i, 0) for i in range(len(template))] + [(len(template), 1)]
        else:
            return []
    
    template = np.diff(template)
    query = np.diff(query)
    
    # Reshape to 2D for compatibility with dtw library
    template = template.reshape(-1, 1)
    query = query.reshape(-1, 1)
    
    dist_fun = lambda x_val, y_val: abs(x_val[0] - y_val[0])
    alignment_default = dtw.dtw(template, query,
                        dist_method=dist_fun,
                        step_pattern=step_pattern, # 或者 rabinerJuangStepPattern(6, "c"))\
                        keep_internals=True)
    # 获取结果
    distance_default = alignment_default.distance         # DTW距离
    path_s1_default = alignment_default.index1            # s1 中点的索引序列
    path_s2_default = alignment_default.index2            # s2 中点的索引序列
    cost_matrix_default = alignment_default.costMatrix    # 累积代价矩阵
    local_cost_matrix_default = alignment_default.localCostMatrix # 局部代价矩阵


    path_pairs_default = list(zip(path_s1_default, path_s2_default))

    # 希望在最前面增加一个(-1,-1), 因为之前采取的是差值匹配，会少一个数值
    path_pairs_default = [(-1, -1)] + path_pairs_default
    # 对所有元素+1，这样回归到从0开始的索引
    path_pairs_default = [(i + 1, j + 1) for i, j in path_pairs_default]

    if verbose:
        print(f"--- 使用 dtw-python ( {step_pattern} 步进模式) ---")
        print(f"DTW 距离: {distance_default:.2f}")
        dtw.dtwPlot(alignment_default,type="twoway")
        dtw.dtwPlot(alignment_default,type="density")
        print(f"匹配结果（template点的id, query点的id） :\n {path_pairs_default}")
        # print(rabinerJuangStepPattern(6,"c"))
        # rabinerJuangStepPattern(6,"c").plot()

    return path_pairs_default

def _purify_pairs(pairs, key_index=0):
    """
    一个通用的配对列表清理函数，确保基于 key_index 的唯一性。

    Args:
        pairs (list): (id_A, id_B) 格式的配对列表。
        key_index (int): 指定作为Key的索引 (0 或 1)。

    Returns:
        list: 清理后的配对列表。
    """
    if not pairs:
        return []

    value_index = 1 - key_index
    
    # 按指定的 key 进行分组
    key_to_values_map = defaultdict(list)
    for pair in pairs:
        key = pair[key_index]
        value = pair[value_index]
        key_to_values_map[key].append(value)
        
    # 解决冲突，并构建纯净的配对列表
    purified_pairs = []
    for key, values in key_to_values_map.items():
        # 决策规则：保留value最小的那个
        best_value = min(values)
        
        # 重新组装配对，注意保持原始的 (id_A, id_B) 顺序
        if key_index == 0:
            purified_pairs.append((key, best_value))
        else:
            purified_pairs.append((best_value, key))
            
    return purified_pairs

def get_paired_ephys_event_index(task_ephys_dtw_pairs, conservative=False):
    """
    【最终重构版】
    将DTW配对转换为task索引到ephys索引的映射数组。
    通过连续调用两次通用清理函数来确保最终映射的双向唯一性。
    """
    # 原始DTW配对可能包含两种冲突
    
    if not conservative:
        # 第一步：清理“一个Ephys对多个Task”的冲突 (key_index=1, 按ephys_id分组)
        clean_pairs_stage1 = _purify_pairs(task_ephys_dtw_pairs, key_index=1)

        # 第二步：清理“一个Task对多个Ephys”的冲突 (key_index=0, 按task_id分组)
        # 输入是上一阶段的输出
        final_clean_pairs = _purify_pairs(clean_pairs_stage1, key_index=0)
    else:
        # 保守模式下，直接进行第二步清理
        final_clean_pairs = _purify_pairs(task_ephys_dtw_pairs, key_index=0)
    
    # 第三步：使用完全纯净的配对列表构建最终数组
    if not final_clean_pairs:
        return np.array([])
        
    max_task_id = max(pair[0] for pair in final_clean_pairs)
    final_numpy_array = np.full(max_task_id + 1, np.nan)
    
    # 因为 final_clean_pairs 中的 task_event_id 已经是唯一的，所以可以直接赋值
    for task_event_id, ephys_event_id in final_clean_pairs:
        final_numpy_array[task_event_id] = ephys_event_id

    return final_numpy_array

def get_pair_via_dtw_minimal(template, query, chunk_size=50, tolerance=0.01, 
                              step_pattern="symmetric2", verbose=False, max_iterations=3):
    """
    Minimal pair method for DTW alignment with chunked processing.
    
    This method improves efficiency when dealing with long queues by:
    1. Taking only a portion from the front of each queue at a time
    2. Performing DTW pairing on these portions
    3. Validating the pairing by checking interval consistency
    4. If mismatch is found, re-pairing with adjusted chunk size
    5. Processing incrementally until all data is paired
    
    Args:
        template: First sequence (reference), array of time points
        query: Second sequence (to be aligned), array of time points
        chunk_size (int): Number of elements to process in each chunk. Default: 50
        tolerance (float): Tolerance for interval consistency check. Default: 0.01
        step_pattern (str): DTW step pattern. Default: "symmetric2"
        verbose (bool): Whether to print detailed information. Default: False
        max_iterations (int): Maximum number of re-pairing attempts per chunk. Default: 3
        
    Returns:
        list: List of paired indices in format [(template_idx, query_idx), ...]
    """
    template = np.asarray(template)
    query = np.asarray(query)
    
    if len(template) == 0 or len(query) == 0:
        return []
    
    # If sequences are short enough, use regular DTW
    if len(template) <= chunk_size and len(query) <= chunk_size:
        return get_pair_via_dtw(template, query, step_pattern, verbose)
    
    all_pairs = []
    template_offset = 0
    query_offset = 0
    
    if verbose:
        print(f"Starting minimal pair DTW with chunk_size={chunk_size}, tolerance={tolerance}")
        print(f"Template length: {len(template)}, Query length: {len(query)}")
    
    while template_offset < len(template) and query_offset < len(query):
        # Determine chunk boundaries with some extra context for better alignment
        template_end = min(template_offset + chunk_size, len(template))
        query_end = min(query_offset + chunk_size, len(query))
        
        # Extract chunks
        template_chunk = template[template_offset:template_end]
        query_chunk = query[query_offset:query_end]
        
        if verbose:
            print(f"\nProcessing chunk: template[{template_offset}:{template_end}], query[{query_offset}:{query_end}]")
        
        # Perform DTW on chunk with retry logic
        iteration = 0
        chunk_valid = False
        current_chunk_size = chunk_size
        chunk_pairs = []
        
        while iteration < max_iterations and not chunk_valid:
            # Adjust chunk size for retry - increase size to get better context
            if iteration > 0:
                # Increase chunk size by 50% for each retry
                current_chunk_size = min(
                    int(chunk_size * (1.5 ** iteration)), 
                    len(template) - template_offset, 
                    len(query) - query_offset
                )
                template_end = min(template_offset + current_chunk_size, len(template))
                query_end = min(query_offset + current_chunk_size, len(query))
                template_chunk = template[template_offset:template_end]
                query_chunk = query[query_offset:query_end]
                
                if verbose:
                    print(f"  Retry {iteration}: Adjusting chunk size to {current_chunk_size}")
            
            # Get pairs for this chunk
            chunk_pairs = get_pair_via_dtw(template_chunk, query_chunk, step_pattern, verbose=False)
            
            # Validate chunk pairs
            chunk_valid = _validate_pairs(template_chunk, query_chunk, chunk_pairs, tolerance)
            
            if verbose:
                if chunk_valid:
                    print(f"  Validation passed with {len(chunk_pairs)} pairs")
                else:
                    print(f"  Validation failed (iteration {iteration + 1})")
            
            iteration += 1
        
        if not chunk_valid and verbose:
            print(f"  Warning: Using result after {max_iterations} validation attempts")
        
        # Adjust indices to global coordinates and add to results
        # For the first chunk, add all pairs
        # For subsequent chunks, skip the first pair to avoid duplicates at boundaries
        start_idx = 1 if len(all_pairs) > 0 else 0
        
        for i, j in chunk_pairs[start_idx:]:
            global_i = template_offset + i
            global_j = query_offset + j
            all_pairs.append((global_i, global_j))
        
        # Update offsets for next iteration
        # Move forward based on the last matched index in the chunk
        if len(chunk_pairs) > 0:
            last_template_idx, last_query_idx = chunk_pairs[-1]
            
            # Move forward but leave a small overlap to ensure continuity
            overlap = min(5, chunk_size // 10)
            template_offset += max(1, last_template_idx - overlap + 1)
            query_offset += max(1, last_query_idx - overlap + 1)
        else:
            # If no pairs found (shouldn't happen), move forward by 1
            template_offset += 1
            query_offset += 1
    
    if verbose:
        print(f"\nCompleted minimal pair DTW with {len(all_pairs)} total pairs")
    
    return all_pairs


def _validate_pairs(template, query, pairs, tolerance=0.01):
    """
    Validate DTW pairs by checking interval consistency.
    
    Args:
        template: Template sequence
        query: Query sequence  
        pairs: List of paired indices
        tolerance: Tolerance for interval consistency
        
    Returns:
        bool: True if pairs are valid, False otherwise
    """
    if len(pairs) < 3:
        # Not enough pairs to validate intervals
        return True
    
    template = np.asarray(template)
    query = np.asarray(query)
    
    # Extract matched values
    template_vals = [template[i] for i, j in pairs]
    query_vals = [query[j] for i, j in pairs]
    
    # Calculate interval differences
    template_diffs = np.diff(template_vals)
    query_diffs = np.diff(query_vals)
    interval_diffs = query_diffs - template_diffs
    
    # Check if all intervals are within tolerance
    inconsistent = np.abs(interval_diffs) > tolerance
    
    # Allow up to 10% of intervals to be inconsistent
    inconsistent_ratio = np.sum(inconsistent) / len(interval_diffs)
    
    return inconsistent_ratio <= 0.1


def get_spikes_in_windows(spike_train, event_windows):
    """
    使用列表推导式，更简洁地提取窗口内的相对spike time。
    """
    spike_train = np.asarray(spike_train)
    event_windows = np.asarray(event_windows)

    # 一行列表推导式完成所有操作
    return [
        spike_train[(spike_train >= start) & (spike_train < end)] - start
        for start, end in event_windows
    ]