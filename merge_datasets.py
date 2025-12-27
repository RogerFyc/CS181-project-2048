#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge multiple 2048 human dataset JSON files into one,
optionally reindex episode IDs, regenerate stats, and export CSV.

Usage examples:
    # 基本用法（默认输入 collected_data/，输出 merged_human_data.json/_stats.json）
    python merge_datasets.py

    # 指定目录与输出前缀
    python merge_datasets.py --input-dir collected_data --output-base merged_human_data

    # 重写 episode ID（默认启用），并导出步骤级 CSV
    python merge_datasets.py --export-steps-csv merged_steps.csv

    # 不重写 ID（保留原始 episode）
    python merge_datasets.py --no-reindex
"""
import argparse
import json
import os
import glob
from datetime import datetime

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _ensure_list(x):
    return x if isinstance(x, list) else []

def _matrix_max_from_vec16(vec16):
    """从 16 维向量计算最大值；若不是 16 维，返回 0。"""
    try:
        if isinstance(vec16, list) and len(vec16) == 16:
            return max(vec16) if vec16 else 0
    except Exception:
        pass
    return 0

def _episode_num_steps(ep):
    # 优先使用 ep['num_steps']；否则用 len(ep['data'])
    if isinstance(ep, dict):
        if 'num_steps' in ep and isinstance(ep['num_steps'], int):
            return ep['num_steps']
        data = ep.get('data', [])
        if isinstance(data, list):
            return len(data)
    return 0

def _episode_last_step(ep):
    data = _ensure_list(ep.get('data', []))
    return data[-1] if data else None

def _episode_max_tile(ep):
    """
    尝试从 ep['data'] 中求最大 tile：
    - 优先用步骤中的 'max_tile_val'
    - 否则取最后一步的 next_state 的最大值
    - 若无 next_state，取最后一步的 state 的最大值
    """
    data = _ensure_list(ep.get('data', []))
    if not data:
        return 0
    # 尝试使用每步的 max_tile_val（若存在）
    max_by_field = 0
    for step in data:
        v = _safe_int(step.get('max_tile_val', 0), 0)
        if v > max_by_field:
            max_by_field = v
    if max_by_field > 0:
        return max_by_field

    # 回退：用最后一步的 next_state 或 state 计算
    last = data[-1]
    next_state = last.get('next_state')
    state = last.get('state')
    if next_state:
        return _matrix_max_from_vec16(next_state)
    if state:
        return _matrix_max_from_vec16(state)
    return 0

def _episode_summary(ep):
    last = _episode_last_step(ep)
    max_tile = _episode_max_tile(ep)
    return {
        'game_id': _safe_int(ep.get('episode', 0), 0),
        'num_steps': _episode_num_steps(ep),
        'score': ep.get('game_score', 0),
        'state': ep.get('game_state', 'unknown'),
        'max_tile': max_tile,
        'times_max_reduced': _safe_int(last.get('times_max_reduced', 0), 0) if last else 0,
        'times_other_reduced': _safe_int(last.get('times_other_reduced', 0), 0) if last else 0,
    }

def _load_episodes_from_file(path):
    """读取单个 JSON 文件的所有 episode（列表）。非列表则返回空。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            # 某些文件可能保存为单个 episode dict
            return [data] if isinstance(data, dict) else []
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return []

def _reindex_episodes(episodes, start_id=1):
    """将 episode ID 重写为从 start_id 开始的连续序列，同时更新每个 step 的 episode 字段（若存在）。"""
    new_eps = []
    next_id = start_id
    for ep in episodes:
        ep_copy = dict(ep)
        ep_copy['episode'] = next_id
        data = _ensure_list(ep_copy.get('data', []))
        new_data = []
        for step in data:
            s = dict(step)
            s['episode'] = next_id
            new_data.append(s)
        ep_copy['data'] = new_data
        new_eps.append(ep_copy)
        next_id += 1
    return new_eps

def _export_steps_csv(episodes, csv_path):
    """导出步骤级 CSV，包含 state_0..state_15 和 next_0..next_15。"""
    import csv
    cols = [
        'episode', 'step', 'action',
        'empty_cells', 'times_max_reduced', 'times_other_reduced', 'max_tile_val',
        'special_pos_row', 'special_pos_col',
    ]
    # 扩展 16 列 state 与 16 列 next_state
    state_cols = [f'state_{i}' for i in range(16)]
    next_cols = [f'next_{i}' for i in range(16)]
    all_cols = cols + state_cols + next_cols

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for ep in episodes:
            ep_id = _safe_int(ep.get('episode', 0), 0)
            for step in _ensure_list(ep.get('data', [])):
                row = {
                    'episode': ep_id,
                    'step': _safe_int(step.get('step', 0), 0),
                    'action': step.get('action', ''),
                    'empty_cells': _safe_int(step.get('empty_cells', 0), 0),
                    'times_max_reduced': _safe_int(step.get('times_max_reduced', 0), 0),
                    'times_other_reduced': _safe_int(step.get('times_other_reduced', 0), 0),
                    'max_tile_val': _safe_int(step.get('max_tile_val', 0), 0),
                }
                # special_pos 可能是 [i, j] 或 None
                sp = step.get('special_pos')
                if isinstance(sp, (list, tuple)) and len(sp) == 2:
                    row['special_pos_row'] = _safe_int(sp[0], 0)
                    row['special_pos_col'] = _safe_int(sp[1], 0)
                else:
                    row['special_pos_row'] = ''
                    row['special_pos_col'] = ''

                # 填充 state / next_state
                st = step.get('state') or []
                nx = step.get('next_state') or []
                for i in range(16):
                    row[state_cols[i]] = st[i] if i < len(st) else ''
                    row[next_cols[i]] = nx[i] if i < len(nx) else ''
                writer.writerow(row)
    print(f"Steps CSV exported: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge 2048 human dataset JSON files.")
    parser.add_argument('--input-dir', type=str, default='collected_data',
                        help='Directory containing dataset JSON files (default: collected_data)')
    parser.add_argument('--pattern', type=str, default='*.json',
                        help='Glob pattern to match dataset files (default: *.json)')
    parser.add_argument('--output-base', type=str, default='merged_human_data',
                        help='Base name for output files (without extension). '
                             'Will produce {output-base}.json and {output-base}_stats.json')
    parser.add_argument('--no-reindex', action='store_true',
                        help='Do not reindex episode IDs (keep originals).')
    parser.add_argument('--export-steps-csv', type=str, default=None,
                        help='Optional path to export steps-level CSV.')
    args = parser.parse_args()

    in_dir = args.input_dir
    pattern = args.pattern
    out_base = args.output_base
    reindex = not args.no_reindex

    if not os.path.isdir(in_dir):
        print(f"Error: input dir not found: {in_dir}")
        return

    # 收集候选文件（跳过 *_stats.json）
    files = sorted(glob.glob(os.path.join(in_dir, pattern)))
    files = [p for p in files if not p.endswith('_stats.json')]
    if not files:
        print(f"No dataset files matched in {in_dir} with pattern {pattern}.")
        return

    print("Merging files:")
    for p in files:
        print(f"  - {p}")

    # 合并所有 episodes
    episodes = []
    for p in files:
        eps = _load_episodes_from_file(p)
        if eps:
            episodes.extend(eps)

    if not episodes:
        print("No episodes loaded; abort.")
        return

    # 可选重写 episode ID
    if reindex:
        episodes = _reindex_episodes(episodes, start_id=1)
        print(f"Reindexed {len(episodes)} episodes from 1..{len(episodes)}")
    else:
        print(f"Keeping original episode IDs ({len(episodes)} episodes).")

    # 写合并后的 JSON
    out_json = f"{out_base}.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, indent=2, separators=(', ', ': '), ensure_ascii=False)
    print(f"✅ Merged dataset written to: {out_json}")

    # 生成统计文件
    total_games = len(episodes)
    total_steps = sum(_episode_num_steps(ep) for ep in episodes)
    stats = {
        'total_games': total_games,
        'total_steps': total_steps,
        'timestamp': datetime.now().isoformat(),
        'games': [_episode_summary(ep) for ep in episodes]
    }
    out_stats = f"{out_base}_stats.json"
    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"📊 Stats written to: {out_stats}")

    # 可选导出 CSV（步骤级）
    if args.export_steps_csv:
        _export_steps_csv(episodes, args.export_steps_csv)

    print("\nDone.")

if __name__ == '__main__':
    main()