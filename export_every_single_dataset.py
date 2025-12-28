#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split each 2048 dataset JSON into per-episode CSV files (one game -> one CSV).

- Scans an input directory (default: collected_data), or a single --file
- For every dataset JSON (excluding *_stats.json), writes one CSV per episode:
    - Default naming (same directory as JSON):
        {base}_ep{episode_id}.csv
      Example:
        collected_data/2048_human_data_10games_20251228_162430_ep1.csv

    - If --into-subdir is set:
        creates a subdir per dataset file:
        collected_data/2048_human_data_10games_20251228_162430_per_episode/ep_1.csv, ep_2.csv, ...

Usage:
    # 按局导出所有 JSON（默认写回 collected_data，同目录生成 *_ep{ID}.csv）
    python export_per_episode_csv.py

    # 指定目录
    python export_per_episode_csv.py --input-dir collected_data

    # 只处理某个文件
    python export_per_episode_csv.py --file collected_data/2048_human_data_10games_20251228_162430.json

    # 重排该文件内部的 episode ID 为 1..N（不改原 JSON，只在导出时重排）
    python export_per_episode_csv.py --reindex

    # 将每个数据文件的 per-episode CSV 导出到专属子目录
    python export_per_episode_csv.py --into-subdir

    # 自定义输出目录（不设置则用原 JSON 同目录）
    python export_per_episode_csv.py --out-dir out_csv
"""
import argparse
import csv
import json
import os
import glob

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _ensure_list(x):
    return x if isinstance(x, list) else []

def _load_episodes_from_file(path):
    """读取单个 JSON 文件的所有 episode（列表）。非列表则返回空。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
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

def _episode_to_csv_rows(ep):
    """将一个 episode 转为 CSV 行（每一步一行）。"""
    rows = []
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
            'special_pos_row': '',
            'special_pos_col': '',
        }
        sp = step.get('special_pos')
        if isinstance(sp, (list, tuple)) and len(sp) == 2:
            row['special_pos_row'] = _safe_int(sp[0], 0)
            row['special_pos_col'] = _safe_int(sp[1], 0)

        st = step.get('state') or []
        nx = step.get('next_state') or []
        # 填充 16 维展开
        for i in range(16):
            row[f'state_{i}'] = st[i] if i < len(st) else ''
            row[f'next_{i}'] = nx[i] if i < len(nx) else ''
        rows.append(row)
    return rows

def _write_episode_csv(rows, csv_path):
    """写一个 episode 的 CSV（多步）。"""
    cols = [
        'episode', 'step', 'action',
        'empty_cells', 'times_max_reduced', 'times_other_reduced', 'max_tile_val',
        'special_pos_row', 'special_pos_col',
    ] + [f'state_{i}' for i in range(16)] + [f'next_{i}' for i in range(16)]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Export per-episode CSVs from 2048 dataset JSON.")
    parser.add_argument('--input-dir', type=str, default='collected_data',
                        help='Directory containing dataset JSON files (default: collected_data)')
    parser.add_argument('--pattern', type=str, default='*.json',
                        help='Glob pattern to match dataset files (default: *.json)')
    parser.add_argument('--file', type=str, default=None,
                        help='If provided, only process this single JSON file.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: same as JSON file directory).')
    parser.add_argument('--reindex', action='store_true',
                        help='Reindex episode IDs starting at 1 within each dataset file.')
    parser.add_argument('--into-subdir', action='store_true',
                        help='Put per-episode CSVs into a subdirectory named <base>_per_episode/')
    args = parser.parse_args()

    files = []
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found: {args.file}")
            return
        files = [args.file]
    else:
        if not os.path.isdir(args.input_dir):
            print(f"Error: input dir not found: {args.input_dir}")
            return
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        files = [p for p in files if not p.endswith('_stats.json')]

    if not files:
        print("No dataset JSON files found to export.")
        return

    print("Per-episode export for files:")
    for p in files:
        print(f"  - {p}")

    for path in files:
        episodes = _load_episodes_from_file(path)
        if not episodes:
            print(f"Skip (no episodes): {path}")
            continue

        if args.reindex:
            episodes = _reindex_episodes(episodes, start_id=1)

        json_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        # 决定输出目录
        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = json_dir

        if args.into_subdir:
            out_dir = os.path.join(out_dir, f"{base_name}_per_episode")
            os.makedirs(out_dir, exist_ok=True)

        # 写每一局
        count = 0
        for ep in episodes:
            ep_id = _safe_int(ep.get('episode', 0), 0)
            if args.into_subdir:
                csv_path = os.path.join(out_dir, f"ep_{ep_id}.csv")
            else:
                csv_path = os.path.join(out_dir, f"{base_name}_ep{ep_id}.csv")
            rows = _episode_to_csv_rows(ep)
            _write_episode_csv(rows, csv_path)
            count += 1

        print(f"Exported {count} episodes from {path} into {out_dir}")

    print("\nDone.")

if __name__ == '__main__':
    main()