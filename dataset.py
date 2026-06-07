import os
import glob
import json
import random
import hashlib
import pickle
import mido
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import *

class MidiVelocityDataset(Dataset):
    """读取 MIDI，抹平力度，返回 (input, target) 对"""

    def __init__(self, file_list, is_train=True):
        self.file_list = file_list
        self.is_train = is_train
        self.samples = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._build_index()

    def _cache_path(self, midi_path):
        stat = os.stat(midi_path)
        cache_key = f"{os.path.abspath(midi_path)}|{stat.st_size}|{stat.st_mtime_ns}|{TIME_TICK}|{MIN_NOTES}"
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return os.path.join(CACHE_DIR, f"{digest}.pkl")

    def _load_cached_notes(self, midi_path):
        cache_path = self._cache_path(midi_path)
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None

        return payload.get("notes") if isinstance(payload, dict) else None

    def _save_cached_notes(self, midi_path, notes):
        cache_path = self._cache_path(midi_path)
        tmp_path = f"{cache_path}.tmp"
        payload = {"notes": notes}
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _extract_notes(self, midi_path):
        """从 MIDI 文件提取音符列表，返回 [(pitch, onset_tick, dur_tick, velocity), ...]"""
        cached_notes = self._load_cached_notes(midi_path)
        if cached_notes is not None:
            return cached_notes

        try:
            mid = mido.MidiFile(midi_path)
        except Exception:
            return []

        # 合并所有音轨
        notes = []
        ticks_per_beat = mid.ticks_per_beat or 480
        tempo = 500000  # 默认 120 BPM (微秒/拍)

        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'pitch': msg.note,
                        'onset_tick': abs_time,
                        'velocity': msg.velocity,
                        'channel': msg.channel,
                    })
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # 找到对应的 note_on，计算 duration
                    for n in reversed(notes):
                        if n.get('pitch') == msg.note and n.get('channel') == msg.channel and 'dur_tick' not in n:
                            n['dur_tick'] = abs_time - n['onset_tick']
                            break

        # 过滤没配对上的
        notes = [n for n in notes if 'dur_tick' in n and n['dur_tick'] > 0]

        if len(notes) < MIN_NOTES:
            return []

        # 按起始时间排序
        notes.sort(key=lambda n: n['onset_tick'])

        # 归一化时间：转成秒
        tick_to_second = tempo / (ticks_per_beat * 1_000_000)
        result = []
        for n in notes:
            onset = round(n['onset_tick'] * tick_to_second / TIME_TICK)
            dur = max(1, round(n['dur_tick'] * tick_to_second / TIME_TICK))
            result.append((n['pitch'], onset, dur, n['velocity']))

        self._save_cached_notes(midi_path, result)
        return result

    def _build_index(self):
        """遍历所有文件，切成 512 音符的段，建立索引"""
        for path in self.file_list:
            notes = self._extract_notes(path)
            if len(notes) < MIN_NOTES:
                continue
            # 切成 MAX_NOTES 长的段
            for start in range(0, len(notes), MAX_NOTES):
                segment = notes[start:start + MAX_NOTES]
                self.samples.append(segment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment = self.samples[idx]
        n = len(segment)
        pad_len = MAX_NOTES - n

        # 输入特征: [pitch/127, onset/max_onset, dur/max_dur, velocity/127]
        pitches = np.array([s[0] for s in segment], dtype=np.float32) / 127.0
        onsets = np.array([s[1] for s in segment], dtype=np.float32)
        durs = np.array([s[2] for s in segment], dtype=np.float32)

        # 时间归一化（除以此段的最大时间值）
        max_time = max(onsets[-1] + durs[-1], 1.0)
        onsets = onsets / max_time
        durs = np.clip(durs / max_time, 0.0, 1.0)

        # 输入力度全部抹平
        input_vel = np.full(n, VELOCITY_NORMALIZE / 127.0, dtype=np.float32)

        # 标签是原始力度 (0-127 整数)
        targets = np.array([s[3] for s in segment], dtype=np.int64)

        # Padding
        if pad_len > 0:
            pitches = np.pad(pitches, (0, pad_len), constant_values=0)
            onsets = np.pad(onsets, (0, pad_len), constant_values=0)
            durs = np.pad(durs, (0, pad_len), constant_values=0)
            input_vel = np.pad(input_vel, (0, pad_len), constant_values=0)
            targets = np.pad(targets, (0, pad_len), constant_values=-100)  # ignore_index

        # 堆叠成 (512, 4)
        inputs = np.stack([pitches, onsets, durs, input_vel], axis=-1).astype(np.float32)

        return torch.from_numpy(inputs), torch.from_numpy(targets)


def get_dataloaders():
    """收集所有 MIDI 文件，划分 train/val，返回 DataLoader"""
    all_files = glob.glob(os.path.join(DATA_DIR, "*", "*.mid"))
    random.seed(42)
    random.shuffle(all_files)

    split = int(len(all_files) * TRAIN_RATIO)
    train_files = all_files[:split]
    val_files = all_files[split:]

    print(f"总文件数: {len(all_files)}, 训练: {len(train_files)}, 验证: {len(val_files)}")

    train_ds = MidiVelocityDataset(train_files, is_train=True)
    val_ds = MidiVelocityDataset(val_files, is_train=False)

    print(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    return train_loader, val_loader
