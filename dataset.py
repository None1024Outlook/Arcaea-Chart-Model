import tensorflow as tf
import soundfile as sf
import numpy as np
import math
import time
import os

class ArcaeaDataset:
    def __init__(self,
                 audio_dir: str = None,
                 chart_dir: str = None,
                 sample_rate: int = 44100,
                 batch_size: int = 1,
                 sequence_length: int = 1024):
        self.audio_dir = audio_dir
        self.chart_dir = chart_dir
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        self.audio_tensor_dtype = tf.float32
        self.time_steps_dtype = np.float32
        self.chart_tensor_dtype = tf.float32
        
        self.audio_files = self._get_dir_files(self.audio_dir, [".ogg", "wav", "mp3"])
        self.chart_files = self._get_dir_files(self.chart_dir, [".aff", ".txt"])
        
        self.audio_lengths = {}
        
    def _get_dir_files(self, directory: str, extensions: list):
        files = []
        for extension in extensions:
            files.extend([os.path.join(directory, f) for f in os.listdir(directory)if f.endswith(extension)])
        return sorted(files)
    
    def _parse_brackets_parameters(self, line: str):
        not_parameters = line.split("(")[1].split(")")[0].split(",")
        parameters = []
        for parameter in not_parameters:
            try:
                if "." in parameter:
                    parameter = float(parameter)
                else:
                    parameter = int(parameter)
            except:
                if parameter == "true":
                    parameter = True
                elif parameter == "false":
                    parameter = False
                elif parameter == "none":
                    parameter = None
                else:
                    parameter.replace("si", "I").replace("so", "O").replace("s", "S")
                    parameter = str(parameter)
            parameters.append(parameter)
        return parameters
    
    def _parse_chart(self, chart_text: str):
        audio_offset = 0
        timing_points = []
        taps = []
        arctaps = []
        arcs = []
        holds = []
        
        lines = chart_text.split("\n")
        isTimingGroup = False
        for line_num, line in enumerate(lines, 1):
            try:
                line = line.strip()
                
                if line.startswith("-") or not line: continue
                elif line.startswith("timinggroup"):
                    isTimingGroup = True
                    continue
                elif line.endswith("};"):
                    isTimingGroup = False
                    continue
                elif line.startswith('scenecontrol'): continue
					
                if isTimingGroup: continue
                
                if line.startswith("AudioOffset"):
                    audio_offset = float(line.split(":")[1])
                elif line.startswith("timing"):
                    timing_parameters = self._parse_brackets_parameters(line)
                    timing_point = {
                        "time": timing_parameters[0],
                        "bpm": timing_parameters[1],
                        "beats": timing_parameters[2]
                    }
                    timing_points.append(timing_point)
                elif line.startswith("hold"):
                    hold_parameters = self._parse_brackets_parameters(line)
                    hold = {
                        "start_time": hold_parameters[0],
                        "end_time": hold_parameters[1],
                        "track": hold_parameters[2], # 轨道 (4轨道: 1/2/3/4)
                    }
                    holds.append(hold)
                elif line.startswith("arc"):
                    if "arctap" in line:
                        not_arctaps = line.split("[")[1].split("]")[0].split(",")
                        arctaps += [{"time": int(arctap.split("(")[1].split(")")[0])} for arctap in not_arctaps]
                    
                    arc_parameters = self._parse_brackets_parameters(line)
                    arc = {
                        "start_time": arc_parameters[0],
                        "end_time": arc_parameters[1],
                        "start_x": arc_parameters[2],
                        "end_x": arc_parameters[3],
                        "curve_type": arc_parameters[4], # s: 直线 / so: 向外突出 / si: 向内突出 / siso / sosi
                        "start_y": arc_parameters[5],
                        "end_y": arc_parameters[6],
                        "color": arc_parameters[7], # 0: 蓝色 / 1: 红色 / 绿色
                        "whatsthis": arc_parameters[8], # none: None
                        "is_skyline": arc_parameters[9], # false: 蛇 / true: 线
                    }
                    arcs.append(arc)
                elif line.startswith("("):
                    tap_parameters = self._parse_brackets_parameters(line)
                    tap = {
                        "time": tap_parameters[0],
                        "track": tap_parameters[1], # 轨道 (4轨道: 1/2/3/4)
                    }
                    taps.append(tap)
            except Exception as e:
                print(f"警告: 错误在 {line}: {str(e)}")
            
        if not any([timing_points, taps, arcs, holds, arctaps]):
            print("警告: 谱面文件解析结果为空")
        
        return {
            'audio_offset': audio_offset,
            'timing_points': timing_points,
            'taps': taps,
            'arctaps': arctaps,
            'arcs': arcs,
            'holds': holds
        }
    
    def _load_audio(self, audio_path: str):
        try:
            data, sr = sf.read(audio_path)
            
            if len(data.shape) > 1: data = np.mean(data, axis=1)
            
            if sr != self.sample_rate:
                print(f"警告: 音频采样率 {sr}Hz 与目标采样率 {self.sample_rate}Hz 不匹配")
            
            return tf.convert_to_tensor(data, dtype=self.audio_tensor_dtype)
        except Exception as e:
            print(f"警告: 加载音频文件时出错 {audio_path}: {str(e)}")
            return None
    
    def _load_chart(self, chart_path: str, audio_length: int):
        with open(chart_path, 'r', encoding='utf-8') as f:
            chart_text = f.read()
        
        chart_data = self._parse_chart(chart_text)
        
        chart_time = 0
        for timing_point in chart_data["timing_points"]:
            chart_time = max(chart_time, timing_point["time"])
        for arc in chart_data["arcs"]:
            chart_time = max(chart_time, arc["end_time"])
        for hold in chart_data["holds"]:
            chart_time = max(chart_time, hold["end_time"])
        for arctap in chart_data["arctaps"]:
            chart_time = max(chart_time, arctap["time"])
        for tap in chart_data["taps"]:
            chart_time = max(chart_time, tap["time"])
        
        audio_time = math.ceil(audio_length / self.sample_rate * 1000)
        
        arcaea_time = max(audio_time, chart_time)
        
        # 时间步长为10ms
        time_steps = np.zeros((arcaea_time + 1, 12), self.time_steps_dtype)

        for tap in chart_data["taps"]:
            try:
                time_idx = min(tap["time"] // 10, arcaea_time // 10)
                track = tap["track"]
                track = min(4, max(1, track)) # 让轨道在1-4之间
                time_steps[time_idx, track-1] = 1.0
            except (ValueError, IndexError) as e:
                print(f"警告: 跳过无效的tap: {tap}")
                
        for hold in chart_data["holds"]:
            try:
                start_time = min(hold["start_time"] // 10, arcaea_time // 10)
                end_time = min(hold["end_time"] // 10, arcaea_time // 10)
                track = hold["track"]
                track = min(4, max(1, track)) # 让轨道在1-4之间
                time_steps[start_time:end_time, track+3] = 1.0
            except (ValueError, IndexError) as e:
                print(f"警告: 跳过无效的hold: {hold}")
        
        for arc in chart_data["arcs"]:
            try:
                start_time = min(arc["start_time"] // 10, arcaea_time // 10)
                end_time = min(arc["end_time"] // 10, arcaea_time // 10)
                
                if arc["is_skyline"]:
                    time_steps[start_time:end_time, 10] = 1.0
                if arc["color"] == 0: # 蓝色
                    time_steps[start_time:end_time, 8] = 1.0
                elif arc["color"] == 1: # 红色
                    time_steps[start_time:end_time, 9] = 1.0
                else:
                    print(f"警告: 跳过无效的arc(非蓝非红非天线): {arc}")
            except (ValueError, IndexError) as e:
                print(f"警告: 跳过无效的arc: {hold}")
        
        for arctap in chart_data["arctaps"]:
            try:
                time_idx = min(arctap["time"] // 10, arcaea_time // 10)
                time_steps[time_idx, 11] = 1.0
            except (ValueError, IndexError) as e:
                print(f"警告: 跳过无效的arctap: {tap}")
        
        note_count = np.sum(time_steps)
        print(f"\n谱面统计 ({os.path.basename(chart_path)}):")
        print(f"总时长: {arcaea_time/1000:.2f}秒")
        print(f"音符数量: {note_count}")
        print(f"平均密度: {note_count/(arcaea_time/1000):.2f}音符/秒")
        
        return tf.convert_to_tensor(time_steps, dtype=self.chart_tensor_dtype)
    
    def _get_audio_length(self, force: bool = False):
        if self.audio_lengths != {} and not force:
            return
        for audio_file in self.audio_files:
            try:
                audio, sr = sf.read(audio_file)
                self.audio_lengths[audio_file] = len(audio)
            except Exception as e:
                print(f"警告: 无法读取音频 {audio_file}: {str(e)}")
       
    def data_generator(self, audio_files, chart_files, sequence_length, force: bool = False):
        for audio_file, chart_file in zip(audio_files, chart_files):
            audio_data = self._load_audio(audio_file)
            if audio_data is None:
                print(f"警告: 跳过无效的音频: {audio_file}")
                continue
            
            self._get_audio_length(force=force)
            chart_data = self._load_chart(chart_file,self.audio_lengths.get(audio_file))
            if chart_data is None:
                print(f"警告: 跳过无效的谱面: {chart_file}")
                continue
            
            min_length = min(len(audio_data), len(chart_data))
            audio_data = audio_data[:min_length]
            chart_data = chart_data[:min_length]
            
            for i in range(0, min_length - sequence_length + 1, sequence_length // 2):
                audio_seq = audio_data[i:i+sequence_length]
                chart_seq = chart_data[i:i+sequence_length]
                
                if len(audio_seq) == sequence_length and len(chart_seq) == sequence_length:
                    yield audio_seq, chart_seq
    
    def create_dataset(self):
        self.dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(self.audio_files, self.chart_files, self.sequence_length),
            output_signature=(
                tf.TensorSpec(shape=(self.sequence_length,), dtype=self.audio_tensor_dtype),
                tf.TensorSpec(shape=(self.sequence_length, 12), dtype=self.chart_tensor_dtype)
            )
        )
        
        self.dataset = self.dataset.batch(
            self.batch_size,
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)
        
        return self.dataset
    
    def get_dataset_cardinality(self):
        return min(len(self.music_files), len(self.chart_files))

    def save_dataset(self, save_dir: str):
        if self.dataset is None:
            print("没有数据集可保存")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"arcaea_{time.time()}.data")
        self.dataset.save(save_path)
        print(f"数据集已保存到 {save_path}")
    
    def load_dataset(self, save_path: str):
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"{save_path} 不存在")
        
        self.dataset = tf.data.Dataset.load(save_path)
        print(f"从 {save_path} 加载了数据集")
        
        return self.dataset
