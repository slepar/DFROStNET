import numpy as np
import tensorflow as tf
from Data import Data
import os 
import matplotlib.pyplot as plt

data_gen = Data()

def data_generator(switch_data, pulses_data):
    for i in range(len(switch_data)):
        # Normalize switch amplitude
        amp = np.abs(switch_data[i])
        amp = (amp - np.min(amp)) / (np.max(amp) - np.min(amp))

        # Normalize pulses
        pulse1 = pulses_data['pulse1'][i]
        pulse2 = pulses_data['pulse2'][i]
        baseline_pulse = pulse1 / np.max(np.abs(pulse1))
        FS_pulse = pulse2 / np.max(np.abs(pulse2))

        # Generate traces
        trace1 = data_gen.generate_trace(baseline_pulse, amp)
        trace2 = data_gen.generate_trace(FS_pulse, amp)

        # Stack pulses and traces
        paired_pulses = np.stack([baseline_pulse, FS_pulse], axis=-1)  # (128, 2)
        paired_traces = np.stack([trace1, trace2], axis=-1)            # (128, 128, 2)
        paired_switches = np.stack([amp, amp], axis=-1)                 # (512, 2)

        yield paired_pulses.astype(np.complex64), paired_switches.astype(np.float64), paired_traces.astype(np.float64)

def create_tf_dataset(switch_data, pulses_data, pulse_shape=(128, 2), switch_shape=(512, 2), trace_shape=(128, 128, 2)):
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(switch_data, pulses_data),
        output_signature=(
            tf.TensorSpec(shape=pulse_shape, dtype=tf.complex64),  # paired pulses
            tf.TensorSpec(shape=switch_shape, dtype=tf.float64),        # switch amplitude
            tf.TensorSpec(shape=trace_shape, dtype=tf.float64)      # paired traces
        )
    )
    return dataset

if __name__ == "__main__":
    pulses_data = np.load(r"D:\DFROStNET_may\Datasets\Pulses\128pts_10fs\FS35mm_1800nm_65535paired.npz")
    switch_data = np.load(r"D:\DFROStNET_may\Datasets\Switches\0730_65536\0730_65536.npz")


    dataset = create_tf_dataset(switch_data, pulses_data)
    tf_dataset_path = os.path.join(r"D:\DFROStNET_may\Datasets", f"0308_FS35mm_0730St_{len(switch_data)}_tf")
    tf.data.Dataset.save(dataset, tf_dataset_path)
    dataset = tf.data.Dataset.load(tf_dataset_path)
    print(dataset)   
    print(len(dataset))
