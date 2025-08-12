import matplotlib.pyplot as plt
from Data import Data
import tensorflow as tf
import os 
import numpy as np

def extract_spectral_amplitude(trace):
    spectrum = tf.sqrt(tf.abs(trace[:, :, 0, 0]))  # shape: (B,)
    spectrum /= tf.reduce_max(spectrum, axis=1, keepdims=True) + 1e-9  # <-- only valid if shape is (B, K)
    return spectrum

def gen_dataset(N_pulses, data_type = "Mixed"):
    # generates a dataset of N pulses, of the requested phase type: [Mixed, Primary_Mixed, Pure]
    accepted_phases, counter, error = [], 0, 0
    dir = os.path.join(os.getcwd(), "Datasets", "256pts_10fs", f"{N_pulses}pulses", data_type)
    os.makedirs(dir, exist_ok = True)
    while counter < N_pulses:
        # cycle through each order from linear to quartic
        for degree in range(1, 5):
            path = os.path.join(dir, f"{counter}.npz")
            # generate pulse
            if data_type == "Mixed":
                data_gen.generate_chirped_pulse(phase_coeff = data_gen.mixed_phase())
            elif data_type == "Primary_Mixed":
                data_gen.generate_chirped_pulse(phase_coeff = data_gen.primary_mixed_phase(degree))
            elif data_type == "Pure":
                data_gen.generate_chirped_pulse(phase_coeff = data_gen.pure_phase(degree))

            # check phase quality
            if data_gen.check_ambiguity(data_gen.phase_t, accepted_phases) is True:
                error += 1
                continue 

            # generate the trace and save the datapoint
            trace = data_gen.interp_trace(data_gen.FROStNET(data_gen.complex_Et_field, switch), data_gen.N_points, data_gen.N_points)
            trace = tf.expand_dims(trace, axis = -1)        # trace must be in format (x, y, 1) for batching
            # trace = tf.expand_dims(trace, axis = 0)        # trace must be in format (x, y, 1) for batching

            # amp = extract_spectral_amplitude(trace)
            plt.subplot(131)
            plt.plot(data_gen.t_s*1e15, np.angle(data_gen.complex_Et_field))
            plt.plot(data_gen.t_s*1e15, np.abs(data_gen.complex_Et_field))
            plt.plot(data_gen.switch_t*1e15, np.abs(switch))

            plt.subplot(132)
            plt.plot(data_gen.w_Hz*1e-15, np.angle(data_gen.complex_Ew_field))
            plt.plot(data_gen.w_Hz*1e-15, np.abs(data_gen.complex_Ew_field) / np.max(np.abs(data_gen.complex_Ew_field)))
            # plt.plot(data_gen.w_Hz*1e-15, np.abs(amp))

            plt.subplot(133)
            plt.imshow(np.abs(trace), aspect= "auto")
            plt.show()

            data_gen.save_datapoint(data_gen.complex_Et_field, switch, trace, path)
            counter += 1
            accepted_phases.append(data_gen.phase_t)
            print(counter / N_pulses)
        plt.show()

    print("Data Generation Completed with Error:       ", error)
    return os.path.dirname(dir)

def compile_data(directory):
    # walks through subdirs and assembles a list of all the datapoints
    pulse_list, switch_list, frost_list = [], [], []
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            dataset = np.load(file_path)

            # Append each array to the respective list
            pulse_list.append(dataset['pulse'])
            switch_list.append(dataset['switch'])
            frost_list.append(dataset['trace'])    
    return pulse_list, switch_list, frost_list

if __name__ == "__main__":
    # Start by initializing necessary variables
    data_gen = Data()
    switch = data_gen.generate_switch()

    # generate some pulses with various types of phases
    N_dataset = 10 #65536 // 3 # 16384                             # Generate a small number of data points
    path = gen_dataset(N_dataset, "Mixed")
    path = gen_dataset(N_dataset, "Primary_Mixed")
    path = gen_dataset(N_dataset, "Pure")

    # Now you should have a folder at path with 3 subfolders: [Mixed, Primary_Mixed, Pure]
    # each subfolder contains N_dataset .npz files where each file contains a single datapoint
    # now we want to assemble them into a single npz file
    pulse_list, switch_list, trace_list = compile_data(path)
    unified_npz_path = os.path.join(path, f"{os.path.basename(path)}.npz")
    np.savez_compressed(unified_npz_path, pulse=pulse_list, switch=switch_list, trace=trace_list)

    # now we want to put that into a tensorflow dataset that can be fetched during training
    dataset = tf.data.Dataset.from_tensor_slices((pulse_list, switch_list, trace_list))
    tf_dataset_path = os.path.join(path, f"{os.path.basename(path)}_tf")
    tf.data.Dataset.save(dataset, tf_dataset_path)

    # read the tf file back to ensure it saved right
    dataset = tf.data.Dataset.load(tf_dataset_path)
    print(dataset)

    # also save the axes to a npz file for later
    np.savez_compressed(os.path.join(path, "x_axes.npz"), time = data_gen.t_s, delay = data_gen.switch_t, freq = data_gen.w_Hz)
