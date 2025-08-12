import os 
from Data import Data
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import random
import tensorflow as tf

def bloch_to_optical(u, v, w):
    """
    Maps Bloch vectors (u, v, w) to amplitude and phase response.
    
    Returns:
        A: amplitude modulation (0 to 1)
        phi: phase modulation in radians
        T: complex transmission function A * exp(i * phi)
    """
    A = np.sqrt((1 - w) / 2)
    phi = np.arctan2(v, u)
    T = A * np.exp(1j * phi)
    return A, phi, T

def adequate_contrast(vec, threshold = 0.25):
    start_ave = np.mean(vec[:10])
    end_ave = np.mean(vec[-10:])
    return np.abs(start_ave - end_ave) < threshold

def generate_random_parameters():
    tau = random.uniform(*TAU_RANGE)
    Delta = random.uniform(*DELTA_RANGE) 
    OmgVar = random.uniform(*VAR_RANGE) 
    delta_sign = 1 if random.getrandbits(1) else -1
    T1 = random.uniform(*T1_RANGE)
    T2 = random.uniform(*T2_RANGE) 
    return tau, delta_sign*Delta, OmgVar, T1, T2

def generate_bloch(time, Omg_var = 0, tau_fwhm = 50, Delta = 0.01, r0 = [0, 0, -1], T1 = 100, T2 = 100):
    # calculate spread for exciting beam
    tau = tau_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Gaussian std dev
    Omega0 = (np.pi + Omg_var) / (tau * np.sqrt(np.pi))

    # Rabi frequency as a function of time
    def Omega(Omega0, t):
        return Omega0 * np.exp(-(t / tau)**2)

    # Bloch equations (dimensionless)
    def bloch(t, r, Omega0, Delta):
        u, v, w = r
        Om = Omega(Omega0, t)
        du = -Delta * v
        dv = Delta * u - Om * w
        dw = Om * v
        return [du, dv, dw]
    
    def bloch_re(t, r, Omega0, Delta, T1 = 1000, T2 = 200):
        u, v, w = r
        Om = Omega(Omega0, t)
        du = -u / T2 + Delta * v
        dv = -v / T2 - Delta * u + Om * w
        dw = -(w + 1) / T1 - Om * v
        return [du, dv, dw]

    # Solve the system
    sol = solve_ivp(bloch_re, (np.min(time), np.max(time)), r0, args=(Omega0, Delta, T1, T2), t_eval=time)

    return sol.y[0], sol.y[1], sol.y[2]

def generate_dataset(n_samples, name):
    # dir = os.path.join(os.getcwd(), "Datasets", f"{name}", f"{n_samples}switches")
    old_hdd_path = r"D:\DFROStNET_may"
    dir = os.path.join(old_hdd_path, "Datasets", "Switches", f"{name}", f"{n_samples}switches")

    os.makedirs(dir, exist_ok = True)
    counter = 0
    fail_rate = 0

    time = np.linspace(np.min(data_gen.t_s*1e15), np.max(data_gen.t_s*1e15), 4*128)
    #data_gen.t_s*1e15
    while counter <= n_samples:
        tau_vec, Delta_vec, var_vec, T1, T2 = generate_random_parameters()
        u, v, w = generate_bloch(time, var_vec, tau_vec, Delta_vec, [0.0, 0.0, -1.0], T1, T2)
        _, _, complex_impulse = bloch_to_optical(u, v, w)
        # amp = (np.abs(complex_impulse) / np.max(np.abs(complex_impulse))) # / (np.max(np.abs(complex_impulse)) - np.min(np.abs(complex_impulse)))

        amp = (np.abs(complex_impulse) - np.min(np.abs(complex_impulse))) / (np.max(np.abs(complex_impulse)) - np.min(np.abs(complex_impulse)))
        phase = np.unwrap(np.angle(complex_impulse))

        if tf.math.reduce_any(tf.math.is_nan(amp)).numpy() or adequate_contrast(amp):
            # print("failed")
            fail_rate += 1
            continue

        if np.abs(complex_impulse[10]) < np.abs(complex_impulse[-10]):
            complex_impulse = np.flip(complex_impulse)

        counter += 1
        print(counter)
        data_gen.save_switch(complex_impulse, os.path.join(dir, f"{counter}.npz"))

    print(fail_rate / n_samples)

    return os.path.dirname(dir)

def compile_data(directory):
    # walks through subdirs and assembles a list of all the datapoints
    switch_list = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                continue  # skip CSV files
            dataset = np.load(os.path.join(subdir, file))
            switch_list.append(dataset['switch'])
    return switch_list

if __name__ == "__main__":
    data_gen = Data()
    data_gen.generate_switch()

    data_gen.generate_FTL_pulse()
    data_gen.generate_chirped_pulse(phase_coeff = [0, 1, 0, 0, 0])

    list = [pow(2, 14), pow(2, 16)]

    TAU_RANGE = (5, 150)        # 50, 150   # fs
    DELTA_RANGE = (0.0, 0.5) #0.25   # 1/fs
    VAR_RANGE = (0, 1*np.pi)   
    T1_RANGE = (200, 1000)
    T2_RANGE = (200, 500)

    for i in range(len(list)):
        N_SAMPLES = list[i]
        path = generate_dataset(N_SAMPLES, f"0731_{N_SAMPLES}")
        
        switch_list = compile_data(path)
        print(len(switch_list))

        save_path = os.path.join(path, f"{os.path.basename(path)}.npz")
        np.savez_compressed(save_path, switch = switch_list) 
    
        np.savez_compressed(os.path.join(path, "x_axes.npz"), time = np.linspace(np.min(data_gen.t_s*1e15), np.max(data_gen.t_s*1e15), 4*128))
        #, delay = data_gen.switch_t, freq = data_gen.w_Hz)
