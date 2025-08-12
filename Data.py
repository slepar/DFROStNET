import numpy as np
from scipy.constants import c
import warnings
import scipy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


"""Class to generate x axis, pulse, simple integrated switch, and FROSt."""
class Data:
    def __init__(self):
        # initialize time and frequency vectors (and other basic properties)
        self.N_points = 128
        self.t_max = 750 #150
        self.switch_duration_s = 55 * 1e-15
        self.wvl_0 = 1.8 * 1e-6
        self.pulse_duration_s = 10*1e-15 #10 * 1e-15
        self.f0_Hz = c / self.wvl_0
        self._w0 = 2 * np.pi * self.f0_Hz

        self.t_s = np.linspace(start = -self.t_max, stop = self.t_max, num = self.N_points) * 1e-15
        self.dt = abs(self.t_s[1] - self.t_s[0])
        self.w_Hz = 2*np.pi*(np.fft.fftshift(np.fft.fftfreq(len(self.t_s), self.dt)))
        self.dw = abs(self.w_Hz[1] - self.w_Hz[0])
        self.generate_switch()
        self.generate_FTL_pulse()

    @staticmethod
    def model_FS(wvl_um, omega, central_wavelength = 1.8, Length = 0.035):
        model_n = np.sqrt((0.6961663*(wvl_um)**2/((wvl_um)**2-0.0046791482584))+(0.4079426*(wvl_um)**2/((wvl_um)**2-0.01351206307396))+(0.8974794*(wvl_um)**2/((wvl_um)**2-(9.896161)**2)+1))
        idx_W = np.argmin(np.abs(wvl_um - central_wavelength))
        eps = np.finfo(float).eps
        dndlambda = np.diff(np.concatenate(([eps], model_n))) / np.diff(np.concatenate(([eps], wvl_um*1e-6)))
        term1 = model_n * Length * omega / c
        term2 = (Length / c) * (model_n[idx_W] - wvl_um[idx_W]*1e-6 * dndlambda[idx_W]) * (omega - omega[idx_W])
        term3 = model_n[idx_W] * Length * omega[idx_W] / c
        phi = term1 - term2 - term3
        phi = -phi
        phi = phi - np.min(phi)

        return phi, idx_W

    def generate_paired_dispersion_pulse(self, disp_phase):
        self.complex_Ew_field_FS = self.complex_Ew_field * np.exp(1j * disp_phase)
        self.complex_Et_field_FS = self._ifft_(self.complex_Ew_field_FS)

    def generate_switch(self):
        # generate switch envelope 
        sigma = self.switch_duration_s / (2 * np.sqrt(2 * np.log(2)))
        self.__gauss_amplitude_t = (np.exp(-0.5*(self.t_s ** 2 / (sigma ** 2))))
        self.__gauss_amplitude_t /= max(self.__gauss_amplitude_t)
        
        # integrate envelope to get one sided switch
        self.switch_amplitude = np.zeros(len(self.__gauss_amplitude_t))
        for i in range(self.switch_amplitude.size - 1):
            self.switch_amplitude[i + 1] = self.switch_amplitude[i] + self.__gauss_amplitude_t[i]
        self.switch_amplitude /= max(self.switch_amplitude)
        self.switch_amplitude = np.flip(self.switch_amplitude)
        self.switch_spectrum = self._fft_(self.switch_amplitude)

    def generate_FTL_pulse(self): 
        # make time pulse with nul phase
        self.sigma = (self.pulse_duration_s / (2*np.sqrt(2*np.log(2))))
        self.gauss_amplitude_t = (1/(self.sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((self.t_s) / self.sigma) ** 2) 
        self.gauss_amplitude_t /= max(self.gauss_amplitude_t)
        self.complex_Et_ftl = np.sqrt(self.gauss_amplitude_t)
        self.complex_Ew_ftl = self._fft_(self.complex_Et_ftl)
    
    def generate_trace(self, pulse, switch):
        if switch.shape[-1] <= pulse.shape[-1]:
            switch = scipy.ndimage.zoom(switch, (3))
        # return tf.expand_dims(self.interp_trace(self.FROStNET(pulse, switch), 128, 128), axis = -1)    
        return self.interp_trace(self.FROStNET(pulse, switch), 128, 128)   
    
    def generate_chirped_pulse(self, phase_coeff = [0, 0, 0, 0, 0]):
        # generate FTL
        self.phase_coeff = phase_coeff     
        dummy_vector = np.linspace(-self.N_points//2, self.N_points//2, self.N_points) / 10 #/ 15 # 25
        self.phase_w = np.polyval(self.phase_coeff, dummy_vector)
        
        # add random chirp to spectral gaussian pulse #
        self.complex_Ew_field = (self.complex_Ew_ftl * np.exp(1j * self.phase_w))
        self.complex_Et_field = self._ifft_(self.complex_Ew_field)

    def FROStNET(self, pulse, switch):
        # generate FROSt trace from pulse and switch
        switch_shifted = np.lib.stride_tricks.sliding_window_view(switch, window_shape = (len(pulse),), axis = 0)
        product = switch_shifted * pulse  
        product = self._fft_(product)
        spec = np.abs(product.T)**2
        spec /= np.max(spec)
        return spec
    
    """ Functions used to generate variety of phases and validate them."""
    @staticmethod
    def check_ambiguity(vector1, all_vectors):          
        for vector2 in all_vectors:           
            # Check for vertical shift (constant difference)
            differences = vector2 - vector1
            if np.allclose(differences, differences[0]):
                return True
            
            # Check for horizontal shift (cyclic permutation)
            if np.array_equal(np.sort(vector1), np.sort(vector2)):
                return True
            
            # Check for negative conjugate reversal
            if np.array_equal(vector1, -1 * np.flip(vector2)):
                return True
    
        return False
    
    @staticmethod
    def mixed_phase(small_coeff = 5):
        phase_coeff = np.zeros(5, dtype = float)
        phase_coeff[:-1] = np.round(np.random.uniform(-small_coeff, small_coeff, size=len(phase_coeff)-1), 3)
        return phase_coeff

    @staticmethod
    def primary_mixed_phase(degree, small_coeff = 5, big_coeff = 10):
        phase_coeff = np.zeros(5, dtype = float)
        position = len(phase_coeff) - degree - 1 
        phase_coeff[:-1] = np.round(np.random.uniform(-small_coeff, small_coeff, size=len(phase_coeff)-1), 3)
        phase_coeff[position] = np.round(np.random.uniform(-big_coeff, big_coeff, size=1), 3)  
        return phase_coeff

    @staticmethod
    def pure_phase(degree, big_coeff = 10):
        phase_coeff = np.zeros(5, dtype = float)
        position = len(phase_coeff) - degree - 1 
        phase_coeff[position] = np.round(np.random.uniform(-big_coeff, big_coeff, size=1), 3)
        return phase_coeff

    """ Misc functions for various low-level calculations. """
    @staticmethod
    def _fft_(input):
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(input)))

    @staticmethod
    def _ifft_(input):
        return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(input)))
    
    @staticmethod
    def extract_components(complex_vector):
        intensity = np.abs(complex_vector) ** 2 / np.max(np.abs(complex_vector) ** 2)
        phase  = np.unwrap(np.angle(complex_vector))
        return intensity, phase

    @staticmethod
    def interp_trace(trace, new_x = 64, new_y = 3*64):
        new_shape = (new_x / trace.shape[0], new_y / trace.shape[1])
        return scipy.ndimage.zoom(trace, new_shape)
    
    @staticmethod
    def save_trace(trace, path):
        np.savez_compressed(path, trace = trace)

    @staticmethod
    def save_pulse(pulse, path):
        np.savez_compressed(path, pulse = pulse)

    @staticmethod
    def save_switch(switch, path):
        np.savez_compressed(path, switch = switch)

    @staticmethod
    def crop_trace(trace):
        indices = np.where(trace[:, 0] / np.max(trace[:, 0]) >= 0.001)[0]
        return trace[indices, :]
    

if __name__ == "__main__":

    # initialize the Data class
    data_gen = Data()

    # generate a switch
    switch = data_gen.generate_switch()

    # generate a FTL pulse
    ftl_pulse, ftl_spectrum = data_gen.generate_FTL_pulse()

    # TIP: make some plots here to see the FTL pulse / spectrum / phase !
    # plt.plot(data_gen.t_s, np.abs(ftl_pulse))
    # plt.plot(data_gen.w_Hz, np.abs(ftl_spectrum))
    # plt.show()

    # generate a chriped pulse
    chirp_pulse, chirp_spectrum = data_gen.generate_chirped_pulse()
    # TIP: plot here to see the chirped data

    # use the switch and chirped pulse to generate a frost trace
    frost = data_gen.FROStNET(chirp_pulse, switch)
    # TIP: plot here to see the frost trace too
    # plt.imshow(np.abs(frost))
    # plt.show()
