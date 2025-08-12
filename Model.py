import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

class DFROStNET(tf.keras.Model):
    def __init__(self, parameters, points, **kwargs):
        super().__init__(**kwargs)
        self.MR_kernels = parameters[0]
        self.conv_kernel = parameters[1]
        self.dropout = parameters[2]
        self.stddev = parameters[3]
        self.n_layers = parameters[4]
        self.dense_neurons = parameters[5]
        self.input_shape_ = parameters[6]
        self.points = points

        self.noise_layer = tf.keras.layers.GaussianNoise(self.stddev)

        # Build model in __init__
        self.model = self.build_model()

    def create_MR_block(self, input_matrix, filters, MR_kernels, conv_kernel):
        input_shape = input_matrix.shape
        filtered = []
        for k in MR_kernels:
            stride = int((input_shape[1] - k + (2 * k // 2)) / (input_shape[1] - 1))
            x = tf.keras.layers.Conv2D(filters=filters,
                                       kernel_size=(k, k),
                                       strides=(stride, stride),
                                       padding='same',
                                       activation='relu')(input_matrix)
            filtered.append(x)

        output_matrix = tf.keras.layers.Concatenate(axis=-1)(filtered)

        post_stride = (output_matrix.shape[1] - conv_kernel + 2 * (conv_kernel // 2)) // (output_matrix.shape[1] // 2 - 1)
        output_matrix = tf.keras.layers.Conv2D(filters=output_matrix.shape[-1] * 2,
                                               kernel_size=(conv_kernel, conv_kernel),
                                               strides=(post_stride, post_stride),
                                               padding='same',
                                               activation='relu')(output_matrix)
        return output_matrix

    @staticmethod
    def combine_complex(x):
        real, imag = x[0], x[1]
        return tf.complex(real, imag)
    
    @staticmethod
    def normalize(vector):
        amp = tf.abs(vector)
        phase = tf.math.angle(vector)
        amp_max = tf.maximum(tf.reduce_max(amp, axis=-1, keepdims=True), 1e-9)
        amp_norm = amp / amp_max
        normed = tf.complex(amp_norm * tf.cos(phase), amp_norm * tf.sin(phase))
        return normed
    
    def build_model(self):
        input_trace = tf.keras.Input(shape = self.input_shape_)        

        x = self.noise_layer(input_trace)

        x = self.create_MR_block(x, x.shape[-1], self.MR_kernels, self.conv_kernel)
        x = self.create_MR_block(x, x.shape[-1], self.MR_kernels, self.conv_kernel)
        x = self.create_MR_block(x, x.shape[-1], self.MR_kernels, self.conv_kernel)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.Dense(self.dense_neurons, activation='relu')(x)
        x = tf.keras.layers.Dense(self.dense_neurons // 2, activation='relu')(x)

        switch_output = tf.keras.layers.Dense(3*self.points, activation='linear')(x)

        # handling pulse processing
        phase_spectral = tf.keras.layers.Dense(self.points, activation='linear')(x)

        return tf.keras.Model(inputs = input_trace, outputs = [phase_spectral, switch_output], name="DFROStNET")

    def call(self, inputs):
        return self.model(inputs)

class FROStNET(tf.keras.layers.Layer):
    def __init__(self, points, **kwargs):
        super().__init__(**kwargs)
        self.points = points

    @staticmethod
    def resize_complex_vector(vector, target_size):
        # vector: shape (B, N), complex64
        real = tf.expand_dims(tf.math.real(vector), -1)  # (B, N, 1)
        imag = tf.expand_dims(tf.math.imag(vector), -1)  # (B, N, 1)

        # Resize along the "length" dimension
        real_resized = tf.image.resize(real, size=(1, target_size))  # (B, target_size, 1)
        imag_resized = tf.image.resize(imag, size=(1, target_size))  # (B, target_size, 1)

        # Remove channel dim and recombine
        real_resized = tf.squeeze(real_resized, -1)
        imag_resized = tf.squeeze(imag_resized, -1)
        return tf.complex(real_resized, imag_resized)  # (B, target_size)

    @staticmethod
    def _fft_(signal, axis=-1):
        return tf.signal.fftshift(tf.signal.fft(tf.signal.ifftshift(signal, axes=axis)), axes=axis)

    def call(self, inputs):
        pulse, switch = inputs  # pulse: (B, N), switch: (B, N)
        pulse = tf.cast(pulse, tf.complex64)
        switch = tf.cast(switch, tf.complex64)
        B = tf.shape(pulse)[0]
        N = tf.shape(pulse)[1]
        P = self.points

        # Build sliding windows manually using tf.TensorArray
        switch = self.resize_complex_vector(switch, 3*P)
        switch_windows = tf.signal.frame(tf.transpose(switch), frame_length=P, frame_step=1, axis=0)  # (N-P+1, P, B)
        switch_windows = tf.transpose(switch_windows, perm=[2, 0, 1])  # (B, N-P+1, P)
        product = switch_windows * pulse[:, tf.newaxis, :]  # broadcast pulse

        fft_result = self._fft_(product, axis=-1)
        trace = tf.abs(fft_result) ** 2  # (B, T, P)

        # Normalize
        trace /= tf.reduce_max(trace, axis=(1, 2), keepdims=True) #+ 1e-9

        # Reshape and resize to (B, P, P, 1)
        trace = tf.transpose(tf.expand_dims(trace, axis=-1), perm=[0, 2, 1, 3])  # (B, P, T, 1)
        trace = tf.image.resize(trace, size=(P, P))
        return trace

from Data import Data

if __name__ == "__main__":
    # Example parameter list: [MR_kernels, conv_kernel, dropout, stddev, n_layers, dense_neurons, input_shape]
    params = [[5, 5, 7, 3], 3, 0.5, 0.1, 3, 1024, (128, 128, 1)]
    model = DFROStNET(parameters=params, points=128)
    
    # train = Training()
    # train_dataset, train_size = train.batch_dataset(train.training_set, 32)
    # pulse_batch, switch_batch, trace_batch = train_dataset[0]

    # # amp = model.extract_spectral_amplitude(trace)
    # # print(trace.shape, amp.shape)

    # output_pulse, output_switch = model(trace_batch)

    # plt.plot(np.abs(output_switch[0]))
    # plt.plot(np.abs(output_pulse[0]))
    # plt.show()