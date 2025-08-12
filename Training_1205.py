import scipy.ndimage
import scipy.signal
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import scipy
from scipy.interpolate import interp1d

class Training():
    def __init__(self, points = 128, duration = 10):
        self.points = points
        self.switch_n = points*3
        self.duration = duration

        if self.points == 64:
            self.training_set = os.path.join(os.getcwd(), 'Datasets', '64pts_10fs', '64pts_10fs_65536pulses_tf')     
            self.validation_set = os.path.join(os.getcwd(), 'Datasets', '64pts_10fs', '64pts_10fs_16384pulses_tf')     
            self.axes = np.load(os.path.join(os.getcwd(), 'Datasets', '64pts_10fs', 'x_axes.npz'))

        # elif self.points == 128:
        self.training_set = os.path.join(os.getcwd(), 'Datasets', '128pts_10fs', '128pts_10fs_65536pulses_tf')     
        
        # self.training_set = os.path.join(os.getcwd(), 'Datasets', '128pts_10fs', '128pts_10fs_16384pulses_tf') 

        self.validation_set = os.path.join(os.getcwd(), 'Datasets', '128pts_10fs', '128pts_10fs_16384pulses_tf') 
        self.axes = np.load(os.path.join(os.getcwd(), 'Datasets', '128pts_10fs', 'x_axes.npz'))

        self.training_set = r"D:\DFROStNET_may\Datasets\3007_65537_tf"    
        self.validation_set = r"D:\DFROStNET_may\Datasets\3007_16385_tf"

        
    def get_dataset_size(self, path):
        dataset = tf.data.Dataset.load(path)
        return len(dataset)
   
    @staticmethod
    def create_MR_block(input_matrix, filters, MR_kernels, conv_kernel):
        input_shape = input_matrix.shape

        # MR convolutions
        filtered = []
        for i in range(len(MR_kernels)):
            stride = (input_shape[1] - MR_kernels[i] + (2*MR_kernels[i]//2)) // (input_shape[1] - 1)
            x = tf.keras.layers.Conv2D(filters = filters, 
                                    kernel_size = (MR_kernels[i], MR_kernels[i]), 
                                    strides = (stride, stride), 
                                    padding = 'same', activation = 'relu')(input_matrix)
            filtered.append(x)
        output_matrix = tf.keras.layers.Concatenate(axis = -1)(filtered)

        # post MR convolution
        stride = (output_matrix.shape[1] - conv_kernel + 2*(conv_kernel//2)) // (output_matrix.shape[1]//2 - 1)
        output_matrix = tf.keras.layers.Conv2D(filters = output_matrix.shape[-1] * 2, 
                                            kernel_size = (conv_kernel, conv_kernel), 
                                            strides = (stride, stride), 
                                            padding = 'same', activation = 'relu')(output_matrix)
        return output_matrix
    
    def create_MR_model(self, parameters):
        MR_kernels = parameters[0]
        conv_kernel = parameters[1]
        dropout = parameters[2]
        stddev = parameters[3]
        n_layers = parameters[4]
        dense_neurons = parameters[5] 
        self.input_shape = parameters[6]
        self.n_outputs = parameters[-1]
        inputs = tf.keras.Input(shape = self.input_shape)

        wgn_layer = tf.keras.layers.GaussianNoise(stddev)(inputs)

        # multi res blocks
        multires_output1 = self.create_MR_block(wgn_layer, self.input_shape[-1], MR_kernels, conv_kernel)
        multires_output2 = self.create_MR_block(multires_output1, multires_output1.shape[-1], MR_kernels, conv_kernel)
        multires_output3 = self.create_MR_block(multires_output2, multires_output2.shape[-1], MR_kernels, conv_kernel)
        
        # flatten, drop
        x = tf.keras.layers.Flatten()(multires_output3)
        x = tf.keras.layers.Dropout(dropout)(x)

        # dense and output layers
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(dense_neurons, activation = 'relu')(x)

        phi_output = tf.keras.layers.Dense(self.points, activation = 'linear')(x)

        if self.n_outputs > 1:
            switch_output = tf.keras.layers.Dense(self.switch_n, activation = 'linear')(x)
            model = tf.keras.Model(inputs = inputs, outputs = [phi_output, switch_output], name = "DFROStNET")
        else:
            model = tf.keras.Model(inputs = inputs, outputs = phi_output, name = "DFROStNET")

        return model
    
    @staticmethod
    def _fft_(pulse, axis = 1):
        return tf.signal.fftshift(tf.signal.fft(tf.signal.ifftshift(pulse, axes = axis)), axes = axis)
    
    @staticmethod
    def _ifft_(spectrum, axis = 1):
        return tf.signal.ifftshift(tf.signal.ifft(tf.signal.fftshift(spectrum, axes = axis)), axes = axis)
    
    @staticmethod
    def extract_components(complex_tensor):
        amplitude = tf.abs(complex_tensor)
        amplitude /= tf.reduce_max(amplitude, axis = 1, keepdims = True)
        phase = tf.math.angle(complex_tensor)
        return amplitude, phase
    
    @staticmethod
    def recombine_complex_vector(amplitude, phase):
        amplitude_complex = tf.cast(amplitude, tf.complex64)
        phase_complex = tf.complex(tf.zeros_like(phase), phase)
        return amplitude_complex * tf.exp(phase_complex)

    @staticmethod
    def normalize_switch(tensor):
        tensor_min = tf.reduce_min(tensor, axis = 1, keepdims = True)
        tensor_max = tf.reduce_max(tensor, axis = 1, keepdims = True)
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return normalized_tensor

    @staticmethod
    def extract_spectral_amp(trace):
        diff = np.sqrt(np.abs(trace[:, :, 0, 0])) #- trace[:, :, -1, 0]))
        diff = scipy.ndimage.gaussian_filter1d(diff, 1, axis = 1)
        spectrum = diff / np.max(diff, axis = 1, keepdims=True)
        return spectrum 
    
    def prep_labels(self, batch):
        pulse, switch, trace = batch
        switch = np.abs(switch)
        pulse /= np.max(np.abs(pulse), axis = 1, keepdims = True)
        spectrum = self.extract_spectral_amp(trace)
        return pulse, switch, trace, spectrum

    def return_prediction(self, amp, phase, xp = False):
        spectrum = self.recombine_complex_vector(amp, phase) 
        spectrum /= np.max(spectrum, axis = 1, keepdims = True)
        pulse = self._ifft_(spectrum)
        # if xp is True:
        #     background = tf.reduce_mean((tf.reduce_mean(pulse[:, :10], axis = -1), tf.reduce_mean(pulse[:, -10:], axis = -1)), axis = 0)
        #     pulse -= background[:, np.newaxis]
        #     pulse *= np.hanning(self.points)[np.newaxis, :]
        pulse /= np.max(np.abs(pulse), axis = 1, keepdims = True)
        return pulse

    def setup_callbacks(self, model, save_path, batch_size, epochs):
        es_cb1 = tf.keras.callbacks.EarlyStopping(monitor = 'val_pulse_loss', patience = 2, restore_best_weights = True, mode = 'min')
        es_cb2 = tf.keras.callbacks.EarlyStopping(monitor = 'val_switch_loss', patience = 2, restore_best_weights = True, mode = 'min')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = save_path, save_weights_only = True, verbose = 0)
        history = tf.keras.callbacks.History()
        callbacks = tf.keras.callbacks.CallbackList([history, cp_callback, es_cb1, es_cb2])
        callbacks.set_model(model)
        callbacks.set_params({'batch_size': batch_size, 'epochs': epochs, 'steps': len(tf.data.Dataset.load(self.training_set)) // batch_size, 
                              'verbose': 1, 'do_validation': True, 
                              'metrics': ['mae', 'mae']})
        return es_cb1, es_cb2, callbacks
    
    def training_loop(self, model, save_path, batch_size = 32, epochs = 1, hybrid_rate = 0, lrn_rate = 0.001):
        optimizer = tf.keras.optimizers.Adam(lrn_rate)
        loss_fn = tf.keras.losses.MeanAbsoluteError()

        train_dataset, train_size = self.batch_dataset(self.training_set, batch_size)
        val_dataset, val_size = self.batch_dataset(self.validation_set, batch_size)
        self.input_shape = (self.points, self.points, 1)

        # training loop starts here
        cb1, cb2, callbacks = self.setup_callbacks(model, save_path, batch_size, epochs)
        callbacks.on_train_begin()   
        for epoch in range(epochs):
            print(f'\n Epoch {epoch + 1}/{epochs}')
            callbacks.on_epoch_begin(epoch)
            progbar = tf.keras.utils.Progbar(target = train_size // batch_size, verbose = 1)
            for i, batch in enumerate(train_dataset):
                callbacks.on_train_batch_begin(i // batch_size)               
                pulse_batch, switch_batch, trace_batch, amp_batch = self.prep_labels(batch)
                with tf.GradientTape(persistent = True) as tape:
                    ai_phase_w, ai_switch = model(trace_batch, training = True)  
                    ai_switch = self.normalize_switch(ai_switch)
                    ai_pulse = self.return_prediction(amp_batch, ai_phase_w)
                    pulse_loss = loss_fn(pulse_batch, ai_pulse)
                    switch_loss = loss_fn(switch_batch, ai_switch)
                    trace_loss = tf.math.abs((pulse_loss*switch_loss)**2)  
                    total_loss = (pulse_loss + switch_loss) + hybrid_rate*trace_loss      

                    if tf.math.reduce_any(tf.math.is_nan(total_loss)):
                        print(pulse_loss.numpy())
                        print(switch_loss.numpy())

                        plt.subplot(221)
                        plt.plot(np.abs(pulse_batch[0]))
                        plt.plot(np.abs(ai_pulse[0]))
                        plt.plot(np.abs(amp_batch[0]))

                        plt.subplot(222)
                        plt.plot(np.angle(ai_pulse[0]))
                        plt.plot(np.angle(pulse_batch[0]))

                        plt.subplot(223)
                        plt.plot(np.abs(ai_switch[0]))
                        plt.plot(np.abs(switch_batch[0]))

                        plt.subplot(224)
                        plt.imshow(np.abs(trace_batch[0]), aspect = "auto")
                        plt.show()

                grads = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                batch_logs = {'batch': i, 'size': train_size, 'pulse_loss': pulse_loss.numpy(), 'switch_loss': switch_loss.numpy(), 'trace_loss': trace_loss}

                progbar.update(i, values = [('pulse_loss', pulse_loss.numpy()), ('switch_loss', switch_loss.numpy()), ('trace_loss', trace_loss)])
                callbacks.on_train_batch_end(i, batch_logs)

            # validation loop
            val_logs = {'val_pulse_loss': 0, 'val_switch_loss': 0, 'val_trace_loss': 0}
            progbar = tf.keras.utils.Progbar(target = val_size // batch_size, verbose = 1)
            for i, batch in enumerate(val_dataset):
                pulse_batch, switch_batch, trace_batch, amp_batch = self.prep_labels(batch)
                ai_phase_w, ai_switch = model(trace_batch, training = False)  
                ai_pulse = self.return_prediction(amp_batch, ai_phase_w)
                ai_switch = self.normalize_switch(ai_switch)
                # ai_switch = self.apply_ptychographic_constraint(self.axes['time'], self.axes['delay'], trace_batch, ai_pulse, switch_batch)
                switch_loss = loss_fn(switch_batch, ai_switch)
                pulse_loss = loss_fn(pulse_batch, ai_pulse)
                trace_loss = tf.math.abs((pulse_loss*switch_loss)**2)  

                val_logs['val_pulse_loss'] += pulse_loss.numpy()
                val_logs['val_switch_loss'] += switch_loss.numpy()
                val_logs['val_trace_loss'] += trace_loss.numpy()
                
                progbar.update(i, values = [('val_pulse_loss', pulse_loss.numpy()), ('val_switch_loss', switch_loss.numpy()), ('val_trace_loss', trace_loss.numpy())])

            val_logs['val_pulse_loss'] /= val_size
            val_logs['val_switch_loss'] /= val_size
            val_logs['val_trace_loss'] /= val_size

            callbacks.on_epoch_end(epoch, val_logs)
            if (cb1.stopped_epoch or cb2.stopped_epoch) > 0:
                break
        callbacks.on_train_end()
        model.save_weights(save_path)

        print(f"\nTraining loop completed and model saved to {save_path}.")

    @staticmethod
    def batch_dataset(path, batch_size):
        dataset = tf.data.Dataset.load(path)
        size = len(dataset)
        dataset = (dataset.shuffle(buffer_size = size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))
        return dataset, size

    def load_model_weights(self, model_path, parameters):
        model = self.create_MR_model(parameters)
        model.load_weights(model_path)
        return model

    def FROStNET(self, pulse, switch):
        points = pulse.shape[-1]
        switch_shifted = np.lib.stride_tricks.sliding_window_view(switch, window_shape = (points, ) , axis = 1)
        product = switch_shifted * pulse[:, np.newaxis, :]    
        trace = np.abs(self._fft_(product, axis = -1))**2
        trace = trace.reshape(trace.shape[0], trace.shape[1], trace.shape[2], -1)
        trace /= np.max(trace, axis = (1, 2), keepdims = True)
        trace = tf.transpose(trace, perm=[0, 2, 1, 3])
        trace = tf.image.resize(trace, (points, points))
        return trace

    @staticmethod
    def trace_error(ai_trace, trace_batch):
        temp = tf.cast(ai_trace, dtype = trace_batch.dtype)
        num = np.sum(trace_batch*temp, axis = (1, 2, 3))**2
        denum = np.sum(trace_batch**2, axis = (1, 2, 3))*np.sum(temp**2, axis = (1, 2, 3))
        spec_loss = np.sqrt(1 - num/denum)
        return spec_loss
    
    @staticmethod
    def vector_error(ai, batch):
        temp = tf.cast(ai, dtype = batch.dtype)
        loss = tf.reduce_mean(tf.abs(batch - temp), axis = 1) 
        return loss

    @staticmethod
    def load_raw_data(path):
        raw_data = np.loadtxt(path)
        raw_trace = raw_data[2:, 2:] / np.max(raw_data[2:, 2:])
        raw_delay, raw_wvl = raw_data[0, 2:], raw_data[2:, 0]
        return raw_trace, raw_delay, raw_wvl

    @staticmethod
    def interpolate_batch_traces(batch, size):
        return tf.image.resize(batch, size, method='bilinear')
    
    def post_sim_val(self, model, save_file):
        val_dataset, val_size = self.batch_dataset(self.validation_set, 32)
        fig_path = os.path.join(os.path.dirname(save_file), "Sim_Figures")
        os.makedirs(fig_path, exist_ok=True)

        for i, batch in enumerate(val_dataset):
            pulse_batch, switch_batch, trace_batch, amp_batch = self.prep_labels(batch)
            ai_phase_w, ai_switch = model.predict(trace_batch, verbose = 0)
            ai_pulse = self.return_prediction(amp_batch, ai_phase_w)
            ai_switch = self.normalize_switch(ai_switch)
            
            # ai_switch = self.apply_ptychographic_constraint(self.axes['time'], self.axes['delay'], trace_batch, ai_pulse, switch_batch)
            ai_trace = self.FROStNET(ai_pulse, ai_switch)

            # get errors
            spec_loss = self.trace_error(ai_trace, trace_batch) 
            switch_loss = self.vector_error(ai_switch, switch_batch)  
            pulse_loss = self.vector_error(ai_pulse, pulse_batch)  
            stats = np.column_stack((switch_loss.numpy(),  
                                    pulse_loss.numpy(),
                                    spec_loss))
            with open(save_file, "a") as f:
                np.savetxt(f, stats, delimiter=",")

            if i % 50 == 0:
                batch_amp_t, batch_phase_t = self.extract_components(pulse_batch)
                ai_amp_t, ai_phase_t = self.extract_components(ai_pulse)

                plt.subplot(231)
                plt.imshow(np.abs(trace_batch[0, :, :, 0]), aspect = 'auto') #,
                plt.subplot(233)
                plt.imshow(np.abs(ai_trace[0, :, :, 0]), aspect = 'auto') #,
                plt.subplot(234)
                plt.plot(np.abs(switch_batch[0]))
                plt.plot(np.abs(ai_switch[0]))
                plt.subplot(235)
                plt.plot(np.abs(batch_amp_t[0]))
                plt.plot(np.abs(ai_amp_t[0]))
                plt.subplot(236)
                plt.plot(np.abs(batch_phase_t[0]))
                plt.plot(np.abs(ai_phase_t[0]))
                plt.savefig(os.path.join(fig_path, f'{i}.png'))
                plt.tight_layout()
                plt.close()

    def compare_vectors(self, pty_x, pty_y, ai_x, ai_y):
        if pty_x.shape != pty_y.shape[-1]:
            pty_x = self.interp_vector(pty_x, pty_y.shape[-1])
        if ai_x.shape != ai_y.shape[-1]:
            ai_x = self.interp_vector(ai_x, ai_y.shape[-1])

        new_x, new_ai_y, _, new_pty_y = self.clip_vector(ai_x, ai_y[0], pty_x, pty_y[0])
        new_pty_y = self.interp_vector(new_pty_y, new_ai_y.shape[-1]).reshape(1, -1)
        new_ai_y = tf.reshape(new_ai_y, (1, -1))

        loss = self.vector_error(new_ai_y, new_pty_y)  
        return loss, new_x, new_pty_y, new_ai_y
    
    def compare_traces(self, pty_x, pty_y, pty_z, ai_x, ai_y, ai_z):
        pty_z = self.interp_vector(pty_z, [self.points, self.points])
        if pty_x.shape != pty_z.shape[0]:
            pty_x = self.interp_vector(pty_x, pty_z.shape[0])
        if pty_y.shape != pty_z.shape[1]:
            pty_y = self.interp_vector(pty_y, pty_z.shape[1])
        if ai_x.shape != ai_z.shape[0]:
            ai_x = self.interp_vector(ai_x, ai_z.shape[0])
        if ai_y.shape != ai_z.shape[1]:
            ai_y = self.interp_vector(ai_y, ai_z.shape[1])
        
        new_x, new_y, new_ai_z, _, _, new_pty_z = self.clip_matrix(ai_x, ai_y, ai_z, pty_x, pty_y, pty_z)
        new_ai_z = self.interp_vector(new_ai_z, [self.points, self.points])
        new_pty_z = self.interp_vector(new_pty_z, [self.points, self.points])
        spec_loss = self.trace_error(tf.reshape(new_ai_z, (1, self.points, self.points, 1)), new_pty_z.reshape(1, self.points, self.points, 1)) 
        return spec_loss, new_x, new_y, new_pty_z, new_ai_z
    
    def post_xp_val(self, model, save_file):
        dir = os.path.join(os.getcwd(), "EZ-access_XP-recovered")
        fig_path = os.path.join(os.path.dirname(save_file), "XP_Figures")
        os.makedirs(fig_path, exist_ok=True)

        for ind_file in os.listdir(dir):
            pty_path = os.path.join(dir, ind_file)
            with open(pty_path, 'rb') as file:
                data = pickle.load(file)
            pty_pulse = data['recovered_field_t'] / np.max(np.abs(data['recovered_field_t']))
            pty_time = data['processed_time']
            pty_trace = data['recovered_trace']
            pty_delay = data['processed_delay']
            pty_wvl = data['processed_omega']
            raw_trace = data['raw_trace'][:, :, 0]
            raw_wvl = data['raw_omega']
            raw_delay = data['raw_delay']
            raw_time, raw_freq = self.gen_f_t_vector(raw_wvl)
            pty_switch = self.normalize_switch(np.abs(data['recovered_switch']).reshape(1, -1))

            # interpolate raw data trace
            interp_trace = self.interp_vector(raw_trace, [self.points, self.points]).reshape(1, self.points, self.points, -1)
            amp_w = self.extract_spectral_amp(interp_trace).reshape(1, self.points)

            # generate prediction
            ai_phase_w, ai_switch = model.predict(interp_trace, verbose = 0)
            ai_pulse = self.return_prediction(amp_w, ai_phase_w, True)
            ai_switch = self.apply_ptychographic_constraint(raw_time, raw_delay, interp_trace, ai_pulse)
            if ai_switch.shape[-1] < self.switch_n:
                ai_switch = self.interp_vector(ai_switch[0], self.switch_n).reshape(1, self.switch_n)
            ai_switch = self.normalize_switch(scipy.ndimage.gaussian_filter1d(ai_switch, sigma = 15))
            ai_trace = self.FROStNET(ai_pulse, ai_switch)

            # # errors 
            xp_spec_loss, _, _, _, _ = self.compare_traces(raw_delay, raw_wvl, interp_trace[0, :, :, 0], raw_delay, raw_wvl, ai_trace[0, :, :, 0])
            switch_loss, new_delay, pty_switch, ai_switch = self.compare_vectors(pty_delay, pty_switch, raw_delay, ai_switch)
            pulse_loss, new_time, pty_pulse, ai_pulse = self.compare_vectors(pty_time, pty_pulse, raw_time, ai_pulse)
            spec_loss, trace_delay, trace_wvl, pty_trace, _ = self.compare_traces(pty_delay, pty_wvl, pty_trace, raw_delay, raw_wvl, ai_trace[0, :, :, 0])

            # stack and save
            stats = np.column_stack((switch_loss.numpy(), pulse_loss.numpy(), spec_loss, xp_spec_loss))
            if not tf.math.is_nan(switch_loss):
                with open(save_file, "a") as f:
                    np.savetxt(f, stats, delimiter=",")

            # plotting
            pty_amp_t, pty_phase_t = self.extract_components(pty_pulse.reshape(1, -1))
            ai_amp_t, ai_phase_t = self.extract_components(tf.reshape(ai_pulse, (1, -1)))

            plt.subplot(231)
            plt.imshow(np.abs(interp_trace[0, :, :, 0]), aspect = 'auto',
                    extent = (np.min(raw_delay), np.max(raw_delay), raw_wvl[0], raw_wvl[-1]))
            plt.subplot(232)
            plt.imshow(np.abs(pty_trace), aspect = 'auto',
                    extent = (trace_delay[0], trace_delay[-1], trace_wvl[0], trace_wvl[-1]))
            plt.subplot(233)
            plt.imshow(np.abs(ai_trace[0, :, :, 0]), aspect = 'auto',
                    extent = (np.min(raw_delay), np.max(raw_delay), raw_wvl[0], raw_wvl[-1]))
                    # extent = (trace_delay[0], trace_delay[-1], trace_wvl[0], trace_wvl[-1]))
            plt.subplot(234)
            plt.plot(new_delay, np.abs(pty_switch[0]))
            plt.plot(new_delay, np.abs(ai_switch[0]))
            plt.subplot(235)
            plt.plot(new_time, np.abs(pty_amp_t[0]))
            plt.plot(new_time, np.abs(ai_amp_t[0]))
            plt.subplot(236)
            plt.plot(new_time, np.abs(pty_phase_t[0]))
            plt.plot(new_time, np.abs(ai_phase_t[0]))
            plt.savefig(os.path.join(fig_path, os.path.splitext(os.path.basename(ind_file))[0]))
            plt.tight_layout()
            plt.close()

    @staticmethod
    def clip_vector(x1, y1, x2, y2):
        common_min = max(min(x1), min(x2))
        common_max = min(max(x1), max(x2))
        mask1 = (x1 >= common_min) & (x1 <= common_max)
        mask2 = (x2 >= common_min) & (x2 <= common_max)
        x1_clipped, y1_clipped = x1[mask1], y1[mask1]
        x2_clipped, y2_clipped = x2[mask2], y2[mask2]
        return x1_clipped, y1_clipped, x2_clipped, y2_clipped
    
    @staticmethod
    def clip_matrix(x1, y1, m1, x2, y2, m2):
        common_x_min = max(min(x1), min(x2))
        common_x_max = min(max(x1), max(x2))
        common_y_min = max(min(y1), min(y2))
        common_y_max = min(max(y1), max(y2))

        mask1_x = (x1 >= common_x_min) & (x1 <= common_x_max)
        mask2_x = (x2 >= common_x_min) & (x2 <= common_x_max)
        mask1_y = (y1 >= common_y_min) & (y1 <= common_y_max)
        mask2_y = (y2 >= common_y_min) & (y2 <= common_y_max)

        y1_clipped = tf.boolean_mask(y1, mask1_y)
        x1_clipped = tf.boolean_mask(x1, mask1_x)
        m1_rows_clipped = tf.boolean_mask(m1, mask1_x, axis=0)  # Clip rows
        m1_clipped = tf.boolean_mask(m1_rows_clipped, mask1_y, axis=1)  # Clip columns

        y2_clipped = tf.boolean_mask(y2, mask2_y)
        x2_clipped = tf.boolean_mask(x2, mask2_x)
        m2_rows_clipped = tf.boolean_mask(m2, mask2_x, axis=0)  # Clip rows
        m2_clipped = tf.boolean_mask(m2_rows_clipped, mask2_y, axis=1)  # Clip columns

        return x1_clipped, y1_clipped, m1_clipped, x2_clipped, y2_clipped, m2_clipped

    @staticmethod
    def interp_vector(vector, new_len):
        new_shape = (new_len / vector.shape[0]) if vector.ndim == 1 else (new_len[0] / vector.shape[0], new_len[1] / vector.shape[1])
        return scipy.ndimage.zoom(vector, new_shape)

    @staticmethod
    def save_parameters(file_path, parameters):
        with open(file_path, "w") as file:
            file.write(str(parameters))  # Convert the list to a string and write it to the file

    @staticmethod
    def gen_f_t_vector(wavelength):
        freq = (3e8 / wavelength * 1e-9)        # wvl should be in nm from the spectrometer
        sorted_indices = np.argsort(freq)
        temp_freq = freq[sorted_indices]
        freq = np.linspace(temp_freq[0], temp_freq[-1], len(temp_freq))
        df = (freq[9] - freq[8])
        Dt = 1 / (len(freq)*df) 
        time = np.arange(-len(freq)//2, len(freq)//2) * Dt
        return time, freq


    @staticmethod
    def new_vector(vector1, vector2):
        min_value = max(vector1[0], vector2[0])
        max_value = min(vector1[-1], vector2[-1])
        step = min(np.abs(vector1[1] - vector1[0]), np.abs(vector2[1] - vector2[0]))
        return np.arange(min_value, max_value + step/2, step)

    @staticmethod
    def crop_trace_intensity(y, trace, threshold = 0.005):
        mask = np.abs(trace) >= threshold  # or just (trace >= 0.01) if no negatives
        row_min, row_max = np.where(np.any(mask, axis=1))[0][[0, -1]]
        trace = trace[row_min:row_max+1, :] # col_min:col_max+1]
        y = y[row_min:row_max+1]
        return y, trace

    def new_post_xp_val(self, model, save_file):
        dir = os.path.join(os.getcwd(), "EZ_Access")
        fig_path = os.path.join(os.path.dirname(save_file), "XP_Figures")
        os.makedirs(fig_path, exist_ok=True)

        for ind_file in os.listdir(dir):
            pty_path = os.path.join(dir, ind_file)
            with open(pty_path, 'rb') as file:
                data = pickle.load(file)
            
            pty_pulse = data['Chp_recons'] / np.max(data['Chp_recons'])
            pty_time = data['V_t'][0]
            pty_trace = data['Trace_Int_Recons'].T
            pty_trace /= np.max(np.abs(pty_trace))
            pty_delay = data['V_tdelais'][0]
            pty_omg = data['V_omg'][0]
            raw_trace = data['trace'][2:, 2:] / np.max(data['trace'][2:, 2:])
            if np.max(raw_trace[:, 10]) <= np.max(raw_trace[:, -10]):
                raw_trace = np.flip(raw_trace, axis = 1)

            if np.max(pty_trace[:, 10]) <= np.max(pty_trace[:, -10]):
                pty_trace = np.flip(pty_trace, axis = 1)

            raw_wvl = data['wvl']
            raw_delay = data['delay']
            raw_time, raw_freq = self.gen_f_t_vector(raw_wvl*1e-3)
            pty_switch = data['Obj_recons'] / np.max(data['Obj_recons'])

            if pty_switch[0, 10] <= pty_switch[0, -10]:
                pty_switch = np.flip(pty_switch, axis = 1)
            pty_spectrum = self._fft_(pty_pulse)

            # # convert    to wvl
            raw_wvl_crop, raw_trace_crop = self.crop_trace_intensity(raw_wvl, raw_trace)
            pty_omg_crop, pty_trace_crop = self.crop_trace_intensity(pty_omg, pty_trace)
            pty_wvl = np.flip(2*np.pi*3e8 / pty_omg_crop)*1e9
            new_married_wvl = self.new_vector(raw_wvl_crop, pty_wvl)
            raw_trace = interp1d(raw_wvl_crop, raw_trace_crop, kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')(new_married_wvl)
            pty_trace = interp1d(pty_wvl, np.flip(pty_trace_crop, axis = 0), kind='cubic', axis=0, bounds_error=False, fill_value='extrapolate')(new_married_wvl)

            # interpolate raw data trace
            interp_trace = self.interp_vector(raw_trace, [self.points, self.points]).reshape(1, self.points, self.points, -1)
            amp_w = self.extract_spectral_amp(interp_trace).reshape(1, self.points)

            # generate prediction
            ai_phase_w, ai_switch = model.predict(interp_trace, verbose = 0)
            ai_pulse = self.return_prediction(amp_w, ai_phase_w)
            ai_switch = self.normalize_switch(ai_switch)
            pty_switch = tf.math.abs(pty_switch)
            pty_pulse /= np.max(np.abs(pty_pulse), axis = 1, keepdims = True)
            ai_trace = self.FROStNET(ai_pulse, ai_switch)

            # # errors 
            xp_spec_loss, _, _, _, _ = self.compare_traces(raw_delay, new_married_wvl, interp_trace[0, :, :, 0], raw_delay, new_married_wvl, ai_trace[0, :, :, 0])
            switch_loss, new_delay, pty_switch, ai_switch = self.compare_vectors(pty_delay, pty_switch, raw_delay, ai_switch)
            pulse_loss, new_time, pty_pulse, ai_pulse = self.compare_vectors(pty_time, pty_pulse, raw_time, ai_pulse)
            spec_loss, trace_delay, trace_wvl, pty_trace, _ = self.compare_traces(pty_delay, new_married_wvl, pty_trace, raw_delay, new_married_wvl, ai_trace[0, :, :, 0])
            ai_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ai_pulse[0])))
            ai_spectrum /= np.max(ai_spectrum)

            # stack and save
            stats = np.column_stack((switch_loss.numpy(), pulse_loss.numpy(), spec_loss, xp_spec_loss))
            if not tf.math.is_nan(switch_loss):
                with open(save_file, "a") as f:
                    np.savetxt(f, stats, delimiter=",")

            # plotting
            pty_amp_t, pty_phase_t = self.extract_components(pty_pulse.reshape(1, -1))
            ai_amp_t, ai_phase_t = self.extract_components(tf.reshape(ai_pulse, (1, -1)))

            plt.subplot(231)
            plt.imshow(np.abs(interp_trace[0, :, :, 0]), aspect = 'auto',
                    extent = (np.min(raw_delay), np.max(raw_delay), new_married_wvl[0], new_married_wvl[-1]))
            plt.subplot(232)
            plt.imshow(np.abs(pty_trace), aspect = 'auto',
                    extent = (trace_delay[0], trace_delay[-1], new_married_wvl[0], new_married_wvl[-1]))
            plt.subplot(233)
            plt.imshow(np.abs(ai_trace[0, :, :, 0]), aspect = 'auto',
                    extent = (np.min(raw_delay), np.max(raw_delay), new_married_wvl[0], new_married_wvl[-1]))
            plt.subplot(234)
            plt.plot(new_delay, np.abs(pty_switch[0]) / np.max(np.abs(pty_switch[0])))
            plt.plot(new_delay, np.abs(ai_switch[0]) / np.max(np.abs(ai_switch[0])))
            ax = plt.twinx()
            ax.plot(new_delay, np.unwrap(np.angle(pty_switch[0])), linestyle = "--")
            ax.plot(new_delay, np.unwrap(np.angle(ai_switch[0])), linestyle = "--")

            plt.subplot(235)
            plt.plot(new_time, np.abs(pty_amp_t[0]))
            plt.plot(new_time, np.abs(ai_amp_t[0]))
            ax = plt.twinx()
            ax.plot(new_time, np.unwrap(pty_phase_t[0]), linestyle = "--")
            ax.plot(new_time, np.unwrap(ai_phase_t[0]), linestyle = "--")

            _, pty_phase_w = self.extract_components(tf.reshape(np.flip(pty_spectrum), (1, -1)))
            _, ai_phase_w = self.extract_components(tf.reshape(np.flip(ai_spectrum), (1, -1)))
            dummy_wvl = np.linspace(new_married_wvl[0], new_married_wvl[-1], 128)
            
            plt.subplot(236)
            plt.plot(dummy_wvl, np.abs(interp_trace[0, :, 0, 0]) / np.max(interp_trace[0, :, 0, 0]), color = "black")
            plt.plot(dummy_wvl, np.abs(pty_trace[:, 0]) / np.max(np.abs(pty_trace[:, 0])))
            plt.plot(dummy_wvl, np.abs(ai_trace[0, :, 0]) / np.max(ai_trace[0, :, 0]))

            ax = plt.twinx()
            ax.plot(np.linspace(new_married_wvl[0], new_married_wvl[-1], len(np.abs(pty_phase_w[0]))), np.unwrap(pty_phase_w[0]), linestyle = "--")
            ax.plot(np.linspace(new_married_wvl[0], new_married_wvl[-1], len(np.abs(ai_phase_w[0]))), np.unwrap(ai_phase_w[0]), linestyle = "--")

            plt.savefig(os.path.join(fig_path, os.path.splitext(os.path.basename(ind_file))[0]))
            plt.tight_layout()
            plt.close()

    @staticmethod
    def batchwise_2D_ifft_(trace):
        dummy_trace = tf.cast(trace[:, :, :, 0], tf.complex64)
        dummy_trace = tf.transpose(tf.signal.fftshift(dummy_trace, axes = 1), perm = [0, 2, 1])  # Swap -2 (axis 2) with the last axis
        trace_t = tf.signal.ifft(dummy_trace)
        trace_t = tf.signal.ifftshift(tf.transpose(trace_t, perm = [0, 2, 1]), axes = 1)
        trace_t = tf.expand_dims(trace_t, axis = -1)
        return trace_t
       
    def apply_ptychographic_constraint(self, time, delay, trace, pulse, true_switch = None):
        og_shape = len(delay)
        new_shape = [1051, 1501] #[1051, 1501]
        delay = self.interp_vector(delay, new_shape[0])
        time = self.interp_vector(time, new_shape[1])
        trace = self.interpolate_batch(trace, new_shape[1], new_shape[0])

        pulse = tf.complex(self.interpolate_batch(tf.math.real(pulse), 0, new_shape[1]), self.interpolate_batch(tf.math.imag(pulse), 0, new_shape[1]))
        
        N, N0 = len(time), len(delay)
        dt, dt0 = time[1] - time[0], delay[1] - delay[0]

        # # step 1 get e field matrix from trace
        trace_t = self.batchwise_2D_ifft_(trace)

        # step 2 create shifted e field matrix
        M_decalage_t0 = self.fc_decalage_temporelle_batch_tf(N0, N, dt0, dt, pulse)

        # step 3 somme sur decales
        M_decalage_t0_sum = self.fc_somme_sur_les_decales_batch_tf(N0, N, dt0, dt, M_decalage_t0)

        # # step 4 equation
        pred_switch = self.sum_colonne_normale_batch(M_decalage_t0_sum, trace_t, N0, N)

        # interpolate back down to # of points
        pred_switch = self.interpolate_batch(pred_switch, 0, og_shape)

        # uses label to determine phase shift if simulation data
        if true_switch is not None:
            pred_switch = self.roll_and_pad(pred_switch, true_switch)

        # pred_switch = np.abs(pred_switch)**(1/2) / np.max(np.abs(pred_switch)**(1/2))

        pred_switch = self.normalize_switch(np.abs(pred_switch))

        return pred_switch

    @staticmethod
    def roll_and_pad(labels, recovereds):
        signal_length = tf.shape(labels)[1]
        batch_size = tf.shape(labels)[0]

        # Convert inputs to complex64 for FFT operations
        labels = tf.cast(labels, tf.complex64)
        recovereds = tf.cast(recovereds, tf.complex64)

        # Compute FFT of labels and recovereds
        label_fft = tf.signal.fftshift(tf.signal.fft(tf.signal.ifftshift(labels, axes=1)), axes=1)
        recovered_fft = tf.signal.fftshift(tf.signal.fft(tf.signal.ifftshift(recovereds, axes=1)), axes=1)

        # Cross-spectrum calculation
        cross_spectrum = label_fft * tf.math.conj(recovered_fft)

        # Compute the phase shift
        phase_shift = tf.math.angle(tf.signal.ifft(tf.signal.fftshift(cross_spectrum, axes=1)))

        # Find the shift for each signal in the batch
        shift_indices = tf.cast(tf.argmax(phase_shift, axis = 1), dtype = tf.int32)  # Shape: (batch_size,)
        shifts = -(shift_indices - signal_length // 2 ) // 152 # division is required to scale down shifts (trial and error, larger division gives smaller error)
        
        # Compute the shifted indices for each signal
        range_indices = tf.range(signal_length, dtype = tf.int32)
        shifted_indices = range_indices[None, :] - shifts[:, None]
        clipped_indices = tf.clip_by_value(shifted_indices, 0, signal_length - 1)
        aligned_signals = tf.gather(recovereds, clipped_indices, batch_dims=1)
        return aligned_signals

    @staticmethod
    def fc_decalage_temporelle_batch_tf(N0, N, dt0, dt, V_t_batch):
        """
        Optimized TensorFlow version for handling a batch of vectors.
        
        Parameters:
            N0 (int): Number of rows in the output matrices.
            N (int): Number of columns in the output matrices.
            dt0 (float): Time step for the rows.
            dt (float): Time step for the columns.
            V_t_batch (Tensor): Tensor of input vectors of shape (batch_size, vector_length).
        
        Returns:
            Tensor: Tensor of output matrices of shape (batch_size, N0, N).
        """
        batch_size, vector_length = tf.shape(V_t_batch)[0], tf.shape(V_t_batch)[1]        
        N_dt0_s_dt = dt0 / dt
        N_pts_plus_decalage = tf.cast((N0 - 1) * N_dt0_s_dt, tf.int32)
        moitie = N_pts_plus_decalage // 2

        # Create row indices for each vector in the batch
        row_indices = tf.range(N0, dtype=tf.float32) * dt0 / dt
        row_indices = tf.cast(tf.round(row_indices), tf.int32)

        # Prepare indices for the columns
        col_indices = tf.range(-moitie, N - moitie, dtype=tf.int32)

        # Create a 2D grid of indices for rows and columns
        col_grid = tf.expand_dims(col_indices, axis=0) + tf.expand_dims(row_indices, axis=1)

        # Apply boundary conditions
        col_grid_clipped = tf.clip_by_value(col_grid, 0, vector_length - 1)

        # Gather values for each batch
        def process_vector(v):
            return tf.gather(v, col_grid_clipped, axis=0)

        # Process each vector in the batch
        M_decalage_t0_batch = tf.map_fn(process_vector, V_t_batch, fn_output_signature = tf.complex64)

        M_decalage_t0_batch = tf.image.flip_left_right(M_decalage_t0_batch)

        return M_decalage_t0_batch

    @staticmethod
    def fc_somme_sur_les_decales_batch_tf(N0, N, dt0, dt, M_decalage_t0):
        batch_size = tf.shape(M_decalage_t0)[0]
        N_dt0_s_dt = dt0 / dt
        N_pts_plus_somme = int(tf.floor((N_dt0_s_dt + 1.0) / 2.0))
        extended_N = N + 2 * N_pts_plus_somme - (0 if N_dt0_s_dt % 2 else 1)

        # Initialize the extended tensor
        M_t_t0_loc = tf.zeros((batch_size, N0, extended_N), dtype=M_decalage_t0.dtype)

        # Fill the initial columns
        initial_columns = tf.repeat(
            tf.expand_dims(M_decalage_t0[:, :, 0], axis=-1), 
            repeats=N_pts_plus_somme - (0 if N_dt0_s_dt % 2 else 1), 
            axis=-1
        )
        M_t_t0_loc = tf.concat([initial_columns, M_t_t0_loc[:, :, tf.shape(initial_columns)[-1]:]], axis=-1)

        # Fill the main columns
        main_columns = M_decalage_t0
        M_t_t0_loc = tf.concat(
            [
                M_t_t0_loc[:, :, :N_pts_plus_somme - 1],
                main_columns,
                M_t_t0_loc[:, :, N_pts_plus_somme - 1 + tf.shape(main_columns)[-1]:],
            ],
            axis=-1,
        )

        # Fill the last columns
        last_columns = tf.repeat(
            tf.expand_dims(M_decalage_t0[:, :, -1], axis=-1), 
            repeats=N_pts_plus_somme, 
            axis=-1
        )
        M_t_t0_loc = tf.concat(
            [
                M_t_t0_loc[:, :, :N_pts_plus_somme - 1 + tf.shape(main_columns)[-1]],
                last_columns,
            ],
            axis=-1,
        )

        # Compute the sum for the shifted matrices
        M_decalage_t0_sum = tf.zeros((batch_size, N0, N), dtype=M_decalage_t0.dtype)
        for i in range(int(N_dt0_s_dt)):
            shifted = M_t_t0_loc[:, :, i : i + N]
            M_decalage_t0_sum += shifted

        return M_decalage_t0_sum

    @staticmethod
    def interpolate_batch(batch_tensor, target_height, target_width):
        if batch_tensor.ndim > 3:
            # Use tf.image.resize for resizing the batch
            resized_tensor = tf.image.resize(
                batch_tensor, 
                size=(target_height, target_width), 
                method="nearest"  # or "nearest", "bicubic", etc., depending on the desired interpolation method
            )
        else:
            expanded_tensor = tf.expand_dims(batch_tensor, axis=-1)  # (batch_size, length, 1)
            expanded_tensor = tf.expand_dims(expanded_tensor, axis=1)  # (batch_size, 1, length, 1)
            resized_tensor = tf.image.resize(
                expanded_tensor, 
                size=(1, target_width),  # Resize only along the length axis
                method="nearest"
            )

            # Squeeze back to remove extra dimensions
            resized_tensor = tf.squeeze(resized_tensor, axis=[1, -1])  # (batch_size, target_length)
        return resized_tensor

    @staticmethod
    def sum_colonne_normale_batch(M_Obj, M_trace_t, N0, N):
        M_Obj_summed = tf.reduce_sum(M_Obj, axis=1)  # Shape: (batch_size, N)
        M_Obj_summed_conj = tf.math.conj(M_Obj_summed)  # Shape: (batch_size, N)
        numerator = tf.einsum('bn,bnij->bni', M_Obj_summed_conj, M_trace_t)  # Shape: (batch_size, N0, N)
        numerator = tf.transpose(numerator, perm=[0, 2, 1])
        denominator = tf.reduce_sum(tf.abs(M_Obj_summed) ** 2, axis=1)  # Shape: (batch_size,)
        Obj_t = tf.abs(tf.reduce_sum(numerator, axis = 2) / tf.expand_dims(tf.cast(denominator, tf.complex64), axis = 1))  # Shape: (batch_size, N0)
        max_values = tf.reduce_max(tf.abs(Obj_t), axis = 1, keepdims = True)  # Shape: (batch_size, 1)
        Obj_t = Obj_t / max_values  # Normalize
        return Obj_t


""" TRAINING FUCNTIONS """
def WGN_sweep(model_dir, epochs = 1):
    os.makedirs(model_dir, exist_ok=True)
    WGN_dB = np.linspace(0, 30, 30)        
    points, duration = [128, 10]
    MR_kernels = [7, 5, 3, 1]
    kernel = 3
    dropout = 0
    n_layers = 1
    n_neurons = 512

    for noise in WGN_dB:
        std = np.sqrt(1 / (10**(noise / 10))) 
        parameters = [MR_kernels, kernel, dropout, std, n_layers, n_neurons, (points, points, 1), 2]       
        train = Training(points, duration)
        name = f'{points}pts_{duration}fs_WGN_{noise}'

        # make folders
        subfolder = os.path.join(model_dir, name)
        os.makedirs(subfolder, exist_ok = True)
        model = train.create_MR_model(parameters)
        model_path = os.path.join(subfolder, f"{name}.weights.h5")
        dict_to_save = {'kernels': parameters[0],
                        'convolution': parameters[1],
                        'dropout': parameters[2],
                        'stddev': parameters[3],
                        'n_layers': parameters[4],
                        'n_neurons': parameters[5],
                        'input_shape': parameters[6],
                        'epochs': epochs, 
                        'points': points, 
                        'pulse_duration': duration, 
                        'n_outputs': parameters[7],
                        'hybrid_rate': 0}
        
        train.save_parameters(os.path.join(subfolder, f"{name}_parameters.txt"), dict_to_save)

        train.training_loop(model, model_path, epochs = epochs)

        # do prediction error on loaded model
        loaded_model = train.load_model_weights(model_path, parameters)

        error_path = os.path.join(subfolder, f"{name}_error_sim.txt")
        train.post_sim_val(loaded_model, error_path)
        data1 = np.loadtxt(error_path, delimiter=",")
        print("Sim Errors:   ", np.mean(data1, axis = 0)*100)

        error_path = os.path.join(subfolder, f"{name}_error_xp.txt")
        train.post_xp_val(loaded_model, error_path)
        data2 = np.loadtxt(error_path, delimiter=",")
        print("XP Errors:   ",  np.mean(data2, axis = 0)*100)

        dict_to_save = {'Sim': np.mean(data1, axis = 0), 'xp': np.mean(data2, axis = 0)}
        train.save_parameters(os.path.join(subfolder, f"{name}_errors.txt"), dict_to_save)

def LAMBDA_sweep(model_dir, epochs = 1):
    os.makedirs(model_dir, exist_ok=True)
    rates = np.linspace(0.6, 1, 8)        
    points, duration = [128, 10]
    MR_kernels = [7, 5, 3, 1]
    kernel = 3
    dropout = 0
    std = 0
    n_layers = 1
    n_neurons = 512

    for rate in rates:
        parameters = [MR_kernels, kernel, dropout, std, n_layers, n_neurons, (points, points, 1), 2]       
        train = Training(points, duration)
        name = f'{points}pts_{duration}fs_Hybrid_{rate}'

        # save parameters
        subfolder = os.path.join(model_dir, name)
        os.makedirs(subfolder, exist_ok = True)
        model = train.create_MR_model(parameters)
        model_path = os.path.join(subfolder, f"{name}.weights.h5")
        dict_to_save = {'kernels': MR_kernels,
                        'convolution': kernel,
                        'dropout': dropout,
                        'stddev': std,
                        'n_layers': n_layers,
                        'n_neurons': n_neurons,
                        'input_shape': parameters[6],
                        'epochs': epochs, 
                        'points': points, 
                        'pulse_duration': duration, 
                        'n_outputs': parameters[7],
                        'hybrid_rate': rate}
        train.save_parameters(os.path.join(subfolder, f"{name}_parameters.txt"), dict_to_save)

        # run training
        train.training_loop(model, model_path, epochs = epochs, hybrid_rate = rate)

        # do prediction error on loaded model
        loaded_model = train.load_model_weights(model_path, parameters)
        
        error_path = os.path.join(subfolder, f"{name}_error_sim.txt")
        train.post_sim_val(loaded_model, error_path)
        data1 = np.loadtxt(error_path, delimiter=",")
        print("Sim Errors:   ", np.mean(data1, axis = 0)*100)

        error_path = os.path.join(subfolder, f"{name}_error_xp.txt")
        train.post_xp_val(loaded_model, error_path)
        data2 = np.loadtxt(error_path, delimiter=",")
        print("XP Errors:   ",  np.mean(data2, axis = 0)*100)

        dict_to_save = {'Sim': np.mean(data1, axis = 0), 'xp': np.mean(data2, axis = 0)}
        train.save_parameters(os.path.join(subfolder, f"{name}_errors.txt"), dict_to_save)


if __name__ == "__main__":
    epochs = 9
    model_dir = r"C:\Users\Sydney\Documents\DFROStNET_may\Models"

    points = 128 #[128, 3*128] #[128, 3*128]
    duration = 10
    MR_kernels = [5, 5, 7, 3]
    kernel = 1
    dropout = 0.532
    std = 0.5011872336272722
    n_layers = 3
    n_neurons = 1860
    rate =  0.0 #3104730398487183

    # for model in test_models:
    parameters = [MR_kernels, kernel, dropout, std, n_layers, n_neurons, (points, points, 1), 2]       
    train = Training()
    name = '3007_DualDataset_SmoothedAmp'

    # save parameters
    subfolder = os.path.join(model_dir, name)
    os.makedirs(subfolder, exist_ok = True)
    model = train.create_MR_model(parameters)
    model_path = os.path.join(subfolder, f"{name}.weights.h5")
    dict_to_save = {'kernels': MR_kernels,
                    'convolution': kernel,
                    'dropout': dropout, 
                    'stddev': std,
                    'n_layers': n_layers,
                    'n_neurons': n_neurons,
                    'input_shape': parameters[6],
                    'epochs': epochs, 
                    'points': points, 
                    'pulse_duration': duration, 
                    'n_outputs': parameters[7],
                    'hybrid_rate': rate}
    train.save_parameters(os.path.join(subfolder, f"{name}_parameters.txt"), dict_to_save)

    # run training
    train.training_loop(model, model_path, epochs = epochs, hybrid_rate = rate, lrn_rate = 0.001)

    # do prediction error on loaded model
    loaded_model = train.load_model_weights(model_path, parameters)
    
    error_path = os.path.join(subfolder, f"{name}_error_sim.txt")
    train.post_sim_val(loaded_model, error_path)
    data1 = np.loadtxt(error_path, delimiter=",")
    print("Sim Errors:   ", np.mean(data1, axis = 0)*100)

    error_path = os.path.join(subfolder, f"{name}_error_xp.txt")
    train.new_post_xp_val(loaded_model, error_path)
    data2 = np.loadtxt(error_path, delimiter=",")
    print("XP Errors:   ",  np.mean(data2, axis = 0)*100)

    dict_to_save = {'Sim': np.mean(data1, axis = 0), 'xp': np.mean(data2, axis = 0)}
    train.save_parameters(os.path.join(subfolder, f"{name}_errors.txt"), dict_to_save)
