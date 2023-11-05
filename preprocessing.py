import numpy as np
import matplotlib.pyplot as plt
import pywt
import h5py
from os.path import join as pjoin
from utils import butter_lowpass_filter


def _preprocessing(data, num_classes, mode):

    fs = 250
    downsample_factor = 3
    num_subjects = 9
    xs, ys = [], []
    for i in range(num_subjects):
        dpath = '/s' + str(i + 1)
        x, y = data[pjoin(dpath, 'X')], data[pjoin(dpath, 'Y')]
        cutoff = x.shape[0] // 2
        test_cutoff = cutoff + x.shape[0] // 4
        if mode == 'Train':
            x, y = x[:cutoff], y[:cutoff]  # train
        elif mode == 'Validation':
            x, y = x[cutoff:test_cutoff], y[cutoff:test_cutoff]  # validation
        elif mode == 'Test':
            x, y = x[test_cutoff:], y[test_cutoff:]  # test

        x = butter_lowpass_filter(x, fs / downsample_factor / 2, fs, order=6)
        x_downsampled = x[:, :, ::downsample_factor][:, :, 13:237]

        freq = np.linspace(1, 41, 224)
        reverse_freq = freq[::-1]
        scale = pywt.frequency2scale('cmor1.5-1.0', reverse_freq / (250 / 3))
        coefficients, frequencies = pywt.cwt(x_downsampled, scales=scale, wavelet='cmor1.5-1.0',
                                             sampling_period=downsample_factor / fs)
        # # Visualization with frequency on the y-axis
        # plt.figure(dpi=500)
        # plt.imshow(np.abs(coefficients.transpose((1, 2, 0, 3))[0][0]), aspect='equal', cmap='jet',
        #            interpolation='bilinear')
        # plt.colorbar(label='Magnitude')
        # plt.ylabel('Frequency (Hz)')
        # plt.xlabel('Time (s)')
        # plt.title('Time-Frequency Representation using CWT')
        # x_ticks_pixel = np.linspace(0, 224, 7)  # Assuming 6 ticks for [0, 1, 2, 3]
        # x_ticks_label = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        # y_ticks_pixel = np.linspace(0, 224, 9)  # Assuming 9 ticks for [0, 5, 10, ..., 40]
        # y_ticks_label = np.arange(40, -1, -5)
        # plt.xticks(x_ticks_pixel, x_ticks_label)
        # plt.yticks(y_ticks_pixel, y_ticks_label)
        # plt.show()

        if num_classes == 18:
            xs.append(coefficients.transpose((1, 2, 0, 3))), ys.append(y[:] + 2 * i)
        elif num_classes == 2:
            xs.append(coefficients.transpose((1, 2, 0, 3))), ys.append(y[:])

    xs, ys = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    # 18 labels
    if num_classes == 18:
        np.savez('data/BCICIV_2a_data_LR_UNIQUELABELs_' + mode, x=xs, y=ys)
    # 2 labels
    elif num_classes == 2:
        np.savez('data/BCICIV_2a_data_LR_SAMELABELs_' + mode, x=xs, y=ys)


def preprocessing():
    print('Preprocessing... It will take some time. Please wait.')
    data_path = 'data/BCICIV_2a_data_LR.h5'
    data = h5py.File(data_path, 'r')
    num_classes_list = [18, 2]
    mode_list = ['Train', 'Validation', 'Test']
    counter = 0
    for num_classes in num_classes_list:
        for mode in mode_list:
            counter += 1
            _preprocessing(data, num_classes=num_classes, mode=mode)
            print(f"{100 * counter / 6:.2f}%")
    print('Preprocessing is done.')


if __name__ == '__main__':
    preprocessing()



