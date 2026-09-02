import os
import numpy as np
from numba import njit

# Stimulation parameters for tACS
F0               = 0.00021  # Base amplitude
frequency        = 11       # Frequency of the AC signal in Hz
t_perturb        = 180000   # Time to start perturbation in ms
perturb_duration = 300000   # Duration of tACS perturbation in ms

# Generate different waveforms
@njit(nogil=True)
def triangle_wave(phase):
    # True symmetric triangular wave in [-1, 1]
    return 4.0 * np.abs(phase - 0.5) - 1.0


@njit(nogil=True)
def sawtooth_wave(phase):
    # Centred sawtooth wave in [-1, 1)
    return 2.0 * phase - 1.0


@njit(nogil=True)
def generate_waveform(shape, time, F0, freq, duty_cycle=0.5):
    # time: scalar in ms
    t_s   = time / 1000.0
    phase = (t_s * freq) % 1.0

    if shape == 0:  # Sinusoidal
        return F0 * (
            np.cos(2.0 * np.pi * freq * t_s)
            + 1j * np.sin(2.0 * np.pi * freq * t_s)
        )

    elif shape == 1:  # Square, bipolar, 90-degree quadrature
        if np.sin(2.0 * np.pi * freq * t_s) >= 0.0:
            real = 1.0
        else:
            real = -1.0

        if np.cos(2.0 * np.pi * freq * t_s) >= 0.0:
            imag = 1.0
        else:
            imag = -1.0

        return F0 * (real + 1j * imag)

    elif shape == 2:  # True triangular, 90-degree quadrature
        real = triangle_wave(phase)
        imag = triangle_wave((phase + 0.25) % 1.0)
        return F0 * (real + 1j * imag)

    elif shape == 3:  # Pulsed sinusoid
        base = F0 * (
            np.cos(2.0 * np.pi * freq * t_s)
            + 1j * np.sin(2.0 * np.pi * freq * t_s)
        )

        if phase < duty_cycle:
            return base
        else:
            return 0.0 + 0.0j

    elif shape == 4:  # Centred sawtooth, 90-degree quadrature
        real = sawtooth_wave(phase)
        imag = sawtooth_wave((phase + 0.25) % 1.0)
        return F0 * (real + 1j * imag)

    else:
        return 0.0 + 0.0j


# Main simulation function with in-phase and out-phase options
@njit(nogil=True)
def simulate_SL_with_tACS(t_len, W_mat, sim_time, store_time, dt, dt_save, a_bif, speed, perturbed_nodes, shape, seed):
    np.random.seed(seed)
    W = W_mat.copy()
    n = t_len.shape[0]
    w = 2 * np.pi * 40
    a = a_bif
    b = 0.001

    if speed > 0:
        delays_raw = t_len / speed
        delays     = np.floor(delays_raw / dt)
        max_delay  = int(np.amax(delays))
        delays     = delays.astype(np.int64)
    else:
        delays    = np.zeros_like(t_len)
        max_delay = 0
        delays    = delays.astype(np.int64)

    state       = b * np.random.rand(n, max_delay + 1) + b * 1j * np.random.rand(n, max_delay + 1)
    output      = np.zeros((n, int(np.floor(store_time / dt_save))), dtype=np.complex_)
    store_count = 0

    # Generate perturbation signal
    time_values  = np.arange(0, sim_time, dt)
    perturbation = np.zeros_like(time_values, dtype=np.complex_)

    for k in range(time_values.shape[0]):
        if t_perturb <= time_values[k] < t_perturb + perturb_duration:
            t_rel = time_values[k] - t_perturb
            perturbation[k] = generate_waveform(shape, t_rel, F0, frequency)

    for st in range(int(np.floor(sim_time / dt))):
        curr_z        = state[:, -1].copy().reshape(n, 1)
        delayed_input = np.zeros_like(curr_z, dtype=np.complex_)

        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    delayed_input[i] += W[i, j] * (state[j, -(delays[i, j] + 1)] - state[i, -1])

        dz_base = curr_z * (a + 1j * w - np.abs(curr_z) ** 2) + delayed_input + b * (np.random.randn(n, 1) + 1j * np.random.randn(n, 1))

        # Apply perturbation
        time_in_ms = st * dt
        index      = int(time_in_ms / dt)
        if t_perturb <= time_in_ms < t_perturb + perturb_duration:
            for node in perturbed_nodes:
                dz_base[node] += perturbation[index]

        new_z         = curr_z + (dt * 1e-3) * dz_base
        state[:, :-1] = state[:, 1:]
        state[:, -1]  = new_z.reshape(n)

        # Store data
        if st >= ((sim_time - store_time) / dt):
            if st % (dt_save / dt) == 0:
                output[:, store_count] = new_z.reshape(n)
                store_count += 1

    return output

