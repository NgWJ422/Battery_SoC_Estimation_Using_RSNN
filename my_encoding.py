from typing import Optional

import torch

# Try to ttfs and population
# poisson is created to detech noisy input

def rate_encoding(
    datum: torch.Tensor,
    maxrate: int,
    time: int,
    dt: float = 1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Rate encoding: spike probability proportional to normalized input.
    """
    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)
    
    # Probability of spike at each time step
    p_spike = datum * maxrate * dt / 1000  # Convert Hz to probability per dt
    
    rand = torch.rand((time_steps, datum.shape[0]), device=device)
    spikes = (rand < p_spike.unsqueeze(0)).byte()

    return spikes.view(time_steps, *shape)

def ttfs_encoding(
    datum: torch.Tensor,
    time: int,
    dt: float = 1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Time-to-First-Spike encoding: stronger inputs spike earlier.
    """
    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)

    # Map normalized input to early spike times
    spike_times = (1.0 - datum.clamp(0, 1)) * (time_steps - 1)
    spike_times = torch.clamp(spike_times, 0, time_steps - 1).long()

    spike_times = spike_times.long()

    spikes = torch.zeros((time_steps, datum.shape[0]), device=device, dtype=torch.uint8)
    spikes[spike_times, torch.arange(datum.shape[0])] = 1

    return spikes.view(time_steps, *shape)


def ttfs2_encoding(
    datum: torch.Tensor,
    time: int,
    dt: float,
    device: str = "cpu",
    clamp_eps: float = 1e-2,
    debug: bool = False,
) -> torch.Tensor:
    """
    Time-To-First-Spike (TTFS) encoding with binary spikes (0 or 1), indicating time of first spike.

    :param datum: Tensor with shape [...], values in [0, 1].
    :param time: Total encoding time (in simulation steps).
    :param dt: Time resolution.
    :param device: Output device.
    :param clamp_eps: Clamping value to avoid edge cases.
    :param debug: Debug mode flag.
    :return: Tensor of shape [time_steps, ...] with binary spikes (0 or 1).
    """

    datum = datum.clamp(clamp_eps, 1.0 - clamp_eps)

    shape = datum.shape
    datum_flat = datum.flatten()

    time_steps = int(time / dt)
    spike_times = ((1.0 - datum_flat) * (time_steps - 1)).round().long()
    spike_times = spike_times.clamp(0, time_steps - 1)

    # Binary spike tensor
    spikes = torch.zeros((time_steps, datum_flat.numel()), dtype=torch.float32, device=device)
    spikes[spike_times, torch.arange(datum_flat.numel(), device=device)] = 1.0

    if debug:
        print(f"[DEBUG] Total spikes: {(spikes > 0).sum().item()} / {datum_flat.numel()}")
        hist = torch.bincount(spike_times, minlength=time_steps)
        print(f"[DEBUG] Spike time distribution:\n{hist}")

    
    return spikes.view(time_steps, *shape)


def fix_normalized(
    datum: torch.Tensor,
    maxrate: int,
    time: int,
    dt: float = 1.0,
    device="cpu"
) -> torch.Tensor:
    # language=rst
    """
    Generates deterministic spike trains based on normalized input data.
    Each neuron's spike times are determined by its input intensity.

    :param datum: Normalized input tensor [n_1, ..., n_k] with values in [0, 1].
    :param maxrate: Maximum firing rate (Hz).
    :param time: Total simulation time (in ms).
    :param dt: Simulation time step (in ms).
    :param device: Target device (e.g., "cpu" or "cuda").
    :return: Tensor of shape [time, n_1, ..., n_k] containing spikes (0 or 1).
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)

    spikes = torch.zeros(time_steps, size, dtype=torch.uint8, device=device)

    # Convert normalized input to firing rate (Hz)
    firing_rates = datum * maxrate  # shape: [size]

    # Convert rate (Hz) to spike interval in ms (avoiding division by zero)
    #intervals = torch.ceil(1000.0 / (firing_rates + 1e-6)).long()
    intervals = torch.ceil(time / (firing_rates + 1e-6)).long()

    for i in range(size):
        if firing_rates[i] > 0:
            # Generate spike times at regular intervals
            spike_times = torch.arange(0, time, step=intervals[i].item()).long()
            spike_indices = (spike_times / dt).long()
            spike_indices = spike_indices[spike_indices < time_steps]
            spikes[spike_indices, i] = 1

    return spikes.view(time_steps, *shape)


def burst_encoding(
    datum: torch.Tensor,
    max_burst: int,
    time: int,
    dt: float = 1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Burst encoding: stronger inputs produce more spikes grouped closely together.
    """
    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)

    num_bursts = (datum * max_burst).long()

    spikes = torch.zeros((time_steps, datum.shape[0]), device=device, dtype=torch.uint8)

    for idx in range(datum.shape[0]):
        burst_size = num_bursts[idx]
        if burst_size == 0:
            continue
        start_time = torch.randint(0, time_steps - burst_size, (1,), device=device)
        spikes[start_time:start_time + burst_size, idx] = 1
    
    return spikes.view(time_steps, *shape)

def phase_encoding(
    datum: torch.Tensor,
    num_phases: int,
    time: int,
    dt: float = 1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Phase encoding: spike phases (timing patterns) encode information.
    """
    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)

    phase_step = time_steps // num_phases
    spikes = torch.zeros((time_steps, datum.shape[0]), device=device, dtype=torch.uint8)

    for idx in range(datum.shape[0]):
        phase = int(datum[idx] * (num_phases - 1))
        spike_time = phase * phase_step
        if spike_time < time_steps:
            spikes[spike_time, idx] = 1

    return spikes.view(time_steps, *shape)


#check later if poisson or not
def population_encoding(
    datum: torch.Tensor,
    num_neurons: int,
    time: int,
    dt: float = 1.0,
    sigma: float = 0.1,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Population encoding: encodes input by activation of neurons with preferred stimulus centers.

    :param datum: Normalized input tensor in [0, 1]
    :param num_neurons: Number of population units (neurons)
    :param time: Total simulation time
    :param dt: Time resolution
    :param sigma: Standard deviation of tuning curve (Gaussian)
    :param device: Target device
    :return: Spiketrain tensor of shape [T, num_neurons, ...original_shape]
    """
    shape = datum.shape
    datum = datum.flatten().to(device)

    time_steps = int(time / dt)
    centers = torch.linspace(0, 1, steps=num_neurons, device=device)
    datum = datum.unsqueeze(1)  # Shape: [N, 1]
    centers = centers.unsqueeze(0)  # Shape: [1, M]

    # Gaussian tuning curve
    responses = torch.exp(-((datum - centers)**2) / (2 * sigma**2))  # [N, M]

    # Normalize to probability
    responses = responses / responses.max(dim=1, keepdim=True)[0]

    rand = torch.rand((time_steps, *responses.shape), device=device)
    spikes = (rand < responses.unsqueeze(0)).byte()  # [T, N, M]
    
    return spikes.permute(0, 2, 1).view(time_steps, num_neurons, *shape)



def dynamic_threshold_encoding(
    datum: torch.Tensor,
    time: int,
    dt: float = 1.0,
    threshold_init: float = 0.5,
    decay: float = 0.95,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Dynamic Threshold encoding: spikes when input exceeds a decaying threshold.

    :param datum: Normalized input in [0, 1]
    :param time: Simulation time
    :param dt: Time resolution
    :param threshold_init: Initial threshold
    :param decay: Decay rate of the threshold
    :param device: Target device
    :return: Spiketrain tensor of shape [T, ...original_shape]
    """
    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)
    spikes = torch.zeros((time_steps, datum.shape[0]), dtype=torch.uint8, device=device)

    thresholds = torch.full_like(datum, threshold_init)
    for t in range(time_steps):
        spike = datum > thresholds
        spikes[t] = spike.byte()
        thresholds = torch.where(spike, datum, thresholds * decay)

    return spikes.view(time_steps, *shape)


def latency_encoding(
    datum: torch.Tensor, time: int, dt: float = 1.0, device="cpu"
) -> torch.Tensor:
    # language=rst
    """
    Latency encoding: higher input → earlier spike.
    Maps input ∈ [0, 1] to time ∈ [1, T] using:
        spike_time = round(1 + (1 - input) * (T - 1))

    :param datum: Tensor of shape [n_samples, ...], values must be ∈ [0, 1]
    :param time: Total simulation time in steps
    :param dt: Time step (default: 1.0)
    :return: Tensor of shape [time, ...] with one spike per input neuron
    """
    assert (datum >= 0).all() and (datum <= 1).all(), "Inputs must be normalized in [0, 1]"

    shape = datum.shape
    datum = datum.flatten().to(device)
    time_steps = int(time / dt)

    # Compute spike time using latency encoding
    spike_times = torch.round(1 + (1 - datum) * (time_steps - 1)).long()

    # Create spike tensor
    spikes = torch.zeros(time_steps, datum.numel(), device=device).byte()
    for i in range(datum.numel()):
        t = spike_times[i].item()
        if 1 <= t <= time_steps:
            spikes[t - 1, i] = 1  # index t-1 due to 0-based indexing

    spikes_debug = spikes.numpy()

    return spikes.view(time_steps, *shape)