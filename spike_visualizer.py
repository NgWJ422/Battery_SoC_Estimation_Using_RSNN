import torch
import matplotlib.pyplot as plt
import my_encoding as encoding
from bindsnet.encoding.encodings import poisson, poisson_normalized

def visualize_spike_train(spike_train: torch.Tensor, title: str = "Spike Train", neuron_labels=None):
    """
    Visualize a spike train as a raster plot.

    Parameters:
        spike_train (torch.Tensor): Tensor of shape [T, ...] containing binary spike events.
        title (str): Title of the plot.
        neuron_labels (list): Optional list of neuron labels for multi-neuron cases.
    """
    spike_train = spike_train.cpu().numpy()
    time_steps = spike_train.shape[0]
    spikes = spike_train.reshape(time_steps, -1)

    plt.figure(figsize=(10, 2))
    for neuron_idx in range(spikes.shape[1]):
        spike_times = (spikes[:, neuron_idx] > 0).nonzero()[0]
        plt.vlines(spike_times, neuron_idx + 0.5, neuron_idx + 1.5, color='black')
    
    plt.xlabel("Time step")
    plt.ylabel("Neuron")
    if neuron_labels:
        plt.yticks(range(1, len(neuron_labels) + 1), neuron_labels)
    else:
        plt.yticks(range(1, spikes.shape[1] + 1))
    plt.title(title)
    plt.ylim(0.5, spikes.shape[1] + 0.5)
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Record in csv file
# Do all encodings at the same time


# Example input
datum = torch.tensor([0.111])  # A single input value between 0 and 1
#spike_train = poisson_normalized(datum=datum, maxrate=50,avgISI=1, time=100, dt=1.0, device="cpu")


#spike_train = encoding.population_encoding(datum=datum,num_neurons= 15,time=100, dt=1.0)

spike_train = encoding.fix_normalized(datum= datum, maxrate=50, time=100, dt=1.0, device="cpu")

#spike_train = encoding.ttfs2_encoding(datum,time=100, dt=1.0, device="cpu")

# Visualize
visualize_spike_train(spike_train, title="Encoding Spike Train")
