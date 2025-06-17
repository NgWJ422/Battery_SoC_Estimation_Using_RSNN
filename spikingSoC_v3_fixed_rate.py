# Project Name: SoH Estimation
# Resources:https://www.kaggle.com/code/rajeevsharma993/battery-health-nasa-dataset/notebook
# Description: Reservoir Spiking Neural Network for SoH prediction (using NASA dataset)
# Date: 29 Mac 2024

import datetime
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import os
import csv
import time as mtime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tracemalloc              #monitor memory usage
import psutil                   #monitor CPU usage

from tqdm import tqdm
from bindsnet.encoding.encodings import poisson, poisson_normalized
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection


# Import encoding functions i write myself
from my_encoding import rate_encoding, ttfs_encoding, burst_encoding, phase_encoding, population_encoding, dynamic_threshold_encoding, fix_normalized

#================== Parameter Setting =======================
n_features = 7               # Number of features
n_neurons = 500              # Number of neuron in reservoir network
n_hidden = 50                # Number of hidden layer in regression model
n_epochs = 20               # Number of epoch (training iteration) on regression model
transfer_function = 'sigmoid'  # Change this to 'sigmoid', 'tanh' or 'relu' as needed
learning_rate = 0.001      # Regression model learning rate
momentum = 0.9              # Regression model momentum
n_batch = 1                 # Number of data that being process during data loader. For now not able to use batch training.
time = 100                   # window time. conversion time window and spike rate (that represent value)
dt = 1.0                    # simulation timestep
max_rate = 50               # max spiking rate each neuron during conversion
avg_ISI = 1                 # average inter spike interval during conversion
trefrac = 2                 # refractory period in LIF
tdecay = 10                 # time decay in LIF
datatype = 'float'          # change between e.g 'float', 'int32' and 'uint8' to reduce memory consumption. KIV-too complicated
examples = 5000              # Number of training data. Reduce the number of training data with "examples"
test_samples = 10000          # Number of test data. Reduce numer of testing data for faster result
train_all = True          # set "True" for use all data during training. set "False" with number of "examples"
test_all = True             # set "True" for use all data during testing. set "False" with number of "test_samples"
sparse_tensor = False        # use sparese tensor to save memory
#-----------------------------------------------------------------------------------------------------------------------
inpConn_density = 1.0       # Connection density between input and reservoir network
resConn_density = 0.5      # Connection density within nodes in reservoir network
                            # set range between [0~1]. 1:all connected; 0.5:averagely 50% connections; 0:not connected
inpConn_weight  = 2.0       # Max value connection weight between input and reservoir network
resConn_weight = 2.0        # Max value connection weight between nodes in reservoir network
                            # The weight also depend on this parameter "inpConn_type"/"resConn_type"
inpConn_type = ['rand','exe', 0.3]          # 1st argument: set 'uni' for uniform weight, set 'rand' for random weight
resConn_type = ['rand','inb', 0.5]          # 2nd argument: set 'exe' for excitatory or 'inb' for inhibitory connection
                                            # 3rd argument: propotion of inhibitory connection
                                            #               (1.0 all inhibitory; 0.0 all excitatory)
#-----------------------------------------------------------------------------------------------------------------------
plot = True                 # set "True" to generate all related figures. set "False" to speed up the simulation
record = True               # set "True" to record simulation result on csv file. set "False" otherwise
n_workers = -1              # Number of CPU to process the calculation.Default -1(all processor)
gpu = False                 # set "True" to use GPU for simulation if available.
seed = 0                    # seed random number for GPU

data_path = "battery_data"
trainBattery = "B0005"
testBattery = "B0006"

# save important result for later analysis. Data structure as follow

analysis_result = [n_neurons, n_hidden, n_epochs, time, avg_ISI, max_rate, datatype,
                    inpConn_density, resConn_density,
                    inpConn_type[0], inpConn_type[1], inpConn_type[2],
                    resConn_type[0], resConn_type[1] ,resConn_type[2]]

# Loihi 2 synaptic operation energy (in Joules)
E_SPIKE_J = 11e-12  # 11 pJ per synaptic operation


def CalculateSoC(pSoc, capacity, current, ptime, ctime):
    # Convert capacity Ah to As
    capacity = capacity * 3600

    # Calculate the time difference
    delta_time = ctime - ptime

    # Calculate the integral assuming constant current
    integral_current = current * delta_time

    # Calculate the SoC
    SoC = pSoc + (1 / capacity) * integral_current
    # Ensure SoC is between 0 and 100
    SoC = max(0, min(SoC, 1))

    return SoC

#----Sets up device----
def setDevice(gpu):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False
    torch.set_num_threads(os.cpu_count() - 1)
    print("Running on Device = ", device)
    return device

#------generate Matrix Connection between layer------
def generateMatrixConnection(connSize, connDensity, connWeight, connType):
    #A = torch.tensor([[1, 0, 0], [0, 0, 3], [4, 0, 0]])   #for debug only

    conn_matrix = torch.zeros(connSize[0],connSize[1])
    connProb = torch.rand(connSize[0], connSize[1])
    if connType[0] == 'uni':
        if connType[1] == 'exe':
           conn_matrix = torch.where(connProb > (1-connDensity),
                                    torch.tensor(connWeight), torch.tensor(0))
        elif connType[1] == 'inb':
            inhbProb = torch.rand(connSize[0], connSize[1])
            inhbConn_flag = inhbProb < connType[2]
            conn_matrix = torch.where(connProb > (1 - connDensity),
                                      torch.where(inhbConn_flag,torch.tensor(-connWeight),torch.tensor(connWeight)),
                                      torch.tensor(0))

    elif connType[0] == 'rand':
        ratio = torch.rand(connSize[0], connSize[1])
        if connType[1] == 'exe':
            conn_matrix = torch.where(connProb > (1 - connDensity),
                                    torch.tensor(connWeight * ratio), torch.tensor(0))
        elif connType[1] == 'inb':
            inhbProb = torch.rand(connSize[0], connSize[1])
            inhbConn_flag = inhbProb < connType[2]
            conn_matrix = torch.where(connProb > (1 - connDensity),
                                      torch.where(inhbConn_flag,torch.tensor(-connWeight*ratio),torch.tensor(connWeight*ratio)),
                                      torch.tensor(0))
    else:
        raise ValueError("Invalid connection type. Use 'uni' or 'random'.")

    #Define connection type using specific type
    #return conn_matrix.to(dtype=eval(f"torch.{datatype}"))     #KIV-weight need to define as float. To many part to change
    conn_matrix = conn_matrix.to(dtype=torch.float)

    if sparse_tensor is True:
        #conn_matrix = conn_matrix.to_sparse()       # use _coo format
        final_conn_matrix = conn_matrix.to_sparse_csr()#test other format like _bsc or _bsc

    else:
        final_conn_matrix = conn_matrix

    #return conn_matrix
    return final_conn_matrix

#----Create Reservoir Network Model----
def createNetModel(device):
    # Create simple Torch NN
    network = Network(dt=dt)
    inpt = Input(n_features, shape=(1, n_features))
    network.add_layer(inpt, name="I")
    #output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
    output = LIFNodes(n_neurons, thresh=-52, refrac=trefrac, tc_decay=tdecay)
    network.add_layer(output, name="O")

    #create tensor with random values in range (min, max)
    inpResConnSize = (inpt.n, output.n)
    ResResConnSize = (output.n, output.n)
    #set connection matrix between input and reservoir
    conn_matrix_1 = generateMatrixConnection(inpResConnSize,inpConn_density,inpConn_weight,inpConn_type)
    conn_matrix_2 = generateMatrixConnection(ResResConnSize,resConn_density,resConn_weight,resConn_type)

    #change connection matrix to specific type

    #for debugging
    #conn_matrix_1_debug = conn_matrix_1.numpy()
    #conn_matrix_2_debug = conn_matrix_2.numpy()

    #C1 = Connection(source=inpt, target=output, w=conn_matrix_1.to(dtype=torch.float))    #Later delete. Original function.Builtin on function
    C1 = Connection(source=inpt, target=output, w=conn_matrix_1)
    C2 = Connection(source=output, target=output, w=conn_matrix_2)

    network.add_connection(C1, source="I", target="O")
    network.add_connection(C2, source="O", target="O")

    # Directs network to GPU
    if device == "cuda":
        network.to("cuda")

    return network

def loadDataset(dataPath,batteryName):
    #load dataset and extract the info from .mat file
    mat = loadmat(dataPath + '/' + batteryName + '.mat')
    print('Total data in dataset: ', len(mat[batteryName][0, 0]['cycle'][0]))
    counter = 0
    dataset = []
    capacity_data = []


    for i in range(len(mat[batteryName][0, 0]['cycle'][0])):
        row = mat[batteryName][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                          int(row['time'][0][1]),
                                          int(row['time'][0][2]),
                                          int(row['time'][0][3]),
                                          int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            for j in range(len(data[0][0]['Voltage_measured'][0])):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time])
            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
            counter = counter + 1
    print(dataset[0])
    return [pd.DataFrame(data=dataset,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity', 'voltage_measured',
                                  'current_measured', 'temperature_measured',
                                  'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                         columns=['cycle', 'ambient_temperature', 'datetime',
                                  'capacity'])]


def trainNetwork(network,train_dataset,device):

    #add number of train sample to result
    trainsample = len(train_dataset) if train_all else examples
    analysis_result.append(trainsample)

    # Monitors spike for visualizing activity
    spikes = {}
    spikes['O'] = Monitor(network.layers['O'], ["s"], time=time)
    network.add_monitor(spikes['O'], name="%s_spikes" % 'O')

    #train_dataset_debug = train_dataset.numpy()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=n_batch, shuffle=True, num_workers=0, pin_memory=gpu
    )
    print(f"PyTorch is using {torch.get_num_threads()} threads.")
    # start training the network - record training time and memory usage
    process = psutil.Process(os.getpid())  # current process
    cpu_start = process.cpu_times().user + process.cpu_times().system
    mem_start = process.memory_info().rss  # Resident memory in bytes
    wall_start = mtime.time()

    

    tracemalloc.start()  # Optional: Python-level memory usage

    # Calculate power efficiency for intel loihi chip
    total_input_spikes = 0
    total_reservoir_spikes = 0
    n_iters = examples
    training_pairs = []             #list of spikes from reservoir network
    pbar = tqdm(enumerate(dataloader))


    for (i, dataPoint) in pbar:
        if train_all is False and i > n_iters:             #increase example value to increase accuracy
            break

        datum = dataPoint[0].float()
        target = dataPoint[1].float()

        
        encoded_datum = fix_normalized(datum, maxrate=max_rate, time=time, dt=dt, device=device)
        num_input_spikes = encoded_datum.sum().item()
        total_input_spikes += num_input_spikes  

        if train_all:
            pbar.set_description_str("Train progress: (%d / %d)" % (i, len(train_dataset)))
        else:
            pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

        # Run network on sample image
        network.run(inputs={"I": encoded_datum}, time=time, input_time_dim=0)
        #spikedata = spikes["O"].get("s").sum(0)
        #spikedata = spikedata.to(dtype=eval(f"torch.{datatype}"))

        reservoir_spike_count = spikes["O"].get("s").sum().item()
        total_reservoir_spikes += reservoir_spike_count
        
        training_pairs.append([spikes["O"].get("s").sum(0).float(), target])
        #training_pairs.append([spikedata, target])

        # use debug to view training pairs
        #training_pairs_debug = training_pairs[i][0].numpy()

        network.reset_state_variables()

    #use debug to view training pairs
    #final_training_pairs_debug = np.concatenate([np.concatenate((np.array(data[0]), np.array(data[1])), axis=1) for data in training_pairs], axis=0)
    all_spike_times = torch.stack([s for s, _ in training_pairs])
    print("Feature variance:", all_spike_times.var(dim=0).mean())
    print("Input data completed..")

    # Define logistic regression model.
    # The input is a spikes count from each neuron on reservoir
    # The output is battery SoH
    torch_function = getattr(torch, transfer_function)
    class LogisticReg(nn.Module):
        def __init__(self, input_size):
            super(LogisticReg, self).__init__()
            self.hidden1 = nn.Linear(input_size, n_hidden)
            self.hidden2 = nn.Linear(n_hidden, 10)
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            x1 = torch_function(self.hidden1(x))         #alternatively, use "out = torch.sigmoid(self.linear(x1))"
            x2 = torch_function(self.hidden2(x1))
            out = torch_function(self.linear(x2))
            return out

    # Create and train logistic regression model on reservoir outputs.
    model = LogisticReg(n_neurons).to(device)
    criterion = nn.MSELoss(reduction="sum")     #Mean Squared Error Loss. Please check the meaning of redcution
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training the Model
    print("\n Training the read out")
    pbar = tqdm(enumerate(range(n_epochs)))
    for epoch, _ in pbar:
        avg_loss = 0
        # Extract spike outputs from reservoir for a training sample
        #       i   -> Loop index
        #       s   -> Reservoir output spikes
        #       t   -> target value
        for i, (s, t) in enumerate(training_pairs):

            # Reset gradients to 0
            optimizer.zero_grad()

            # Run spikes through logistic regression model
            outputs = model(s)

            # Calculate MSE
            loss = criterion(outputs, t)
            avg_loss += loss.item()         #similar to "loss.data" but this give tensor itself

            # Optimize parameters
            loss.backward()
            optimizer.step()


        pbar.set_description_str(
            "Epoch: %d/%d, Loss: %.4f"
            % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
        )

    wall_end = mtime.time()
    cpu_end = process.cpu_times().user + process.cpu_times().system
    mem_end = process.memory_info().rss
    _, peakRam = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Report results
    print("\nTraining Completed ✅")
    print(f"Wall-clock Time     : {wall_end - wall_start:.2f} seconds")
    print(f"CPU Time (User+Sys) : {cpu_end - cpu_start:.2f} seconds")
    print(f"Memory Usage Change : {(mem_end - mem_start) / (1024**2):.2f} MB")
    print(f"Memory at End       : {mem_end / (1024 ** 2):.2f} MB")
    print(f"Peak Memory (Python): {peakRam / (1024 ** 2):.2f} MB")

    analysis_result.append(avg_loss / len(training_pairs))
    analysis_result.append(wall_end - wall_start)
    analysis_result.append(cpu_end - cpu_start)  # CPU time in seconds
    analysis_result.append(peakRam / (1024 ** 2))  # convert to MB
    analysis_result.append(mem_end / (1024 ** 2))  # convert to MB

    input_fanout = int(n_neurons * inpConn_density)
    reservoir_fanout = int(n_neurons * resConn_density)
    total_input_energy = total_input_spikes * input_fanout * E_SPIKE_J
    total_reservoir_energy = total_reservoir_spikes * reservoir_fanout * E_SPIKE_J
    total_estimated_energy = total_input_energy + total_reservoir_energy

    print("\n🔋 Estimated Neuromorphic Energy Consumption (Loihi 2)")
    print(f"Total Input Spikes     : {total_input_spikes}")
    print(f"Total Reservoir Spikes : {total_reservoir_spikes}")
    print(f"Input Layer Energy     : {total_input_energy * 1e6:.3f} µJ")
    print(f"Reservoir Energy       : {total_reservoir_energy * 1e6:.3f} µJ")
    print(f"Total Energy Estimate  : {total_estimated_energy * 1e6:.3f} µJ")

    analysis_result.append(total_estimated_energy)  # in Joules
    return model


def testNetwork(network,model,test_dataset,device):

    # add number of test sample to result
    testsample = len(test_dataset) if test_all else test_samples
    analysis_result.append(testsample)

    # Monitors spike for visualizing activity
    spikes = {}
    spikes['O'] = Monitor(network.layers['O'], ["s"], time=time)
    network.add_monitor(spikes['O'], name="%s_spikes" % 'O')

    # convert inputs as spike train using encoder
    # Create a dataloader to iterate and batch data. For test data, do not shuffle arrangment

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=n_batch, shuffle=True, num_workers=0, pin_memory=gpu
    )

    start_time = mtime.time()  # for record training time

    testing_pairs = []  # list of spikes from reservoir network
    pbar = tqdm(enumerate(dataloader))

    for (i, dataPoint) in pbar:
        if  test_all is False and i > test_samples:  # increase example value to increase accuracy
            break
        cycle = dataPoint[0]
        datum = dataPoint[1].float()
        target = dataPoint[2].float()
        SoCtime = dataPoint[3].float()

        #encoded_datum = poisson(datum, time=time, dt=dt)
        encoded_datum = fix_normalized(datum, maxrate=max_rate, time=time, dt=dt, device=device)
        # debug_encoded_datum = (encoded_datum.squeeze()).numpy()
        if train_all:
            pbar.set_description_str("Test progress: (%d / %d)" % (i, len(test_dataset)))
        else:
            pbar.set_description_str("Test progress: (%d / %d)" % (i, test_samples))

        # Run network on sample image
        network.run(inputs={"I": encoded_datum}, time=time, input_time_dim=0)
        testing_pairs.append([cycle, SoCtime, spikes["O"].get("s").sum(0).float(), target])

        # use debug to view testing pairs
        #testing_pairs_debug = testing_pairs[i][0].numpy()

        network.reset_state_variables()

    # Test model with previously trained logistic regression classifier
    results = []
    for c, t, s, calc_soc in testing_pairs:
        pred_soc = model(s)
        results.append([c.item(), t.item(), calc_soc.item(), pred_soc.item()])

    end_time = mtime.time()
    print("Testing Time:%d", end_time - start_time)
    analysis_result.append(end_time-start_time)

    return pd.DataFrame(data=results,columns=['cycle', 'time', 'SoC', 'PredSoC'])


def performanceAnalysis(data):
    # Ensure data is sorted properly
    data = data.sort_values(by=['cycle', 'time'], ascending=[True, True]).reset_index(drop=True)
    
    # Display a sample of the data for verification
    print(data.sample(10))

    # Compute RMSE
    rms = np.sqrt(mean_squared_error(data['SoC'], data['PredSoC']))
    print('Root Mean Square Error: ', rms)
    analysis_result.append(rms)  # Append RMSE to global analysis_result

    # Directory to save figures and analysis
    save_dir = "SoC_Pred_Results_fixed_rate"
    os.makedirs(save_dir, exist_ok=True)

    # Plot and save SoC vs Time figures for selected cycles
    for cycle_id in [1, 10, 90, 150]:
        cycle_data = data[data['cycle'] == cycle_id]

        plt.figure(figsize=(8, 4))
        plt.plot(cycle_data['time'], cycle_data['SoC'], label='Actual SoC', linewidth=2)
        plt.plot(cycle_data['time'], cycle_data['PredSoC'], label='Predicted SoC', linestyle='--', linewidth=2)

        plt.title(f"SoC vs Time — Cycle {cycle_id}")
        plt.xlabel("Time (s)")
        plt.ylabel("State of Charge (SoC)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        if record:
            filename = f"SoC_vs_Time_Cycle_{cycle_id}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)

        plt.show()

    # Optionally record additional outputs
    if record:
        # Save the complete analysis_result with headers
        csv_path = os.path.join(save_dir, 'SoC_simulation_analysis.csv')
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header if file is new
            if not file_exists:
                header = [
                    'n_neurons', 'n_hidden', 'n_epochs', 'time', 'avg_ISI', 'max_rate', 'datatype',
                    'inpConn_density', 'resConn_density',
                    'inpConn_type_0', 'inpConn_type_1', 'inpConn_type_2',
                    'resConn_type_0', 'resConn_type_1', 'resConn_type_2',
                    'train_sample', 'Last epoch train loss', 'train_time', 'cpu_time', 'peak_ram_MB', 'final_ram_MB','total energy_J',
                    'test_sample', 'test_time', 'rmse'
                ]
                writer.writerow(header)

            # Write analysis result row
            writer.writerow(analysis_result)

        print(f"Simulation analysis metrics saved to '{csv_path}'.")

        # Save the SoC prediction results
        result_csv_path = os.path.join(save_dir, 'SoC_Prediction_results.csv')
        data.to_csv(result_csv_path, index=False)
        print(f"SoC prediction results saved to '{result_csv_path}'.")

    return


def main():
    device = setDevice(gpu)       #set run on CPU/GPU
    tracemalloc.start()
    memorySize_before = tracemalloc.take_snapshot()

    network = createNetModel(device)

    memorySize_after = tracemalloc.take_snapshot()
    stats = memorySize_after.compare_to(memorySize_before, 'lineno')

    total_model_memory_usage = sum(stat.size for stat in stats)

    print(f"Total memory usage by the model: {total_model_memory_usage / 1024 ** 2:.2f} MB")
    print("in kB:",{total_model_memory_usage})

    print("\nMemory usage difference after creating the model:")
    for stat in stats[:10]:  # Print top 10 memory allocations
        print(stat)

    tracemalloc.stop()

    #=================== Training Network ========================
    train_data, capacity_data = loadDataset(data_path, trainBattery)    #load training data
    #calculate theoretical SoC using  dataset
    attrib = ['cycle', 'time', 'capacity', 'current_measured']
    SoC_data = train_data[attrib]
    SoC = [1.0]

    for i in range(1, len(SoC_data)):
        # Check if the cycle has changed
        if SoC_data['cycle'][i] != SoC_data['cycle'][i - 1]:
            SoC.append(1.0)  # reset SoC list to 100%
            continue  # skip to the next iteration

        ptime = SoC_data.loc[i - 1, 'time']  # previous time
        ctime = SoC_data.loc[i, 'time']  # current time
        pSoc = SoC[i - 1]  # previous SoC value
        current = SoC_data.loc[i, 'current_measured']  # current value
        capacity = SoC_data.loc[i, 'capacity']  # capacity value
        cSoC = CalculateSoC(pSoc, capacity, current, ptime, ctime)
        SoC.append(cSoC)

    SoC = pd.DataFrame(data=SoC, columns=['SoC'])
    print(SoC_data.head(10))
    print(SoC.head(10))
    print(SoC.shape)


    # NOTE: add different battery properties as training dataset input
    #       Later can reduce number of inputs or use statistical process as input
    attribs = ['capacity', 'voltage_measured', 'current_measured',
               'temperature_measured', 'current_load', 'voltage_load', 'time']
    train_dataset = train_data[attribs]
    #train_dataset_debug = train_dataset
    sc = MinMaxScaler(feature_range=(0, 1))
    train_dataset = sc.fit_transform(train_dataset)  # narmalize input value at range [0,1]
    #train_dataset_normalize_debug = train_dataset

    #x = (soh['SoH'].values).reshape(-1,1)
    #combine inputs/features and target as single Tensor Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_dataset),torch.tensor( (SoC['SoC'].values).reshape(-1,1) ) )
    trained_model = trainNetwork(network, train_dataset, device)        # training the network

    # =================== Testing Network ========================
    test_data, capacity_data = loadDataset(data_path, testBattery)  # load testing data
    # calculate SoC using dataset

    attrib = ['cycle', 'time', 'capacity', 'current_measured']
    SoC_data_test = test_data[attrib]
    SoC_test = [1.0]

    for i in range(1, len(SoC_data_test)):
        # Check if the cycle has changed
        if SoC_data_test['cycle'][i] != SoC_data_test['cycle'][i - 1]:
            SoC_test.append(1.0)  # reset SoC list to 100%
            continue  # skip to the next iteration

        ptime = SoC_data_test.loc[i - 1, 'time']  # previous time
        ctime = SoC_data_test.loc[i, 'time']  # current time
        pSoc = SoC_test[i - 1]  # previous SoC value
        current = SoC_data_test.loc[i, 'current_measured']  # current value
        capacity = SoC_data_test.loc[i, 'capacity']  # capacity value
        cSoC = CalculateSoC(pSoc, capacity, current, ptime, ctime)
        SoC_test.append(cSoC)

    SoC_test = pd.DataFrame(data=SoC_test, columns=['SoC'])

    # NOTE: The properties should similar with training data
    attribs = ['capacity', 'voltage_measured', 'current_measured',
               'temperature_measured', 'current_load', 'voltage_load', 'time']
    test_dataset = sc.fit_transform(test_data[attribs])  # narmalize testdataset at range [0,1]

    # combine inputs/features and target as single Tensor Dataset
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data['cycle']),
                                                  torch.tensor(test_dataset),
                                                   torch.tensor((SoC_test['SoC'].values).reshape(-1, 1)),
                                                  torch.tensor(test_data['time']))

    results = testNetwork(network, trained_model, test_dataset, device)  # testing the network

    #=============Analyze the Model Performance============
    performanceAnalysis(results)

    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()