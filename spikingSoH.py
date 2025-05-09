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
from tqdm import tqdm
from bindsnet.encoding.encodings import poisson, poisson_normalized
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

#================== Parameter Setting =======================
n_neurons = 100              # Number of neuron in reservoir network
n_hidden = 20                # Number of hidden layer in regression model
n_epochs = 10               # Number of epoch (training iteration) on regression model
transfer_function = 'sigmoid'  # Change this to 'sigmoid', 'tanh' or 'relu' as needed
learning_rate = 0.01        # Regression model learning rate
momentum = 0.9              # Regression model momentum
n_batch = 1                 # Number of data that being process during data loader. For now not able to use batch training.
time = 30                   # window time. conversion time window and spike rate (that represent value)
dt = 1.0                    # simulation timestep
max_rate = 20               # max spiking rate each neuron during conversion
avg_ISI = 1                 # average inter spike interval during conversion
trefrac = 2                 # refractory period in LIF
tdecay = 30                 # time decay in LIF
datatype = 'float'          # change between e.g 'float', 'int32' and 'uint8' to reduce memory consumption. KIV-too complicated
examples = 5000              # Number of training data. Reduce the number of training data with "examples"
test_samples = 10000           # Number of test data. Reduce numer of testing data for faster result
train_all = True           # set "True" for use all data during training. set "False" with number of "examples"
test_all = True             # set "True" for use all data during testing. set "False" with number of "test_samples"
#-----------------------------------------------------------------------------------------------------------------------
inpConn_density = 0.5       # Connection density between input and reservoir network
resConn_density = 0.5       # Connection density within nodes in reservoir network
                            # set range between [0~1]. 1:all connected; 0.5:averagely 50% connections; 0:not connected
inpConn_weight  = 5.0       # Max value connection weight between input and reservoir network
resConn_weight = 5.0        # Max value connection weight between nodes in reservoir network
                            # The weight also depend on this parameter "inpConn_type"/"resConn_type"
inpConn_type = ['uni','exe', 0.5]          # 1st argument: set 'uni' for uniform weight, set 'rand' for random weight
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
'''
n_neuron; n_hidden; n_epochs; windowtime; avgISI; maxrate; datatype; inptConnDense; ResConnDense;
inptConnType; inptExeInh; inptInhbProb; ResConnType; ResExeInh; ResInhbProb;
trainexamples; avgloss; traintime; trainpeakmemory; testsamples; testtime; MSE
'''
analysis_result = [n_neurons, n_hidden, n_epochs, time, avg_ISI, max_rate, datatype,
                    inpConn_density, resConn_density,
                    inpConn_type[0], inpConn_type[1], inpConn_type[2],
                    resConn_type[0], resConn_type[1] ,resConn_type[2]]


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


    #return conn_matrix.to(dtype=eval(f"torch.{datatype}"))     #KIV-weight need to define as float. To many part to change
    return conn_matrix

#----Create Reservoir Network Model----
def createNetModel(device):
    # Create simple Torch NN
    network = Network(dt=dt)
    inpt = Input(7, shape=(1, 7))
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

    #for debugging
    conn_matrix_1_debug = conn_matrix_1.numpy()
    conn_matrix_2_debug = conn_matrix_2.numpy()

    C1 = Connection(source=inpt, target=output, w=conn_matrix_1.to(dtype=torch.float))    #KIV-defined weight using other type
    C2 = Connection(source=output, target=output, w=conn_matrix_2.to(dtype=torch.float))

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

    # start training the network - record training time and memory usage
    tracemalloc.start()  # start memory usage monitoring
    start_time = mtime.time()  # for record training time

    n_iters = examples
    training_pairs = []             #list of spikes from reservoir network
    pbar = tqdm(enumerate(dataloader))


    for (i, dataPoint) in pbar:
        if train_all is False and i > n_iters:             #increase example value to increase accuracy
            break

        datum = dataPoint[0].float()
        target = dataPoint[1].float()

        #encoded_datum = poisson(datum, time=time, dt=dt)
        encoded_datum = poisson_normalized(datum, maxrate=max_rate, avgISI=avg_ISI, time=time, dt=dt)
        encoded_datum = encoded_datum.to(dtype=eval(f"torch.{datatype}"))
        encoded_datum_debug = (encoded_datum.squeeze()).numpy()

        if train_all:
            pbar.set_description_str("Train progress: (%d / %d)" % (i, len(train_dataset)))
        else:
            pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

        # Run network on sample image
        network.run(inputs={"I": encoded_datum}, time=time, input_time_dim=1)
        #spikedata = spikes["O"].get("s").sum(0)
        #spikedata = spikedata.to(dtype=eval(f"torch.{datatype}"))
        training_pairs.append([spikes["O"].get("s").sum(0).float(), target])
        #training_pairs.append([spikedata, target])

        # use debug to view training pairs
        #training_pairs_debug = training_pairs[i][0].numpy()

        network.reset_state_variables()

    #use debug to view training pairs
    #final_training_pairs_debug = np.concatenate([np.concatenate((np.array(data[0]), np.array(data[1])), axis=1) for data in training_pairs], axis=0)
    print("Input data completed..")

    # Define logistic regression model.
    # The input is a spikes count from each neuron on reservoir
    # The output is battery SoH
    torch_function = getattr(torch, transfer_function)
    class LogisticReg(nn.Module):
        def __init__(self, input_size):
            super(LogisticReg, self).__init__()
            self.hidden1 = nn.Linear(input_size, n_hidden)
            #self.hidden2 = nn.Linear(10, n_hidden)
            self.linear = nn.Linear(n_hidden, 1)

        def forward(self, x):
            x1 = torch_function(self.hidden1(x))         #alternatively, use "out = torch.sigmoid(self.linear(x1))"
            #x2 = torch_function(self.hidden2(x1))
            out = torch_function(self.linear(x1))
            return out

    # Create and train logistic regression model on reservoir outputs.
    model = LogisticReg(n_neurons).to(device)
    criterion = nn.MSELoss(reduction="sum")     #Mean Squared Error Loss. Please check the meaning of redcution
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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

    print("Training Completed ... ")
    end_time = mtime.time()
    print("Training Time:%d", end_time - start_time)
    _, peakRam = tracemalloc.get_traced_memory()
    print("Peak Memory usage:", peakRam)
    tracemalloc.stop()

    analysis_result.append(avg_loss/len(training_pairs))
    analysis_result.append(end_time-start_time)
    analysis_result.append(peakRam)


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

        #encoded_datum = poisson(datum, time=time, dt=dt)
        encoded_datum = poisson_normalized(datum, maxrate=max_rate, avgISI=avg_ISI, time=time, dt=dt)
        # debug_encoded_datum = (encoded_datum.squeeze()).numpy()
        pbar.set_description_str("Testing progress: (%d / %d)" % (i, len(test_dataset)))

        # Run network on sample image
        network.run(inputs={"I": encoded_datum}, time=time, input_time_dim=1)
        testing_pairs.append([cycle, spikes["O"].get("s").sum(0).float(), target])

        # use debug to view testing pairs
        #testing_pairs_debug = testing_pairs[i][0].numpy()

        network.reset_state_variables()

    # Test model with previously trained logistic regression classifier
    results = []
    for c, s, calc_soh in testing_pairs:
        pred_soh = model(s)
        results.append([c.item(), calc_soh.item(), pred_soh.item()])

    end_time = mtime.time()
    print("Testing Time:%d", end_time - start_time)
    analysis_result.append(end_time-start_time)

    return pd.DataFrame(data=results,columns=['cycle', 'SoH', 'PredSoH'])


def performanceAnalysis(data):
    data = data.sort_values(by='cycle')
    data = data.groupby(['cycle']).mean().reset_index()
    print(data.sample(10))
    rms = np.sqrt(mean_squared_error(data['SoH'], data['PredSoH']))
    print('Root Mean Square Error: ', rms)
    analysis_result.append(rms)

    # plot final result
    if plot is True:
        plot_df = data.loc[(data['cycle'] >= 1), ['cycle', 'SoH', 'PredSoH']]
        sns.set_style("white")
        plt.figure(figsize=(8, 5))
        plt.plot(plot_df['cycle'], plot_df['SoH'], label='Calculated SoH')
        plt.plot(plot_df['cycle'], plot_df['PredSoH'], label='Predicted SoH')
        # Draw threshold
        # plt.plot([0.,len(capacity)], [0.70, 0.70], label='Threshold')
        plt.ylabel('SOH')
        # make x-axis ticks legible
        adf = plt.gca().get_xaxis().get_major_formatter()
        plt.xlabel('cycle')
        plt.legend()
        plt.title('Comparison Calculated vs Predicted SoH')
        plt.show(block=True)

    if record is True:
        #record simulation log
        with open('simulation_analysis.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(analysis_result)

        #record SoH result
        data.to_csv('SoH_results.csv', index=False)


    return

def main():
    device = setDevice(gpu)       #set run on CPU/GPU
    network = createNetModel(device)

    #=================== Training Network ========================
    train_data, capacity_data = loadDataset(data_path, trainBattery)    #load training data
    #calculate SoH based on dataset for each cycle -
    # NOTE: Later try to optimize by insert it on loadDataset function
    #       only use for plot SoH graph
    attrib = ['cycle', 'capacity']
    SoH_data = capacity_data[attrib]
    initcap = capacity_data['capacity'][0]
    for i in range(len(SoH_data)):
        SoH_data['SoH'] = (SoH_data['capacity']) / initcap  # current_capacity/initial_capacity

    # plot graph battery SoH vs cycle
    if plot is True:
        plot_df = SoH_data.loc[(SoH_data['cycle'] >= 1), ['cycle', 'SoH']]
        sns.set_style("white")
        plt.figure(figsize=(8, 5))
        plt.plot(plot_df['cycle'], plot_df['SoH'])
        # Draw threshold at 70% of battery SoH
        plt.plot([0., len(capacity_data)], [0.70, 0.70])
        plt.ylabel('SoH')
        # make x-axis ticks legible
        adf = plt.gca().get_xaxis().get_major_formatter()
        plt.xlabel('cycle')
        plt.title('Battery Discharge(' + trainBattery + ')')
        plt.show(block=True)

    # create training dataset
    initcap = train_data['capacity'][0]
    soh = []
    for i in range(len(train_data)):  # add SoH for each cycle on training dataset output
        soh.append([train_data['capacity'][i] / initcap])
    soh = pd.DataFrame(data=soh, columns=['SoH'])

    # NOTE: add different battery properties as training dataset input
    #       Later can reduce number of inputs or use statistical process as input
    attribs = ['capacity', 'voltage_measured', 'current_measured',
               'temperature_measured', 'current_load', 'voltage_load', 'time']
    train_dataset = train_data[attribs]
    #train_dataset_debug = train_dataset
    sc = MinMaxScaler(feature_range=(0, 1))
    train_dataset = sc.fit_transform(train_dataset)  # narmalize input value at range [0,1]
    #train_dataset_normalize_debug = train_dataset
    print(soh.head(10))
    #x = (soh['SoH'].values).reshape(-1,1)
    #combine inputs/features and target as single Tensor Dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_dataset),torch.tensor( (soh['SoH'].values).reshape(-1,1) ) )
    trained_model = trainNetwork(network, train_dataset, device)        # training the network

    # =================== Testing Network ========================
    test_data, capacity_data = loadDataset(data_path, testBattery)  # load testing data
    # calculate SoH based on dataset for each cycle -
    # NOTE: Later try to optimize by insert it on loadDataset function
    #       only use for plot SoH graph
    attrib = ['cycle', 'capacity']
    SoH_data = capacity_data[attrib]
    initcap = capacity_data['capacity'][0]
    for i in range(len(SoH_data)):
        SoH_data['SoH'] = (SoH_data['capacity']) / initcap  # current_capacity/initial_capacity

    # create testing dataset
    initcap = test_data['capacity'][0]
    soh = []
    for i in range(len(test_data)):  # add SoH for each cycle on test dataset output
        soh.append([test_data['capacity'][i] / initcap])
    soh = pd.DataFrame(data=soh, columns=['SoH'])  #need as testing label

    # NOTE: The properties should similar with training data
    attribs = ['capacity', 'voltage_measured', 'current_measured',
               'temperature_measured', 'current_load', 'voltage_load', 'time']
    test_dataset = sc.fit_transform(test_data[attribs])  # narmalize testdataset at range [0,1]

    # combine inputs/features and target as single Tensor Dataset
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data['cycle']),
                                                  torch.tensor(test_dataset),
                                                   torch.tensor((soh['SoH'].values).reshape(-1, 1)))

    results = testNetwork(network, trained_model, test_dataset, device)  # testing the network

    #=============Analyze the Model Performance============
    performanceAnalysis(results)

    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()