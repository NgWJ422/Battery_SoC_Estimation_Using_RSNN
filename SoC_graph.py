from scipy.io import loadmat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import time
import psutil
from memory_profiler import profile
import threading

# Ensure the output directory exists
BatteryName = 'B0018'       # Battery Name: B0005, B0006, B0007, B0018
output_dir = os.path.join('plots', f'{BatteryName}_plots')
os.makedirs(output_dir, exist_ok=True)

#load .mat file
dataset = loadmat('battery_data/' + BatteryName+ '.mat')
alldata = dataset[BatteryName][0,0]['cycle'][0]

def monitor_memory(interval=0.1, duration=5):
    memory_usage = []
    timestamps = []
    process = psutil.Process()
    start_time = time.time()
    while time.time() - start_time < duration:
        memory_usage.append(process.memory_info().rss / 1024 / 1024)  # Convert to MB
        timestamps.append(time.time() - start_time)
        time.sleep(interval)
    return timestamps, memory_usage

#Discharge Cycle starts from 1
def DischargeNumber():
    NumDischarge = 0
    for i in range(len(alldata)):
        row = alldata[i]
        if row['type'][0] != 'discharge':
            continue

        NumDischarge = NumDischarge + 1

    print("Number of Discharge cycle: " + str(NumDischarge))


def LoopAndStoreSoC(DcycleSelected = -1):
    DischargeCycleIndex = 0 # Initialize DischargeCycleIndex
    nfield = 0
    timeTo50_list = []
    timeTo20_list = []
    timeTo0_list = []
    average_gradient_list = []
    for i in range(len(alldata)):
        row = alldata[i]
        nfield += 1
        if row['type'][0] == 'discharge':
            DischargeCycleIndex = DischargeCycleIndex + 1
            Dcycle_output_dir = os.path.join(output_dir, f'Dcycle_{DischargeCycleIndex}({nfield})')
            os.makedirs(Dcycle_output_dir, exist_ok=True)
            if DcycleSelected != -1 and DischargeCycleIndex != DcycleSelected:
                continue

            capacity = row['data'][0,0]['Capacity'][0,0] * 3600  # Convert from Ah to As
            SoC = [np.float64(100)]  # Initial SoC value
            Current_measured = [row['data'][0,0]['Current_measured'][0][0]]
            Time = [row['data'][0,0]['Time'][0][0]]
            Voltage_measured = [row['data'][0,0]['Voltage_measured'][0,0]]
            Temperature = [row['data'][0,0]['Temperature_measured'][0,0]]
            Current_load = [row['data'][0,0]['Current_load'][0,0]]
            Voltage_load = [row['data'][0,0]['Voltage_load'][0,0]]
            for j in range(1, len(row['data'][0,0]['Time'][0])):
                ptime = row['data'][0,0]['Time'][0][j-1]
                ctime = row['data'][0,0]['Time'][0][j]
                pSoc = SoC[j-1]
                current = row['data'][0,0]['Current_measured'][0][j]
                Time.append(ctime)
                Current_measured.append(current)
                Voltage_measured.append(row['data'][0,0]['Voltage_measured'][0,j])
                Temperature.append(row['data'][0,0]['Temperature_measured'][0,j])
                Current_load.append(row['data'][0,0]['Current_load'][0,j])
                Voltage_load.append(row['data'][0,0]['Voltage_load'][0,j])

                # if(SoC[j-1] == 0):
                #     SoC.append(0)
                #     continue
                SoC.append(CalculateSoC(pSoc, capacity, current, ptime, ctime))
            print("Last SoC: " + str(SoC[-1]))
            # Create a DataFrame
            data = {
                'Time': Time,
                'Current_measured': Current_measured,
                'SoC': SoC,
                'Capacity': [capacity] * len(Time),
                'Voltage_measured': Voltage_measured,
                'Temperature': Temperature,
                'Current_load': Current_load,
                'Voltage_load': Voltage_load
            }
            first_length = len(data['Time'])
            for key, value in data.items():
                if len(value) != first_length:
                    raise ValueError(f"Error: The length of '{key}' is {len(value)}, which is different from the reference length of {first_length}, Discharged cycle: {DischargeCycleIndex}({nfield}).")
            print(f"All arrays have the same length of {first_length},  Discharged cycle: {DischargeCycleIndex}({nfield}).")
            df = pd.DataFrame(data)
            df['Gradient'] = np.gradient(df['SoC'], df['Time'])
            time_to_50 = timeToDropTo(50, df)
            time_to_20 = timeToDropTo(20, df)
            time_to_0 = timeToDropTo(0, df)
            timeTo50_list.append(time_to_50)
            timeTo20_list.append(time_to_20)
            timeTo0_list.append(time_to_0)
            average_gradient = averageGradient(df)
            average_gradient_list.append(average_gradient)
            if(DcycleSelected == DischargeCycleIndex):
                return df
            else:                
                saveplot_and_data(df, DischargeCycleIndex, Dcycle_output_dir)
                save_txt_data(df, DischargeCycleIndex, average_gradient, nfield, time_to_50, time_to_20, time_to_0, Dcycle_output_dir)
            if (DcycleSelected != -1):
                break
    
    SummaryPlot(timeTo50_list, timeTo20_list, timeTo0_list, average_gradient_list, DischargeCycleIndex)

def SummaryPlot(timeTo50_list, timeTo20_list, timeTo0_list, average_gradient, DischargeCycleIndex):
    # Helper function to create and save plot
    def create_and_save_plot(data_list, title, filename):
        x = list(range(1, DischargeCycleIndex + 1))  # X-axis values: 1, 2, 3, ..., len(data_list)
        plt.figure()
        plt.plot(x, data_list)
        plt.xlabel('Index')
        plt.ylabel(title)
        plt.title(f'{title} Plot')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # Create and save each plot
    create_and_save_plot(timeTo50_list, 'Time to 50%', 'time_to_50.png')
    create_and_save_plot(timeTo20_list, 'Time to 20%', 'time_to_20.png')
    create_and_save_plot(timeTo0_list, 'Time to 0%', 'time_to_0.png')
    create_and_save_plot(average_gradient, 'Average Gradient', 'average_gradient.png')


def saveplot_and_data(df, cycle_number, dir = output_dir):
    plotParameter(df, cycle_number, 'Current_measured', dir)
    plotParameter(df, cycle_number, 'Voltage_measured', dir)
    plotParameter(df, cycle_number, 'Temperature', dir)
    plotParameter(df, cycle_number, 'Current_load', dir)
    plotParameter(df, cycle_number, 'Voltage_load', dir)
    plotParameter(df, cycle_number, 'Gradient', dir)
    plotandsaveSoC(df, cycle_number, dir)

def save_txt_data(df, cycle_number,average_gradient,field_number, timeto50, timeto20, timeto0, dir = output_dir):
    # Create a text file to save the data
    txt_output_path = os.path.join(dir, f'Dcycle_{cycle_number}_data.txt')
    with open(txt_output_path, 'w') as file:
        # Save the last value in the SoC column
        last_soc = df['SoC'].iloc[-1]
        file.write(f"Last SoC: {last_soc}\n")
        
        # Save the number of discharge cycles
        file.write(f"Discharge Cycle Number: {cycle_number}\n")

        # Save the number of cycles
        file.write(f"Cycle(Field) Number: {field_number}\n")

        # Save the average gradient
        file.write(f"Average Gradient of SoC: {average_gradient}\n")
        
        # Save the time to drop to 50%, 20%, and 0% SoC

        file.write(f"Time to drop to 50% SoC: {timeto50}\n")
        file.write(f"Time to drop to 20% SoC: {timeto20}\n")
        file.write(f"Time to drop to 0% SoC: {timeto0}\n")
    
    print(f"Saved data for cycle {cycle_number} at {txt_output_path}")

def CalculateSoC(pSoc, capacity, current, ptime, ctime):
    # Calculate the time difference
    delta_time = ctime - ptime
    
    # Calculate the integral assuming constant current
    integral_current = current * delta_time
    
    # Calculate the SoC
    SoC = pSoc + (100 / capacity) * integral_current
    # Ensure SoC is between 0 and 100
    #SoC = max(0, min(SoC, 100))  
    
    return SoC


def plotParameter(df, cycle_number, parameter, dir = output_dir):
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create a line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y=parameter, data=df, label=parameter)

    # Customize the plot
    plt.title(f'{parameter} over Time')
    plt.xlabel('Time (s)')
    plt.ylabel(parameter)

    output_path = os.path.join(dir, f'{parameter}_Dcycle_{cycle_number}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot for cycle {cycle_number} at {output_path}")

def plotandsaveSoC(df, cycle_number, dir = output_dir):
    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create a line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Time', y='SoC', data=df, label='SoC')

    plt.axhline(y=50, color='g', linestyle='--')
    plt.axhline(y=20, color='b', linestyle='--')
    plt.axhline(y=10, color='r', linestyle='--')

    # Customize the plot
    plt.title('State of Charge (SoC) over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('State of Charge (%)')

    output_path = os.path.join(dir, f'SoC_{cycle_number}_{len(df["Time"])}.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved plot for cycle {cycle_number} at {output_path}")


def averageGradient(df_gradient):
    # Identify the steady region
    Assume_steady_current = -1.9
    steady_region = df_gradient[df_gradient['Current_measured'] < Assume_steady_current]
    steady_region_start = steady_region.index[0]
    steady_region_end = steady_region.index[-1]
    start_time = df_gradient.loc[steady_region_start, 'Time']
    end_time = df_gradient.loc[steady_region_end, 'Time']

    steady_df = df_gradient[(df_gradient['Time'] >= start_time) & (df_gradient['Time'] <= end_time)]

    return steady_df['Gradient'].mean()
    

def timeToDropTo(SoCPercent, df):
    for i in range(len(df['SoC'])):
        if df['SoC'][i] < SoCPercent:
            return df['Time'][i]
    raise ValueError(f"Error: The SoC never drops below {SoCPercent}%.")
   
@profile
def main():

    LoopAndStoreSoC()
    DischargeNumber()
    pass
    
if __name__ == "__main__":
    # Measure computational time
    start_time = time.time()

    # Start memory monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_memory, args=(0.1, 10))
    monitor_thread.start()

    # Call the main function
    main()

    # Wait for the monitoring thread to finish
    monitor_thread.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total computational time: {elapsed_time} seconds")

    # Plot memory usage
    timestamps, memory_usage = monitor_memory()
    plt.plot(timestamps, memory_usage)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.grid(True)
    plt.show()
