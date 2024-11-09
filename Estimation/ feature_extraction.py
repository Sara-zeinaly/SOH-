# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:50:57 2023

@author: West
"""
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis


def findFirstPeak(arr):
    n = len(arr)
    if n == 1:
        return (arr[0],0)
    if arr[0] >= arr[1]:
        return (arr[0],0)
    for i in range(1, n-1):
        if arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
            return (arr[i],i)
    if arr[n-1] >= arr[n-2]:
        return (arr[n-1],n-1)
    return -1  # No peak found

def findLastPeak(arr):
    n = len(arr)
    if n == 1:
        return (arr[0],0)
    if arr[n-1] >= arr[n-2]:
        return (arr[n-1],n-1)
    for i in range(n-2, 0, -1):
        if arr[i] >= arr[i-1] and arr[i] >= arr[i+1]:
            return (arr[i],i)
    if arr[0] >= arr[1]:
        return (arr[0],0)
    return -1  # No peak found

def smooth(df_smooth):
    for i in tqdm(range(len(df_smooth) - 1)):
        if df_smooth['Voltage'][i] - df_smooth['Voltage'][i+1] <= 0:
            df_smooth = df_smooth.drop(i)
    df_smooth = df_smooth.reset_index(drop=True)   
    return df_smooth['Voltage']

def trend(x):
    if len(smooth(x)) == len(x['Voltage']):
         return False
    else: 
        return True
    
shape = (10 , 22, 15) 
features = np.zeros(shape)

Cells = ['W3', 'W4', 'W5', 'W7', 'W8', 'W9', 'W10', 'G1', 'V4', 'V5']
# ew3333333312Cells = ['W5', 'W8', 'W9', 'W10']
for k, Cell in enumerate(Cells):
    for j in tqdm(range(1,16)):
        if Cell == 'W3':
            if j in range(1,4):
                os.chdir("F:\Data\Capacity_Test_Diags\W3")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W3_Channel_3 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_3_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W4':
            if j in range(1,4):
                os.chdir("F:\Data\Capacity_Test_Diags\W4")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W4_Channel_4 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_4_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
            if j in range(4,9):
                os.chdir("F:\Data\Capacity_Test_Diags\W4")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W4_Channel_2 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_2_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W5':
            os.chdir("F:\Data\Capacity_Test_Diags\W5")
            if j in range(1,4):
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W5_Channel_5 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_5_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
            else:
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W5_Channel_2 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_2_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W7':
            if j in range(1,5):
                os.chdir("F:\Data\Capacity_Test_Diags\W7")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W7_Channel_3 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_3_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W8': 
                os.chdir("F:\Data\Capacity_Test_Diags\W8")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W8_Channel_4 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_4_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W9':
                os.chdir("F:\Data\Capacity_Test_Diags\W9")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W9_Channel_5 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_5_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'W10':
                os.chdir("F:\Data\Capacity_Test_Diags\W10")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_W10_Channel_6 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_6_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'G1':
            if j in range(1,12):
                os.chdir("F:\Data\Capacity_Test_Diags\G1")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_G1_Channel_3 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_3_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'V4':
            if j in range(1,12):
                os.chdir("F:\Data\Capacity_Test_Diags\V4")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_V4_Channel_1 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_1_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])
        if Cell == 'V5':
            if j in range(1,5):
                os.chdir("F:\Data\Capacity_Test_Diags\V5")
                Batch = f'INR21700_M50T_T23_Aging_0_05C_V5_Channel_1 ({j}).xlsx'
                df = pd.read_excel(Batch, sheet_name='Channel_1_1', usecols=['Test_Time(s)', 'Voltage(V)','Current(A)','Discharge_Capacity(Ah)','Charge_Capacity(Ah)','Step_Index','Aux_Temperature(¡æ)_1'])

        
        current = []
        Test_Time = []    
        for index, row in df.iterrows():
            if row['Step_Index'] == 3:
                current.append(row['Current(A)'])
                Test_Time.append(row['Test_Time(s)'])
                
        plt.plot(df['Test_Time(s)'], df['Voltage(V)'], label='Voltage')
        plt.plot(df['Test_Time(s)'], df['Charge_Capacity(Ah)'], label='Charge Capacity')
        plt.plot(df['Test_Time(s)'], df['Step_Index'], label='Step Index')
        plt.legend()
        
        #1st feature
        L  = Test_Time[-1]-Test_Time[0]
        #2nd feature
        changes_made = True
        while changes_made:
            changes_made = False  # Reset the flag at the beginning of each iteration
            indices_to_delete = []
            for i in tqdm(range(len(current) - 1, 0, -1)):
                if current[i] - current[i - 1] >= 0:
                    indices_to_delete.append(i)
                    changes_made = True  # Set the flag to True when changes are made
            if indices_to_delete:
                current = np.delete(current, indices_to_delete)
                Test_Time = np.delete(Test_Time, indices_to_delete)
        
        interp_func = interp1d(Test_Time, current, kind='linear')
        Interpolated_Test_Time = np.arange(Test_Time[0], Test_Time[-1], 1)
        Interpolated_current = interp_func(Interpolated_Test_Time)
        
        changes_made = True
        while changes_made:
            changes_made = False  # Reset the flag at the beginning of each iteration
            indices_to_delete = []
            for i in tqdm(range(len(Interpolated_current) - 1, 0, -1)):
                if Interpolated_current[i] - Interpolated_current[i - 1] >= 0:
                    indices_to_delete.append(i)
                    changes_made = True  # Set the flag to True when changes are made
            if indices_to_delete:
                Interpolated_current = np.delete(Interpolated_current, indices_to_delete)
                Interpolated_Test_Time = np.delete(Interpolated_Test_Time, indices_to_delete)
        
        dy_dx = np.gradient(Interpolated_current, Interpolated_Test_Time)
        d2y_dx2 = np.gradient(dy_dx, Interpolated_Test_Time)
        expression = (1 + dy_dx**2)**(3/2) / d2y_dx2
        reciprocal_expression = 1 / expression
        
        plt.figure(figsize=(8, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(Interpolated_Test_Time, Interpolated_current, label='Current')
        plt.xlabel('Test Time')
        plt.ylabel('Current')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(Interpolated_Test_Time, reciprocal_expression, label='Reciprocal')
        plt.xlabel('Test Time')
        plt.ylabel('1/m Reciprocal')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        r = np.max(np.abs(reciprocal_expression))
        #3rd feature 
        A = np.trapz(current, Test_Time)
        #4th feature
        Voltage = []
        Test_Time = []
        i=0
        for index, row in df.iterrows():
            if row['Step_Index'] == 5:
                Voltage.append(row['Voltage(V)'])
                Test_Time.append(row['Test_Time(s)'])
                i+=1
            if i==500:
                break
            
        changes_made = True
        while changes_made:
            changes_made = False  # Reset the flag at the beginning of each iteration
            indices_to_delete = []
            for i in tqdm(range(len(Voltage) - 1, 0, -1)):
                if Voltage[i] - Voltage[i - 1] >= 0:
                    indices_to_delete.append(i)
                    changes_made = True  # Set the flag to True when changes are made
            if indices_to_delete:
                Voltage = np.delete(Voltage, indices_to_delete)
                Test_Time = np.delete(Test_Time, indices_to_delete)
        
        interp_func = interp1d(Test_Time, Voltage, kind='cubic')
        Interpolated_Test_Time = np.arange(Test_Time[0], Test_Time[-1], 1)
        Interpolated_Voltage = interp_func(Interpolated_Test_Time)
        
        changes_made = True
        while changes_made:
            changes_made = False  # Reset the flag at the beginning of each iteration
            indices_to_delete = []
            for i in tqdm(range(len(Interpolated_Voltage) - 1, 0, -1)):
                if Interpolated_Voltage[i] - Interpolated_Voltage[i - 1] >= 0:
                    indices_to_delete.append(i)
                    changes_made = True  # Set the flag to True when changes are made
            if indices_to_delete:
                Interpolated_Voltage = np.delete(Interpolated_Voltage, indices_to_delete)
                Interpolated_Test_Time = np.delete(Interpolated_Test_Time, indices_to_delete)
                
        max_slope = 0.0
        for i in range(len(Interpolated_Test_Time) - 1):
            slope = abs((Interpolated_Voltage[i + 1] - Interpolated_Voltage[i]) / (Interpolated_Test_Time[i + 1] - Interpolated_Test_Time[i]))
            max_slope = max(max_slope, slope)
        S = max_slope   
        
        plt.plot(Interpolated_Test_Time, Interpolated_Voltage, label='Voltage')
        plt.legend()
        
        plt.plot(df['Test_Time(s)'],df['Step_Index'])
        plt.plot(df['Test_Time(s)'],df['Voltage(V)'])
        plt.legend()
        plt.show()

        for index, row in df.iterrows():
            if row['Step_Index'] == 2:
                Rest_Time = df['Test_Time(s)'][index-1]
                break

        for index, row in df.iterrows():
            if row['Step_Index'] == 3:
                CC_Time = df['Test_Time(s)'][index-1]-Rest_Time
                break

        for index, row in df.iterrows():
            if row['Step_Index'] == 4:
                CV_Time = df['Test_Time(s)'][index-1]-CC_Time-Rest_Time
                break

            
        for index, row in df.iterrows():
            if df['Step_Index'][index] == 4 and df['Step_Index'][index+1] == 5:
                i = index 
            if df['Step_Index'][index] == 5 and df['Step_Index'][index+1] == 6:
                Discharge_Time = df['Test_Time(s)'][index]- df['Test_Time(s)'][i]
                break

        F1 = CC_Time
        F2 = CV_Time
        F3 = CC_Time + CV_Time
        F4 = Discharge_Time
        
        plt.plot(df['Test_Time(s)'],df['Discharge_Capacity(Ah)'])
        plt.plot(df['Test_Time(s)'],df['Voltage(V)'])
        plt.legend()
        plt.show()
        plt.plot(df['Voltage(V)'],df['Discharge_Capacity(Ah)'])
        plt.show()

        discharge_capacity = []
        Voltage = []
        for i in range(len(df)):
            if df.loc[i,'Step_Index'] == 5 and df.loc[i,'Voltage(V)'] <3.4:
                Voltage.append({'Voltage': df.loc[i,'Voltage(V)']})
                discharge_capacity.append({'Discharge_Capacity':  df.loc[i,'Discharge_Capacity(Ah)']})
        discharge_capacity = pd.DataFrame(discharge_capacity)
        Voltage = pd.DataFrame(Voltage)
        f = interp1d(Voltage['Voltage'], discharge_capacity['Discharge_Capacity'], bounds_error=False)
        Voltage_interpolated = np.arange(3.40, 2.50, -0.0001)
        Discharge_Capacity_interpolated = f(Voltage_interpolated)
        plt.plot(Voltage_interpolated,Discharge_Capacity_interpolated)
        plt.show()
        

        #F7 = min(df['Discharge_Capacity(Ah)'])
        F8 = np.mean(df['Discharge_Capacity(Ah)'])
        F9 = np.max(df['Discharge_Capacity(Ah)'])    
        F10 = skew(df['Discharge_Capacity(Ah)'])   
        F11 = kurtosis(df['Discharge_Capacity(Ah)'])   
        
        df_D_1 = pd.DataFrame()
        for i in range(len(df)):
            if df.loc[i, 'Step_Index'] == 5:
                new_row = {'Voltage': df.loc[i, 'Voltage(V)'], 'Discharge_Capacity': df.loc[i, 'Discharge_Capacity(Ah)']}
                df_D_1 = pd.concat([df_D_1, pd.DataFrame(new_row, index=[0])], axis=0, ignore_index=True)


        f1 = interp1d(df_D_1['Voltage'], df_D_1['Discharge_Capacity'], bounds_error=False)
        Voltage_interpolated = np.arange(4.2, 2.7, -0.02)
        Discharge_Capacity_interpolated_1 = f1(Voltage_interpolated)  

        incremental_capacity_1 = []
        for i in range(1, len(Voltage_interpolated)):
            diff = Voltage_interpolated[i] - Voltage_interpolated[i-1] 
            ic = (Discharge_Capacity_interpolated_1[i] - Discharge_Capacity_interpolated_1[i-1]) / diff    
            incremental_capacity_1 = np.append(incremental_capacity_1, ic)
        peaks, _ = find_peaks(incremental_capacity_1, distance=200)
        peak_values = incremental_capacity_1[peaks]
        first_peak_1 = peak_values[0]
        first_peak_1_Voltage = Voltage_interpolated[peaks[0]]
        print("Peak values:", peak_values)
        plt.plot(incremental_capacity_1)
        plt.plot(peaks, incremental_capacity_1[peaks], "x")
        plt.show()
        
        F12 = first_peak_1
        F13 = first_peak_1_Voltage
        
        
        first_peak_1,i_1 = findFirstPeak(df['Aux_Temperature(¡æ)_1'])
        last_peak_1,i_2 = findLastPeak(df['Aux_Temperature(¡æ)_1'])

        F23 = first_peak_1
        F24 = df['Test_Time(s)'][i_1] 
        F25 = last_peak_1
        F26 = df['Test_Time(s)'][i_2] 
        plt.plot(df['Test_Time(s)'], df['Aux_Temperature(¡æ)_1'])
        plt.show()

        integral = trapz(df['Aux_Temperature(¡æ)_1'], x=df['Test_Time(s)'])

        F27 = integral
        F28 = df['Aux_Temperature(¡æ)_1'].min() 
        F29 = df['Aux_Temperature(¡æ)_1'].mean() 
        F30 = df['Aux_Temperature(¡æ)_1'].max() 

        
        features[k][0][j-1] = F1
        features[k][1][j-1] = F2
        features[k][2][j-1] = F3
        features[k][3][j-1] = F4
        features[k][4][j-1] = F8
        features[k][5][j-1] = F9
        features[k][6][j-1] = F10
        features[k][7][j-1] = F11
        features[k][8][j-1] = F12
        features[k][9][j-1] = F13
        features[k][10][j-1] = F23
        features[k][11][j-1] = F24
        features[k][12][j-1] = F25
        features[k][13][j-1] = F26
        features[k][14][j-1] = F27
        features[k][15][j-1] = F28
        features[k][16][j-1] = F29
        features[k][17][j-1] = F30
        
        features[k][18][j-1] = L
        features[k][19][j-1] = r
        features[k][20][j-1] = A
        features[k][21][j-1] = S
