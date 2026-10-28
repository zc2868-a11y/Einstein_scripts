# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:13:17 2026

@author: theov
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
import tkinter
import pandas as pd
from os.path import isfile
from tkinter import filedialog
import os
from pyprep.find_noisy_channels import NoisyChannels
from pyprep.prep_pipeline import PrepPipeline
import re
import glob

#Loading filepaths
batch_names = [
    "ds005505",
    "ds005506",
    "ds005507",
    "ds005508",
    "ds005509",
    "ds005510",
    "ds005511",
    "ds005512",
    "ds005514",]

file_paths = [
    rf"Z:\Analysis\HBN-EEG\Automated_Preprocessed_data_{batch}"
    for batch in batch_names]

epoch_type = 'dots'
exclude_subjects = [
    "sub-NDARKA085YRG",
    "sub-NDARKM784VA7",
    "sub-NDARWP732GD1",
    "sub-NDARZE389XF0",
    "sub-NDARUC715YN4",
    "sub-NDARBN611GRA",
    "sub-NDARCB627KDN"]

list_files_8target = []
list_files_6target = []


for batch_path in file_paths:

    list_files_8target.extend(
        glob.glob(
            os.path.join(
                batch_path,
                'sub-*',
                'eeg',
                f'*seqLearning8target_epochs_{epoch_type}_EEG_epo.fif')))

    list_files_6target.extend(
        glob.glob(
            os.path.join(
                batch_path,
                'sub-*',
                'eeg',
                f'*seqLearning6target_epochs_{epoch_type}_EEG_epo.fif')))

list_files_8target = [
    f for f in list_files_8target
    if not any(subj in f for subj in exclude_subjects)]

list_files_6target = [
    f for f in list_files_6target
    if not any(subj in f for subj in exclude_subjects)]


# Exclude participant with mismatched memory labels from dot power analysis
list_files_8target = [
    f for f in list_files_8target
    if "sub-NDARCJ011TTG" not in f
]


list_files_8target.sort()
list_files_6target.sort()

print("8-target participants:", len(list_files_8target))
print("6-target participants:", len(list_files_6target))
print(
    "Combined participants:",
    len(list_files_8target) + len(list_files_6target))



#Stimulus type
if epoch_type == 'dots':
    Stimulus_type_8target = [
        'dot_no1_ON', 'dot_no2_ON', 'dot_no3_ON', 'dot_no4_ON',
        'dot_no5_ON', 'dot_no6_ON', 'dot_no7_ON', 'dot_no8_ON']
    Stimulus_type_6target = [
        'dot_no1_ON', 'dot_no2_ON', 'dot_no3_ON',
        'dot_no4_ON', 'dot_no5_ON', 'dot_no6_ON']

elif epoch_type == 'learningBlock':
    Stimulus_type_8target = [
        'learningBlock_1', 'learningBlock_2', 'learningBlock_3',
        'learningBlock_4', 'learningBlock_5']
    Stimulus_type_6target = Stimulus_type_8target

elif epoch_type == 'continuous':
    Stimulus_type_8target = ['seqLearning_start']
    Stimulus_type_6target = Stimulus_type_8target

#%% match EEG dot epochs with remembered/forgotten status

def get_behavior_files(eeg_file):
    behavior_file = eeg_file.replace(
        '_epochs_dots_EEG_epo.fif',
        '_accuracy_df_behavioral.csv')
    events_file = eeg_file.replace(
        '_epochs_dots_EEG_epo.fif',
        '_events_df_behavioral.csv')
    return behavior_file, events_file

def add_memory_status(eeg_epochs, eeg_file):
    behavior_file, events_file = get_behavior_files(eeg_file)
    accuracy_df = pd.read_csv(behavior_file)
    events_df = pd.read_csv(events_file)
    memory_status = []
    for _, row in accuracy_df.iterrows():
        user = str(row['user_answer']).split('-')
        correct = str(row['correct_answer']).split('-')
        status = [
            'remembered' if u == c else 'forgotten'
            for u, c in zip(user, correct)]
        memory_status.extend(status)

    dot_rows = events_df[
        events_df['value'].isin([
            'dot_no1_ON',
            'dot_no2_ON',
            'dot_no3_ON',
            'dot_no4_ON',
            'dot_no5_ON',
            'dot_no6_ON',
            'dot_no7_ON',
            'dot_no8_ON'])].copy()
    dot_rows = dot_rows.sort_values('sample').reset_index(drop=True)

    if len(dot_rows) != len(memory_status):
        raise ValueError(f'Dot events ({len(dot_rows)}) do not match 'f'memory labels ({len(memory_status)})')

    # Give every ORIGINAL dot its remembered/forgotten label
    dot_rows['memory_status'] = memory_status

    # Match only dots that survived EEG artifact rejection
    surviving_samples = eeg_epochs.events[:, 0]
    metadata = []
    for sample in surviving_samples:
        match = dot_rows[dot_rows['sample'] == sample]
        if len(match) == 1:
            metadata.append(match.iloc[0]['memory_status'])
        else:
            metadata.append(None)

    eeg_epochs.metadata = pd.DataFrame({'memory_status': metadata})
    print(os.path.basename(eeg_file),
        eeg_epochs.metadata['memory_status'].value_counts())
    return eeg_epochs

#remembered vs forgotten power
def calculate_power(list_files_eeg):
    Power_remembered = []
    Power_forgotten = []
    # Empty list for ITPC and NPL-Power
    ITPC_remembered = []
    ITPC_forgotten = []
    Power_remembered_NPL = []
    Power_forgotten_NPL = []
    
    for i in range(len(list_files_eeg)):
        eeg_epochs = mne.read_epochs(list_files_eeg[i],preload=True)
        eeg_epochs = add_memory_status(eeg_epochs,list_files_eeg[i])
        remembered = eeg_epochs["memory_status == 'remembered'"]
        forgotten = eeg_epochs["memory_status == 'forgotten'"]
        print(f"Remembered: {len(remembered)}, "f"Forgotten: {len(forgotten)}")

        # Participant needs both conditions
        if len(remembered) == 0 or len(forgotten) == 0:
            print("Skipping participant: missing one condition")
            continue

        freqs = np.linspace(*np.array([3, 40]),num=100)
        n_cycles = np.linspace(*np.array([3, 10]),num=100)

        power_remembered, itpc_remembered = mne.time_frequency.tfr_morlet(remembered,freqs=freqs,n_cycles=n_cycles,use_fft=True,return_itc=True,decim=4,n_jobs=1,average=True)
        power_forgotten, itpc_forgotten = mne.time_frequency.tfr_morlet(forgotten,freqs=freqs,n_cycles=n_cycles,use_fft=True,return_itc=True,decim=4,n_jobs=1,average=True)

        Power_remembered.append(power_remembered)
        Power_forgotten.append(power_forgotten)
        
        ITPC_remembered.append(itpc_remembered)
        ITPC_forgotten.append(itpc_forgotten)
        
        # Calculate the non-phase-locked activity (induced)
        remembered_erp = remembered.average()
        NPL_epochs_remembered =remembered.copy()
        
        #Remove that ERP from all trials one by one
        for i in range(0,len(NPL_epochs_remembered)):
            NPL_epochs_remembered._data[i, :, :] = (NPL_epochs_remembered._data[i, :, :] - remembered_erp.data)
            
        # Calculate the erp
        forgotten_erp = forgotten.average()
        NPL_epochs_forgotten =forgotten.copy()
        
        #Remove that ERP from all trials one by one
        for i in range(0,len(NPL_epochs_forgotten)):
            NPL_epochs_forgotten._data[i, :, :] = (NPL_epochs_forgotten._data[i, :, :] - forgotten_erp.data)
            
        power_remembered_NPL, itpc_remembered_NPL = mne.time_frequency.tfr_morlet(NPL_epochs_remembered,freqs=freqs,n_cycles=n_cycles,use_fft=True,return_itc=True,decim=4,n_jobs=1,average=True)
        power_forgotten_NPL, itpc_forgotten_NPL = mne.time_frequency.tfr_morlet(NPL_epochs_forgotten,freqs=freqs,n_cycles=n_cycles,use_fft=True,return_itc=True,decim=4,n_jobs=1,average=True)
    
        Power_remembered_NPL.append(power_remembered_NPL)
        Power_forgotten_NPL.append(power_forgotten_NPL)
        

    return Power_remembered, Power_forgotten, Power_remembered_NPL, Power_forgotten_NPL, ITPC_remembered, ITPC_forgotten

Power_remembered_8target, Power_forgotten_8target, \
Power_remembered_NPL_8target, Power_forgotten_NPL_8target, \
ITPC_remembered_8target, ITPC_forgotten_8target = \
    calculate_power(list_files_8target)
    
Power_remembered_6target, Power_forgotten_6target, \
Power_remembered_NPL_6target, Power_forgotten_NPL_6target, \
ITPC_remembered_6target, ITPC_forgotten_6target = \
    calculate_power(list_files_6target)
    
Power_remembered_combined = (Power_remembered_8target +Power_remembered_6target)
Power_forgotten_combined = (Power_forgotten_8target +Power_forgotten_6target)
Power_remembered_NPL_combined = (Power_remembered_NPL_8target + Power_remembered_NPL_6target)
Power_forgotten_NPL_combined = (Power_forgotten_NPL_8target +Power_forgotten_NPL_6target)
ITPC_remembered_combined = (ITPC_remembered_8target + ITPC_remembered_6target)
ITPC_forgotten_combined = (ITPC_forgotten_8target +ITPC_forgotten_6target)

#%% save file
import pickle
output_path = ('Z://Analysis/HBN-EEG/''Power_dots/all_batches_Power')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 8-target Power
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_remembered_8target.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_8target, fp)
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_forgotten_8target.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_8target, fp)

# 6-target Power
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_remembered_6target.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_6target, fp)
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_forgotten_6target.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_6target, fp)

# Combined Power
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_remembered_combined.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_combined, fp)
with open(os.path.join(
        output_path,
        f'Power_{epoch_type}_forgotten_combined.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_combined, fp)

# 8-target NPL
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_remembered_8target.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_NPL_8target, fp)
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_forgotten_8target.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_NPL_8target, fp)

# 8-target ITPC
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_remembered_8target.pkl'), 'wb') as fp:
    pickle.dump(ITPC_remembered_8target, fp)
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_forgotten_8target.pkl'), 'wb') as fp:
    pickle.dump(ITPC_forgotten_8target, fp)

# 6-target NPL
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_remembered_6target.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_NPL_6target, fp)
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_forgotten_6target.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_NPL_6target, fp)

# 6-target ITPC
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_remembered_6target.pkl'), 'wb') as fp:
    pickle.dump(ITPC_remembered_6target, fp)
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_forgotten_6target.pkl'), 'wb') as fp:
    pickle.dump(ITPC_forgotten_6target, fp)


# Combined NPL
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_remembered_combined.pkl'), 'wb') as fp:
    pickle.dump(Power_remembered_NPL_combined, fp)
with open(os.path.join(
        output_path,
        f'Power_NPL_{epoch_type}_forgotten_combined.pkl'), 'wb') as fp:
    pickle.dump(Power_forgotten_NPL_combined, fp)


# Combined ITPC
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_remembered_combined.pkl'), 'wb') as fp:
    pickle.dump(ITPC_remembered_combined, fp)
with open(os.path.join(
        output_path,
        f'ITPC_{epoch_type}_forgotten_combined.pkl'), 'wb') as fp:
    pickle.dump(ITPC_forgotten_combined, fp)

print("\nFiles saved successfully:")


#%% LearningBlock power extraction

# LearningBlock files
list_files_8target_learningBlock = []
list_files_6target_learningBlock = []


for batch_path in file_paths:

    list_files_8target_learningBlock.extend(
        glob.glob(
            os.path.join(
                batch_path,
                'sub-*',
                'eeg',
                '*seqLearning8target_epochs_learningBlock_EEG_epo.fif')))

    list_files_6target_learningBlock.extend(
        glob.glob(
            os.path.join(
                batch_path,
                'sub-*',
                'eeg',
                '*seqLearning6target_epochs_learningBlock_EEG_epo.fif')))

list_files_8target_learningBlock.sort()
list_files_6target_learningBlock.sort()

# Exclude noisy participants
list_files_8target_learningBlock = [
    f for f in list_files_8target_learningBlock
    if not any(subj in f for subj in exclude_subjects)
]

list_files_6target_learningBlock = [
    f for f in list_files_6target_learningBlock
    if not any(subj in f for subj in exclude_subjects)
]

learning_blocks = [
    'learningBlock_1',
    'learningBlock_2',
    'learningBlock_3',
    'learningBlock_4',
    'learningBlock_5']

#%%

def calculate_power_learningBlock(list_files_eeg):
    Power_blocks = {
        block: [] for block in learning_blocks}

    freqs = np.linspace(*np.array([3, 40]), num=100)
    n_cycles = np.linspace(*np.array([3, 10]), num=100)

    for i in range(len(list_files_eeg)):
        eeg_epochs = mne.read_epochs(
            list_files_eeg[i],
            preload=True)

        print(
            "\nParticipant:",
            os.path.basename(list_files_eeg[i]))

        for block in learning_blocks:
            if block not in eeg_epochs.event_id:
                print(block, "not found")
                continue

            block_epochs = eeg_epochs[block]
            print(
                block,
                "epochs:",
                len(block_epochs))
            if len(block_epochs) == 0:
                continue

            power_block = mne.time_frequency.tfr_morlet(
                block_epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                decim=4,
                n_jobs=1,
                average=True)
            Power_blocks[block].append(power_block)
    return Power_blocks
#%%

Power_learningBlock_8target = calculate_power_learningBlock(
    list_files_8target_learningBlock)

Power_learningBlock_6target = calculate_power_learningBlock(
    list_files_6target_learningBlock)

Power_learningBlock_combined = {}

for block in learning_blocks:
    Power_learningBlock_combined[block] = (
        Power_learningBlock_8target[block]
        + Power_learningBlock_6target[block])
    
#%% Save LearningBlock power
import pickle
output_path = ('Z://Analysis/HBN-EEG/''Power_learningBlock/all_batches_total_Power')
if not os.path.exists(output_path):
    os.makedirs(output_path)

for block in learning_blocks:
    block_num = block.replace('learningBlock_', '')

    # 8-target
    with open(
        os.path.join(
            output_path,
            f'Power_learningBlock_{block_num}_8target.pkl'),'wb') as fp:
        pickle.dump(
            Power_learningBlock_8target[block],
            fp)

    # 6-target
    with open(
        os.path.join(
            output_path,
            f'Power_learningBlock_{block_num}_6target.pkl'),'wb') as fp:
        pickle.dump(
            Power_learningBlock_6target[block],
            fp)

    # Combined
    with open(
        os.path.join(
            output_path,
            f'Power_learningBlock_{block_num}_combined.pkl'),'wb') as fp:
        pickle.dump(
            Power_learningBlock_combined[block],
            fp)

print("\nLearningBlock files saved successfully")
