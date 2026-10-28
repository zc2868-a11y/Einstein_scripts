# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:24:11 2026

@author: theov
"""


import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#============================================
#     AVSRT Analysis - Visual Stimulation
#============================================

#loading needed toolboxes 
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
import ast


batch_names = [
    "ds005505",
    "ds005506",
    "ds005507",
    "ds005508",
    "ds005509",
    "ds005510",
    "ds005511",
    "ds005512",
    "ds005514",
]

file_paths = [
    rf"Z:\Analysis\HBN-EEG\Automated_Preprocessed_data_{batch}"
    for batch in batch_names
]


# Load 8-target and 6-target files separately

list_files_eeg_8target = []
list_files_RT_8target = []

list_files_eeg_6target = []
list_files_RT_6target = []


for file_path in file_paths:

    # 8-target
    list_files_eeg_8target.extend(
        glob.glob(
            os.path.join(
                file_path,'sub-*','eeg','*seqLearning8target_epochs_dots_EEG_epo.fif')))

    list_files_RT_8target.extend(
        glob.glob(
            os.path.join(
                file_path,'sub-*','eeg','*seqLearning8target_accuracy_df_behavioral.csv')))

    # 6-target
    list_files_eeg_6target.extend(
        glob.glob(
            os.path.join(
                file_path,'sub-*','eeg','*seqLearning6target_epochs_dots_EEG_epo.fif')))

    list_files_RT_6target.extend(
        glob.glob(
            os.path.join(
                file_path,'sub-*','eeg','*seqLearning6target_accuracy_df_behavioral.csv')))


# Sort
list_files_eeg_8target.sort()
list_files_RT_8target.sort()

list_files_eeg_6target.sort()
list_files_RT_6target.sort()
print("8-target EEG files:", len(list_files_eeg_8target))
print("8-target behavioral files:", len(list_files_RT_8target))

print("6-target EEG files:", len(list_files_eeg_6target))
print("6-target behavioral files:", len(list_files_RT_6target))


# Separate lists
filtered_EEG_lst_8target = []
filtered_RT_lst_8target = []

filtered_EEG_lst_6target = []
filtered_RT_lst_6target = []


# Load 8-target
for i in range(len(list_files_eeg_8target)):

    eeg_epochs = mne.read_epochs(list_files_eeg_8target[i])
    if len(eeg_epochs) == 0:
        print(f"Skipping empty epochs file: {list_files_eeg_8target[i]}")
        continue

    eeg_epochs._data = mne.filter.notch_filter(
    eeg_epochs.get_data(),
    Fs=eeg_epochs.info['sfreq'],
    freqs=60
    )
    eeg_epochs._data = mne.filter.notch_filter(
        eeg_epochs.get_data(),
        Fs=eeg_epochs.info['sfreq'],
        freqs=[60])

    rt_data = pd.read_csv(
        list_files_RT_8target[i],
        sep=',',
        encoding='utf-8')

    rt_data['Subject'] = os.path.basename(
        os.path.dirname(os.path.dirname(list_files_eeg_8target[i])))

    rt_data['task'] = '8target'

    filtered_EEG_lst_8target.append(eeg_epochs)
    filtered_RT_lst_8target.append(rt_data)


# Load 6-target
for i in range(len(list_files_eeg_6target)):

    eeg_epochs = mne.read_epochs(list_files_eeg_6target[i])
    if len(eeg_epochs) == 0:
        print(f"Skipping empty epochs file: {list_files_eeg_6target[i]}")
        continue
    eeg_epochs._data = mne.filter.notch_filter(
        eeg_epochs.get_data(),
        Fs=eeg_epochs.info['sfreq'],
        freqs=[60])

    rt_data = pd.read_csv(
        list_files_RT_6target[i],
        sep=',',
        encoding='utf-8')

    rt_data['Subject'] = os.path.basename(
        os.path.dirname(os.path.dirname(list_files_eeg_6target[i])))

    rt_data['task'] = '6target'

    filtered_EEG_lst_6target.append(eeg_epochs)
    filtered_RT_lst_6target.append(rt_data)


# BOTH tasks together
filtered_EEG_lst = filtered_EEG_lst_8target + filtered_EEG_lst_6target
filtered_RT_lst = filtered_RT_lst_8target + filtered_RT_lst_6target

#%% Attach remembered/forgotten label to each dot epoch
def parse_memory_status(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)

for i, (rt_df, epochs) in enumerate(zip(filtered_RT_lst, filtered_EEG_lst)):
    flat_status = []
    for value in rt_df['memory_status']:
        flat_status.extend(parse_memory_status(value))

    if len(flat_status) == len(epochs):
        epochs.metadata = pd.DataFrame({'memory_status': flat_status})
    else:
        print(f"Subject {i} ({rt_df['Subject'].iloc[0]}): "
              f"{len(flat_status)} memory labels vs {len(epochs)} dot epochs — "
              f"metadata NOT attached, check for dropped epochs/trials")
#%%
# Concatenate all dataframes into one
# 8-target behavioral data
df_8target = pd.concat(filtered_RT_lst_8target, ignore_index=True)

# 6-target behavioral data
df_6target = pd.concat(filtered_RT_lst_6target, ignore_index=True)

# Both together
df_combined = pd.concat(filtered_RT_lst, ignore_index=True)

# Check the combined dataframe
df_combined.head()

# #%% Check number of subjects for dot epochs

# print("Number of subjects:", len(filtered_EEG_lst))
# print("Number of dot epoch files:", len(list_files_eeg))

# for i, epochs in enumerate(filtered_EEG_lst):
#     print(f"Subject {i+1}: {len(epochs)} dot epochs")

#%% Dot-level behavioral accuracy: remembered vs forgotten

def dot_accuracy(df, title):
    df = df.copy()
    df["learning_block"] = (df.groupby("Subject").cumcount() + 1)
    df["memory_status"] = df["memory_status"].apply(parse_memory_status)

    dot_df = df.explode("memory_status")

    dot_df["dot_accuracy"] = (dot_df["memory_status"] == "remembered").astype(int)

    # Mean accuracy for each learning block
    accuracy = dot_df.groupby("learning_block")["dot_accuracy"].agg(
        Mean="mean",
        SD="std",
        N="count")

    print(f"\nDot accuracy - {title}")
    print(accuracy)
    return dot_df, accuracy

dot_df_combined, dot_accuracy_combined = dot_accuracy(
    df_combined, "Combined")

dot_df_8target, dot_accuracy_8target = dot_accuracy(
    df_8target, "8 Target")

dot_df_6target, dot_accuracy_6target = dot_accuracy(
    df_6target, "6 Target")


#%% Overall remembered vs forgotten dot counts

def remembered_forgotten_summary(dot_df, title):
    counts = dot_df["memory_status"].value_counts()
    proportions = dot_df["memory_status"].value_counts(normalize=True)
    
    print(f"\n{title}")
    print("Counts:")
    print(counts)
    print("\nProportions:")
    print(proportions)

remembered_forgotten_summary(dot_df_combined, "Combined")
remembered_forgotten_summary(dot_df_8target, "8 Target")
remembered_forgotten_summary(dot_df_6target, "6 Target")

#%% Option 1: Mixed ANOVA: 6-target vs 8-target across learning blocks
#Does trial accuracy differ between 6-target and 8-target tasks, does accuracy change across the 5 learning blocks, and does that change depend on task version?
from pingouin import rm_anova, pairwise_tests, mixed_anova
import matplotlib.pyplot as plt
import numpy as np

df_combined = df_combined.copy()
df_combined["learning_block"] = (df_combined.groupby("Subject").cumcount() + 1)

anova = mixed_anova(
    data=df_combined,
    dv='trial_accuracy',
    within='learning_block',
    between='task',
    subject='Subject')

print(anova)

#%% Descriptive statistics + Boxplots

import matplotlib.pyplot as plt
import seaborn as sns

def describe_and_plot(df, title):
    df = df.copy()
    df["learning_block"] = (df.groupby("Subject").cumcount() + 1)

    # Descriptive statistics
    desc = df.groupby("learning_block")["trial_accuracy"].agg(
        Mean="mean",
        SD="std",
        N="count"
    )

    print(f"\nDescriptive Statistics {title}:")
    print(desc)

    # Separate plot
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="learning_block",
        y="trial_accuracy"
    )

    plt.xlabel("Learning Block")
    plt.ylabel("Trial Accuracy")
    plt.title(f"Trial Accuracy - {title}")
    plt.ylim(0, 1)

    plt.show()

# Combined
describe_and_plot(df_combined, "Combined")
# 6-target
describe_and_plot(df_6target, "6 Target")
# 8-target
describe_and_plot(df_8target, "8 Target")


#%% ERP

tmin_epochs = -0.7
tmax_epochs =  1.5


# Adjust the calculate_erp function to include the new baseline correction
def calculate_erp(filtered_epochs_lst):
    erp_list = []
    number_stim = []

    # Loop through each subject's epochs
    for epochs in filtered_epochs_lst:
        # epochs.set_eeg_reference(ref_channels=['E17','B18'])
        # epochs.apply_baseline(baseline=(-1.05, -0.95))
        
        # epochs.set_eeg_reference(ref_channels='average')
        epochs.apply_baseline(baseline=(-0.2, 0))

        # Select epochs based on stimulation type
        epochs_stim = epochs
        
        # Calculate the ERP (mean across epochs)
        erp = epochs_stim.average()
        erp_list.append(erp)
        number_stim.append(len(epochs_stim))
    
    return erp_list, number_stim


#Calculate ERP separately for each task

# 8-target only
erp_dots_8target, number_stim_dots_8target = calculate_erp(
    filtered_EEG_lst_8target)

# 6-target only
erp_dots_6target, number_stim_dots_6target = calculate_erp(
    filtered_EEG_lst_6target)

# Both 6-target + 8-target
erp_dots, number_stim_dots = calculate_erp(
    filtered_EEG_lst)

#%% Remembered vs. forgotten ERP (per-subject, then grand average)

def calculate_erp_by_condition(filtered_epochs_lst, query, baseline=(-0.2, 0)):
    erp_list, n_list = [], []
    for epochs in filtered_epochs_lst:
        if epochs.metadata is None:
            erp_list.append(None)
            n_list.append(0)
            continue
        subset = epochs[query]
        if len(subset) == 0:
            erp_list.append(None)
            n_list.append(0)
            continue
        subset = subset.copy().apply_baseline(baseline=baseline)
        erp_list.append(subset.average())
        n_list.append(len(subset))
    return erp_list, n_list


erp_dots_remembered, n_remembered = calculate_erp_by_condition(
    filtered_EEG_lst, "memory_status == 'remembered'")
erp_dots_forgotten, n_forgotten = calculate_erp_by_condition(
    filtered_EEG_lst, "memory_status == 'forgotten'")

print("Remembered trials per subject:", n_remembered)
print("Forgotten trials per subject:", n_forgotten)

#%% Grand-average ERP

Evoked_dots_8target = mne.grand_average(erp_dots_8target)

Evoked_dots_6target = mne.grand_average(erp_dots_6target)

Evoked_dots = mne.grand_average(erp_dots)

Evoked_remembered = mne.grand_average([erp for erp in erp_dots_remembered if erp is not None])
Evoked_forgotten = mne.grand_average([erp for erp in erp_dots_forgotten if erp is not None])

times = [-0.2, 0.00, 0.05, 0.10, 0.15, 0.20, 0.6, 1.5]

Evoked_dots.plot_topomap(times=times, size=1, vlim=(-1.5, 1.5), cmap='jet', average=0.05)
# Evoked_dots_8target.plot_topomap(times=times, size=1, vlim=(-1.5, 1.5), cmap='jet', average=0.05)
# Evoked_dots_6target.plot_topomap(times=times, size=1, vlim=(-1.5, 1.5), cmap='jet', average=0.05)
Evoked_remembered.plot_topomap(times=times,size=1,vlim=(-1.5, 1.5),cmap='jet', average=0.05)
Evoked_forgotten.plot_topomap(times=times,size=1,vlim=(-1.5, 1.5),cmap='jet', average=0.05)

Evoked_difference = mne.combine_evoked([Evoked_remembered, Evoked_forgotten],weights=[1, -1])
Evoked_difference.plot_topomap(times=times,average=0.05,cmap='RdBu_r')


#%% save

plt.savefig('ERP_Topomap_dots_TD.svg')

#%% Plot ERP - within group (Average ERP across all dot trials)

import numpy as np
import matplotlib.pyplot as plt

tmin_epochs = -0.7
tmax_epochs =  1.5

# Function to average ERP over specific channels
def average_erp_channels(erp, channel_list):
    # Select the specified channels and average the ERP over these channels
    selected_channels = erp.copy().pick_channels(channel_list)
    avg_erp = np.mean(selected_channels.data, axis=0)  # Average over channels
    return avg_erp

# Function to calculate SEM (Standard Error of the Mean)
def calculate_sem(erp_list):
    erp_array = np.array([erp.data for erp in erp_list])  # Convert list of ERPs to a 2D array (n_subjects, n_timepoints)
    sem = np.std(erp_array, axis=0) / np.sqrt(erp_array.shape[0])  # SEM = std / sqrt(n_subjects)
    return sem

# Function to plot the average ERP with SEM for the three groups for a given stimulation type
def plot_avg_erp_with_sem(erp_dots, sem_dots, channels, sfreq, plot_tmin=-0.7, plot_tmax=1.5):
    # Calculate time points based on the sampling frequency (sfreq)
    times = filtered_EEG_lst[0].times
    
    # Find the index range corresponding to the desired plot time window (-0.1 to 0.6 seconds)
    idx_min = np.searchsorted(times, plot_tmin)
    idx_max = np.searchsorted(times, plot_tmax)

    # Slice the ERP data, SEM, and time points for the plot time window
    times_plot = times[idx_min:idx_max]
    
    erp_dots_plot = erp_dots[idx_min:idx_max]
    sem_dots_plot = sem_dots[idx_min:idx_max]
    
    # Plotting the ERPs for the three groups with SEM shaded areas
    plt.figure(figsize=(10, 6))

    # dots
    plt.plot(times_plot, erp_dots_plot, label='dots', color='#f768a1')
    plt.fill_between(times_plot, erp_dots_plot - sem_dots_plot, erp_dots_plot + sem_dots_plot, color='#f768a1', alpha=0.3)

    # Add title and labels
    plt.title(f'Average ERP with SEM ({", ".join(channels)})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.axvline(x=times_plot[12], color='black', linestyle='--', label='Cue')

    plt.grid(False)  # No grid lines
    plt.tick_params(bottom=True, left=True)  # Show ticks
    for spine in plt.gca().spines.values():
        spine.set_visible(True)  # Ensure axis frame is visible
    
    # plt.show()
    # plt.ylim([-2e-6,28e-6])
    
    # plt.xlim([-0.95, 0])
    plt.ylim([-5.0e-6, 8.0e-6])
    
    # plt.grid(True)
    plt.show()
    
#%% Plot ERP

channels = ['E66', 'E71', 'E72', 'E77', 'E84', 'E85']
# channels = ['E70', 'E74', 'E75', 'E83', 'E89', 'E90']

# 8 TARGET
avg_erp_dots_8target = np.mean(
    [average_erp_channels(erp, channels)
     for erp in erp_dots_8target],
    axis=0)

sem_erp_dots_8target = np.std(
    [average_erp_channels(erp, channels)
     for erp in erp_dots_8target],
    axis=0
) / np.sqrt(len(erp_dots_8target))

plot_avg_erp_with_sem(
    avg_erp_dots_8target,
    sem_erp_dots_8target,
    channels,
    erp_dots_8target[0].info['sfreq'],
    plot_tmin=-0.2,
    plot_tmax=1.5)

# 6 TARGET
avg_erp_dots_6target = np.mean(
    [average_erp_channels(erp, channels)
     for erp in erp_dots_6target],
    axis=0)

sem_erp_dots_6target = np.std(
    [average_erp_channels(erp, channels)
     for erp in erp_dots_6target],
    axis=0
) / np.sqrt(len(erp_dots_6target))

plot_avg_erp_with_sem(
    avg_erp_dots_6target,
    sem_erp_dots_6target,
    channels,
    erp_dots_6target[0].info['sfreq'],
    plot_tmin=-0.2,
    plot_tmax=1.5)

# BOTH
avg_erp_dots = np.mean(
    [average_erp_channels(erp, channels)
     for erp in erp_dots],
    axis=0)

sem_erp_dots = np.std(
    [average_erp_channels(erp, channels)
     for erp in erp_dots],
    axis=0
) / np.sqrt(len(erp_dots))

plot_avg_erp_with_sem(
    avg_erp_dots,
    sem_erp_dots,
    channels,
    erp_dots[0].info['sfreq'],
    plot_tmin=-0.2,
    plot_tmax=1.5)

#%% Plot all three ERP conditions on the same graph

times = erp_dots[0].times

idx_min = np.searchsorted(times, -0.2)
idx_max = np.searchsorted(times, 1.5)

times_plot = times[idx_min:idx_max]

plt.figure(figsize=(10, 6))

conditions = [
    (avg_erp_dots_8target, sem_erp_dots_8target, '8 Target'),
    (avg_erp_dots_6target, sem_erp_dots_6target, '6 Target'),
    (avg_erp_dots, sem_erp_dots, 'Combined')]

for avg, sem, label in conditions:

    avg_plot = avg[idx_min:idx_max]
    sem_plot = sem[idx_min:idx_max]

    plt.plot(
        times_plot,
        avg_plot,
        label=label)

    plt.fill_between(
        times_plot,
        avg_plot - sem_plot,
        avg_plot + sem_plot,
        alpha=0.2)

plt.axvline(0, color='black', linestyle='--', label='Dot onset')
plt.axhline(0, color='gray', linewidth=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.title(f'ERP by Sequence Length ({", ".join(channels)})')
plt.legend()
plt.ylim([-5.0e-6, 8.0e-6])

plt.show()


#%% Remembered vs. forgotten

def plot_condition_comparison(erp_list_A, label_A, color_A,
                               erp_list_B, label_B, color_B,
                               channels, plot_tmin=-0.7, plot_tmax=1.5, title=None):
    times = filtered_EEG_lst[0].times
    idx_min = np.searchsorted(times, plot_tmin)
    idx_max = np.searchsorted(times, plot_tmax)
    times_plot = times[idx_min:idx_max]

    def prep(erp_list):
        chans = [average_erp_channels(erp, channels) for erp in erp_list if erp is not None]
        avg = np.mean(chans, axis=0)[idx_min:idx_max]
        sem = (np.std(chans, axis=0) / np.sqrt(len(chans)))[idx_min:idx_max]
        return avg, sem, len(chans)

    avg_A, sem_A, n_A = prep(erp_list_A)
    avg_B, sem_B, n_B = prep(erp_list_B)

    plt.figure(figsize=(10, 6))
    plt.plot(times_plot, avg_A, label=f'{label_A} (n={n_A})', color=color_A)
    plt.fill_between(times_plot, avg_A - sem_A, avg_A + sem_A, color=color_A, alpha=0.3)
    plt.plot(times_plot, avg_B, label=f'{label_B} (n={n_B})', color=color_B)
    plt.fill_between(times_plot, avg_B - sem_B, avg_B + sem_B, color=color_B, alpha=0.3)

    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8, label='Dot onset')
    plt.title(title or f'{label_A} vs. {label_B} ({", ".join(channels)})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.legend()
    plt.ylim([-5.0e-6, 8.0e-6])
    plt.grid(False)
    plt.show()

# BOTH
erp_dots_remembered, n_remembered = calculate_erp_by_condition(
    filtered_EEG_lst, "memory_status == 'remembered'")

erp_dots_forgotten, n_forgotten = calculate_erp_by_condition(
    filtered_EEG_lst, "memory_status == 'forgotten'")


# 8 TARGET
erp_dots_remembered_8target, n_remembered_8target = calculate_erp_by_condition(
    filtered_EEG_lst_8target, "memory_status == 'remembered'")

erp_dots_forgotten_8target, n_forgotten_8target = calculate_erp_by_condition(
    filtered_EEG_lst_8target, "memory_status == 'forgotten'")


# 6 TARGET
erp_dots_remembered_6target, n_remembered_6target = calculate_erp_by_condition(
    filtered_EEG_lst_6target, "memory_status == 'remembered'")

erp_dots_forgotten_6target, n_forgotten_6target = calculate_erp_by_condition(
    filtered_EEG_lst_6target, "memory_status == 'forgotten'")


plot_condition_comparison(
    erp_dots_remembered, 'Remembered', '#31a354',
    erp_dots_forgotten, 'Forgotten', '#de2d26',
    channels,
    plot_tmin=-0.2,
    plot_tmax=1.5,
    title='Remembered vs Forgotten - Combined')


plot_condition_comparison(
    erp_dots_remembered_8target, 'Remembered', '#31a354',
    erp_dots_forgotten_8target, 'Forgotten', '#de2d26',
    channels,
    plot_tmin=-0.2,
    plot_tmax=1.5,
    title='Remembered vs Forgotten - 8 Target')

plot_condition_comparison(
    erp_dots_remembered_6target, 'Remembered', '#31a354',
    erp_dots_forgotten_6target, 'Forgotten', '#de2d26',
    channels,
    plot_tmin=-0.2,
    plot_tmax=1.5,
    title='Remembered vs Forgotten - 6 Target')


#%%

def plot_individual_erps_per_subject(erp_list_dots, channels,
                                     plot_tmin=-2.2, plot_tmax=0.8, subject_ids=None):
    """
    Plot one figure per subject showing all three cue_probability conditions.
    """
    times = filtered_EEG_lst[0].times
    idx_min = np.searchsorted(times, plot_tmin)
    idx_max = np.searchsorted(times, plot_tmax)
    times_plot = times[idx_min:idx_max]

    n_subjects = len(erp_list_dots)

    for i in range(n_subjects):
        subj_label = subject_ids[i] if subject_ids is not None else f'Subject {i+1}'

        erp_dots_plot = average_erp_channels(erp_list_dots[i], channels)[idx_min:idx_max]

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(times_plot, erp_dots_plot, label='dots', color='#f768a1', linewidth=1.5)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        ax.set_title(f'{subj_label} — ({", ".join(channels)})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_ylim([-5.0e-6, 6.0e-6])
        ax.legend(fontsize=9)
        ax.tick_params(bottom=True, left=True)
        for spine in ax.spines.values():
            spine.set_visible(True)

        plt.tight_layout()
        plt.show()

print("8-target EEG files:", len(list_files_eeg_8target))
print("6-target EEG files:", len(list_files_eeg_6target))
print("Total EEG files:", len(list_files_eeg_8target) + len(list_files_eeg_6target))

print("\n8-target ERPs:", len(erp_dots_8target))
print("6-target ERPs:", len(erp_dots_6target))
print("Combined ERPs:", len(erp_dots))
#%% Plot one figure per subject
plot_individual_erps_per_subject(erp_dots, channels,
                                 plot_tmin=-2.2, plot_tmax=2.0,
                                 subject_ids=None)  # e.g. subject_ids=['S01', 'S02', ...]
