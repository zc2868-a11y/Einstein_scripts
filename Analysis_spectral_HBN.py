# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:25:08 2026

@author: theov
"""

#%% Load power data

import os
import pickle
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

epoch_type = 'dots'
# epoch_type = 'learningBlock'
# epoch_type = 'continuous'

target_type = '8target'
# target_type = '6target'
# target_type = 'combined'

measure_type = 'Power'
# measure_type = 'NPL'
# measure_type = 'ITPC'

if measure_type == 'Power':
    file_prefix = 'Power'
    measure_label = 'Total power'
elif measure_type == 'NPL':
    file_prefix = 'Power_NPL'
    measure_label = 'Non-phase-locked power'
elif measure_type == 'ITPC':
    file_prefix = 'ITPC'
    measure_label = 'ITPC'

if epoch_type == 'dots':
    power_path = (
        'Z://Analysis/HBN-EEG/'
        'Power_dots/all_batches_Power')

elif epoch_type == 'learningBlock':
    power_path = (
        'Z://Analysis/HBN-EEG/'
        'Power_learningBlock/all_batches_total_Power')
    
#%% Load condition data

if epoch_type == 'dots':
    conditions= ['remembered', 'forgotten']
    Power_conditions = {}
    for condition in conditions:
        file = os.path.join (
            power_path,
            f' {file_prefix}_{epoch_type}_{condition}_{target_type}.pkl')
        with open (file, 'rb') as f:
            Power_conditions[condition]=pickle.load(f)
elif epoch_type == 'learningBlock':
    conditions = [
        'learningBlock_1',
        'learningBlock_2',
        'learningBlock_3',
        'learningBlock_4',
        'learningBlock_5']
    Power_conditions = {}
    for condition in conditions:
        block_num = condition.replace( 'learningBlock_', '')
        file = os.path.join (power_path, f'Power_learningBlock_{block_num}_{target_type}.pkl')
        with open (file, 'rb') as f:
            Power_conditions[condition]=pickle.load(f)
for condition in conditions:
    print(condition, "participants:", len (Power_conditions[condition]))

#%% Baseline correction

if epoch_type == 'continuous': baseline = (-1.0, -0.1)
elif epoch_type == 'learningBlock': baseline = (-0.4, -0.1)
elif epoch_type == 'dots': baseline = (-0.4, -0.1)

if measure_type != 'ITPC':
    for condition in conditions:
        for power in Power_conditions[condition]:
            power.apply_baseline(baseline=baseline,mode='percent')

M_Power = {}
for condition in conditions:
    M_Power[condition] = mne.grand_average(Power_conditions[condition])

#%% Topomap: Remembered vs Forgotten

# Alpha frequency range
fmin = 8
fmax = 13

# Period after dot onset
tmin1 = 0.0
tmax1 = 0.6

if measure_type == 'ITPC':
    vlim1 = 0
    vlim2 = 0.3
else:
    vlim1 = -0.3
    vlim2 = 0.3

for condition in conditions:
    fig = M_Power[condition].plot_topomap(tmin=tmin1,tmax=tmax1,fmin=fmin,fmax=fmax,baseline=None,cmap='jet',vlim=(vlim1, vlim2))
    if measure_type == 'ITPC':
        fig.axes[-1].set_title('ITPC')
    else:
        fig.axes[-1].set_title('Power change\nfrom baseline')

    fig.suptitle(
        f'Alpha {measure_label}: '
        f'{condition} - {target_type}')
    plt.show()

#%% Time-frequency map

Electrodes = ['E66', 'E71', 'E72','E77', 'E84', 'E85']
if measure_type == 'ITPC':
    tf_vlim = (0, 0.3)
else:
    tf_vlim = (-0.6, 0.6)

for condition in conditions:
    M_Power[condition].plot(Electrodes,baseline=None,cmap='jet',combine='mean',tmin=-0.3,tmax=1.0,vlim=tf_vlim,fmin=3,fmax=40)
    plt.suptitle(
        f'{measure_label}: '
        f'{condition} - {target_type}')
    plt.show()

#%% Extract theta/alpha/gamma power

Electrodes = ['E66', 'E71', 'E72', 'E77', 'E84', 'E85']
tmin = -0.3
tmax = 1.0

def extract_band_power(power_list,fmin,fmax):
    Band_power = []
    for power in power_list:
        p = (power.copy().pick(Electrodes).crop(tmin=tmin,tmax=tmax))

        freq_mask = ((p.freqs >= fmin) & (p.freqs <= fmax))
        # Average across:
        # channels + frequencies
        band = p.data[:, freq_mask, :].mean(axis=(0, 1))
        Band_power.append(band)
    return np.array(Band_power)

Theta = {}
Alpha = {}
Gamma = {}

for condition in conditions:
    Theta[condition] = extract_band_power(
        Power_conditions[condition],4,7)

    Alpha[condition] = extract_band_power(
        Power_conditions[condition],8,13)

    Gamma[condition] = extract_band_power(
        Power_conditions[condition],30,40)
    
#%% Time-frequency plot (theta/alpha/gamma power)
def plot_band(Band,band_name,freq_label):
    first_condition = conditions[0]
    x = (Power_conditions[first_condition][0].copy().
         crop(tmin=tmin,tmax=tmax).times)
    fig, ax = plt.subplots()

    for condition in conditions:
        mean = np.mean(Band[condition],axis=0)

        sem = (np.std(Band[condition],axis=0)
            / np.sqrt(len(Band[condition])))

        ax.plot(x,mean,label=condition)
        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            alpha=0.2)
    ax.axvline(0,linestyle='--')

    if measure_type == 'ITPC':
        ax.set_ylim(0, 0.5)
        ax.set_ylabel('ITPC')
    else:
        ax.axhline(0,linestyle='--')
        ax.set_ylabel('Power change from baseline')
    if epoch_type == 'dots':
        ax.set_xlabel('Time from dot onset (s)')
    elif epoch_type == 'learningBlock':
        ax.set_xlabel('Time from learning block onset (s)')

    ax.set_title(
        f'{band_name} '
        f'{measure_label} '
        f'{freq_label}: '
        f'{target_type}')
    ax.legend()
    plt.show()
    return x

x = plot_band(
    Theta,
    'Theta',
    '4–7 Hz')

plot_band(
    Alpha,
    'Alpha',
    '8–13 Hz')

plot_band(
    Gamma,
    'Gamma',
    '30–40 Hz')
# #%% Theta: remembered vs forgotten

# x = (Power_remembered[0].copy().crop(tmin=tmin,tmax=tmax).times)

# theta_mean_rem = np.mean(Theta_remembered,axis=0)
# theta_mean_forg = np.mean(Theta_forgotten,axis=0)

# theta_sem_rem = (np.std(Theta_remembered,axis=0) / np.sqrt(len(Theta_remembered)))
# theta_sem_forg = (np.std(Theta_forgotten,axis=0) / np.sqrt(len(Theta_forgotten)))

# fig, ax = plt.subplots()
# ax.plot(x,theta_mean_rem,label='Remembered')
# ax.plot(x,theta_mean_forg,label='Forgotten')

# ax.fill_between(x,theta_mean_rem - theta_sem_rem,theta_mean_rem + theta_sem_rem,alpha=0.2)
# ax.fill_between(x,theta_mean_forg - theta_sem_forg,theta_mean_forg + theta_sem_forg,alpha=0.2)
# ax.axvline(0, linestyle='--')

# if measure_type == 'ITPC':
#     ax.set_ylim(0, 0.5)
# else:
#     ax.axhline(0, linestyle='--')
# ax.set_xlabel('Time from dot onset (s)')
# if measure_type == 'ITPC':
#     ax.set_ylabel('ITPC')
# else:
#     ax.set_ylabel('Power change from baseline')
# ax.set_title(f'Theta {measure_label} 4–7 Hz: {target_type}')
# ax.legend()
# plt.show()
# #%% Alpha: remembered vs forgotten

# x = (Power_remembered[0].copy().crop(tmin=tmin,tmax=tmax).times)
# alpha_mean_rem = np.mean(Alpha_remembered,axis=0)

# alpha_mean_forg = np.mean(Alpha_forgotten,axis=0)
# alpha_sem_rem = (np.std(Alpha_remembered,axis=0)/ np.sqrt(len(Alpha_remembered)))
# alpha_sem_forg = (np.std(Alpha_forgotten,axis=0)/ np.sqrt(len(Alpha_forgotten)))

# fig, ax = plt.subplots()
# ax.plot(x,alpha_mean_rem,label='Remembered')

# ax.plot(x,alpha_mean_forg,label='Forgotten')
# ax.fill_between(x,alpha_mean_rem - alpha_sem_rem,alpha_mean_rem + alpha_sem_rem,alpha=0.2)
# ax.fill_between(x,alpha_mean_forg - alpha_sem_forg,alpha_mean_forg + alpha_sem_forg,alpha=0.2)

# ax.axvline(0, linestyle='--')
# if measure_type == 'ITPC':
#     ax.set_ylim(0, 0.5)
# else:
#     ax.axhline(0, linestyle='--')
# ax.set_xlabel('Time from dot onset (s)')
# if measure_type == 'ITPC':
#     ax.set_ylabel('ITPC')
# else:
#     ax.set_ylabel('Power change from baseline')
# ax.set_title(
#     f'Alpha {measure_label} 8–13 Hz: {target_type}')
# ax.legend()
# plt.show()

# #%% Gamma: remembered vs forgotten

# gamma_mean_rem = np.mean(Gamma_remembered,axis=0)
# gamma_mean_forg = np.mean(Gamma_forgotten,axis=0)
# gamma_sem_rem = (np.std(Gamma_remembered,axis=0)
#     / np.sqrt(len(Gamma_remembered)))

# gamma_sem_forg = (np.std(Gamma_forgotten,axis=0)
#     / np.sqrt(len(Gamma_forgotten)))

# fig, ax = plt.subplots()
# ax.plot(x,gamma_mean_rem,label='Remembered')

# ax.plot(x,gamma_mean_forg,label='Forgotten')
# ax.fill_between(x,gamma_mean_rem - gamma_sem_rem,gamma_mean_rem + gamma_sem_rem,alpha=0.2)
# ax.fill_between(x,gamma_mean_forg - gamma_sem_forg,gamma_mean_forg + gamma_sem_forg,alpha=0.2)

# ax.axvline(0, linestyle='--')
# if measure_type == 'ITPC':
#     ax.set_ylim(0, 0.3)
# else:
#     ax.axhline(0, linestyle='--')
# ax.set_xlabel('Time from dot onset (s)')
# if measure_type == 'ITPC':
#     ax.set_ylabel('ITPC')
# else:
#     ax.set_ylabel('Power change from baseline')
# ax.set_title(
#     f'Gamma {measure_label} 30–40 Hz: {target_type}')
# ax.legend()
# plt.show()


#%% Statistical test

from scipy.stats import ttest_rel
import pingouin as pg
import pandas as pd

analysis_tmin = 0.0
analysis_tmax = 0.6

time_mask = (
    (x >= analysis_tmin)
    & (x <= analysis_tmax))


#%% DOTS: remembered vs forgotten
if epoch_type == 'dots':
    theta_rem_subject = (
        Theta['remembered'][:, time_mask]
        .mean(axis=1))
    theta_forg_subject = (
        Theta['forgotten'][:, time_mask]
        .mean(axis=1))

    alpha_rem_subject = (
        Alpha['remembered'][:, time_mask]
        .mean(axis=1))
    alpha_forg_subject = (
        Alpha['forgotten'][:, time_mask]
        .mean(axis=1))

    gamma_rem_subject = (
        Gamma['remembered'][:, time_mask]
        .mean(axis=1))
    gamma_forg_subject = (
        Gamma['forgotten'][:, time_mask]
        .mean(axis=1))


    # Paired t-tests
    t_theta, p_theta = ttest_rel(
        theta_rem_subject,
        theta_forg_subject)

    t_alpha, p_alpha = ttest_rel(
        alpha_rem_subject,
        alpha_forg_subject)

    t_gamma, p_gamma = ttest_rel(
        gamma_rem_subject,
        gamma_forg_subject)


    print("MEASURE:", measure_label)
    print("GROUP:", target_type)
    print("\nTHETA: 4–7 Hz")
    print("Remembered mean:",
        np.mean(theta_rem_subject))
    print("Forgotten mean:",
        np.mean(theta_forg_subject))
    print(
        f"t = {t_theta:.3f}, "
        f"p = {p_theta:.4f}")

    print("\nALPHA: 8–13 Hz")
    print("Remembered mean:",
        np.mean(alpha_rem_subject))
    print(
        "Forgotten mean:",
        np.mean(alpha_forg_subject))
    print(
        f"t = {t_alpha:.3f}, "
        f"p = {p_alpha:.4f}")

    print("\nGAMMA: 30–40 Hz")
    print("Remembered mean:",
        np.mean(gamma_rem_subject))
    print("Forgotten mean:",
        np.mean(gamma_forg_subject))
    print(
        f"t = {t_gamma:.3f}, "
        f"p = {p_gamma:.4f}")

#%% LEARNING BLOCK: repeated-measures ANOVA
elif epoch_type == 'learningBlock':
    def block_stats(Band, band_name):
        rows = []
        for block in conditions:
            subject_power = (
                Band[block][:, time_mask]
                .mean(axis=1))
            for subject, value in enumerate(
                    subject_power):
                rows.append({
                    'Subject': subject,
                    'learning_block': block,
                    'power': value})

        df = pd.DataFrame(rows)
        print("\nMEASURE:", measure_label)
        print("GROUP:", target_type)
        print(f"\n{band_name.upper()}")

        # Mean for each block
        print(
            df.groupby('learning_block')
            ['power']
            .mean())

        # Repeated-measures ANOVA
        print("\nRepeated-measures ANOVA:")
        print(
            pg.rm_anova(
                data=df,
                dv='power',
                within='learning_block',
                subject='Subject',
                detailed=True))

        # Pairwise comparisons
        print("\nPairwise comparisons:")
        print(
            pg.pairwise_tests(
                data=df,
                dv='power',
                within='learning_block',
                subject='Subject',
                padjust='bonf'))
        return df


    theta_stats = block_stats(
        Theta,
        'Theta 4–7 Hz')

    alpha_stats = block_stats(
        Alpha,
        'Alpha 8–13 Hz')

    gamma_stats = block_stats(
        Gamma,
        'Gamma 30–40 Hz')
#%% Stats
# analysis_tmin = 0.0
# analysis_tmax = 0.6

# time_mask = ((x >= analysis_tmin) &(x <= analysis_tmax))
# alpha_rem_subject = (Alpha_remembered[:, time_mask].mean(axis=1))

# theta_rem_subject = (Theta_remembered[:, time_mask].mean(axis=1))
# theta_forg_subject = (Theta_forgotten[:, time_mask].mean(axis=1))

# alpha_forg_subject = (Alpha_forgotten[:, time_mask].mean(axis=1))
# gamma_rem_subject = (Gamma_remembered[:, time_mask].mean(axis=1))

# gamma_forg_subject = (Gamma_forgotten[:, time_mask].mean(axis=1))

# # Theta
# t_theta, p_theta = ttest_rel(theta_rem_subject,theta_forg_subject)
# # Alpha
# t_alpha, p_alpha = ttest_rel(alpha_rem_subject,alpha_forg_subject)
# # Gamma
# t_gamma, p_gamma = ttest_rel(gamma_rem_subject,gamma_forg_subject)
# print("MEASURE:", measure_label)
# print("GROUP:", target_type)

# print("\nTHETA: 4–7 Hz")
# print("Remembered mean:",np.mean(theta_rem_subject))
# print("Forgotten mean:",np.mean(theta_forg_subject))
# print(f"t = {t_theta:.3f}, p = {p_theta:.4f}")

# print("\nALPHA: 8–13 Hz")
# print("Remembered mean:",np.mean(alpha_rem_subject))
# print("Forgotten mean:",np.mean(alpha_forg_subject))
# print(f"t = {t_alpha:.3f}, p = {p_alpha:.4f}")

# print("\nGAMMA: 30–40 Hz")
# print("Remembered mean:",np.mean(gamma_rem_subject))
# print("Forgotten mean:",np.mean(gamma_forg_subject))
# print(f"t = {t_gamma:.3f}, p = {p_gamma:.4f}")