# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:28:10 2025

@author: zhichen
"""

import matplotlib as mpl

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

#%% 
#============================================================
# FEATURE EXTRACTION FROM CLEAN EEG/ET EPOCHS
# ============================================================

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
from mne import EpochsArray
from ebosc.eBOSC_wrapper import eBOSC_wrapper
from scipy.signal import hilbert
from mne.filter import filter_data
from scipy.stats import zscore
from scipy.signal import windows
from mne import EpochsArray
from scipy.stats import zscore
from scipy.signal import correlate
from scipy.signal import resample
from mne.filter import filter_data
from scipy.ndimage import gaussian_filter1d


group = 'TD'  # change to 'TD' or 'SIB' when running the other groups

file_path = (
    'Z://Analysis/SFARI_Analysis/Resting_State/'
    'Clean_Epochs_EEG_ET/' + group)

list_file = os.listdir(file_path)
list_file.sort()

list_files_epochs = [
    s for s in list_file
    if s.endswith('_RS_EEG_epo.fif')]

all_subjects_blocks = []

print(f"Group: {group}")
print(f"Number of EEG files: {len(list_files_epochs)}")

for i in range(len(list_files_epochs)):

    eeg_filename = list_files_epochs[i]

    Subject_id = eeg_filename.split('_')[0]

    # Match ET file directly from EEG filename
    et_filename = eeg_filename.replace(
        '_RS_EEG_epo.fif',
        '_RS_ET_epo.fif')

    eeg_file = os.path.join(file_path, eeg_filename)
    et_file = os.path.join(file_path, et_filename)

    # Skip participant if paired ET file is missing
    if not os.path.exists(et_file):
        print(f"⚠️ Missing ET file for {Subject_id}")
        continue

    print(f"\nProcessing {Subject_id}")

    eeg_epochs = mne.read_epochs(
        eeg_file,
        preload=True)

    et_epochs = mne.read_epochs(
        et_file,
        preload=True)

    # Match existing resting-state analysis
    eeg_epochs.set_eeg_reference(
        ref_channels='average')

    # Sanity check
    if len(eeg_epochs) != len(et_epochs):
        print(
            f"⚠️ EEG/ET block mismatch for {Subject_id}: "
            f"EEG={len(eeg_epochs)}, ET={len(et_epochs)}")
        continue
    print(f"Number of resting blocks: {len(eeg_epochs)}")
        
    for block_idx in range(len(eeg_epochs)):
        print(f"  Block {block_idx + 1}/{len(eeg_epochs)}")
        block_type = 'resting-state'
        
        # Extract one resting-state block
        eeg_epoch_final = eeg_epochs[block_idx].copy().crop(
            tmin=10,
            tmax=50
            )

        et_epoch_final = et_epochs[block_idx].copy().crop(
            tmin=10,
            tmax=50
            )
    # Put EEG and ET onto the same sampling grid
        target_sfreq = 128

        eeg_epoch_final.resample(target_sfreq)
        et_epoch_final.resample(target_sfreq)

        sfreq_eeg = eeg_epoch_final.info['sfreq']
        sfreq_et = et_epoch_final.info['sfreq']
        print(f"    EEG sfreq={sfreq_eeg}, "f"ET sfreq={sfreq_et}")
    
        # Time-frequency analysis for each quartile 
        freqs = np.linspace(*np.array([2, 40]), num=200) #num = step / bin allows a precision of the frequency resolution
        n_cycles = np.linspace(*np.array([3, 10]), num=200)  # different number of cycle per frequency
        
        # Total power for each quartile
        power_eeg = mne.time_frequency.tfr_morlet(eeg_epoch_final, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=1, n_jobs=1, average=False)
        
        # Log transformation of the power (now in dB)
        power_eeg.data = 10 * np.log10(power_eeg.data)

        
        # Define clusters
        parieto_occipital_cluster = ['O1', 'O2', 'PO7', 'PO3', 'PO4', 'PO8']
        alpha_range = (7, 13)

        # Get alpha frequency indices
        alpha_freq_idx = np.where((power_eeg.freqs >= alpha_range[0]) & (power_eeg.freqs <= alpha_range[1]))[0]

        def compute_power_cluster(power_obj, cluster, freq_idx):
            picks = mne.pick_channels(power_obj.info['ch_names'], include=cluster)
            # Average across epochs, channels, frequencies
            fb_power = power_obj.data[:, :, :, :]  # shape: (n_epochs, n_channels, n_freqs, n_times)
            fb_power = fb_power[:, picks][:, :, freq_idx].mean(axis=(1, 2))  # shape: (n_epochs, n_times)
            return fb_power.mean(axis=0)  # average across epochs → final shape: (n_times,)

       
        alpha_parocc = compute_power_cluster(power_eeg, parieto_occipital_cluster, alpha_freq_idx)

        # Cross-correlation between z-score power and pupil
        time_vector = eeg_epoch_final.times  # Shape: (n_times,)
        
        # Window and step in samples
        # 0.2-s sliding window
        window_sec = 0.2
        step_sec = 0.1

        win_size = int(window_sec * sfreq_eeg)
        step_size = int(step_sec * sfreq_eeg)

        print(f"    Sliding window: {window_sec}s, "f"step ≈ {step_size / sfreq_eeg:.4f}s")

        # Hamming window (optional for smoothing)
        window_func = windows.hamming(win_size)
    
        alpha_z_parocc = zscore(alpha_parocc)
        
        # Sliding average of alpha
        alpha_parocc_means = []
        
        for start in range(0, len(alpha_z_parocc) - win_size + 1, step_size):
            end = start + win_size
            
            segment_alpha_parocc = alpha_z_parocc[start:end] * window_func
            alpha_parocc_means.append(np.mean(segment_alpha_parocc))
        
        alpha_means_parocc = np.array(alpha_parocc_means)
        
        # Step 1: Extract pupil signal
        pupil_idx = et_epoch_final.ch_names.index('pupil')
        pupil_signal = (et_epoch_final.get_data()[0, pupil_idx, :])
        
        # Match pupil signal length to alpha time series if needed
        n_samples_target = len(alpha_parocc)
        if len(pupil_signal) != n_samples_target:
            pupil_resampled = resample(
                pupil_signal,
                n_samples_target)
        else:
            pupil_resampled = pupil_signal
        
        
        # Filtration of the pupil data in 4 different frequency bands
        
        # Very low-frequency -> Negtive load with PCA : 0.01 to 0.1
        pupil_very_low = filter_data(pupil_resampled, sfreq=128, l_freq=0.01, h_freq=0.1)
        pupil_very_low_z = zscore(pupil_very_low)
        
        # Low frequency: 0.1 to 0.5
        pupil_low = filter_data(pupil_resampled, sfreq=128, l_freq=0.1, h_freq=0.5)
        pupil_low_z = zscore(pupil_low)
        
        # Middle frequency: 0.6 to 1.6
        pupil_middle = filter_data(pupil_resampled, sfreq=128, l_freq=0.6, h_freq=1.6)
        pupil_middle_z = zscore(pupil_middle)
        
        # High frequency: 1.8 to 4Hz
        pupil_high = filter_data(pupil_resampled, sfreq=128, l_freq=1.8, h_freq=4.0)
        pupil_high_z = zscore(pupil_high)
        
        # Tonic pupil (slowly varying pupil diameter)
        pupil_tonic = gaussian_filter1d(pupil_resampled, sigma=2 * sfreq_et)
        pupil_tonic_z = zscore(pupil_tonic)

        # Sliding mean
        means_very_low = []
        means_low = []
        means_middle = []
        means_high = []
        means_tonic = []
        win_times = []
        
        for start in range(0, len(pupil_very_low_z) - win_size + 1, step_size):
            end = start + win_size
            segment_very_low = pupil_very_low_z[start:end] * window_func
            segment_low = pupil_low_z[start:end] * window_func
            segment_middle = pupil_middle_z[start:end] * window_func
            segment_high = pupil_high_z[start:end] * window_func
            segment_tonic = pupil_tonic_z[start:end] * window_func

            means_very_low.append(np.mean(segment_very_low))
            means_low.append(np.mean(segment_low))
            means_middle.append(np.mean(segment_middle))
            means_high.append(np.mean(segment_high))
            means_tonic.append(np.mean(segment_tonic))
            
            win_times.append(time_vector[start + win_size // 2])  # center of window
            
        pupil_means_very_low = np.array(means_very_low)
        pupil_means_low = np.array(means_low)
        pupil_means_middle = np.array(means_middle)
        pupil_means_high = np.array(means_high)
        pupil_means_tonic = np.array(means_tonic)
        win_times = np.array(win_times)
        
        print(f"    Alpha windows: {len(alpha_means_parocc)}")
        print(f"    Pupil very-low windows: {len(pupil_means_very_low)}")
        print(f"    Pupil low windows: {len(pupil_means_low)}")
        print(f"    Pupil middle windows: {len(pupil_means_middle)}")
        print(f"    Pupil high windows: {len(pupil_means_high)}")

        lengths = [
            len(alpha_means_parocc),
            len(pupil_means_very_low),
            len(pupil_means_low),
            len(pupil_means_middle),
            len(pupil_means_high),
            len(pupil_means_tonic)]

        if len(set(lengths)) != 1:
            print(f"⚠️ Length mismatch for "
                  f"{Subject_id}, block {block_idx}: "
                  f"{lengths}")
            continue
        
        all_subjects_blocks.append({
            "Subject": Subject_id,
            "Block": block_idx,
            "BlockType": block_type,
            "alpha_means_parocc": alpha_means_parocc,
            "pupil_means_very_low": pupil_means_very_low,
            "pupil_means_low": pupil_means_low,
            "pupil_means_middle": pupil_means_middle,
            "pupil_means_high": pupil_means_high,
            "pupil_means_tonic": pupil_means_tonic,
            "win_times": win_times,})
        
#%%
import pickle
output_dir = (
    'Z://Analysis/SFARI_Analysis/Resting_State/'
    'Alpha_Parocc_pupil_RS')
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(
    output_dir,
    f'Alpha_Parocc_pupil_RS_{group}.pkl')

with open(output_file, 'wb') as f:
    pickle.dump(all_subjects_blocks, f)
print(f"Saved extraction file to:\n{output_file}")
    

#%%

import numpy as np
import pickle
from collections import defaultdict
from scipy.signal import correlate, correlation_lags
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Config
# -----------------------
sfreq = 128
step_size = int(0.1 * sfreq)  # 100 ms step used when creating *_means_* vectors
dt = step_size / sfreq
max_lag_s = 5.0
pupil_key = 'pupil_means_tonic'  # change if you want another band
conditions = ['resting-state']
group_files = {
    'TD': os.path.join(
        output_dir,
        "Alpha_Parocc_pupil_RS_TD.pkl"),
    'ASD': os.path.join(
        output_dir,
        "Alpha_Parocc_pupil_RS_ASD.pkl"),
    'SIB': os.path.join(
        output_dir,
        "Alpha_Parocc_pupil_RS_SIB.pkl")}

# -----------------------
# Helpers
# -----------------------
def build_by_subject(block_list):
    by_subject = defaultdict(lambda: defaultdict(list))
    for blk in block_list:
        by_subject[blk['Subject']][blk['BlockType']].append(blk)
    return by_subject

def xcorr_overlap_norm(x, y, dt, max_lag_s):
    """Approximate Pearson r per lag: z-score, cross-correlate, divide by overlap."""
    x = zscore(np.asarray(x)); y = zscore(np.asarray(y))
    L = min(len(x), len(y)); x, y = x[:L], y[:L]
    cc   = correlate(y, x, mode='full', method='auto')   # y vs x; +lag => x leads y
    lags = correlation_lags(L, L, mode='full')
    overlap = (L - np.abs(lags)).astype(float)
    r = cc / overlap
    max_lag = int(round(max_lag_s / dt))
    keep = np.where(np.abs(lags) <= max_lag)[0]
    return (lags[keep] * dt), r[keep]

def fisher_mean(r_rows, axis=0, weights=None):
    r = np.clip(np.asarray(r_rows), -0.999999, 0.999999)
    z = np.arctanh(r)
    zbar = np.average(z, axis=axis, weights=weights) if weights is not None else np.nanmean(z, axis=axis)
    return np.tanh(zbar)

def subject_curves(by_subject, btype, dt, max_lag_s, pupil_key='pupil_means_low'):
    """Return lags (s) and dict: subj -> Fisher-averaged correlogram across that subject's blocks."""
    subj_curves = {}
    lags_s_shared = None
    for subj, subj_blocks in by_subject.items():
        blocks = subj_blocks.get(btype, [])
        if not blocks:
            continue
        r_blocks = []
        for blk in blocks:
            alpha = blk['alpha_means_parocc']
            pupil = blk[pupil_key]
            lags_s, r = xcorr_overlap_norm(alpha, pupil, dt=dt, max_lag_s=max_lag_s)
            r_blocks.append(r)
            if lags_s_shared is None:
                lags_s_shared = lags_s
        subj_curves[subj] = fisher_mean(np.vstack(r_blocks), axis=0)
    return lags_s_shared, subj_curves


# Circular-shift surrogate
def circ_shift(vec, min_shift_s, dt):
    L = len(vec)
    min_shift = max(1, int(round(min_shift_s / dt)))
    shift = np.random.randint(min_shift,L - min_shift)
    return np.roll(vec, shift)

# Group null distribution using circularly shifted pupil
def group_null_maxstat(
        by_subject,
        btype,
        dt,
        n_perm=1000,
        max_lag_s=5.0,
        min_shift_s=5.0,
        pupil_key='pupil_means_low'):

    # Real subject-level curves
    lags_s, subj_real = subject_curves(
        by_subject,
        btype,
        dt=dt,
        max_lag_s=max_lag_s,
        pupil_key=pupil_key)
    subjects = list(subj_real.keys())
    real_mat = np.vstack([subj_real[s] for s in subjects])
    real_mean = fisher_mean(real_mat,axis=0)

    # Null distribution
    n_lags = real_mat.shape[1]
    null_group_means = np.empty((n_perm, n_lags))
    max_stats = np.empty(n_perm)
    for p in range(n_perm):
        subj_perm_curves = []
        for subj in subjects:
            blocks = by_subject[subj][btype]
            r_perm_blocks = []
            for blk in blocks:
                alpha = blk['alpha_means_parocc']
                pupil = blk[pupil_key]
                pupil_surr = circ_shift(
                    pupil,
                    min_shift_s=min_shift_s,
                    dt=dt)
                _, r = xcorr_overlap_norm(
                    alpha,
                    pupil_surr,
                    dt=dt,
                    max_lag_s=max_lag_s)
                r_perm_blocks.append(r)

            subj_perm_curves.append(
                fisher_mean(
                    np.vstack(r_perm_blocks),
                    axis=0))

        perm_group = fisher_mean(
            np.vstack(subj_perm_curves),
            axis=0)
        null_group_means[p] = perm_group
        max_stats[p] = np.max(
            np.abs(perm_group))

    lo = np.percentile(null_group_means,2.5,axis=0)
    hi = np.percentile(null_group_means,97.5,axis=0)
    thr = np.percentile(max_stats,95.0)
    sig_mask = (np.abs(real_mean) >= thr)
    return (lags_s,real_mean,(lo, hi),thr,sig_mask)



def features_from_curve(lags_s, r_curve, lo=0.0, hi=1.0, peak_mode='pos'):
    """
    Extract features in [lo,hi] seconds.
    peak_mode: 'pos' (max positive), 'neg' (min), or 'abs' (max |r|).
    Returns: peak_r, peak_lag_s, auc_signed
    """
    m = (lags_s >= lo) & (lags_s <= hi)
    l = lags_s[m]; r = r_curve[m]
    if r.size == 0 or np.all(np.isnan(r)):
        return np.nan, np.nan, np.nan
    if peak_mode == 'pos':
        idx = np.nanargmax(r)
    elif peak_mode == 'neg':
        idx = np.nanargmin(r)
    else:  # 'abs'
        idx = np.nanargmax(np.abs(r))
    peak_r = float(r[idx]); peak_lag = float(l[idx])
    # auc = float(np.trapz(r, x=l))  # signed area under curve
    auc = float(np.trapezoid(r, x=l))
    return peak_r, peak_lag, auc

# -----------------------
# Load, compute subject features
# -----------------------
rows = []
for group, pkl in group_files.items():
    with open(pkl, "rb") as f:
        blocks = pickle.load(f)
    by_subject = build_by_subject(blocks)

    for cond in conditions:
        lags_s, subj_curves = subject_curves(by_subject, cond, dt=dt, max_lag_s=max_lag_s, pupil_key=pupil_key)
        if lags_s is None:
            continue  # no data for this condition
        for subj, curve in subj_curves.items():
            peak_r, peak_lag, auc = features_from_curve(lags_s, curve, lo=0.0, hi=0.6, peak_mode='pos')
            # peak_r, peak_lag, auc = features_from_curve(lags_s, curve, lo=0.0, hi=1, peak_mode='pos')
            
            rows.append({
                'Subject': subj,
                'Group': group,
                'Condition': cond,
                'PupilBand': pupil_key,
                'peak_r_0to0p6s': peak_r,
                'peak_lag_s_0to0p6s': peak_lag,
                'auc_signed_0to0p6s': auc
            })

df = pd.DataFrame(rows).sort_values(['Condition','Group','Subject'])
print(df.head())
# -> df now has per-subject features you can export or analyze in LMM

# -----------------------
# Plot: peak r and AUC per subject (by Group × Condition)
# -----------------------
def jitter(n, scale=0.06):
    return (np.random.rand(n) - 0.5) * scale

palette = {'TD':'#4e79a7', 'SIB':'#f28e2b', 'ASD':'#e15759'}

for metric, ylabel in [('peak_r_0to0p6s','Peak r (0–0.6 s)'),
                       ('auc_signed_0to0p6s','AUC signed (0–0.6 s)')]:
    fig, axes = plt.subplots(1, len(conditions), figsize=(12,4), sharey=False)
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        sub = df[df['Condition']==cond]
        x_positions = {'TD':0, 'SIB':1, 'ASD':2}
        # points
        for g, x0 in x_positions.items():
            vals = sub[sub['Group']==g][metric].values
            n = len(vals)
            ax.scatter(np.full(n, x0)+jitter(n), vals, s=28, alpha=0.8, color=palette[g], label=g)
            # mean line
            if n:
                m = np.nanmean(vals)
                ax.hlines(m, x0-0.18, x0+0.18, colors=palette[g], linewidth=3)
        ax.axhline(0, ls='--', c='k', lw=1)
        ax.set_xticks(list(x_positions.values()), list(x_positions.keys()))
        ax.set_title(cond)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(alpha=0.25)
    fig.suptitle(ylabel + ' by Group', y=0.98, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

#%% Delete '_' for the SIB group before saving to run stats
df['Subject'] = df['Subject'].astype(str).str.replace(r'_$', '', regex=True)

df.to_csv("RS_Cross_correlation_Peak_Values_All.csv", index=False)

#%% ============================================================
# GROUP-LEVEL CROSS-CORRELATION + CIRCULAR-SHIFT NULL
# ============================================================

for group, pkl in group_files.items():
    with open(pkl, "rb") as f:
        blocks = pickle.load(f)
    by_subject = build_by_subject(blocks)
    lags_s, real_mean, (lo, hi), thr, sig_mask = group_null_maxstat(
        by_subject,
        btype='resting-state',
        dt=dt,
        n_perm=1000,
        max_lag_s=5.0,
        min_shift_s=5.0,
        pupil_key=pupil_key)
    plt.figure(figsize=(9, 4))
    plt.fill_between(
        lags_s,
        lo,
        hi,
        alpha=0.25,
        label='Circular-shift null 95% CI')
    plt.plot(
        lags_s,
        real_mean,
        linewidth=2,
        label=f'{group} mean')
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.axhline(thr, linestyle=':')
    plt.axhline(-thr, linestyle=':')
    plt.scatter(
        lags_s[sig_mask],
        real_mean[sig_mask],
        s=12,
        label='FWER-significant')
    plt.xlabel('Lag (s)')
    plt.ylabel('Alpha–pupil correlation (r)')
    plt.title(f'{group}: resting-state alpha–pupil cross-correlation')
    plt.legend()
    plt.tight_layout()
    plt.show()
#%% ============================================================
# RESTING-STATE GROUP STATISTICS
# ============================================================

import statsmodels.formula.api as smf

df_stats = pd.read_csv(
    "RS_Cross_correlation_Peak_Values_All.csv")
print("\nSubjects per group:")
print(df_stats.groupby("Group")["Subject"].nunique())
print("\nDescriptive statistics:")
print(df_stats.groupby("Group")[
        ["auc_signed_0to0p6s",
            "peak_r_0to0p6s",
            "peak_lag_s_0to0p6s"]].agg(["mean", "std", "count"]))

# AUC
model_auc = smf.ols(
    "auc_signed_0to0p6s ~ C(Group)",
    data=df_stats).fit()
print("\nSIGNED AUC")
print(model_auc.summary())

# Peak correlation
model_peak = smf.ols(
    "peak_r_0to0p6s ~ C(Group)",
    data=df_stats).fit()
print("\nPEAK CORRELATION")
print(model_peak.summary())

# Peak lag
model_lag = smf.ols(
    "peak_lag_s_0to0p6s ~ C(Group)",
    data=df_stats
).fit()

print("\nPEAK LAG")
print(model_lag.summary())

#%% Plot resting-state group statistics

import matplotlib.pyplot as plt
import numpy as np

groups = ['ASD', 'SIB', 'TD']

metrics = [
    ('auc_signed_0to0p6s',
     'Signed AUC (0–0.6 s)',
     model_auc),

    ('peak_r_0to0p6s',
     'Peak r (0–0.6 s)',
     model_peak),

    ('peak_lag_s_0to0p6s',
     'Peak lag (s)',
     model_lag)
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (metric, ylabel, model) in zip(axes, metrics):

    for x, group in enumerate(groups):

        vals = df_stats.loc[
            df_stats['Group'] == group,
            metric
        ].dropna().values

        # Individual participants
        jitter = np.random.uniform(-0.07, 0.07, len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            alpha=0.55,
            s=30
        )

        # Mean ± SEM
        mean = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals))

        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt='o',
            markersize=8,
            capsize=5,
            linewidth=2
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)

    # Overall group-effect p-value
    overall_p = model.f_pvalue
    ax.set_title(
        f'Group effect: p = {overall_p:.4f}'
    )

    ax.axhline(0, linestyle='--', linewidth=1)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()