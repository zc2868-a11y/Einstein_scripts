# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:57:47 2026

@author: tvanneau
"""

#%% Optimized FOOOF: participant-level aperiodic and periodic parameters

import os
import gc
from time import perf_counter

import numpy as np
import pandas as pd

from fooof import FOOOFGroup

# =========================
# User settings
# =========================
groups = ["TD", "ASD", "PMS"]

base_psd_dir = (
    r"Z:\Analysis\Shlomit_MFF_RS\PSD_median"
)

fooof_output_dir = os.path.join(
    base_psd_dir,
    "FOOOF_results_median"
)

os.makedirs(
    fooof_output_dir,
    exist_ok=True
)

# Frequency range fitted by FOOOF
fit_range = [1.0, 40.0]

# FOOOF settings
peak_width_limits = [0.5, 12.0]
max_n_peaks = 6
min_peak_height = 0.10
peak_threshold = 2.0
aperiodic_mode = "fixed"

# Frequency bands
# Alpha: 7 <= frequency < 13 Hz
# Beta: 13 <= frequency <= 30 Hz
alpha_band = (7.0, 13.0)
beta_band = (13.0, 30.0)

# On Windows/Spyder, start with 1.
# You can test 4 later to see whether it is faster.
fooof_n_jobs = 1

# Saving complete FOOOF models can be slow on a network drive.
# The CSV contains all extracted parameters, so False is recommended.
save_fooof_models = False

# If complete models are saved, omit the original PSD data
# to reduce file size and network-writing time.
save_fooof_data = False

tiny = np.finfo(float).tiny

all_results = []


# =========================
# Helper function:
# summarize peaks in a band
# =========================
def summarize_band_peaks(
    peak_parameters,
    fmin,
    fmax,
    include_upper=False
):
    """
    Summarize FOOOF peaks within a frequency band.

    Parameters
    ----------
    peak_parameters : ndarray
        FOOOF peak parameters with columns:
        0 = center frequency
        1 = peak power above aperiodic fit
        2 = bandwidth

    fmin, fmax : float
        Band limits.

    include_upper : bool
        If True, include peaks exactly equal to fmax.

    Returns
    -------
    results : dict
        Number of peaks, summed power, maximum power,
        dominant frequency, mean frequency and bandwidth.
    """

    empty_results = {
        "n_peaks": 0,
        "sum_peak_power": np.nan,
        "max_peak_power": np.nan,
        "dominant_frequency": np.nan,
        "mean_frequency": np.nan,
        "mean_bandwidth": np.nan
    }

    if (
        peak_parameters is None
        or np.size(peak_parameters) == 0
    ):
        return empty_results

    peak_parameters = np.atleast_2d(
        peak_parameters
    )

    center_frequency = (
        peak_parameters[:, 0]
    )

    if include_upper:

        band_mask = (
            (center_frequency >= fmin)
            & (center_frequency <= fmax)
        )

    else:

        band_mask = (
            (center_frequency >= fmin)
            & (center_frequency < fmax)
        )

    band_peaks = peak_parameters[
        band_mask
    ]

    if len(band_peaks) == 0:
        return empty_results

    band_frequency = band_peaks[:, 0]
    band_power = band_peaks[:, 1]
    band_bandwidth = band_peaks[:, 2]

    strongest_peak_idx = np.nanargmax(
        band_power
    )

    return {
        "n_peaks": len(band_peaks),

        "sum_peak_power": np.nansum(
            band_power
        ),

        "max_peak_power": band_power[
            strongest_peak_idx
        ],

        "dominant_frequency": band_frequency[
            strongest_peak_idx
        ],

        "mean_frequency": np.nanmean(
            band_frequency
        ),

        "mean_bandwidth": np.nanmean(
            band_bandwidth
        )
    }


# =========================
# Loop through groups
# =========================
total_start = perf_counter()

for group in groups:

    group_start = perf_counter()

    group_dir = os.path.join(
        base_psd_dir,
        group
    )

    group_fooof_dir = os.path.join(
        fooof_output_dir,
        group
    )

    os.makedirs(
        group_fooof_dir,
        exist_ok=True
    )

    if not os.path.isdir(group_dir):
        print(
            f"\nSkipping {group}: "
            f"folder not found:\n{group_dir}"
        )
        continue

    psd_files = sorted([
        fname
        for fname in os.listdir(group_dir)
        if fname.endswith("_PSD.npz")
    ])

    print(
        f"\n{group}: found "
        f"{len(psd_files)} PSD files"
    )

    # =========================
    # Participant loop
    # =========================
    for file_number, fname in enumerate(
        psd_files,
        start=1
    ):

        participant_start = perf_counter()

        print(
            f"\nProcessing {group} participant "
            f"{file_number}/{len(psd_files)}: "
            f"{fname}"
        )

        fpath = os.path.join(
            group_dir,
            fname
        )

        # =========================
        # Load participant PSD
        # =========================
        with np.load(
            fpath,
            allow_pickle=True
        ) as data:

            # Shape:
            # epochs × channels × frequencies
            psd_epochs = data["psd"]

            freqs = data["freqs"]

            ch_names = (
                data["ch_names"]
                .astype(str)
            )

            subject_id = str(
                data["subject_id"].item()
            )

            n_epochs = psd_epochs.shape[0]
            n_channels = psd_epochs.shape[1]

            # Average epochs in linear power:
            # channels × frequencies
            subject_psd = np.nanmean(
                psd_epochs,
                axis=0
            )

        print(
            f"  Subject: {subject_id}"
        )

        print(
            f"  Epochs: {n_epochs}; "
            f"channels: {n_channels}"
        )

        # Convert V²/Hz to µV²/Hz
        subject_psd_uv = (
            subject_psd * 1e12
        )

        # Prevent log10 errors inside FOOOF
        subject_psd_uv = np.maximum(
            subject_psd_uv,
            tiny
        )

        # =========================
        # Fit all channels
        # =========================
        fit_start = perf_counter()

        fg = FOOOFGroup(
            peak_width_limits=(
                peak_width_limits
            ),
            max_n_peaks=max_n_peaks,
            min_peak_height=(
                min_peak_height
            ),
            peak_threshold=(
                peak_threshold
            ),
            aperiodic_mode=(
                aperiodic_mode
            ),
            verbose=False
        )

        fg.fit(
            freqs,
            subject_psd_uv,
            fit_range,
            n_jobs=fooof_n_jobs
        )

        fit_duration = (
            perf_counter() - fit_start
        )

        print(
            f"  FOOOF fitting time: "
            f"{fit_duration:.1f} seconds"
        )

        # =========================
        # Optionally save model
        # =========================
        if save_fooof_models:

            save_start = perf_counter()

            fg.save(
                file_name=(
                    f"{subject_id}_"
                    f"FOOOF_median"
                ),
                file_path=group_fooof_dir,
                save_results=True,
                save_settings=True,
                save_data=save_fooof_data
            )

            print(
                f"  Model saving time: "
                f"{perf_counter() - save_start:.1f} "
                f"seconds"
            )

        # =========================
        # Extract fitted parameters
        # =========================
        extraction_start = perf_counter()

        # Fixed aperiodic mode gives:
        # column 0 = offset
        # column 1 = exponent
        aperiodic_parameters_all = (
            fg.get_params(
                "aperiodic_params"
            )
        )

        r_squared_all = fg.get_params(
            "r_squared"
        )

        fit_error_all = fg.get_params(
            "error"
        )

        # =========================
        # Channel loop
        # =========================
        for channel_idx, channel_name in enumerate(
            ch_names
        ):

            # Default values if extraction failed
            fit_success = False

            offset = np.nan
            exponent = np.nan
            r_squared = np.nan
            fit_error = np.nan
            total_n_peaks = 0

            alpha_results = {
                "n_peaks": 0,
                "sum_peak_power": np.nan,
                "max_peak_power": np.nan,
                "dominant_frequency": np.nan,
                "mean_frequency": np.nan,
                "mean_bandwidth": np.nan
            }

            beta_results = alpha_results.copy()

            try:
                # -------------------------
                # Aperiodic parameters
                # -------------------------
                offset = (
                    aperiodic_parameters_all[
                        channel_idx,
                        0
                    ]
                )

                exponent = (
                    aperiodic_parameters_all[
                        channel_idx,
                        1
                    ]
                )

                r_squared = (
                    r_squared_all[
                        channel_idx
                    ]
                )

                fit_error = (
                    fit_error_all[
                        channel_idx
                    ]
                )

                # -------------------------
                # Periodic parameters
                # -------------------------
                # Access stored results directly.
                # This avoids slow get_fooof(
                # regenerate=True) calls.
                channel_result = (
                    fg.group_results[
                        channel_idx
                    ]
                )

                peak_parameters = np.asarray(
                    channel_result.peak_params
                )

                if peak_parameters.size == 0:

                    peak_parameters = np.empty(
                        (0, 3),
                        dtype=float
                    )

                else:

                    peak_parameters = np.atleast_2d(
                        peak_parameters
                    )

                total_n_peaks = len(
                    peak_parameters
                )

                # Alpha:
                # 7 <= frequency < 13 Hz
                alpha_results = (
                    summarize_band_peaks(
                        peak_parameters,
                        alpha_band[0],
                        alpha_band[1],
                        include_upper=False
                    )
                )

                # Beta:
                # 13 <= frequency <= 30 Hz
                beta_results = (
                    summarize_band_peaks(
                        peak_parameters,
                        beta_band[0],
                        beta_band[1],
                        include_upper=True
                    )
                )

                fit_success = bool(
                    np.isfinite(offset)
                    and np.isfinite(exponent)
                    and np.isfinite(r_squared)
                    and np.isfinite(fit_error)
                )

            except Exception as error:

                print(
                    f"  Failed extraction: "
                    f"{subject_id}, "
                    f"{channel_name}: {error}"
                )

            # =========================
            # Store channel results
            # =========================
            all_results.append({
                "subject": subject_id,
                "group": group,
                "channel": channel_name,
                "n_epochs": n_epochs,
                "fit_success": fit_success,

                # Aperiodic parameters
                "offset": offset,
                "exponent": exponent,
                "r_squared": r_squared,
                "fit_error": fit_error,

                # All detected peaks
                "total_n_peaks": (
                    total_n_peaks
                ),

                # Alpha parameters
                "alpha_n_peaks": (
                    alpha_results[
                        "n_peaks"
                    ]
                ),

                "alpha_peak_present": int(
                    alpha_results[
                        "n_peaks"
                    ] > 0
                ),

                "alpha_sum_peak_power": (
                    alpha_results[
                        "sum_peak_power"
                    ]
                ),

                "alpha_max_peak_power": (
                    alpha_results[
                        "max_peak_power"
                    ]
                ),

                "alpha_dominant_frequency": (
                    alpha_results[
                        "dominant_frequency"
                    ]
                ),

                "alpha_mean_frequency": (
                    alpha_results[
                        "mean_frequency"
                    ]
                ),

                "alpha_mean_bandwidth": (
                    alpha_results[
                        "mean_bandwidth"
                    ]
                ),

                # Beta parameters
                "beta_n_peaks": (
                    beta_results[
                        "n_peaks"
                    ]
                ),

                "beta_peak_present": int(
                    beta_results[
                        "n_peaks"
                    ] > 0
                ),

                "beta_sum_peak_power": (
                    beta_results[
                        "sum_peak_power"
                    ]
                ),

                "beta_max_peak_power": (
                    beta_results[
                        "max_peak_power"
                    ]
                ),

                "beta_dominant_frequency": (
                    beta_results[
                        "dominant_frequency"
                    ]
                ),

                "beta_mean_frequency": (
                    beta_results[
                        "mean_frequency"
                    ]
                ),

                "beta_mean_bandwidth": (
                    beta_results[
                        "mean_bandwidth"
                    ]
                )
            })

        extraction_duration = (
            perf_counter()
            - extraction_start
        )

        participant_duration = (
            perf_counter()
            - participant_start
        )

        print(
            f"  Parameter extraction time: "
            f"{extraction_duration:.1f} seconds"
        )

        print(
            f"  Total participant time: "
            f"{participant_duration:.1f} seconds"
        )

        del (
            fg,
            subject_psd,
            subject_psd_uv,
            aperiodic_parameters_all,
            r_squared_all,
            fit_error_all
        )

        gc.collect()

    print(
        f"\n{group} completed in "
        f"{(perf_counter() - group_start) / 60:.1f} "
        f"minutes"
    )

    # =========================
    # Save a checkpoint after
    # completing each group
    # =========================
    checkpoint_df = pd.DataFrame(
        all_results
    )

    checkpoint_file = os.path.join(
        fooof_output_dir,
        "FOOOF_aperiodic_periodic_"
        f"parameters_checkpoint_{group}.csv"
    )

    checkpoint_df.to_csv(
        checkpoint_file,
        index=False
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_file}"
    )

# =========================
# Save final combined table
# =========================
df_fooof = pd.DataFrame(
    all_results
)

output_csv = os.path.join(
    fooof_output_dir,
    "FOOOF_aperiodic_periodic_"
    "parameters_all_channels_MEDIAN_PSD.csv"
)

df_fooof.to_csv(
    output_csv,
    index=False
)

total_duration = (
    perf_counter() - total_start
)

print("\nFOOOF extraction completed.")

print(
    f"Total duration: "
    f"{total_duration / 60:.1f} minutes"
)

print(f"Saved to: {output_csv}")

print("\nFirst rows:")
print(df_fooof.head())

print("\nNumber of participants per group:")
print(
    df_fooof
    .groupby("group")["subject"]
    .nunique()
)

print("\nSuccessful fits:")
print(
    df_fooof["fit_success"]
    .value_counts(
        dropna=False
    )
)

#%% Generic FOOOF topomaps by group

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# =========================
# User settings
# =========================
fooof_file = (
    r"Z:\Analysis\Shlomit_MFF_RS\PSD_median"
    r"\FOOOF_results_median"
    r"\FOOOF_aperiodic_periodic_"
    r"parameters_all_channels_MEDIAN_PSD.csv"
)

# Choose the parameter here
parameter = "fit_error"

groups = ["TD", "ASD", "PMS"]

group_titles = {
    "TD": "TD",
    "ASD": "ASD",
    "PMS": "PMS"
}

montage_name = "GSN-HydroCel-128"
cmap = "viridis"

# Set to None for automatic common scaling
manual_vlim = None

# Use percentiles for automatic scaling to reduce
# the influence of extreme channels
use_percentile_vlim = False
vlim_percentiles = (2, 98)

# Relevant only for periodic power parameters:
#
# False:
# Mean power is calculated only across participants
# with a detected peak at that channel.
#
# True:
# No detected peak is assigned zero power.
include_peak_absence_as_zero = False

# Save figure
save_figure = False

output_dir = (
    r"Z:\Analysis\Shlomit_MFF_RS\PSD_median"
    r"\FOOOF_results_median\Topomaps"
)
os.makedirs(output_dir, exist_ok=True)

# =========================
# Parameter labels
# =========================
parameter_titles = {
    # Aperiodic parameters
    "offset":
        "Aperiodic offset",
    "exponent":
        "Aperiodic exponent",
    "r_squared":
        r"FOOOF model fit ($R^2$)",
    "fit_error":
        "FOOOF model fitting error",

    # Alpha parameters
    "alpha_n_peaks":
        "Number of alpha peaks",
    "alpha_peak_present":
        "Proportion with an alpha peak",
    "alpha_sum_peak_power":
        "Summed alpha peak power",
    "alpha_max_peak_power":
        "Maximum alpha peak power",
    "alpha_dominant_frequency":
        "Dominant alpha frequency",
    "alpha_mean_frequency":
        "Mean alpha frequency",
    "alpha_mean_bandwidth":
        "Mean alpha peak bandwidth",

    # Beta parameters
    "beta_n_peaks":
        "Number of beta peaks",
    "beta_peak_present":
        "Proportion with a beta peak",
    "beta_sum_peak_power":
        "Summed beta peak power",
    "beta_max_peak_power":
        "Maximum beta peak power",
    "beta_dominant_frequency":
        "Dominant beta frequency",
    "beta_mean_frequency":
        "Mean beta frequency",
    "beta_mean_bandwidth":
        "Mean beta peak bandwidth"
}

parameter_labels = {
    # Aperiodic
    "offset":
        r"Offset [log$_{10}$(µV$^2$/Hz)]",
    "exponent":
        "Exponent",
    "r_squared":
        r"$R^2$",
    "fit_error":
        "Model error",

    # Alpha
    "alpha_n_peaks":
        "Number of peaks",
    "alpha_peak_present":
        "Proportion of participants",
    "alpha_sum_peak_power":
        "Summed peak power above aperiodic fit",
    "alpha_max_peak_power":
        "Peak power above aperiodic fit",
    "alpha_dominant_frequency":
        "Frequency (Hz)",
    "alpha_mean_frequency":
        "Frequency (Hz)",
    "alpha_mean_bandwidth":
        "Bandwidth (Hz)",

    # Beta
    "beta_n_peaks":
        "Number of peaks",
    "beta_peak_present":
        "Proportion of participants",
    "beta_sum_peak_power":
        "Summed peak power above aperiodic fit",
    "beta_max_peak_power":
        "Peak power above aperiodic fit",
    "beta_dominant_frequency":
        "Frequency (Hz)",
    "beta_mean_frequency":
        "Frequency (Hz)",
    "beta_mean_bandwidth":
        "Bandwidth (Hz)"
}

# Periodic power parameters for which no peak
# may optionally be represented by zero
peak_power_parameters = [
    "alpha_sum_peak_power",
    "alpha_max_peak_power",
    "beta_sum_peak_power",
    "beta_max_peak_power"
]

# =========================
# Load data
# =========================
df_fooof = pd.read_csv(fooof_file)

required_columns = [
    "subject",
    "group",
    "channel",
    parameter
]

missing_columns = [
    column
    for column in required_columns
    if column not in df_fooof.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if parameter not in parameter_titles:
    raise ValueError(
        f"No title defined for parameter: "
        f"{parameter}"
    )

# Ensure selected parameter is numeric
df_fooof[parameter] = pd.to_numeric(
    df_fooof[parameter],
    errors="coerce"
)

# =========================
# Optional treatment of
# participants without peaks
# =========================
if (
    include_peak_absence_as_zero
    and parameter in peak_power_parameters
):

    peak_present_column = (
        "alpha_peak_present"
        if parameter.startswith("alpha")
        else "beta_peak_present"
    )

    if peak_present_column not in df_fooof.columns:
        raise ValueError(
            f"{peak_present_column} is needed "
            "to assign absent peaks a value of zero."
        )

    no_peak = (
        df_fooof[peak_present_column] == 0
    )

    df_fooof.loc[
        no_peak,
        parameter
    ] = 0.0

# =========================
# Create montage and Info
# =========================
montage = mne.channels.make_standard_montage(
    montage_name
)

data_channels = set(
    df_fooof["channel"].astype(str)
)

available_channels = [
    channel
    for channel in montage.ch_names
    if channel in data_channels
]

if len(available_channels) < 3:
    raise ValueError(
        "Fewer than three channels matched "
        "the selected montage."
    )

print(
    f"Channels used: "
    f"{len(available_channels)}"
)

info = mne.create_info(
    ch_names=available_channels,
    sfreq=1.0,
    ch_types="eeg"
)

info.set_montage(
    montage,
    match_case=False,
    on_missing="raise"
)

# =========================
# Participant-level values
# =========================
# This also makes the code compatible with a file
# containing duplicate rows or epoch-level results:
# epochs are averaged within participant/channel first.
subject_channel_values = (
    df_fooof
    .groupby(
        ["subject", "group", "channel"],
        as_index=False
    )
    .agg(
        value=(parameter, "mean")
    )
)

# =========================
# Group-level channel means
# =========================
group_channel_values = (
    subject_channel_values
    .groupby(
        ["group", "channel"],
        as_index=False
    )
    .agg(
        group_mean=("value", "mean"),
        group_median=("value", "median"),
        n_valid_subjects=("value", "count")
    )
)

# Choose mean or median group maps
group_statistic = "group_mean"

# =========================
# Arrange channels
# =========================
topomap_data = {}

for group in groups:

    group_df = (
        group_channel_values[
            group_channel_values["group"]
            == group
        ]
        .set_index("channel")
        .reindex(available_channels)
    )

    topomap_data[group] = (
        group_df[group_statistic]
        .to_numpy(dtype=float)
    )

    valid_n = (
        group_df["n_valid_subjects"]
        .dropna()
    )

    if len(valid_n) > 0:
        print(
            f"{group}: valid participants per "
            f"channel = "
            f"{int(valid_n.min())}–"
            f"{int(valid_n.max())}"
        )

# =========================
# Common color scale
# =========================
all_values = np.concatenate([
    values[np.isfinite(values)]
    for values in topomap_data.values()
])

if len(all_values) == 0:
    raise ValueError(
        f"All values are NaN for {parameter}."
    )

if manual_vlim is not None:

    current_vlim = manual_vlim

elif use_percentile_vlim:

    current_vlim = (
        np.nanpercentile(
            all_values,
            vlim_percentiles[0]
        ),
        np.nanpercentile(
            all_values,
            vlim_percentiles[1]
        )
    )

else:

    current_vlim = (
        np.nanmin(all_values),
        np.nanmax(all_values)
    )

print(
    f"{parameter} color limits: "
    f"{current_vlim}"
)

# =========================
# Plot topomaps
# =========================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4.5),
    constrained_layout=True
)

topomap_image = None

for ax, group in zip(
    axes,
    groups
):

    values = topomap_data[group]

    # Periodic parameters may contain NaN where
    # no peak was detected
    good_channels = np.isfinite(values)

    if good_channels.sum() < 3:

        ax.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.set_title(group_titles[group])
        ax.axis("off")
        continue

    values_plot = values[
        good_channels
    ]

    info_plot = mne.pick_info(
        info,
        sel=np.where(good_channels)[0],
        copy=True
    )

    topomap_image, _ = mne.viz.plot_topomap(
        values_plot,
        info_plot,
        axes=ax,
        cmap=cmap,
        vlim=current_vlim,
        sensors=True,
        contours=6,
        outlines="head",
        extrapolate="head",
        sphere="auto",
        show=False
    )

    n_group = (
        subject_channel_values.loc[
            subject_channel_values["group"]
            == group,
            "subject"
        ]
        .nunique()
    )

    ax.set_title(
        f"{group_titles[group]} "
        f"(n={n_group})",
        fontsize=14
    )

# Shared colorbar
if topomap_image is not None:

    colorbar = fig.colorbar(
        topomap_image,
        ax=axes,
        shrink=0.80,
        pad=0.03
    )

    colorbar.set_label(
        parameter_labels[parameter],
        fontsize=11
    )

fig.suptitle(
    parameter_titles[parameter],
    fontsize=16
)

# # =========================
# # Save figure
# # =========================
# if save_figure:

#     suffix = (
#         "zero_for_absent_peaks"
#         if (
#             include_peak_absence_as_zero
#             and parameter
#             in peak_power_parameters
#         )
#         else "detected_peaks_only"
#     )

#     # output_file = os.path.join(
#     #     output_dir,
#     #     f"Topomap_{parameter}_{suffix}.png"
#     # )

#     # fig.savefig(
#     #     output_file,
#     #     dpi=300,
#     #     bbox_inches="tight"
#     # )

#     # print(f"Saved: {output_file}")

plt.show()

#%% FOOOF cluster-level raincloud plots

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from time import perf_counter
from fooof import FOOOFGroup


# =========================
# User settings
# =========================
fooof_file = (
    r"Z:\Analysis\Shlomit_MFF_RS\PSD_median"
    r"\FOOOF_results_median"
    r"\FOOOF_aperiodic_periodic_"
    r"parameters_all_channels_MEDIAN_PSD.csv"
)

output_dir = (
    r"Z:\Analysis\Shlomit_MFF_RS\PSD_median"
    r"\FOOOF_results_median\Cluster_Raincloud_Plots"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# Select channel cluster
# cluster_channels = [
#     "E60",
#     "E67",
#     "E73",
#     "E78",
#     "E86"
# ]

# cluster_name = "Posterior_cluster"

cluster_channels = [
    "E12",
    "E5",
    "E11",
    "E6"
]

cluster_name = "Frontal_cluster"

groups = ["TD", "ASD", "PMS"]

group_colors = {
    "TD": "#377EB8",
    "ASD": "#FF7F00",
    "PMS": "#E41A1C"
}

# =========================
# Select parameters
# =========================
# Remove parameters you do not want to plot.
parameters_to_plot = {
    # Aperiodic parameters
    "offset":
        r"Aperiodic offset [log$_{10}$(µV$^2$/Hz)]",

    "exponent":
        "Aperiodic exponent",

    "r_squared":
        r"FOOOF model fit ($R^2$)",

    "fit_error":
        "FOOOF model fitting error",

    # Alpha parameters
    "alpha_n_peaks":
        "Number of alpha peaks",

    "alpha_peak_present":
        "Proportion of channels with an alpha peak",

    "alpha_sum_peak_power":
        "Summed alpha peak power above aperiodic fit",

    "alpha_max_peak_power":
        "Maximum alpha peak power above aperiodic fit",

    "alpha_dominant_frequency":
        "Dominant alpha frequency (Hz)",

    "alpha_mean_frequency":
        "Mean alpha frequency (Hz)",

    "alpha_mean_bandwidth":
        "Mean alpha bandwidth (Hz)",

    # Beta parameters
    "beta_n_peaks":
        "Number of beta peaks",

    "beta_peak_present":
        "Proportion of channels with a beta peak",

    "beta_sum_peak_power":
        "Summed beta peak power above aperiodic fit",

    "beta_max_peak_power":
        "Maximum beta peak power above aperiodic fit",

    "beta_dominant_frequency":
        "Dominant beta frequency (Hz)",

    "beta_mean_frequency":
        "Mean beta frequency (Hz)",

    "beta_mean_bandwidth":
        "Mean beta bandwidth (Hz)"
}

# To plot only selected parameters, for example:
# parameters_to_plot = {
#     "offset": "Aperiodic offset",
#     "exponent": "Aperiodic exponent",
#     "alpha_sum_peak_power":
#         "Summed alpha peak power above aperiodic fit"
# }

# How to average channels within participants:
# "mean" or "median"
cluster_aggregation = "mean"

# Require every selected cluster channel to be present
require_all_cluster_channels = True

# Exclude failed FOOOF fits
only_successful_fits = True

# =========================
# Handling absent peaks
# =========================
# False:
# Peak power is averaged only across channels where a
# peak was detected.
#
# True:
# Channels without a detected peak are assigned zero
# peak power. This combines peak presence and strength.
include_peak_absence_as_zero = False

peak_power_parameters = [
    "alpha_sum_peak_power",
    "alpha_max_peak_power",
    "beta_sum_peak_power",
    "beta_max_peak_power"
]

# Optional manual y-axis limits
# Use None for automatic scaling
ylim_dict = {
    "offset": None,
    "exponent": None,
    "r_squared": None,
    "fit_error": None,
    "alpha_n_peaks": None,
    "alpha_peak_present": (0, 1.05),
    "alpha_sum_peak_power": None,
    "alpha_max_peak_power": None,
    "alpha_dominant_frequency": (7, 13),
    "alpha_mean_frequency": (7, 13),
    "alpha_mean_bandwidth": None,
    "beta_n_peaks": None,
    "beta_peak_present": (0, 1.05),
    "beta_sum_peak_power": None,
    "beta_max_peak_power": None,
    "beta_dominant_frequency": (13, 30),
    "beta_mean_frequency": (13, 30),
    "beta_mean_bandwidth": None
}

save_png = False
save_svg = False

# Reproducible jitter
rng = np.random.default_rng(
    seed=42
)


# =========================
# Load FOOOF data
# =========================
df_fooof = pd.read_csv(
    fooof_file
)

required_columns = [
    "subject",
    "group",
    "channel"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df_fooof.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

missing_parameters = [
    parameter
    for parameter in parameters_to_plot
    if parameter not in df_fooof.columns
]

if missing_parameters:
    raise ValueError(
        f"Missing FOOOF parameters: "
        f"{missing_parameters}"
    )

# Make parameter columns numeric
for parameter in parameters_to_plot:

    df_fooof[parameter] = pd.to_numeric(
        df_fooof[parameter],
        errors="coerce"
    )

# Make peak-presence columns numeric if available
for column in [
    "alpha_peak_present",
    "beta_peak_present"
]:

    if column in df_fooof.columns:

        df_fooof[column] = pd.to_numeric(
            df_fooof[column],
            errors="coerce"
        )

# Keep specified groups
df_fooof = df_fooof[
    df_fooof["group"].isin(groups)
].copy()

# Optionally remove failed fits
if (
    only_successful_fits
    and "fit_success" in df_fooof.columns
):

    # Handles Boolean values and strings saved in CSV
    fit_success = (
        df_fooof["fit_success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    df_fooof = df_fooof[
        fit_success
    ].copy()

# Keep cluster channels
df_cluster = df_fooof[
    df_fooof["channel"].isin(
        cluster_channels
    )
].copy()

if len(df_cluster) == 0:
    raise ValueError(
        "None of the selected cluster channels "
        "were found in the FOOOF table."
    )

found_channels = sorted(
    df_cluster["channel"].unique()
)

missing_cluster_channels = [
    channel
    for channel in cluster_channels
    if channel not in found_channels
]

print(
    f"Cluster channels found: "
    f"{found_channels}"
)

if missing_cluster_channels:

    print(
        f"Missing cluster channels: "
        f"{missing_cluster_channels}"
    )


# =========================
# Count channels present
# per participant
# =========================
participant_channel_counts = (
    df_cluster
    .groupby(
        ["subject", "group"],
        as_index=False
    )
    .agg(
        n_cluster_channels_present=(
            "channel",
            "nunique"
        )
    )
)

if require_all_cluster_channels:

    complete_participants = (
        participant_channel_counts[
            participant_channel_counts[
                "n_cluster_channels_present"
            ]
            == len(cluster_channels)
        ][["subject", "group"]]
    )

    df_cluster = df_cluster.merge(
        complete_participants,
        on=["subject", "group"],
        how="inner"
    )


# =========================
# Raincloud plotting function
# =========================
def plot_raincloud(
    participant_values,
    parameter,
    ylabel
):
    """
    Plot a left-half violin, boxplot and
    individual participant points.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    group_positions = np.arange(
        len(groups)
    )

    group_data = []

    # Prepare group arrays
    for group in groups:

        values = (
            participant_values.loc[
                participant_values["group"]
                == group,
                "value"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        group_data.append(values)

        print(
            f"{parameter}, {group}: "
            f"n={len(values)}"
        )

    # -------------------------
    # Half violins
    # -------------------------
    for position, group, values in zip(
        group_positions,
        groups,
        group_data
    ):

        # At least two different values are needed
        # to draw a meaningful density
        if (
            len(values) < 2
            or np.nanstd(values) == 0
        ):
            continue

        violin = ax.violinplot(
            dataset=[values],
            positions=[position],
            widths=0.70,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )

        body = violin["bodies"][0]

        body.set_facecolor(
            group_colors[group]
        )

        body.set_edgecolor(
            group_colors[group]
        )

        body.set_alpha(0.45)

        # Keep only the left half
        for path in body.get_paths():

            vertices = path.vertices

            vertices[:, 0] = np.minimum(
                vertices[:, 0],
                position
            )

    # -------------------------
    # Boxplots
    # -------------------------
    box_positions = (
        group_positions + 0.08
    )

    boxplot = ax.boxplot(
        group_data,
        positions=box_positions,
        widths=0.13,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={
            "color": "black",
            "linewidth": 2
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.2
        },
        capprops={
            "color": "black",
            "linewidth": 1.2
        },
        boxprops={
            "edgecolor": "black",
            "linewidth": 1.2
        }
    )

    for patch, group in zip(
        boxplot["boxes"],
        groups
    ):

        patch.set_facecolor(
            group_colors[group]
        )

        patch.set_alpha(0.75)

    # -------------------------
    # Individual points
    # -------------------------
    for position, group, values in zip(
        group_positions,
        groups,
        group_data
    ):

        if len(values) == 0:
            continue

        jitter = rng.normal(
            loc=0,
            scale=0.025,
            size=len(values)
        )

        x_values = (
            position
            + 0.25
            + jitter
        )

        ax.scatter(
            x_values,
            values,
            s=32,
            color=group_colors[group],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.75,
            zorder=3
        )

    # -------------------------
    # Labels
    # -------------------------
    group_labels = [
        f"{group}\n(n={len(values)})"
        for group, values in zip(
            groups,
            group_data
        )
    ]

    ax.set_xticks(
        group_positions
    )

    ax.set_xticklabels(
        group_labels,
        fontsize=11
    )

    ax.set_xlim(
        -0.55,
        len(groups) - 0.45
    )

    ax.set_xlabel(
        "Group",
        fontsize=12
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12
    )

    ax.set_title(
        f"{ylabel}\n{cluster_name}",
        fontsize=14
    )

    current_ylim = ylim_dict.get(
        parameter
    )

    if current_ylim is not None:

        ax.set_ylim(
            current_ylim
        )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    fig.tight_layout()

    # -------------------------
    # Save figure
    # -------------------------
    suffix = (
        "zero_for_absent_peaks"
        if (
            include_peak_absence_as_zero
            and parameter
            in peak_power_parameters
        )
        else "detected_peaks_only"
    )

    if save_png:

        png_file = os.path.join(
            output_dir,
            f"{cluster_name}_{parameter}_"
            f"{suffix}.png"
        )

        fig.savefig(
            png_file,
            dpi=300,
            bbox_inches="tight"
        )

    if save_svg:

        svg_file = os.path.join(
            output_dir,
            f"{cluster_name}_{parameter}_"
            f"{suffix}.svg"
        )

        fig.savefig(
            svg_file,
            bbox_inches="tight"
        )

    plt.show()
    # plt.close(fig)


# =========================
# Calculate participant
# values and plot
# =========================
all_participant_values = []

for parameter, ylabel in parameters_to_plot.items():

    parameter_df = df_cluster[
        [
            "subject",
            "group",
            "channel",
            parameter
        ]
    ].copy()

    parameter_df = parameter_df.rename(
        columns={
            parameter: "value"
        }
    )

    # -------------------------
    # Handle channels without
    # detected peaks
    # -------------------------
    if (
        include_peak_absence_as_zero
        and parameter
        in peak_power_parameters
    ):

        peak_present_column = (
            "alpha_peak_present"
            if parameter.startswith("alpha")
            else "beta_peak_present"
        )

        peak_information = df_cluster[
            [
                "subject",
                "group",
                "channel",
                peak_present_column
            ]
        ].copy()

        parameter_df = parameter_df.merge(
            peak_information,
            on=[
                "subject",
                "group",
                "channel"
            ],
            how="left"
        )

        no_peak = (
            parameter_df[
                peak_present_column
            ] == 0
        )

        parameter_df.loc[
            no_peak,
            "value"
        ] = 0.0

    # -------------------------
    # Protect against duplicate
    # subject-channel rows
    # -------------------------
    # If an epoch-level file is loaded, this first
    # averages epochs within each subject/channel.
    subject_channel_values = (
        parameter_df
        .groupby(
            ["subject", "group", "channel"],
            as_index=False
        )
        .agg(
            value=("value", "mean")
        )
    )

    # Number of valid channels for this parameter
    valid_parameter_counts = (
        subject_channel_values
        .groupby(
            ["subject", "group"],
            as_index=False
        )
        .agg(
            n_valid_parameter_channels=(
                "value",
                "count"
            )
        )
    )

    # -------------------------
    # Average selected channels
    # within participants
    # -------------------------
    if cluster_aggregation == "mean":

        participant_values = (
            subject_channel_values
            .groupby(
                ["subject", "group"],
                as_index=False
            )
            .agg(
                value=("value", "mean")
            )
        )

    elif cluster_aggregation == "median":

        participant_values = (
            subject_channel_values
            .groupby(
                ["subject", "group"],
                as_index=False
            )
            .agg(
                value=("value", "median")
            )
        )

    else:

        raise ValueError(
            "cluster_aggregation must be "
            "'mean' or 'median'."
        )

    participant_values = (
        participant_values
        .merge(
            participant_channel_counts,
            on=["subject", "group"],
            how="left"
        )
        .merge(
            valid_parameter_counts,
            on=["subject", "group"],
            how="left"
        )
    )

    participant_values[
        "parameter"
    ] = parameter

    all_participant_values.append(
        participant_values
    )

    # Plot selected parameter
    plot_raincloud(
        participant_values=participant_values,
        parameter=parameter,
        ylabel=ylabel
    )


# =========================
# Save all participant-level
# cluster values
# =========================
participant_values_long = pd.concat(
    all_participant_values,
    ignore_index=True
)

participant_output_file = os.path.join(
    output_dir,
    f"{cluster_name}_FOOOF_"
    f"participant_values_long.csv"
)

participant_values_long.to_csv(
    participant_output_file,
    index=False
)

print(
    "\nParticipant-level FOOOF values "
    f"saved to:\n{participant_output_file}"
)

print("\nNumber of participants per parameter and group:")

print(
    participant_values_long
    .groupby(
        ["parameter", "group"]
    )["subject"]
    .nunique()
)

