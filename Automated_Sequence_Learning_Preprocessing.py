# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:18:22 2026

@author: theov
"""


from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels

try:
    from mne_icalabel import label_components
except ImportError:
    label_components = None


# =============================================================================
# CONFIGURATION
# =============================================================================
BIDS_ROOT = Path(r"Z:\Data_new\HBN-EEG\ds005511")
OUTPUT_ROOT = Path(r"Z:\Analysis\HBN-EEG\Automated_Preprocessed_data_ds005511")
print(BIDS_ROOT.exists())
print(BIDS_ROOT)
print(list(BIDS_ROOT.glob("*"))[:10])

TASK_PATTERN = "task-seqLearning*target"
SUBJECTS_TO_RUN: list[str] | None = None  # e.g. ["sub-NDARAC904DMU"]
OVERWRITE = False

BAD_CHANNEL_THRESHOLD = 20.0
LOWPASS = 50.0
HIGHPASS_FINAL = 0.01
HIGHPASS_ICA = 1.0
LINE_FREQ = 60.0
RANDOM_STATE = 42

# Automatic ICLabel exclusion. Probabilities prevent weak labels being removed.
AUTO_ICA = True
ICA_LABELS_TO_EXCLUDE = {"eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"}
ICA_MIN_PROBABILITY = 0.80

# Set True only after testing the automatic ICA decisions on a representative subset.
APPLY_AUTOMATIC_ICA = True

MONTAGE_NAME = "GSN-HydroCel-128"


# =============================================================================
# UTILITIES
# =============================================================================
def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def discover_subjects(root: Path) -> list[Path]:
    subjects = sorted(p for p in root.glob("sub-*") if p.is_dir())
    if SUBJECTS_TO_RUN is not None:
        requested = set(SUBJECTS_TO_RUN)
        subjects = [p for p in subjects if p.name in requested]
    return subjects


def discover_sequence_files(subject_dir: Path) -> list[dict[str, Path]]:
    """Find complete .set + events.tsv bundles, including 6- and 8-target tasks."""
    eeg_dir = subject_dir / "eeg"
    if not eeg_dir.exists():
        return []

    bundles: list[dict[str, Path]] = []
    for set_path in sorted(eeg_dir.glob(f"{subject_dir.name}_{TASK_PATTERN}_eeg.set")):
        stem = set_path.name.removesuffix("_eeg.set")
        events_path = eeg_dir / f"{stem}_events.tsv"
        channels_path = eeg_dir / f"{stem}_channels.tsv"
        json_path = eeg_dir / f"{stem}_eeg.json"

        if not events_path.exists():
            raise FileNotFoundError(f"Missing events TSV for {set_path.name}")

        bundles.append({
            "stem": Path(stem),
            "set": set_path,
            "events": events_path,
            "channels": channels_path,
            "json": json_path,
        })
    return bundles


def load_raw_and_tables(bundle: dict[str, Path]) -> tuple[mne.io.BaseRaw, pd.DataFrame, pd.DataFrame | None]:
    raw = mne.io.read_raw_eeglab(bundle["set"], preload=True, verbose="ERROR")
    events_df = pd.read_csv(bundle["events"], sep="\t")
    channels_df = pd.read_csv(bundle["channels"], sep="\t") if bundle["channels"].exists() else None
    
    data = raw.get_data()
    empty_chs = [ch for ch, row in zip(raw.ch_names, data) if not np.isfinite(row).any()]
    if empty_chs:
       print(f"Dropping channel(s) with no valid data: {empty_chs}")
       raw.drop_channels(empty_chs)

    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, on_missing="ignore")
    raw.info["line_freq"] = LINE_FREQ
    return raw, events_df, channels_df
    
    
def calculate_behavior(events_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    needed = {"user_answer", "correct_answer"}

    if not needed.issubset(events_df.columns):
        behavior_summary = {
            "n_behavior_trials": 0,
            "exact_accuracy_pct": None,
            "element_accuracy_pct": None,
        }
        return behavior_summary, pd.DataFrame()

    # Keep only behavioral-response rows
    accuracy_df = events_df[
        events_df["user_answer"].notna() &
        events_df["correct_answer"].notna()
    ].copy()

    if accuracy_df.empty:
        behavior_summary = {
            "n_behavior_trials": 0,
            "exact_accuracy_pct": None,
            "element_accuracy_pct": None,
        }
        return behavior_summary, accuracy_df

    # Exact whole-sequence accuracy
    accuracy_df["correct"] = (
        accuracy_df["user_answer"].astype(str).str.strip() ==
        accuracy_df["correct_answer"].astype(str).str.strip()
    )

    exact_accuracy = accuracy_df["correct"].mean() * 100


    # Trial-by-trial / element accuracy
    def sequence_accuracy(user, correct):
        user = str(user).split("-")
        correct = str(correct).split("-")

        matches = [u == c for u, c in zip(user, correct)]

        return np.mean(matches)

    accuracy_df["trial_accuracy"] = accuracy_df.apply(
        lambda row: sequence_accuracy(
            row["user_answer"],
            row["correct_answer"]),axis=1)

    average_trial_accuracy = (
        accuracy_df["trial_accuracy"].mean() * 100)

    accuracy_df["average_trial_accuracy"] = average_trial_accuracy

    # ---------------------------------------------------------
    # Dot-level remembered / forgotten status
    # ---------------------------------------------------------
    def dot_accuracy(user, correct):

        if pd.isna(user) or pd.isna(correct):
            return None

        user = str(user).split("-")
        correct = str(correct).split("-")

        return [
            "remembered" if u == c else "forgotten"
            for u, c in zip(user, correct)]

    accuracy_df["memory_status"] = accuracy_df.apply(
        lambda row: dot_accuracy(
            row["user_answer"],
            row["correct_answer"]),axis=1)

    # ---------------------------------------------------------
    # Summary for preprocessing QC
    # ---------------------------------------------------------
    behavior_summary = {
        "n_behavior_trials": len(accuracy_df),
        "exact_accuracy_pct": exact_accuracy,
        "element_accuracy_pct": average_trial_accuracy,}

    return behavior_summary, accuracy_df


def get_bids_bad_channels(raw: mne.io.BaseRaw, channels_df: pd.DataFrame | None) -> list[str]:
    if channels_df is None or not {"name", "status"}.issubset(channels_df.columns):
        return []
    bads = channels_df.loc[
        channels_df["status"].astype(str).str.lower().eq("bad"), "name"
    ].astype(str).tolist()
    return sorted(set(bads).intersection(raw.ch_names))


def detect_and_interpolate_bad_channels(raw,bids_bads=None,threshold_pct=BAD_CHANNEL_THRESHOLD):
    
    nd = NoisyChannels(raw, random_state=42)
    
    nd.find_bad_by_nan_flat(flat_threshold=1e-15)
    nd.find_bad_by_deviation(deviation_threshold=5.0)
    nd.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
    nd.find_bad_by_correlation(correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.05)
    nd.find_bad_by_SNR()
    nd.find_bad_by_PSD(zscore_threshold=5.0, fmin=1.0, fmax=45.0)
    nd.find_bad_by_ransac( n_samples=50,
        sample_prop=0.25,
        corr_thresh=0.75,
        frac_bad=0.5,
        corr_window_secs=5.0,
        channel_wise=True,
        max_chunk_size=None)
    
    bad_channels = nd.get_bads()
    if bids_bads is None:
        bids_bads = []
    bad_channels = sorted(set(bad_channels).union(bids_bads).intersection(raw.ch_names))
    bad_pct      = len(bad_channels) / 128 * 100
    print(f"  Bad channels ({bad_pct:.1f}%): {bad_channels}")
    
    if bad_pct > threshold_pct:
        print(f"  /!\\ Subject excluded: {len(bad_channels)} bad channels ({bad_pct:.1f}%)")
        return raw, bad_channels, True

    raw.info["bads"] = bad_channels
    if 'Cz' in raw.ch_names:
        raw.drop_channels(['Cz'])
        raw.interpolate_bads()
    print(f"  Interpolated {len(bad_channels)} channels.")
    return raw, bad_channels, False


def interpolate_or_exclude(
    raw: mne.io.BaseRaw,
    detected_bads: list[str],
    bids_bads: list[str],
) -> tuple[mne.io.BaseRaw, list[str], float, bool]:
    all_bads = sorted(set(detected_bads).union(bids_bads).intersection(raw.ch_names))
    n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    bad_pct = 100 * len(all_bads) / n_eeg if n_eeg else 100.0
    excluded = bad_pct > BAD_CHANNEL_THRESHOLD

    raw.info["bads"] = all_bads
    if not excluded and all_bads:
        raw.interpolate_bads(reset_bads=False, verbose="ERROR")
    return raw, all_bads, bad_pct, excluded


def run_ica(raw: mne.io.BaseRaw) -> tuple[ICA, list[int], list[str], list[float]]:
    raw_for_ica = raw.copy().filter(HIGHPASS_ICA, LOWPASS, verbose="ERROR")
    raw_for_ica.set_eeg_reference("average", projection=False, verbose="ERROR")
    

    ica = ICA(
        n_components=None,
        method="picard", fit_params=dict(ortho=False, extended=True),
        max_iter=1000,
        random_state=RANDOM_STATE,)
    ica.fit(raw_for_ica, picks="eeg", reject_by_annotation=True, verbose="ERROR")

    labels: list[str] = []
    probabilities: list[float] = []
    exclude: list[int] = []

    if AUTO_ICA:
        if label_components is None:
            raise ImportError("AUTO_ICA=True requires mne-icalabel")
        result = label_components(raw_for_ica, ica, method="iclabel")
        labels = list(result["labels"])
        probabilities = [float(x) for x in result["y_pred_proba"]]
        exclude = [
            i for i, (label, prob) in enumerate(zip(labels, probabilities))
            if label in ICA_LABELS_TO_EXCLUDE and prob >= ICA_MIN_PROBABILITY
        ]

    return ica, exclude, labels, probabilities

# =============================================================================
# EPOCH CREATION
# =============================================================================

def create_epochs(
    raw_clean: mne.io.BaseRaw,
    events_df: pd.DataFrame,
) -> tuple[mne.Epochs | None, mne.Epochs | None, mne.Epochs | None]:

    events_df = events_df.copy()
    events_df = (
    events_df[~events_df["value"].isin(["boundary", "break cnt"])]
    .sort_values("sample")
    .reset_index(drop=True))

    events = np.column_stack([
    events_df["sample"].astype(int),
    np.zeros(len(events_df), dtype=int),
    events_df["event_code"].astype(int),])

    # =========================================================
    # 1. LEARNING-BLOCK EPOCHS
    # =========================================================
    learning_event_dict = {
        "learningBlock_1": 31,
        "learningBlock_2": 32,
        "learningBlock_3": 33,
        "learningBlock_4": 34,
        "learningBlock_5": 35,}

    # Only retain event IDs that actually occur in this participant
    available_codes = set(events[:, 2])

    learning_event_dict = {
        name: code
        for name, code in learning_event_dict.items()
        if code in available_codes}

    if learning_event_dict:
        epochs_learningBlock = mne.Epochs(
            raw_clean,
            events,
            event_id=learning_event_dict,
            tmin=-1.0,
            tmax=17.5,
            baseline=None,
            detrend=1,
            reject=None,
            preload=True,
            decim=2,
            reject_by_annotation=True,
            verbose="ERROR",)
    else:
        epochs_learningBlock = None

    # =========================================================
    # 2. DOT-ONSET EPOCHS
    # =========================================================

    dot_event_dict = {
        "dot_no1_ON": 11,
        "dot_no2_ON": 12,
        "dot_no3_ON": 13,
        "dot_no4_ON": 14,
        "dot_no5_ON": 15,
        "dot_no6_ON": 16,
        "dot_no7_ON": 17,
        "dot_no8_ON": 18,}

    # Again, only use events actually present
    dot_event_dict = {
        name: code
        for name, code in dot_event_dict.items()
        if code in available_codes}

    dot_codes = list(dot_event_dict.values())

    print("\nDOT EVENT CHECK")
    print(events_df[events_df["event_code"].isin(dot_codes)][["value", "event_code", "onset"]])
    
    print("Number of dot events before MNE epoching:",events_df["event_code"].isin(dot_codes).sum())

    print("Counts by dot identity:")
    print(events_df.loc[events_df["event_code"].isin(dot_codes),"value"].value_counts())
    
    
    if dot_event_dict:
        epochs_dots = mne.Epochs(
            raw_clean,
            events,
            event_id=dot_event_dict,
            tmin=-0.7,
            tmax=1.5,
            baseline=None,
            detrend=1,
            reject=None,
            preload=True,
            decim=2,
            reject_by_annotation=True,
            verbose="ERROR",)
    else:
        epochs_dots = None
        
    # =========================================================
    # 3. CONTINUOUS EPOCH
    # =========================================================

    tmin_epochs = -1

    tmax_epochs = (events_df.loc[events_df["value"] == "seqLearning_stop", "onset"].iloc[0]-events_df.loc[events_df["value"] == "seqLearning_start", "onset"].iloc[0])

    epochs_continuous = mne.Epochs(
    raw_clean,
    events,
    event_id={"seqLearning_start": 91},
    tmin=tmin_epochs,
    tmax=tmax_epochs,
    baseline=None,
    detrend=1,
    reject=None,
    preload=True,
    decim=2,
    reject_by_annotation=True,
    verbose="ERROR",)

    # ---------------------------------------------------------
    # Quick QC
    # ---------------------------------------------------------
    print("\nEpoch creation:")
    print(
        "  Learning blocks:",
        0 if epochs_learningBlock is None else len(epochs_learningBlock))
    print(
        "  Dot epochs:",
        0 if epochs_dots is None else len(epochs_dots))
    print(
        " Continuous Epoch:",
        0 if epochs_continuous is None else len(epochs_continuous))

    return epochs_learningBlock, epochs_dots, epochs_continuous

def save_outputs(
    raw_clean: mne.io.BaseRaw,
    epochs_learningBlock: mne.Epochs | None,
    epochs_dots: mne.Epochs | None,
    epochs_continuous: mne.Epochs | None,
    ica: ICA,
    events_df: pd.DataFrame,
    accuracy_df: pd.DataFrame, 
    qc: dict[str, Any],
    output_dir: Path,
    stem: str,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    learning_path = (output_dir /f"{stem}_epochs_learningBlock_EEG_epo.fif")
    dots_path = (output_dir /f"{stem}_epochs_dots_EEG_epo.fif")
    continuous_path = (output_dir /f"{stem}_epochs_continuous_EEG_epo.fif")

    ica_path = output_dir / f"{stem}_ica.fif"

    accuracy_path = (output_dir /f"{stem}_accuracy_df_behavioral.csv")

    events_path = (output_dir /f"{stem}_events_df_behavioral.csv")

    qc_path = (output_dir /f"{stem}_preprocessing_info.json")



    if epochs_learningBlock is not None:
        epochs_learningBlock.save(learning_path,overwrite=OVERWRITE)

    if epochs_dots is not None:
        epochs_dots.save(dots_path,overwrite=OVERWRITE)

    if epochs_continuous is not None:
        epochs_continuous.save(continuous_path,overwrite=OVERWRITE)


    ica.save(ica_path,overwrite=OVERWRITE)

    accuracy_df.to_csv(accuracy_path,index=False)

    events_df.to_csv(events_path,index=False)

    save_json(qc,qc_path)


# =============================================================================
# ONE RECORDING / ONE SUBJECT
# =============================================================================
def preprocess_recording(subject_id: str, bundle: dict[str, Path]) -> dict[str, Any]:
    stem = str(bundle["stem"])
    output_dir = OUTPUT_ROOT / subject_id / "eeg"
    expected_raw = output_dir / f"{stem}_epochs_continuous_EEG_epo.fif"

    if expected_raw.exists() and not OVERWRITE:
        return {"subject_id": subject_id, "recording": stem, "status": "skipped_existing"}

    raw, events_df, channels_df = load_raw_and_tables(bundle)
    original_duration = raw.times[-1]
    original_n_channels = len(raw.ch_names)

    task_start = events_df.loc[events_df["value"] == "seqLearning_start", "onset"].iloc[0]

    task_stop = events_df.loc[events_df["value"] == "seqLearning_stop", "onset"].iloc[0]

    behavior, accuracy_df = calculate_behavior(events_df)

    bids_bads = get_bids_bad_channels(raw, channels_df)

    raw, all_bads, excluded = detect_and_interpolate_bad_channels(raw,bids_bads=bids_bads)
    detected_bads = all_bads
    n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    bad_pct = 100 * len(all_bads) / n_eeg if n_eeg else 100.0

    qc: dict[str, Any] = {
        "subject_id": subject_id,
        "recording": stem,
        "status": "excluded" if excluded else "processed",
        "reason": "too_many_bad_channels" if excluded else None,
        "input_set": bundle["set"],
        "original_duration_s": original_duration,
        "task_start_s": task_start,
        "task_stop_s": task_stop,
        "duration_s": raw.times[-1],
        "original_n_channels": original_n_channels,
        "n_eeg_channels": len(mne.pick_types(raw.info, eeg=True, exclude=[])),
        "bids_bad_channels": bids_bads,
        "pyprep_bad_channels": detected_bads,
        "all_bad_channels": all_bads,
        "bad_channel_pct": bad_pct,
        **behavior,}

    if excluded:
        save_json(qc, output_dir / f"{stem}_preprocessing_info.json")
        return qc

    # Final continuous data: filter and average-reference after interpolation.
    raw.filter(HIGHPASS_FINAL, LOWPASS, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    ica, exclude_idx, labels, probabilities = run_ica(raw)
    raw_clean = raw.copy()
    if APPLY_AUTOMATIC_ICA:
        ica.apply(raw_clean, exclude=exclude_idx, verbose="ERROR")
    
    epochs_learningBlock, epochs_dots, epochs_continuous = create_epochs(raw_clean,events_df)

    qc.update({
        "ica_applied": APPLY_AUTOMATIC_ICA,
        "ica_components_removed": exclude_idx,
        "ica_labels": labels,
        "ica_label_probabilities": probabilities,
        "n_ica_components_removed": len(exclude_idx),

        # Epoch QC
        "n_learningBlock_epochs":
            0 if epochs_learningBlock is None
            else len(epochs_learningBlock),

        "n_dot_epochs":
            0 if epochs_dots is None
            else len(epochs_dots),
        "n_continuous_epochs":
            0 if epochs_continuous is None
            else len(epochs_continuous),})


    save_outputs(raw_clean,epochs_learningBlock,epochs_dots,epochs_continuous,ica, events_df,accuracy_df, qc, output_dir, stem)
    return qc


def preprocess_subject(subject_dir: Path) -> list[dict[str, Any]]:
    subject_id = subject_dir.name
    bundles = discover_sequence_files(subject_dir)
    if not bundles:
        return [{"subject_id": subject_id, "status": "no_sequence_learning_files"}]

    results = []
    for bundle in bundles:
        try:
            print(f"\n{'=' * 72}\nProcessing {subject_id}: {bundle['set'].name}\n{'=' * 72}")
            results.append(preprocess_recording(subject_id, bundle))
        except Exception as exc:
            results.append({
                "subject_id": subject_id,
                "recording": str(bundle["stem"]),
                "status": "error",
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"ERROR: {subject_id}: {exc}")
    return results


# =============================================================================
# BATCH RUNNER
# =============================================================================
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(BIDS_ROOT)
    print(f"Found {len(subjects)} subject folders")

    all_results: list[dict[str, Any]] = []
    for subject_dir in subjects:
        all_results.extend(preprocess_subject(subject_dir))

        # Checkpoint after every subject, so a crash does not lose group QC.
        pd.DataFrame(all_results).to_csv(
            OUTPUT_ROOT / "sequence_learning_preprocessing_summary.csv", index=False
        )
        save_json(
            {"results": all_results},
            OUTPUT_ROOT / "sequence_learning_preprocessing_summary.json",
        )

    summary = pd.DataFrame(all_results)
    print("\nBatch preprocessing complete")
    if not summary.empty and "status" in summary:
        print(summary["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
