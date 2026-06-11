"""Online sensor-fault detection and correction script.

This script loads offline-trained model utilities, trains a CAE for reconstruction,
trains a CNN-LSTM classifier for fault classification, and applies the detection
pipeline to online water-quality data.
"""

import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Model

from offline_code import train_cae, train_cnn, train_cnnlstm, train_model


def has_consecutive_zeros(data, window_size=20):
    """Return True if the input contains at least `window_size` consecutive zeros."""
    convolved = np.convolve(data == 0, np.ones(window_size, dtype=int), mode="valid")
    return np.any(convolved == window_size)


# Fault labels: missing=0, incremental=1, bias=2, spiky=3
scaler = MinMaxScaler(feature_range=(0, 1))

# CNN-LSTM settings
time_steps_cnn = 48
features_cnn = 1
n_classes = 4

# CAE settings
time_steps = 336
features_cae = 6

input_shape_cae = (time_steps, features_cae)
input_shape_cnn = (time_steps_cnn, features_cnn)

# Normal data for CAE training
df_normal = pd.read_csv("normal_noisy.csv")
df_normal = df_normal.loc[336:8400 - 1, :]
x_train_normal = df_normal.to_numpy().reshape(-1, time_steps, features_cae)

# Online data
df_arr = pd.read_csv("hanoi_n5_com_bias_noisy.csv")
df_arr = df_arr.loc[336:8400 - 1, :]
x_arr_orig = df_arr.to_numpy().reshape(-1, time_steps, features_cae)

# Anomalous data for CNN-LSTM training
df_ano = pd.read_csv("train_reducedecoded_noisy.csv")
X_normal = df_normal.to_numpy()
x_train_ano = df_ano.drop("class", axis=1).to_numpy()
X_normal = X_normal.reshape(-1, time_steps_cnn, features_cnn)
x_train_ano = x_train_ano.reshape(-1, time_steps_cnn, features_cnn)
y_train_ano = np.array([1] * 35 + [2] * 35 + [3] * 35 + [4] * 35) - 1


def run_detection_once(
    run_id,
    x_train_normal,
    x_arr_input,
    x_train_ano,
    y_train_ano,
    input_shape_cae,
    input_shape_cnn,
    replace,
):
    """Run one independent detection experiment."""
    cae, thre_list, _ = train_cae(x_train_normal, input_shape_cae)
    model = train_model("cnnlstm", x_train_ano, y_train_ano, input_shape_cnn)

    x_arr = x_arr_input.copy()
    run_logs = [f"===== Run {run_id} ====="]

    reconstruction_loss_batch9 = 0
    correction_loss_batch9 = 0
    batch9_count = 0
    missing = 0

    print(f"Starting Run {run_id}")

    for i in range(x_arr.shape[0]):
        batch = x_arr[i:i + 1, :, :]
        decoded_batch = cae.predict(batch)
        anomalous_count = 0
        fault_index = []
        batch_normal = x_train_normal[i:i + 1, :, :]

        for j in range(features_cae):
            if has_consecutive_zeros(batch[:, :, j].flatten()) and missing == 0:
                print(i, "missing fault detected; threshold adjusted", j)
                thre_list[j] = thre_list[j] * 5
                missing = 1

        for j in range(features_cae):
            reconstruction_loss = np.mean(np.square(batch[:, :, j] - decoded_batch[:, :, j]))
            if reconstruction_loss > thre_list[j]:
                print(
                    "batch",
                    i,
                    "feature:",
                    j,
                    "reconstruction loss:",
                    reconstruction_loss,
                    "threshold:",
                    thre_list[j],
                )
                anomalous_count += 1
                fault_index.append(j)

                if i == 9:
                    correction_loss = np.mean(np.square(batch_normal[:, :, j] - decoded_batch[:, :, j]))
                    reconstruction_loss_batch9 += reconstruction_loss
                    correction_loss_batch9 += correction_loss
                    batch9_count += 1

        if anomalous_count >= 1:
            for feat_idx in fault_index:
                if not has_consecutive_zeros(batch[:, :, feat_idx].flatten()):
                    x_pred = batch[:, :, feat_idx] - decoded_batch[:, :, feat_idx]
                else:
                    x_pred = batch[:, :, feat_idx]

                x_pred = x_pred.reshape(-1, time_steps_cnn, features_cnn)
                prediction = model.predict(x_pred)
                predicted_class = np.argmax(prediction, axis=1)

                log_msg = f"batch {i}, feature {feat_idx}, {predicted_class.tolist()}"
                print(log_msg)
                run_logs.append(log_msg)

                if replace == 1 and i + 1 < x_arr.shape[0]:
                    x_arr[i + 1, :, feat_idx] = decoded_batch[0, :, feat_idx]

            if anomalous_count > 1:
                print("possible contamination", fault_index)
            else:
                print("faulty sensor values replaced")

    batch9_reconstruction_loss = reconstruction_loss_batch9 / batch9_count if batch9_count > 0 else 0
    batch9_correction_loss = correction_loss_batch9 / batch9_count if batch9_count > 0 else 0

    return run_logs, batch9_reconstruction_loss, batch9_correction_loss


num_runs = 1
final_all_logs = []
metrics_summary = []

# Replacement mode: 1 = with replacement, 0 = without replacement
replace = 1

for run_id in range(1, num_runs + 1):
    logs, reconstruction_loss_b9, correction_loss_b9 = run_detection_once(
        run_id,
        x_train_normal,
        x_arr_orig,
        x_train_ano,
        y_train_ano,
        input_shape_cae,
        input_shape_cnn,
        replace,
    )
    final_all_logs.extend(logs)
    final_all_logs.append("\n")
    metrics_summary.append((run_id, reconstruction_loss_b9, correction_loss_b9))

output_path = "detection_results_summary_obvious_hanoi_bias_contam.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for line in final_all_logs:
        f.write(line + "\n")

    f.write("\n===== Batch 9 Metrics Summary =====\n")
    for run_id, reconstruction_loss_b9, correction_loss_b9 in metrics_summary:
        f.write(
            f"Run {run_id}: Mean Reconstruction Loss B9: {reconstruction_loss_b9}, "
            f"Mean Correction Loss B9: {correction_loss_b9}\n"
        )

print(f"All results saved to {output_path}")
