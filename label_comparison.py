import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def compute_syllable_score( prediction_on_offset_list, label_on_offset_list, tolerance  ):
    n_positive_in_prediction = len(prediction_on_offset_list)
    n_positive_in_label = len(label_on_offset_list)

    n_true_positive = 0
    for pred_onset, pred_offset, pred_cluster in prediction_on_offset_list:
        is_matched = False
        for count, (label_onset, label_offset, label_cluster) in enumerate( label_on_offset_list ):
            if np.abs( pred_onset -  label_onset )<=tolerance and np.abs( pred_offset - label_offset )<= tolerance and pred_cluster == label_cluster:
                # print( (pred_onset, pred_offset), (label_onset, label_offset) )
                n_true_positive += 1
                is_matched = True
                break  # early stop for the predicted value
        if is_matched:
            ## remove the already matched syllable from the ground-truth
            label_on_offset_list.pop(count)

    return n_true_positive, n_positive_in_prediction, n_positive_in_label

def segment_score( predicted_df, gt_df, target_cluster = None, tolerance = 0.02 ):

    prediction_on_offset_list = []
    for pos in range(len(predicted_df["onset"])):
        if target_cluster is None or str(target_cluster) == str(predicted_df["clustername"][pos]):
            prediction_on_offset_list.append([ predicted_df["onset"][pos], predicted_df["offset"][pos], str(predicted_df["clustername"][pos]) ])

    label_on_offset_list = []
    for pos in range(len(gt_df["onset"])):
        if target_cluster is None or str(target_cluster) == str( gt_df["clustername"][pos] ):
            label_on_offset_list.append([ gt_df["onset"][pos], gt_df["offset"][pos], str(gt_df["clustername"][pos]) ])

    if target_cluster is not None and len(label_on_offset_list) == 0:
        print("Warning: the specified target cluster '%s' does not exist in the ground-truth labels."%(str(target_cluster)))

    TP, P_pred, P_label = compute_syllable_score( prediction_on_offset_list, label_on_offset_list, tolerance  )

    precision = TP / max(P_pred, 1e-12  )
    recall = TP / max( P_label, 1e-12 )
    f1 = 2/(1/ max(precision, 1e-12) + 1/max(recall, 1e-12)  )

    return {
        "Number of True Positive": int(TP),
        "Number of Positive in Prediction": int(P_pred),
        "Number of Positive in Label": int(P_label),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1)
    }

def frame_score( predicted_df, gt_df, target_cluster = None, time_per_frame_for_scoring = 0.001 ):

    
    prediction_segments = {
        "cluster" : predicted_df["clustername"],
        "onset" : predicted_df["onset"],
        "offset" : predicted_df["offset"]
        
    }
    
    label_segments = {
        
        "cluster" : gt_df["clustername"],
        "onset" : gt_df["onset"],
        "offset" : gt_df["offset"]
        
    }
    
    #prediction_segments = prediction
    #label_segments = label

    prediction_segments["cluster"] = list( map(str, prediction_segments["cluster"]) )
    label_segments["cluster"] = list( map(str, label_segments["cluster"]) )

    cluster_to_id_mapper = {}
    for cluster in list(prediction_segments["cluster"]) + list(label_segments["cluster"]):
        if cluster not in cluster_to_id_mapper:
            cluster_to_id_mapper[cluster] = len( cluster_to_id_mapper )

    all_timestamps = list(prediction_segments["onset"]) + list(prediction_segments["offset"]) + \
                        list(label_segments["onset"]) + list( label_segments["offset"] )
    if len(all_timestamps) == 0:
        max_time = 1.0
    else:
        max_time = np.max( all_timestamps )

    num_frames = int(np.round( max_time / time_per_frame_for_scoring )) + 1

    frame_wise_prediction = np.ones( num_frames ) * -1
    for idx in range( len( prediction_segments["onset"] ) ):
        onset_pos = int(np.round( prediction_segments["onset"][idx] / time_per_frame_for_scoring ))
        offset_pos = int(np.round( prediction_segments["offset"][idx] / time_per_frame_for_scoring ))
        frame_wise_prediction[onset_pos:offset_pos] = cluster_to_id_mapper[ prediction_segments["cluster"][idx] ]

    frame_wise_label = np.ones( num_frames ) * -1
    for idx in range( len( label_segments["onset"] ) ):
        onset_pos = int(np.round( label_segments["onset"][idx] / time_per_frame_for_scoring ))
        offset_pos = int(np.round( label_segments["offset"][idx] / time_per_frame_for_scoring ))
        frame_wise_label[onset_pos:offset_pos] = cluster_to_id_mapper[ label_segments["cluster"][idx] ]

    if target_cluster is None:
        TP = np.logical_and( frame_wise_label != -1, frame_wise_prediction == frame_wise_label ).sum()
        P_in_pred = (frame_wise_prediction != -1).sum()
        P_in_label = (frame_wise_label != -1).sum()
    else:
        target_cluster_id = cluster_to_id_mapper[target_cluster]
        TP = np.logical_and( frame_wise_label == target_cluster_id, frame_wise_prediction == frame_wise_label ).sum()
        P_in_pred = (frame_wise_prediction == target_cluster_id).sum()
        P_in_label = (frame_wise_label == target_cluster_id).sum()


    precision = TP / max(P_in_pred, 1e-12)
    recall = TP / max(P_in_label, 1e-12)
    f1 = 2/( 1/max( precision, 1e-12 ) + 1/max( recall, 1e-12 ) )

    return {
        "Number of True Positive": int(TP),
        "Number of Positive in Prediction": int(P_in_pred),
        "Number of Positive in Label": int(P_in_label),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1)
    }
    
def spectrogram_comparison(clip_path, df1, df2, target_cluster):
    # Load audio
    y, sr = librosa.load(clip_path)
    
    # Create spectrogram
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='hz', cmap='gray')
    
    # Get time axis limits
    duration = len(y) / sr
    
    # Add annotations for df1 (green)
    for _, row in df1.iterrows():
        if str(row['clustername']) == str(target_cluster):
            plt.axvspan(row['onset'], row['offset'], alpha=0.3, color='green', ymin=0, ymax=1)
    
    # Add annotations for df2 (red)
    for _, row in df2.iterrows():
        if str(row['clustername']) == str(target_cluster):
            plt.axvspan(row['onset'], row['offset'], alpha=0.3, color='red', ymin=0, ymax=1)
    
    # Add legend
    legend_elements = [Patch(facecolor='green', alpha=0.3, label='DF1'),
                      Patch(facecolor='red', alpha=0.3, label='DF2')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Overlapping regions will appear yellow due to red+green transparency
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram with {target_cluster} annotations')
    plt.tight_layout()
    plt.show()
    