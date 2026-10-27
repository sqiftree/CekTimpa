"""
Modul Metrik Evaluasi Kinerja Forensik Citra & Kualitas Citra
============================================================
Materi Pengolahan Citra Digital & Forensik:
- Image Quality Assessment: MSE (Mean Squared Error) & PSNR (Peak Signal-to-Noise Ratio)
- Klasifikasi Tingkat Piksel: Precision, Recall, F1-Score (Dice), IoU (Jaccard Index)
- Klasifikasi Tingkat Citra: True Positive Rate (TPR), False Positive Rate (FPR), Akurasi
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


def calculate_mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """
    Menghitung Mean Squared Error (MSE) antara dua citra berukuran sama.
    MSE = (1 / (H * W * C)) * Σ (I_A - I_B)^2
    """
    if image_a.shape != image_b.shape:
        raise ValueError(f"Ukuran citra harus identik: {image_a.shape} vs {image_b.shape}")

    diff = image_a.astype(np.float64) - image_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray, max_val: float = 255.0) -> float:
    """
    Menghitung Peak Signal-to-Noise Ratio (PSNR) dalam satuan desibel (dB).
    PSNR = 10 * log10(MAX^2 / MSE)

    Returns:
        Nilai PSNR (float). Jika citra identik (MSE=0), mengembalikan float('inf').
    """
    mse = calculate_mse(image_a, image_b)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def calculate_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Menghitung metrik evaluasi tingkat piksel (Pixel-Level Metrics).
    
    Binerisasi:
      Piksel > 0 dianggap Positif (Terindikasi Pemalsuan).
      Piksel == 0 dianggap Negatif (Area Citra Asli).

    Metrik yang dihasilkan:
      - Precision: TP / (TP + FP)
      - Recall (Sensitivity): TP / (TP + FN)
      - F1-Score (Dice): 2 * TP / (2 * TP + FP + FN)
      - IoU (Jaccard Index): TP / (TP + FP + FN)
      - Specificity (TNR): TN / (TN + FP)
      - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Dimensi mask prediksi {pred_mask.shape} tidak cocok dengan ground truth {gt_mask.shape}")

    p_pos = pred_mask > 0
    g_pos = gt_mask > 0

    tp = float(np.sum(p_pos & g_pos))
    fp = float(np.sum(p_pos & (~g_pos)))
    fn = float(np.sum((~p_pos) & g_pos))
    tn = float(np.sum((~p_pos) & (~g_pos)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if fp == 0 and fn == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 1.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def calculate_image_metrics(predictions: Sequence[bool], true_labels: Sequence[bool]) -> Dict[str, float]:
    """
    Menghitung metrik evaluasi tingkat citra (Image-Level Detection Metrics):
    - TPR (True Positive Rate / Recall) = TP / (TP + FN)
    - FPR (False Positive Rate / Fall-out) = FP / (FP + TN)
    - Akurasi = (TP + TN) / Total
    - Precision = TP / (TP + FP)
    - F1-Score = 2 * P * R / (P + R)
    """
    if len(predictions) != len(true_labels):
        raise ValueError("Jumlah prediksi dan label aktual harus sama.")

    tp = sum(1 for p, y in zip(predictions, true_labels) if p and y)
    fp = sum(1 for p, y in zip(predictions, true_labels) if p and not y)
    fn = sum(1 for p, y in zip(predictions, true_labels) if not p and y)
    tn = sum(1 for p, y in zip(predictions, true_labels) if not p and not y)
    total = len(predictions)

    if total == 0:
        return {"tpr": 0.0, "fpr": 0.0, "accuracy": 0.0, "precision": 0.0, "f1": 0.0}

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total
    f1 = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
