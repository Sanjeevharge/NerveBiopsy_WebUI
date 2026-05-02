import React, { useState, useRef, useEffect } from 'react';
import type { MouseEvent, WheelEvent } from 'react';
import Editor from '@monaco-editor/react';
import './App.css';

const DEFAULT_CODE = `import cv2
import numpy as np
import os
import pandas as pd
import json
import base64
import math
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import random
from collections import deque

# --------------------------
# CONFIG (UI ADAPTED)
# --------------------------
scale_factor = 0.136

# --------------------------
# Ray casting config
# --------------------------
N_RAYS = 36          
RAY_STEP = 0.5       

# --------------------------
# Tunables
# --------------------------
TISSUE_OVERLAP_MIN = 0.50
MAX_MEAN_INTENSITY = 245

MIN_CIRCULARITY = 0.05
MIN_CONTOUR_AREA = 8
MAX_CONTOUR_AREA = 2000000
MIN_SOLIDITY = 0.15

RAY_ANGLES             = np.linspace(0, 360, N_RAYS, endpoint=False)
OUTER_RAY_COL_LABELS   = [f"outer_ray_{int(a):03d}deg_um"     for a in RAY_ANGLES]
INNER_RAY_COL_LABELS   = [f"inner_ray_{int(a):03d}deg_um"     for a in RAY_ANGLES]
THICK_RAY_COL_LABELS   = [f"thickness_ray_{int(a):03d}deg_um" for a in RAY_ANGLES]

# --------------------------
# Helpers
# --------------------------
def darken_and_sharpen(image):
    darkened = cv2.convertScaleAbs(image, alpha=1.5, beta=-20)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    return cv2.filter2D(darkened, -1, sharpen_kernel)

def build_tissue_mask(full_image):
    hsv = cv2.cvtColor(full_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    sat_mask = cv2.inRange(S, 15, 255)
    not_too_bright = cv2.inRange(V, 0, 250)
    mask = cv2.bitwise_and(sat_mask, not_too_bright)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def contour_overlap_ratio(contour, mask):
    c_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(c_mask, [contour], -1, 255, -1)
    inter = cv2.bitwise_and(c_mask, mask)
    area_c = cv2.countNonZero(c_mask)
    area_i = cv2.countNonZero(inter)
    return area_i / float(area_c) if area_c > 0 else 0.0

def circularity(contour):
    a = cv2.contourArea(contour)
    p = cv2.arcLength(contour, True)
    return 4 * np.pi * a / (p * p + 1e-6) if p > 0 else 0

def cast_rays_to_boundary(contour, cx, cy, n_rays=N_RAYS, step=RAY_STEP):
    x_b, y_b, w_b, h_b = cv2.boundingRect(contour)
    pad = 20
    max_r = int(max(w_b, h_b) + pad)

    local_cx = max_r
    local_cy = max_r
    shifted = contour - np.array([[int(cx) - max_r, int(cy) - max_r]])

    mask_size = max_r * 2 + 2
    local_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    cv2.drawContours(local_mask, [shifted], -1, 255, -1)

    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ray_distances_px = np.zeros(n_rays)

    for i, angle in enumerate(angles):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        r_vals = np.arange(0, max_r, step)
        xs = (local_cx + r_vals * cos_a).astype(int)
        ys = (local_cy + r_vals * sin_a).astype(int)

        valid = (xs >= 0) & (xs < mask_size) & (ys >= 0) & (ys < mask_size)
        xs_v = xs[valid]
        ys_v = ys[valid]
        r_v  = r_vals[valid]

        if len(xs_v) == 0:
            ray_distances_px[i] = 0
            continue

        pixel_vals = local_mask[ys_v, xs_v]

        inside_idx = np.where(pixel_vals == 255)[0]
        if len(inside_idx) == 0:
            ray_distances_px[i] = 0
            continue

        outside_after = np.where((pixel_vals == 0) & (np.arange(len(pixel_vals)) > inside_idx[0]))[0]
        if len(outside_after) > 0:
            ray_distances_px[i] = r_v[outside_after[0]]
        else:
            ray_distances_px[i] = r_v[-1]

    return ray_distances_px, np.degrees(angles)

def draw_rays_on_image(output_image, cx, cy, ray_distances_px, ray_angles_deg, chosen_idx, x_offset, y_offset):
    angles_rad = np.deg2rad(ray_angles_deg)
    for i, (r, ang) in enumerate(zip(ray_distances_px, angles_rad)):
        end_x = int(cx + r * np.cos(ang))
        end_y = int(cy + r * np.sin(ang))
        color = (0, 0, 255) if i == chosen_idx else (200, 200, 0)
        cv2.line(output_image,
                 (int(cx + x_offset), int(cy + y_offset)),
                 (end_x + x_offset, end_y + y_offset),
                 color, 1)

def local_laplacian_variance(lap_img, k):
    mean = cv2.blur(lap_img, (k, k))
    mean_sq = cv2.blur(lap_img ** 2, (k, k))
    var = mean_sq - mean ** 2
    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX)
    return var.astype(np.uint8)

def has_nonzero_neighbor(img, x, y):
    h, w = img.shape
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if img[nx, ny] > 0:
                    return True
    return False

def multi_source_clustering(img, seeds):
    h, w = img.shape
    labels = np.zeros((h, w), dtype=int)
    queue = deque()

    for i, (x, y) in enumerate(seeds):
        labels[x, y] = i + 1
        queue.append((x, y, i + 1))

    while queue:
        x, y, cid = queue.popleft()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    if labels[nx, ny] == 0:
                        if img[nx, ny] > 0 or has_nonzero_neighbor(img, nx, ny):
                            labels[nx, ny] = cid
                            queue.append((nx, ny, cid))
    return labels

def run_fullimage_analysis(image, output_folder, img_name):
    orig = image.copy()
    mask = np.any(orig != 0, axis=2)
    mask_img = (mask * 255).astype(np.uint8)
    
    metrics = {}

    total_fascicle_area = np.sum(mask)
    total_fascicle_area_um2 = total_fascicle_area * (scale_factor ** 2)
    metrics["Total fascicle area (um2)"] = round(float(total_fascicle_area_um2), 2)
    
    print("[*] Total fascicle area (pixels):", total_fascicle_area)
    print("[*] Total fascicle area (um2):", total_fascicle_area_um2)

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(gray_denoised, cv2.CV_64F)
    lap = np.abs(lap)

    img_blur = local_laplacian_variance(lap, 7)

    plt.figure(figsize=(5, 5))

    plt.imshow(img_blur, cmap="gray")
    plt.title("Blur Map (7x7)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "debug_blur_map.png"), dpi=150)
    plt.close()

    k_values = [100]
    nonzero_pixels = list(zip(*np.where(img_blur > 0)))

    if len(nonzero_pixels) == 0:
        print("[!] No non-zero pixels in blur map, skipping clustering.")
        return None, metrics

    results = {}
    for k in k_values:
        seeds = random.sample(nonzero_pixels, min(k, len(nonzero_pixels)))
        labels = multi_source_clustering(img_blur, seeds)
        results[k] = labels

    final_median_mask = None

    for k in k_values:
        labels = results[k]
        binary = np.zeros_like(labels, dtype=np.uint8)
        binary[labels > 0] = 255
        kernel_sizes = [65]

        n = len(kernel_sizes)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))

        plt.figure(figsize=(4 * cols, 4 * rows))
        plt.suptitle(f"Median Filters (k = {k})", fontsize=16)

        for i, ks in enumerate(kernel_sizes):
            median = cv2.medianBlur(binary, ks)

            nonblurred_area = np.sum(median == 255)
            nonblurred_area_um2 = nonblurred_area * (scale_factor ** 2)
            metrics["Non-blurred area (um2)"] = round(float(nonblurred_area_um2), 2)

            blurred_area = total_fascicle_area - nonblurred_area
            blurred_area_um2 = blurred_area * (scale_factor ** 2)
            metrics["Blurred area (um2)"] = round(float(blurred_area_um2), 2)

            plt.subplot(rows, cols, i + 1)
            plt.title(f"{ks}x{ks}")
            plt.imshow(median, cmap="gray")
            plt.axis("off")

            final_median_mask = median

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"debug_median_filter_k{k}.png"), dpi=150)
        plt.close()

    return final_median_mask, metrics

BLURRED_REGION_THRESHOLD = 75.0   

def pct_contour_in_blurred_region(contour, blurred_region_mask, x_offset, y_offset):
    if blurred_region_mask is None:
        return 0.0

    h_full, w_full = blurred_region_mask.shape[:2]
    contour_abs = contour + np.array([[x_offset, y_offset]])
    tmp = np.zeros((h_full, w_full), dtype=np.uint8)
    cv2.drawContours(tmp, [contour_abs], -1, 255, -1)

    total_px = cv2.countNonZero(tmp)
    if total_px == 0:
        return 0.0

    blurred_zone = (blurred_region_mask == 0).astype(np.uint8) * 255
    inter = cv2.bitwise_and(tmp, blurred_zone)
    blurred_px = cv2.countNonZero(inter)
    return 100.0 * blurred_px / total_px

def process_patch(patch, mask_patch, x_offset, y_offset, axon_data, object_counter,
                  output_image, axons_image, blurred_region_mask):

    patch = darken_and_sharpen(patch)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 5, 20, 20)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(smooth)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 2
    )
    thresh = cv2.bitwise_or(otsu, adaptive)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist_transform.max() > 0:
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    else:
        sure_fg = np.zeros_like(opening)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    wshed_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(wshed_input, markers)

    cleaned = thresh.copy()
    cleaned[markers == -1] = 0

    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return object_counter
    hierarchy = hierarchy[0]

    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue

        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        overlap = contour_overlap_ratio(contour, mask_patch)
        if overlap < TISSUE_OVERLAP_MIN:
            continue

        hull = cv2.convexHull(contour)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        circ = circularity(contour)
        if solidity < MIN_SOLIDITY or circ < MIN_CIRCULARITY:
            continue

        inner_children = []
        for j, child in enumerate(contours):
            if hierarchy[j][3] == i and cv2.contourArea(child) > 30:
                inner_children.append(child)

        if len(inner_children) == 0:
            continue

        axon_type = "mature" if len(inner_children) <= 2 else "regrowth_cluster"

        (x_outer, y_outer), outer_radius_enc = cv2.minEnclosingCircle(contour)
        contour_shifted = contour + np.array([x_offset, y_offset])
        x_outer_abs = x_outer + x_offset
        y_outer_abs = y_outer + y_offset

        outer_area_px  = cv2.contourArea(contour)
        outer_area_um2 = outer_area_px * (scale_factor ** 2)

        pct_blurred = pct_contour_in_blurred_region(
            contour, blurred_region_mask, x_offset, y_offset
        )
        is_in_blurred = pct_blurred >= BLURRED_REGION_THRESHOLD

        if is_in_blurred:
            outer_color = (0, 0, 255)          
        elif axon_type == "mature":
            outer_color = (255, 0, 0)          
        else:
            outer_color = (0, 255, 255)        

        cv2.drawContours(output_image, [contour_shifted], -1, outer_color, 1)
        cv2.drawContours(axons_image,  [contour_shifted], -1, outer_color, 1)

        inner_radii = []
        inner_areas  = []
        for inner_c in inner_children:
            (_, _), ir = cv2.minEnclosingCircle(inner_c)
            inner_shifted = inner_c + np.array([x_offset, y_offset])
            cv2.drawContours(output_image, [inner_shifted], -1, (0, 255, 0), 1)
            cv2.drawContours(axons_image,  [inner_shifted], -1, (0, 255, 0), 1)
            inner_radii.append(ir)
            inner_areas.append(cv2.contourArea(inner_c))

        inner_radius   = max(inner_radii)
        inner_area_px  = max(inner_areas)
        inner_area_um2 = inner_area_px * (scale_factor ** 2)

        outer_radius_um  = outer_radius_enc * scale_factor
        inner_radius_um  = inner_radius      * scale_factor

        outer_ray_dict  = {col: np.nan for col in OUTER_RAY_COL_LABELS}
        inner_ray_dict  = {col: np.nan for col in INNER_RAY_COL_LABELS}
        thick_ray_dict  = {col: np.nan for col in THICK_RAY_COL_LABELS}

        chosen_outer_um  = outer_radius_um   
        chosen_inner_um  = inner_radius_um
        chosen_thick_um  = np.nan            
        chosen_ray_angle = np.nan
        chosen_idx       = -1

        if axon_type == "mature" and not is_in_blurred:
            outer_rays_px, ray_angles_deg = cast_rays_to_boundary(
                contour, x_outer, y_outer, n_rays=N_RAYS, step=RAY_STEP
            )
            outer_rays_um = outer_rays_px * scale_factor

            largest_inner_c = inner_children[np.argmax(inner_areas)]
            inner_rays_px, _ = cast_rays_to_boundary(
                largest_inner_c, x_outer, y_outer, n_rays=N_RAYS, step=RAY_STEP
            )
            inner_rays_um = inner_rays_px * scale_factor

            thick_per_dir = outer_rays_um - inner_rays_um   
            for o_col, i_col, t_col, o_val, i_val, t_val in zip(
                OUTER_RAY_COL_LABELS, INNER_RAY_COL_LABELS, THICK_RAY_COL_LABELS,
                outer_rays_um, inner_rays_um, thick_per_dir
            ):
                outer_ray_dict[o_col] = round(float(o_val), 4)
                inner_ray_dict[i_col] = round(float(i_val), 4)
                thick_ray_dict[t_col] = round(float(t_val), 4)

            positive_mask_ray = thick_per_dir > 0
            if positive_mask_ray.any():
                valid_thick = np.where(positive_mask_ray, thick_per_dir, np.inf)
                chosen_idx       = int(np.argmin(valid_thick))
                chosen_outer_um  = float(outer_rays_um[chosen_idx])
                chosen_inner_um  = float(inner_rays_um[chosen_idx])
                chosen_thick_um  = float(thick_per_dir[chosen_idx])
                chosen_ray_angle = float(ray_angles_deg[chosen_idx])
            else:
                chosen_thick_um  = max(outer_radius_um - inner_radius_um, 0.0)

            draw_rays_on_image(
                output_image, x_outer, y_outer, outer_rays_px, ray_angles_deg, chosen_idx, x_offset, y_offset
            )

        if is_in_blurred:
            g_ratio    = np.nan
            area_ratio = np.nan
            thickness_um = np.nan
            diameter_um  = np.nan
        else:
            thickness_um = chosen_thick_um
            diameter_um  = round(2 * chosen_outer_um, 4)
            g_ratio      = chosen_inner_um / (chosen_outer_um + 1e-6)
            area_ratio   = inner_area_um2  / (outer_area_um2  + 1e-6)

        record = {
            "axon_id"                  : object_counter,
            "axon_type"                : axon_type,
            "num_inner_contours"       : len(inner_children),
            "center_x_px"             : round(float(x_outer_abs), 2),
            "center_y_px"             : round(float(y_outer_abs), 2),
            "pct_in_blurred_region"   : round(pct_blurred, 2),
            "blurred_axon_flag"        : "YES" if is_in_blurred else "NO",
            "outer_radius_enc_um"     : round(outer_radius_um, 4),
            "inner_radius_enc_um"     : round(inner_radius_um, 4),
            "chosen_ray_angle_deg"    : (round(chosen_ray_angle, 2)
                                         if axon_type == "mature" and not is_in_blurred
                                         else np.nan),
            "chosen_outer_ray_um"     : round(chosen_outer_um, 4),
            "chosen_inner_ray_um"     : round(chosen_inner_um, 4),
            "thickness_um"            : (round(thickness_um, 4)
                                         if not (isinstance(thickness_um, float) and np.isnan(thickness_um))
                                         else np.nan),
            "diameter_um"             : (round(diameter_um, 4)
                                         if not (isinstance(diameter_um, float) and np.isnan(diameter_um))
                                         else np.nan),
            "outer_area_um2"          : round(outer_area_um2, 4),
            "inner_area_um2"          : round(inner_area_um2, 4),
            "area_ratio"              : (round(area_ratio, 4)
                                         if not (isinstance(area_ratio, float) and np.isnan(area_ratio))
                                         else np.nan),
            "g_ratio"                 : (round(g_ratio, 4)
                                         if not (isinstance(g_ratio, float) and np.isnan(g_ratio))
                                         else np.nan),
        }

        record.update(outer_ray_dict)
        record.update(inner_ray_dict)
        record.update(thick_ray_dict)

        axon_data.append(record)

        cv2.putText(output_image, str(object_counter),
                    (int(x_outer_abs), int(y_outer_abs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(axons_image, str(object_counter),
                    (int(x_outer_abs), int(y_outer_abs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        object_counter += 1

    return object_counter

def process_image(image_path, img_output_folder, patch_size=1024):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[X] Cannot read image: {image_path}")
        return None, None, None, {}

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process Full Image Metrics
    return_tuple = run_fullimage_analysis(image, img_output_folder, img_name)
    blurred_region_mask = return_tuple[0]
    area_metrics = return_tuple[1]

    tissue_mask_full = build_tissue_mask(image)
    h, w = image.shape[:2]

    output_image  = image.copy()
    axons_image   = image.copy()   
    axon_data     = []
    object_counter = 1

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch      = image[y:y + patch_size, x:x + patch_size]
            mask_patch = tissue_mask_full[y:y + patch_size, x:x + patch_size]
            object_counter = process_patch(
                patch, mask_patch, x, y, axon_data, object_counter, output_image, axons_image,
                blurred_region_mask
            )

    return axon_data, output_image, axons_image, area_metrics

def save_to_excel(axon_data, output_folder, image_name):
    if not axon_data:
        return None
    df = pd.DataFrame(axon_data)
    base_cols = [
        "axon_id", "axon_type", "num_inner_contours",
        "center_x_px", "center_y_px",
        "pct_in_blurred_region", "blurred_axon_flag",
        "outer_radius_enc_um", "inner_radius_enc_um",
        *OUTER_RAY_COL_LABELS, *INNER_RAY_COL_LABELS, *THICK_RAY_COL_LABELS,
        "chosen_ray_angle_deg", "chosen_outer_ray_um", "chosen_inner_ray_um",
        "thickness_um", "diameter_um", "outer_area_um2", "inner_area_um2",
        "area_ratio", "g_ratio",
    ]
    ordered_cols = [c for c in base_cols if c in df.columns]
    df = df[ordered_cols]

    excel_path = os.path.join(output_folder, f"{image_name}_axon_measurements.xlsx")
    df.to_excel(excel_path, index=False, sheet_name="Axon Measurements")

    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    HEADER_FILL    = PatternFill("solid", fgColor="1F4E79")   
    OUTER_RAY_FILL = PatternFill("solid", fgColor="E8F4FD")   
    INNER_RAY_FILL = PatternFill("solid", fgColor="EAF4E8")   
    THICK_RAY_FILL = PatternFill("solid", fgColor="FFF2CC")   
    MATURE_FILL    = PatternFill("solid", fgColor="D9F0D3")   
    REGROWTH_FILL  = PatternFill("solid", fgColor="FCE4D6")   
    CHOSEN_FILL    = PatternFill("solid", fgColor="FFD700")   
    THICK_FILL     = PatternFill("solid", fgColor="F4CCCC")   
    BLURRED_FILL   = PatternFill("solid", fgColor="FF6B6B")   

    WHITE_FONT     = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
    HEADER_BORDER  = Border(bottom=Side(style="medium", color="FFFFFF"))
    CELL_BORDER    = Border(bottom=Side(style="thin",   color="D0D0D0"))

    col_names = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    def col_idx(name):
        try:    return col_names.index(name) + 1
        except: return None

    outer_ray_col_indices = [col_idx(c) for c in OUTER_RAY_COL_LABELS if col_idx(c)]
    inner_ray_col_indices = [col_idx(c) for c in INNER_RAY_COL_LABELS if col_idx(c)]
    thick_ray_col_indices = [col_idx(c) for c in THICK_RAY_COL_LABELS if col_idx(c)]
    chosen_cols           = {col_idx(c) for c in ("chosen_ray_angle_deg", "chosen_outer_ray_um", "chosen_inner_ray_um") if col_idx(c)}
    thickness_col         = col_idx("thickness_um")
    axon_type_col         = col_idx("axon_type")
    blurred_cols          = {col_idx(c) for c in ("pct_in_blurred_region", "blurred_axon_flag") if col_idx(c)}

    for cell in ws[1]:
        cell.fill      = HEADER_FILL
        cell.font      = WHITE_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = HEADER_BORDER

    for ci in outer_ray_col_indices: ws.cell(1, ci).fill = PatternFill("solid", fgColor="1565C0")
    for ci in inner_ray_col_indices: ws.cell(1, ci).fill = PatternFill("solid", fgColor="2E7D32")
    for ci in thick_ray_col_indices: ws.cell(1, ci).fill = PatternFill("solid", fgColor="F57F17")
    for ci in chosen_cols: ws.cell(1, ci).fill = PatternFill("solid", fgColor="B8860B")
    for ci in blurred_cols:
        ws.cell(1, ci).fill = PatternFill("solid", fgColor="C0392B")
        ws.cell(1, ci).font = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
    if thickness_col:
        ws.cell(1, thickness_col).fill = PatternFill("solid", fgColor="8B0000")
        ws.cell(1, thickness_col).font = Font(color="FFFFFF", bold=True, name="Calibri", size=10)

    for row_idx in range(2, ws.max_row + 1):
        atype = ws.cell(row_idx, axon_type_col).value if axon_type_col else ""
        row_fill = MATURE_FILL if atype == "mature" else REGROWTH_FILL
        for col_idx_ in range(1, ws.max_column + 1):
            cell = ws.cell(row_idx, col_idx_)
            cell.border    = CELL_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.fill      = row_fill   
            if col_idx_ in outer_ray_col_indices: cell.fill = OUTER_RAY_FILL
            elif col_idx_ in inner_ray_col_indices: cell.fill = INNER_RAY_FILL
            elif col_idx_ in thick_ray_col_indices: cell.fill = THICK_RAY_FILL
            elif col_idx_ in chosen_cols:
                cell.fill = CHOSEN_FILL
                cell.font  = Font(bold=True, name="Calibri", size=10)
            elif thickness_col and col_idx_ == thickness_col:
                cell.fill = THICK_FILL
                cell.font  = Font(bold=True, name="Calibri", size=10)
            elif col_idx_ in blurred_cols:
                cell.fill = BLURRED_FILL
                cell.font  = Font(bold=True, name="Calibri", size=10)

    for col_num in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_num)
        header_val = str(ws.cell(1, col_num).value or "")
        if header_val.startswith(("outer_ray_", "inner_ray_", "thickness_ray_")):
            ws.column_dimensions[col_letter].width = 14
        elif header_val in ("axon_id", "axon_type", "num_inner_contours"):
            ws.column_dimensions[col_letter].width = 16
        else:
            ws.column_dimensions[col_letter].width = 20
    ws.row_dimensions[1].height = 40
    ws.freeze_panes = "A2"

    ws_summary = wb.create_sheet("Summary")
    mature_df   = df[df["axon_type"] == "mature"]
    regrowth_df = df[df["axon_type"] == "regrowth_cluster"]

    summary_rows = [
        ("Image",                image_name),
        ("Total axons",          len(df)),
        ("Mature axons",         len(mature_df)),
        ("Regrowth clusters",    len(regrowth_df)),
        ("", ""),
        ("--- Mature Axons (ray-based measurements) ---", ""),
        ("Mean outer radius enc. (µm)",  round(mature_df["outer_radius_enc_um"].mean(), 4)  if len(mature_df) else "N/A"),
        ("Mean inner radius enc. (µm)",  round(mature_df["inner_radius_enc_um"].mean(), 4)  if len(mature_df) else "N/A"),
        ("Mean chosen outer ray (µm)",   round(mature_df["chosen_outer_ray_um"].mean(), 4)  if len(mature_df) else "N/A"),
        ("Mean chosen inner ray (µm)",   round(mature_df["chosen_inner_ray_um"].mean(), 4)  if len(mature_df) else "N/A"),
        ("Mean thickness (µm)",          round(mature_df["thickness_um"].mean(), 4)          if len(mature_df) else "N/A"),
        ("Mean g-ratio",                 round(mature_df["g_ratio"].mean(), 4)               if len(mature_df) else "N/A"),
        ("Mean diameter (µm)",           round(mature_df["diameter_um"].mean(), 4)           if len(mature_df) else "N/A"),
        ("", ""),
        ("--- Regrowth Clusters ---", ""),
        ("Mean outer radius enc. (µm)",  round(regrowth_df["outer_radius_enc_um"].mean(), 4) if len(regrowth_df) else "N/A"),
        ("Mean thickness (µm)",          round(regrowth_df["thickness_um"].mean(), 4)        if len(regrowth_df) else "N/A"),
    ]

    for r, (label, value) in enumerate(summary_rows, start=1):
        c1 = ws_summary.cell(r, 1, value=label)
        c2 = ws_summary.cell(r, 2, value=value)
        c1.font = Font(bold=True, name="Calibri", size=10)
        c2.alignment = Alignment(horizontal="center")
        if str(label).startswith("---"):
            c1.fill = HEADER_FILL
            c1.font = WHITE_FONT

    ws_summary.column_dimensions["A"].width = 38
    ws_summary.column_dimensions["B"].width = 20
    wb.save(excel_path)
    return df

def save_plots(df, output_folder):
    mature_df = df[df["axon_type"] == "mature"]
    if len(mature_df) == 0: return

    # Plot 1: G-ratio distribution
    g_vals = mature_df["g_ratio"].dropna()
    if len(g_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(g_vals, bins=30, color="mediumpurple", edgecolor="white")
        ax.axvline(g_vals.mean(), color="red", linestyle="--", label=f"Mean = {g_vals.mean():.4f}")
        ax.set_xlabel("G-ratio (inner / outer radius)")
        ax.set_ylabel("Number of mature axons")
        ax.set_title("G-ratio distribution (mature axons)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "plot_gratio_distribution.png"), dpi=150)
        plt.close(fig)

    # Plot 2: Outer radius distribution
    outer_vals = mature_df["chosen_outer_ray_um"].dropna()
    if len(outer_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(outer_vals, bins=30, color="royalblue", edgecolor="white")
        ax.axvline(outer_vals.mean(), color="red", linestyle="--", label=f"Mean = {outer_vals.mean():.2f} µm")
        ax.set_xlabel("Outer radius (µm)")
        ax.set_ylabel("Number of mature axons")
        ax.set_title("Outer radius distribution (mature axons)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "plot_outer_radius_distribution.png"), dpi=150)
        plt.close(fig)

    # Plot 3: Inner radius distribution
    inner_vals = mature_df["chosen_inner_ray_um"].dropna()
    if len(inner_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(inner_vals, bins=30, color="seagreen", edgecolor="white")
        ax.axvline(inner_vals.mean(), color="red", linestyle="--", label=f"Mean = {inner_vals.mean():.2f} µm")
        ax.set_xlabel("Inner radius (µm)")
        ax.set_ylabel("Number of mature axons")
        ax.set_title("Inner radius distribution (mature axons)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "plot_inner_radius_distribution.png"), dpi=150)
        plt.close(fig)

    # Plot 4: Thickness distribution
    thick_vals = mature_df["thickness_um"].dropna()
    if len(thick_vals) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(thick_vals, bins=30, color="darkorange", edgecolor="white")
        ax.axvline(thick_vals.mean(), color="red", linestyle="--", label=f"Mean = {thick_vals.mean():.2f} µm")
        ax.set_xlabel("Myelin thickness (µm)")
        ax.set_ylabel("Number of mature axons")
        ax.set_title("Myelin thickness distribution (mature axons)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "plot_thickness_distribution.png"), dpi=150)
        plt.close(fig)

# --------------------------
# UI EXECUTION HUB
# --------------------------
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(i) for i in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

image_path = 'input.png'
img_output_folder = '.'

# Run Core Execution
axon_data, output_image, axons_image, area_metrics = process_image(image_path, img_output_folder)

if axon_data and len(axon_data) > 0:
    df = save_to_excel(axon_data, img_output_folder, "analysis")
    
    cv2.imwrite(os.path.join(img_output_folder, "numbered_output.png"), output_image)
    cv2.imwrite(os.path.join(img_output_folder, "axons_only.png"), axons_image)
    if df is not None: save_plots(df, img_output_folder)
    
    mature_df = df[df["axon_type"] == "mature"]
    regrowth_df = df[df["axon_type"] == "regrowth_cluster"]

    total_axons = len(df)
    density = round(len(mature_df) / total_axons, 4) if total_axons > 0 else 0.0

    results = {
        "Total Axons Detected": total_axons,
        "Mature Axons": len(mature_df),
        "Regrowth Clusters": len(regrowth_df),
        "Axon Density (%)": round(density * 100, 1),
        "axon_list": axon_data
    }

    # Inject Area metrics automatically!
    if area_metrics:
        results.update(area_metrics)

    if len(mature_df) > 0:
        if not math.isnan(mature_df["thickness_um"].mean()):
            results["Average Thickness (um)"] = round(float(mature_df["thickness_um"].mean()), 4)
        if not math.isnan(mature_df["diameter_um"].mean()):
            results["Average Diameter (um)"] = round(float(mature_df["diameter_um"].mean()), 4)

    # Encode the created Excel file into base64 to send to UI Download Button!
    excel_path = os.path.join(img_output_folder, "analysis_axon_measurements.xlsx")
    if os.path.exists(excel_path):
        with open(excel_path, "rb") as xl:
            results["excel_base64"] = base64.b64encode(xl.read()).decode("utf-8")

    with open('results.json', 'w') as f:
        json.dump(clean_nans(results), f)
        
    print("[*] Analysis complete. Metrics, images, and Excel file exported!")
else:
    with open('results.json', 'w') as f:
        json.dump({"Status": "No axons found in this image"}, f)
    print("[!] No axons detected.")
`;

function App() {
  const [view, setView] = useState<'main' | 'pipeline'>('main');

  const [code, setCode] = useState(DEFAULT_CODE);
  const [file, setFile] = useState<File | null>(null);
  const [previewURL, setPreviewURL] = useState<string | null>(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<any | null>(null);
  const [scaleFactor, setScaleFactor] = useState(0.136);
  const [nRays, setNRays] = useState(36);
  const [showEditor, setShowEditor] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);

  // Advanced Interaction State
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredAxon, setHoveredAxon] = useState<any>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const imageRef = useRef<HTMLImageElement>(null);
  const metricEntries = result?.metrics
    ? Object.entries(result.metrics).filter(([k]) => k !== "excel_base64" && k !== "axon_list")
    : [];
  const getMetricTintClass = (key: string) => {
    const lower = key.toLowerCase();
    if (lower.includes("axon") || lower.includes("count")) return "metric-tint-blue";
    if (lower.includes("area")) return "metric-tint-purple";
    return "metric-tint-teal";
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewURL(URL.createObjectURL(selectedFile));
    }
  };


  const handleRun = async () => {
    if (!file) {
      alert("Please upload an image first!");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    // Inject the selected parameters globally into the script before running
    let processedCode = code.replace(/scale_factor\s*=\s*[\d.]+/, `scale_factor = ${scaleFactor}`);
    processedCode = processedCode.replace(/N_RAYS\s*=\s*\d+/, `N_RAYS = ${nRays}`);
    formData.append("code", processedCode);

    // Hardcoded backend URL due to isolated local env
    try {
      const backendUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const res = await fetch(`${backendUrl}/execute`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ status: 'error', error: "Failed to connect to backend. Is it running?" });
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadExcel = () => {
    if (!result?.metrics?.excel_base64) return;
    const bytes = atob(result.metrics.excel_base64);
    const byteNumbers = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) byteNumbers[i] = bytes.charCodeAt(i);
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'axon_analysis.xlsx';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleWheel = (e: WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const sf = e.deltaY < 0 ? 1.1 : 0.9;
    setScale((prev) => Math.max(0.2, Math.min(prev * sf, 20)));
  };

  const handleMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (isDragging) {
      setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    if (imageRef.current && result?.metrics?.axon_list && selectedImage?.name && (selectedImage.name.includes("numbered") || selectedImage.name.includes("axons_only"))) {
      const rect = imageRef.current.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;

      const x = clientX - rect.left;
      const y = clientY - rect.top;

      const scaleX = imageRef.current.naturalWidth / rect.width;
      const scaleY = imageRef.current.naturalHeight / rect.height;
      const realX = x * scaleX;
      const realY = y * scaleY;

      setMousePos({ x: clientX, y: clientY });

      let closest = null;
      let minDst = 20 * scaleX; // Detection threshold

      for (const axon of result.metrics.axon_list) {
        if (!axon.center_x_px || !axon.center_y_px) continue;
        const dx = realX - axon.center_x_px;
        const dy = realY - axon.center_y_px;
        const dst = Math.sqrt(dx * dx + dy * dy);
        if (dst < minDst) {
          minDst = dst;
          closest = axon;
        }
      }
      setHoveredAxon(closest);
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const resetModal = () => {
    setSelectedImage(null);
    setScale(1);
    setPosition({ x: 0, y: 0 });
    setHoveredAxon(null);
  };

  // Prevent default scroll behavior when modal is open
  useEffect(() => {
    const preventScroll = (e: Event) => e.preventDefault();
    if (selectedImage) {
      window.addEventListener('wheel', preventScroll, { passive: false });
    } else {
      window.removeEventListener('wheel', preventScroll);
    }
    return () => window.removeEventListener('wheel', preventScroll);
  }, [selectedImage]);

  return (
    <div className={`app-shell ${isDarkMode ? '' : 'light-mode'}`}>
      <header className="top-navbar">
        <div className="navbar-content navbar-layout">
          <div className="navbar-left">
            <button className="theme-toggle-btn" onClick={() => setIsDarkMode(!isDarkMode)}>
              {isDarkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </button>
          </div>
          <div className="navbar-center">
            <h1 className="app-title">Nerve Biopsy Analyzer</h1>
            <p className="eyebrow app-subtitle">AI-Assisted Histology Workspace</p>
          </div>
          <div className="navbar-right">
            <span className="status-badge status-badge-large"><span className="status-dot status-dot-large" />Ready</span>
          </div>
        </div>
      </header>

      <div className="app-container">
        {view === 'main' ? (
          <div className="main-content">
            {/* LEFT PANE */}
            <div className="left-pane panel-shell">
              <div className="panel-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Input Operations</span>
                <button className="btn-outline-small" onClick={() => setView('pipeline')} style={{ cursor: 'pointer' }}>
                  Pipeline View
                </button>
              </div>

              <div className={`uploader glass ${file ? 'has-file' : ''}`} style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                {!file ? (
                  <>
                    <svg width="48" height="48" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style={{ marginBottom: "1rem", color: "var(--accent)" }}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <p className="uploader-title">Drag and drop a fascicle image</p>
                    <p className="uploader-subtitle">Use PNG, JPG, or TIFF for best results</p>
                    <label className="browse-link">
                      Browse File
                      <input type="file" style={{ display: "none" }} accept="image/*" onChange={handleFileChange} />
                    </label>
                  </>
                ) : (
                  <div className="uploaded-file-content uploaded-file-content-expanded">
                    <img src={previewURL!} className="uploaded-image uploaded-image-expanded" alt="Preview" />
                    <div className="uploaded-file-row">
                      <span className="uploaded-file-name">Image: {file.name}</span>
                      <label className="btn-outline-small">
                        Change
                        <input type="file" style={{ display: "none" }} accept="image/*" onChange={handleFileChange} />
                      </label>
                    </div>
                  </div>
                )}
              </div>

              <div className="action-row" style={{ display: 'flex', gap: '0.5rem' }}>
                <button className="settings-btn" onClick={() => setShowSettings(!showSettings)} title="Advanced Settings">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                  </svg>
                </button>
                <button className="btn-run primary" onClick={handleRun} disabled={loading || !file} style={{ flex: 1, padding: "1.2rem", fontSize: "1.1rem" }}>
                  {loading ? 'Running...' : 'Run Analysis 🚀'}
                </button>
              </div>

              {showSettings && (
                <div className="settings-container glass animated fade-in">
                  <div style={{ marginBottom: "1.5rem" }}>
                    <label className="settings-label">Scale Factor (µm/px):</label>
                    <input
                      type="number"
                      step="0.001"
                      value={scaleFactor}
                      onChange={(e) => setScaleFactor(parseFloat(e.target.value) || scaleFactor)}
                      className="settings-input"
                    />
                  </div>
                  <div style={{ marginBottom: "1.5rem" }}>
                    <label className="settings-label">N Rays (Number of Rays):</label>
                    <input
                      type="number"
                      step="1"
                      value={nRays}
                      onChange={(e) => setNRays(parseInt(e.target.value) || nRays)}
                      className="settings-input"
                    />
                  </div>
                  <button className="btn-run settings-open-editor" onClick={() => { setShowSettings(false); setShowEditor(true); }}>
                    Edit Pipeline Source Code
                  </button>
                </div>
              )}
            </div>

            {/* RIGHT PANE */}
            <div className="right-pane panel-shell glass results-container">
              <div className="panel-label">Output</div>
              <div className="quick-stats">
                <div className="quick-stat">
                  <span>Status</span>
                  <strong>{loading ? "Running" : result?.status === "success" ? "Completed" : result?.status === "error" ? "Error" : "Idle"}</strong>
                </div>
                <div className="quick-stat">
                  <span>Metrics</span>
                  <strong>{metricEntries.length}</strong>
                </div>
                <div className="quick-stat">
                  <span>Images</span>
                  <strong>{result?.images?.length || 0}</strong>
                </div>
              </div>

              {loading && (
                <div className="empty-state">
                  <div className="spinner">⌛</div>
                  <p>Executing your Python script...</p>
                  <p className="empty-state-sub">This might take a moment.</p>
                </div>
              )}

              {!loading && !result && (
                <div className="empty-state">
                  <div className="empty-state-icon">🧪</div>
                  <p>Upload an image and run your script to see results here.</p>
                </div>
              )}

              {!loading && result && result.status === 'success' && (
                <>
                  <div className="result-header">
                    <h2 style={{ margin: 0 }}>Execution Results</h2>
                    {result.metrics?.excel_base64 && (
                      <button className="btn-run success" onClick={handleDownloadExcel}>
                        📥 Download Excel
                      </button>
                    )}
                  </div>

                  {/* Metrics */}
                  {metricEntries.length > 0 && (
                    <div className="metrics-grid">
                      {metricEntries.map(([key, val], idx) => (
                        <div className={`metric-card animated slide-up ${getMetricTintClass(key)}`} style={{ animationDelay: `${idx * 0.05}s` }} key={key}>
                          <span>{key}</span>
                          <strong>{String(val)}</strong>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Logs if any */}
                  {result.stdout && (
                    <div className="logs-block">
                      <span className="section-caption">Standard Output</span>
                      <div className="logs-terminal">
                        <div className="terminal-header">
                          <span className="dot red" />
                          <span className="dot yellow" />
                          <span className="dot green" />
                        </div>
                        <pre className="logs">
                          {result.stdout.split('\n').map((line: string, idx: number) => (
                            <div className={line.trim().startsWith('[*]') ? 'log-highlight' : ''} key={`${idx}-${line}`}>
                              {line || ' '}
                            </div>
                          ))}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Image Gallery */}
                  {result.images && result.images.length > 0 && (
                    <div className="gallery-block">
                      <h3 className="generated-images-title">Generated Images</h3>
                      <div className="gallery">
                        {result.images.map((img: any, idx: number) => (
                          <div
                            className="gallery-item animated fade-in"
                            style={{ animationDelay: `${idx * 0.1}s` }}
                            key={idx}
                            onClick={() => setSelectedImage(img)}
                          >
                            <div className="img-wrapper">
                              <img src={img.data} alt={img.name} title="Click to view full screen" />
                              <div className="img-overlay">
                                <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path>
                                </svg>
                                <span>Zoom</span>
                              </div>
                            </div>
                            <span className="img-name">{img.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {!loading && result && result.status === 'error' && (
                <div style={{ color: "var(--error)" }}>
                  <h2>Execution Failed</h2>
                  <p>{result.error}</p>
                  <pre className="logs" style={{ color: "var(--error)" }}>{result.stderr}</pre>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="pipeline-view" style={{ display: 'flex', flexDirection: 'column', padding: '2rem', height: '100%', gap: '1.5rem', flex: 1 }}>
            <div className="pipeline-header" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
              <button className="btn-run" onClick={() => setView('main')} style={{ padding: '0.8rem 1.5rem', backgroundColor: 'var(--panel)', border: '1px solid var(--glass-border)', cursor: 'pointer', borderRadius: '8px', color: 'var(--text-main)' }}>
                ← Back to Analysis
              </button>
              <h2 style={{ margin: 0, color: 'var(--text-main)' }}>Pipeline Overview</h2>
            </div>

            <div className="pipeline-content" style={{ display: 'flex', flexDirection: 'row', height: '100%', gap: '1.5rem', overflow: 'hidden' }}>
              {/* 1st half: HTML Output */}
              <div className="pipeline-html-container glass panel-shell" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <div className="panel-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>HTML Output Viewer</span>
                </div>
                <div style={{ flex: 1, backgroundColor: '#fff', borderRadius: '0 0 12px 12px', overflow: 'hidden' }}>
                  <iframe src="/pipeline.html" width="100%" height="100%" title="Pipeline Output" style={{ border: 'none', width: '100%', height: '100%' }}></iframe>
                </div>
              </div>

              {/* 2nd half: Text Content */}
              <div className="pipeline-text-container glass panel-shell" style={{ flex: 1, overflowY: 'auto', padding: '1.5rem' }}>
                <div className="panel-label">Pipeline Steps</div>
                <ul style={{ lineHeight: '1.8', fontSize: '1rem', paddingLeft: '1.5rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.8rem', marginTop: '1rem' }}>
                  <li><strong>Image input</strong> — Reads the source PNG from disk using OpenCV. If the file can't be loaded, the pipeline halts immediately. Output: a raw BGR image array.</li>
                  <li><strong>Full-image analysis</strong> — Converts the image to grayscale, applies Laplacian edge detection, then builds a local variance "blur map" using a 7×7 kernel. Multi-source BFS clustering is run on that map, and a median filter (65×65) smooths the result into a binary mask. Outputs: the blurred-region mask + area metrics (total fascicle area, blurred area, non-blurred area in µm²).</li>
                  <li><strong>Tissue mask builder</strong> — Converts the image to HSV colour space and thresholds on saturation (S &gt; 15) and brightness (V &lt; 250) to isolate actual tissue from background. Morphological open/close operations clean up noise. Output: a binary tissue mask used to discard detections that fall outside tissue.</li>
                  <li><strong>Patch-tiling loop</strong> — Divides the full image into 1024 × 1024 overlapping tiles and iterates over them. This keeps memory usage manageable for large slides. Output: individual patches fed one at a time into the downstream steps.</li>
                  <li><strong>Patch pre-process</strong> — Each tile is darkened, sharpened with a Laplacian kernel, then bilaterally filtered to smooth noise while preserving edges. CLAHE (contrast-limited adaptive histogram equalisation) is applied to boost local contrast. Output: a contrast-enhanced grayscale tile ready for thresholding.</li>
                  <li><strong>Thresholding</strong> — Two independent thresholds are computed: global Otsu on a Gaussian-blurred image, and local adaptive Gaussian thresholding. The two binary masks are OR-combined to catch both high-contrast and subtly dark axon profiles. Output: a combined binary foreground mask.</li>
                  <li><strong>Watershed segmentation</strong> — Distance transform + foreground seeding is used to separate touching axons before finding contours. The watershed boundary (markers == -1) is zeroed out to prevent merged detections. Output: a cleaned binary mask with touching objects split.</li>
                  <li><strong>Contour detection</strong> — cv2.findContours with RETR_TREE extracts all contours plus their parent/child hierarchy. Each outer contour is filtered on: area (8–2,000,000 px²), tissue overlap (≥ 50%), solidity (≥ 0.15), circularity (≥ 0.05), and must have at least one inner child contour (the axon lumen). Output: a validated list of outer myelin contours.</li>
                  <li><strong>Axon classification</strong> — The number of inner child contours decides the type. ≤ 2 inner contours → mature axon (single well-myelinated fibre). &gt; 2 → regrowth_cluster (a bundle of re-myelinating fibres sharing one outer sheath). Output: axon type label used to branch subsequent measurement logic.</li>
                  <li><strong>Ray casting (36 rays)</strong> — For mature axons only: 36 evenly-spaced rays are cast from the centroid outward toward both the outer myelin contour and the inner axon contour, in a local coordinate frame. Each ray records the distance to the boundary in pixels, then converted to µm. Output: 36 outer-ray distances + 36 inner-ray distances per axon.</li>
                  <li><strong>Thickness + g-ratio</strong> — Per-direction myelin thickness = outer_ray − inner_ray. The minimum positive thickness across all 36 directions (the "thinnest wall") is chosen as the representative measurement. G-ratio = chosen inner radius ÷ chosen outer radius. Diameter = 2 × chosen outer radius. Output: thickness_um, g_ratio, diameter_um.</li>
                  <li><strong>Regrowth record</strong> — For regrowth clusters the ray-based measurements are skipped (geometry is too irregular). Only min-enclosing-circle radius and area are recorded; per-direction ray columns are filled with NaN. Output: partial record with area + radius fields.</li>
                  <li><strong>Blur region flag</strong> — Checks what fraction of the axon's area falls inside the blurred zone from the full-image mask. If ≥ 75% of the axon overlaps the blurred region, it is flagged (blurred_axon_flag = YES) and all geometry measurements are set to NaN to avoid noisy data corrupting statistics. Output: pct_in_blurred_region and blurred_axon_flag fields.</li>
                  <li><strong>Outputs</strong> —All per-axon measurements are written to analysis_axon_measurements.xlsx with colour-coded columns and a Summary sheet of aggregate statistics, alongside results.json. Four histogram PNGs (g-ratio, outer radius, inner radius, myelin thickness) are saved with a red dashed mean line each. The annotated image is exported twice — numbered_output.png with full ray and contour overlays, and axons_only.png without the ray clutter, for visual quality-checking.</li>

                </ul>
              </div>
            </div>
          </div>
        )}

        {/* FULL SCREEN MODAL */}
        {selectedImage && (
          <div className="modal-overlay animated fade-in" onClick={resetModal}>
            <button className="modal-close" onClick={resetModal} title="Close">✕</button>
            <div
              className="modal-interactive-area"
              onClick={(e) => e.stopPropagation()}
              onWheel={handleWheel}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <div
                className="image-transform-layer"
                style={{
                  transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                  cursor: isDragging ? 'grabbing' : 'grab'
                }}
              >
                <img ref={imageRef} src={selectedImage.data} alt={selectedImage.name} draggable={false} />
              </div>

              {/* Hover Tooltip */}
              {hoveredAxon && (
                <div
                  className="axon-tooltip glass fade-in"
                  style={{
                    left: mousePos.x + 15,
                    top: mousePos.y + 15,
                  }}
                >
                  <div className="axon-tooltip-header">Axon #{hoveredAxon.axon_id}</div>
                  <div className="axon-tooltip-body">
                    <div><span>Type:</span> {hoveredAxon.axon_type}</div>
                    <div><span>Angle:</span> {hoveredAxon.chosen_ray_angle_deg !== null ? `${hoveredAxon.chosen_ray_angle_deg}°` : 'N/A'}</div>
                    <div><span>Outer Ray:</span> {hoveredAxon.chosen_outer_ray_um !== null ? `${hoveredAxon.chosen_outer_ray_um} µm` : 'N/A'}</div>
                    <div><span>Inner Ray:</span> {hoveredAxon.chosen_inner_ray_um !== null ? `${hoveredAxon.chosen_inner_ray_um} µm` : 'N/A'}</div>
                    <div><span>Thickness:</span> <strong>{hoveredAxon.thickness_um !== null ? `${hoveredAxon.thickness_um} µm` : 'N/A (Blurred)'}</strong></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* CODE EDITOR MODAL */}
        {showEditor && (
          <div className="modal-overlay animated fade-in" onClick={() => setShowEditor(false)}>
            <div className="editor-modal glass scale-in" onClick={(e) => e.stopPropagation()}>
              <div className="editor-header" style={{ padding: "1.5rem", borderBottom: "1px solid var(--glass-border)" }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: "1.5rem" }}>Python Pipeline Code</h2>
                  <p className="editor-subtitle" style={{ opacity: 0.8 }}>View and modify the source code powering the analysis</p>
                </div>
                <button className="btn-run primary" onClick={() => setShowEditor(false)}>Save & Close</button>
              </div>
              <div className="editor-body" style={{ height: "calc(100% - 100px)" }}>
                <Editor
                  height="100%"
                  defaultLanguage="python"
                  theme={isDarkMode ? "vs-dark" : "vs-light"}
                  value={code}
                  onChange={(val) => setCode(val || "")}
                  options={{ minimap: { enabled: false }, fontSize: 14 }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
