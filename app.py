import warnings
import os
import streamlit as st
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import tempfile
import time
from io import BytesIO

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Photoquadrat Analysis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM STYLING ==========
st.markdown("""
<style>
    /* Overall theme */
    :root {
        --primary-color: #2d6a4f;
        --accent-color: #40916c;
        --light-accent: #95d5b2;
        --warning-color: #d62828;
        --text-color: #1b4332;
    }
    
    * {
        box-sizing: border-box;
    }
    
    /* Landing page hero section */
    .hero-container {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 100%);
        padding: 40px 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(45, 106, 79, 0.2);
    }
    
    .hero-container h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0 0 20px 0;
        letter-spacing: -1px;
    }
    
    .hero-container p {
        font-size: 1.1em;
        margin: 0;
        opacity: 0.95;
        line-height: 1.6;
    }
    
    /* STYLE 4: Gradient Border Feature Cards - 2x2 Grid Layout */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 25.5px;
        margin-top: 10px;
    }
    
    .feature-card {
        position: relative;
        background: white;
        padding: 25px 20px;
        border-radius: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        padding: 2px;
        background: linear-gradient(135deg, #40916c, #52b788, #74c69d, #95d5b2);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        transition: padding 0.4s ease;
    }

    .feature-card:hover::before {
        padding: 3px;
        background: linear-gradient(135deg, #2d6a4f, #40916c, #52b788, #74c69d);
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(64, 145, 108, 0.2);
    }

    .feature-icon {
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #40916c, #74c69d);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 15px;
        transition: all 0.4s ease;
        flex-shrink: 0;
    }

    .feature-card:hover .feature-icon {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        transform: scale(1.1) rotate(-5deg);
    }

    .feature-card h3 {
        color: #2d6a4f;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 8px;
        font-weight: 600;
        line-height: 1.3;
    }

    .feature-card p {
        color: #52796f;
        line-height: 1.5;
        font-size: 0.85rem;
        margin: 0;
        flex-grow: 1;
    }
    
    /* Settings card styling */
    .settings-card {
        background: white;
        padding: 10px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 2px solid #e8f5e9;
        margin-top: 20px;
    }
    
    .settings-card h4 {
        color: #2d6a4f;
        margin-bottom: 15px;
        font-size: 1.9rem;
    }
    
    /* Toggle switch styling */
    .stCheckbox {
        margin: 10px 0;
    }
    
    /* Info box styling to match height */
    .info-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        border-left: 4px solid #40916c;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .info-box strong {
        color: #2d6a4f;
        font-size: 1.05rem;
    }
    
    /* Upload section styling */
    .upload-section {
        text-align: center;
        padding: 20px 20px;
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        border-radius: 9px;
        margin: 30px 0;
    }
    
    .upload-section h2 {
        color: #2d6a4f;
        margin-bottom: 10px;
        font-size: 1.8rem;
    }
    
    .upload-section p {
        color: #52796f;
        font-size: 1.1rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 5px;
    }
    
    .badge-ready {
        background-color: #d4f1d4;
        color: #2d6a4f;
        border: 1px solid #40916c;
    }
    
    .badge-processing {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
    }
    
    .badge-complete {
        background-color: #d1e7dd;
        color: #0f5132;
        border: 1px solid #198754;
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin: 20px 0;
        padding: 15px 10px;
        background: #f8fffe;
        border-radius: 8px;
        flex-wrap: wrap;
    }
    
    .step-item {
        text-align: center;
        flex: 1;
        min-width: 70px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .step-item:hover {
        transform: scale(1.05);
    }
    
    .step-number {
        display: inline-block;
        width: 40px;
        height: 40px;
        background: #40916c;
        color: white;
        border-radius: 50%;
        line-height: 40px;
        font-weight: 700;
        margin-bottom: 8px;
        font-size: 1em;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .step-item:hover .step-number {
        background: #52b788;
    }
    
    .step-item.active .step-number {
        background: #2d6a4f;
        box-shadow: 0 0 0 7px #95d5b2;
    }
    
    .step-label {
        color: #2d6a4f;
        font-weight: 600;
        font-size: 0.85em;
        display: block;
        word-break: break-word;
        cursor: pointer;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f5f1;
        border-left: 4px solid #40916c;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.95em;
    }
            
    .info-box2 {
        color: #1b4332;
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f4 100%);
        border-left: 4px solid #40916c;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        line-height: 1.8;
        font-size: 0.95rem;
    }
    
    .info-box2 strong {
        color: #2d6a4f;
        font-size: 1em;
    }
            
    .info-box3 strong {
        color: #2d6a4f;
    }
            
    .info-box3 {
        color: #2d6a4f;
        font-size: 0.95rem;
    }       
            
    /* Results section */
    .results-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 4px solid #40916c;
        margin-top: 20px;
    }
    
    /* Links */
    a {
        color: #40916c;
        text-decoration: none;
        font-weight: 600;
    }
    
    a:hover {
        color: #2d6a4f;
        text-decoration: underline;
    }
    
    /* Mobile responsive - Tablets */
    @media (max-width: 768px) {
        .hero-container {
            padding: 30px 15px;
            margin-bottom: 20px;
        }
        
        .hero-container h1 {
            font-size: 2em;
        }
        
        .hero-container p {
            font-size: 0.95em;
        }
        
        .features-grid {
            grid-template-columns: 1fr;
            gap: 15px;
        }
        
        .feature-card {
            padding: 15px;
        }
        
        .feature-card h3 {
            font-size: 1em;
        }
        
        .feature-card p {
            font-size: 0.8rem;
        }
        
        .step-indicator {
            gap: 5px;
            padding: 12px 8px;
        }
        
        .step-item {
            min-width: 60px;
        }
        
        .step-number {
            width: 36px;
            height: 36px;
            line-height: 36px;
            font-size: 0.9em;
        }
        
        .step-label {
            font-size: 0.75em;
        }
        
        .upload-section {
            margin: 20px 0;
            padding: 15px 15px;
        }
        
        .upload-section h2 {
            font-size: 1.5em;
        }
        
        .upload-section p {
            font-size: 0.95em;
        }
        
        .info-box {
            padding: 12px;
            margin: 8px 0;
            font-size: 0.9em;
        }
        
        .info-box2 {
            padding: 15px;
            font-size: 0.9rem;
        }
        
        button {
            min-height: 44px;
            font-size: 0.95em;
        }
    }
    
    /* Mobile responsive - Small phones */
    @media (max-width: 480px) {
        .hero-container h1 {
            font-size: 1.5em;
        }
        
        .hero-container p {
            font-size: 0.85em;
        }
        
        .features-grid {
            gap: 10px;
        }
        
        .feature-card {
            padding: 12px;
        }
        
        .feature-card h3 {
            font-size: 0.95em;
            margin-bottom: 5px;
        }
        
        .feature-card p {
            font-size: 0.75rem;
        }
        
        .feature-icon {
            width: 40px;
            height: 40px;
            font-size: 18px;
        }
        
        .step-indicator {
            gap: 3px;
            padding: 10px 5px;
        }
        
        .step-item {
            min-width: 50px;
        }
        
        .step-number {
            width: 32px;
            height: 32px;
            line-height: 32px;
            font-size: 0.75em;
        }
        
        .step-label {
            font-size: 0.65em;
        }
        
        .upload-section h2 {
            font-size: 1.3em;
        }
        
        .info-box2 {
            padding: 12px;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "image_array" not in st.session_state:
    st.session_state.image_array = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "use_grid" not in st.session_state:
    st.session_state.use_grid = True
if "grid_shape" not in st.session_state:
    st.session_state.grid_shape = (4, 4)
if "frame_size_cm" not in st.session_state:
    st.session_state.frame_size_cm = 50


# ========== CONFIGURATION ==========
CONFIDENCE_THRESHOLD = 0.10
FRAME_CONFIDENCE = 0.5

# ========== UTILITY FUNCTIONS ==========
@st.cache_resource
def load_models():
    """Load YOLO models - searches in models folder relative to app"""
    try:
        # Get the directory where this script is located
        app_dir = Path(__file__).parent
        models_dir = app_dir / "models"
        
        # Model filenames to search for
        frame_model_names = ["new_frame_detector.pt", "frame_detector.pt", "frame_detection_model.pt"]
        seg_model_names = ["final_detector.pt", "species_segmentation.pt", "custom_photoquad.pt"]
        
        # Find frame model
        frame_model_path = None
        for model_name in frame_model_names:
            candidate = models_dir / model_name
            if candidate.exists():
                frame_model_path = candidate
                break
        
        # Find segmentation model
        seg_model_path = None
        for model_name in seg_model_names:
            candidate = models_dir / model_name
            if candidate.exists():
                seg_model_path = candidate
                break
        
        if frame_model_path is None or seg_model_path is None:
            missing = []
            if frame_model_path is None:
                missing.append("Frame detection model (new_frame_detector.pt, frame_detector.pt, or frame_detection_model.pt)")
            if seg_model_path is None:
                missing.append("Species segmentation model (final_detector.pt, species_segmentation.pt, or custom_photoquad.pt)")
            
            st.error(f"Could not find models:\n" + "\n".join(f"- {m}" for m in missing) + 
                    f"\n\nPlease place your model files in: {models_dir}")
            return None, None
        
        st.info(f"Loading models from: {models_dir}")
        frame_model = YOLO(str(frame_model_path))
        seg_model = YOLO(str(seg_model_path))
        return frame_model, seg_model
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def detect_frame(image, frame_model, conf_threshold=FRAME_CONFIDENCE):
    """Detect the photoquadrat frame in the image."""
    results = frame_model(image, conf=conf_threshold)
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        h, w = image.shape[:2]
        return (0, 0, w, h), None, 0.0
    
    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)
    
    frame_box = boxes[best_idx].xyxy[0].cpu().numpy()
    frame_coords = tuple(map(int, frame_box))
    detection_confidence = float(confidences[best_idx])
    
    frame_polygon = None
    if results[0].masks is not None and len(results[0].masks) > best_idx:
        mask = results[0].masks[best_idx]
        mask_np = mask.data.cpu().numpy()
        
        if len(mask_np.shape) == 3:
            mask_np = mask_np[0]
        
        if mask_np.shape != image.shape[:2]:
            mask_resized = cv.resize(mask_np, (image.shape[1], image.shape[0]))
        else:
            mask_resized = mask_np
        
        mask_binary = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours:
            frame_polygon = max(contours, key=cv.contourArea)
    
    return frame_coords, frame_polygon, detection_confidence


def crop_to_frame(image, frame_coords, frame_polygon=None, padding=0):
    """Crop image to the detected frame region."""
    x1, y1, x2, y2 = frame_coords
    
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    cropped_img = image[y1:y2, x1:x2].copy()
    
    frame_width_px = x2 - x1
    frame_height_px = y2 - y1
    avg_frame_size_px = (frame_width_px + frame_height_px) / 2
    frame_size_cm = st.session_state.frame_size_cm
    scale_factor = avg_frame_size_px / frame_size_cm
    
    crop_offset = (x1, y1)
    
    if frame_polygon is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [frame_polygon], -1, 255, -1)
        mask_cropped = mask[y1:y2, x1:x2]
        cropped_img = cv.bitwise_and(cropped_img, cropped_img, mask=mask_cropped)
    
    return cropped_img, scale_factor, crop_offset


def get_class_colors(num_classes):
    """Generate distinct colors for each class."""
    colors = {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 255, 255),
        4: (255, 0, 255), 5: (255, 255, 0), 6: (255, 128, 0), 7: (128, 0, 255),
        8: (0, 255, 128), 9: (255, 128, 128), 10: (128, 255, 128), 11: (128, 128, 255),
        12: (255, 255, 128), 13: (255, 128, 255), 14: (128, 255, 255),
    }
    return {i: colors.get(i, colors[i % 15]) for i in range(num_classes)}


def calculate_adaptive_font_scale(cell_width, cell_height, num_lines=1):
    """Calculate adaptive font scale based on cell dimensions."""
    min_dimension = min(cell_width, cell_height)
    
    if min_dimension < 100:
        font_scale = 0.5
        thickness = 1
    elif min_dimension < 200:
        font_scale = 0.7
        thickness = 1
    elif min_dimension < 300:
        font_scale = 0.9
        thickness = 2
    elif min_dimension < 500:
        font_scale = 1.2
        thickness = 2
    else:
        font_scale = 1.5
        thickness = 2
    
    if num_lines > 1:
        font_scale *= (1.0 / (1 + (num_lines - 1) * 0.2))
    
    return font_scale, thickness


def grid_lines(image, grid_shape, color=(0, 255, 0), thickness=1):
    """Draw grid on image."""
    img_height, img_width = image.shape[:2]
    rows, cols = grid_shape
    row_height = img_height // rows
    col_width = img_width // cols

    for i in range(1, rows):
        y = i * row_height
        cv.line(image, (0, y), (img_width, y), color, thickness)

    for j in range(1, cols):
        x = j * col_width
        cv.line(image, (x, 0), (x, img_height), color, thickness)

    return image


def calculate_segmentation_coverage(image, masks, results, use_grid=True, grid_shape=(4, 4)):
    """Calculate segmentation coverage with optional grid analysis."""
    img_height, img_width = image.shape[:2]
    class_names = results[0].names
    class_masks = {}
    
    for class_id in range(len(class_names)):
        class_masks[class_id] = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if results[0].masks is not None:
        for mask_idx in range(len(results[0].masks.data)):
            mask = results[0].masks.data[mask_idx]
            class_id = int(results[0].boxes.cls[mask_idx].item())
            
            mask_np = mask.cpu().numpy().astype(np.float32)
            
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            
            if mask_np.shape != (img_height, img_width):
                mask_resized = cv.resize(mask_np, (img_width, img_height), interpolation=cv.INTER_LINEAR)
            else:
                mask_resized = mask_np
            
            mask_resized = (mask_resized * 255).astype(np.uint8)
            _, mask_resized = cv.threshold(mask_resized, 127, 255, cv.THRESH_BINARY)
            
            class_masks[class_id] = cv.bitwise_or(class_masks[class_id], mask_resized)
    
    if use_grid:
        return _calculate_grid_coverage(image, grid_shape, class_masks, class_names, results)
    else:
        return _calculate_total_coverage(image, class_masks, class_names, results)


def _calculate_grid_coverage(image, grid_shape, class_masks, class_names, results):
    """Calculate coverage per grid cell."""
    img_height, img_width = image.shape[:2]
    rows, cols = grid_shape
    row_height = img_height // rows
    col_width = img_width // cols
    
    grid_areas = {}
    grid_detections = {}
    
    for i in range(rows):
        for j in range(cols):
            grid_areas[(i, j)] = {}
            grid_detections[(i, j)] = []
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        mask = class_masks[class_id]
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * row_height
                y_end = (i + 1) * row_height
                x_start = j * col_width
                x_end = (j + 1) * col_width
                
                grid_mask = mask[y_start:y_end, x_start:x_end]
                segmented_pixels = np.sum(grid_mask > 127)
                grid_cell_area = (y_end - y_start) * (x_end - x_start)
                percentage_coverage = (segmented_pixels / grid_cell_area) * 100
                
                if percentage_coverage > 0:
                    grid_areas[(i, j)][class_name] = percentage_coverage
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            grid_row = center_y // row_height
            grid_col = center_x // col_width
            
            if grid_row < rows and grid_col < cols:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = class_names[class_id]
                
                grid_detections[(grid_row, grid_col)].append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
    
    return grid_areas, grid_detections, class_masks


def _calculate_total_coverage(image, class_masks, class_names, results):
    """Calculate total coverage without grid."""
    img_height, img_width = image.shape[:2]
    total_area = img_height * img_width
    
    total_areas = {}
    total_detections = []
    
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        mask = class_masks[class_id]
        
        segmented_pixels = np.sum(mask > 127)
        percentage_coverage = (segmented_pixels / total_area) * 100
        
        if percentage_coverage > 0:
            total_areas[class_name] = percentage_coverage
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = class_names[class_id]
            
            total_detections.append({
                'class': class_name,
                'confidence': confidence,
                'box': (int(x1), int(y1), int(x2), int(y2))
            })
    
    return total_areas, total_detections, class_masks


def visualize_results(image, grid_areas, grid_detections, class_masks, class_colors, 
                      use_grid=True, grid_shape=(4, 4)):
    """Visualize segmentation with optional grid overlay and adaptive text sizing."""
    viz_img = image.copy()
    overlay = viz_img.copy()
    
    for class_id, mask in class_masks.items():
        color = class_colors.get(class_id, (0, 255, 0))
        overlay[mask > 127] = color
    
    alpha = 0.6
    viz_img = cv.addWeighted(viz_img, 1 - alpha, overlay, alpha, 0)
    
    if use_grid:
        viz_img = grid_lines(viz_img, grid_shape, color=(255, 255, 255), thickness=2)
        
        img_height, img_width = image.shape[:2]
        rows, cols = grid_shape
        row_height = img_height // rows
        col_width = img_width // cols
        
        font = cv.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 0)
        bg_color = (0, 0, 0)
        
        for i in range(rows):
            for j in range(cols):
                y_start = i * row_height
                x_start = j * col_width
                
                classes_in_cell = grid_areas[(i, j)]
                
                if classes_in_cell:
                    text_lines = [f"{cls}: {cov:.1f}%" for cls, cov in classes_in_cell.items()]
                else:
                    text_lines = ["Bare"]
                
                num_lines = len(text_lines)
                font_scale, thickness = calculate_adaptive_font_scale(col_width, row_height, num_lines)
                
                test_size = cv.getTextSize("Test", font, font_scale, thickness)[0]
                line_height = int(test_size[1] * 1.8)
                
                available_height = row_height - 10
                max_lines = max(1, available_height // line_height)
                text_lines = text_lines[:max_lines]
                
                y_offset = y_start + line_height
                
                for text in text_lines:
                    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
                    text_width = text_size[0]
                    text_height = text_size[1]
                    
                    text_x = x_start + max(3, (col_width - text_width) // 2)
                    text_x = min(text_x, x_start + col_width - text_width - 3)
                    text_y = y_offset
                    
                    padding = 3
                    cv.rectangle(viz_img,
                                (text_x - padding, text_y - text_height - padding),
                                (text_x + text_width + padding, text_y + padding),
                                bg_color, -1)
                    
                    cv.putText(viz_img, text, (text_x, text_y),
                              font, font_scale, text_color, thickness, cv.LINE_AA)
                    
                    y_offset += line_height
    
    return viz_img


# ========== PAGE: LANDING ==========
def page_landing():
    """Landing page with welcome and feature overview."""
    
    st.markdown("""
    <div class="hero-container">
        <h1>Photoquadrat Analysis</h1>
        <p>A.I image analysis for species coverage quantification</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item active">
            <div class="step-number">1</div>
            <div class="step-label">Welcome</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-label">Upload</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-label">Analyze</div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-label">Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pass
    with col2:
        if st.button("Go to Upload", use_container_width=True, key="nav_2"):
            st.session_state.page = "upload"
            st.rerun()
    with col3:
        pass
    with col4:
        pass
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.divider()
        st.markdown("""
        #### A tool for analyzing photoquadrat images. 
        #### It uses advanced computer vision and deep learning to:
        - **Detect** photoquadrat frames with high precision
        - **Segment** species and vegetation coverage automatically
        - **Quantify** coverage percentages across grid cells
        - **Report** findings in structured formats
        """)
        
        st.markdown("""
        <div class="info-box info-box2">
            <strong>Tip:</strong> Best results with well-lit, straight-on quadrat photos
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        st.markdown("### Analysis Settings")
            
        use_grid = st.checkbox(
            "Enable Grid Analysis",
            value=st.session_state.use_grid,
            key="grid_toggle",
            help="Toggle between grid-based analysis and whole-frame analysis"
        )
        st.session_state.use_grid = use_grid
        
        if use_grid:
            st.markdown("**Grid Configuration:**")
            col_a, col_b = st.columns(2)
            with col_a:
                rows = st.number_input("Rows", min_value=2, max_value=10, value=st.session_state.grid_shape[0], key="grid_rows")
            with col_b:
                cols = st.number_input("Columns", min_value=2, max_value=10, value=st.session_state.grid_shape[1], key="grid_cols")
            
            st.session_state.grid_shape = (rows, cols)
            st.info(f"Current grid: {rows}×{cols} cells ({rows * cols} total cells)")
        else:
            st.info("Whole-frame analysis mode: Coverage will be calculated for the entire image")
        
    
    with col2:
        st.markdown("### Key Features")
        st.markdown("""
        <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🔲</div>
            <h3>Frame Detection</h3>
            <p>Automatically locates and extracts the photoquadrat frame from your image</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🍃</div>
            <h3>Species Segmentation</h3>
            <p>Identifies and segments different species or plant classes with deep learning</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Coverage Analysis</h3>
            <p>Calculates coverage percentages per grid cell for detailed spatial analysis</p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <h3>Report Generation</h3>
            <p>Exports analysis results as Excel reports with detailed metrics</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Frame Configuration:**")
        
        frame_size = st.number_input(
            "Frame Size (cm)",
            min_value=10,
            max_value=200,
            value=st.session_state.frame_size_cm,
            step=5,
            key="frame_size_input",
            help="Physical size of your photoquadrat frame in centimeters (e.g., 50cm for a 50×50cm frame)"
        )

        st.session_state.frame_size_cm = frame_size

        st.info(f"Frame dimensions: **{frame_size}×{frame_size} cm**")

    st.divider()

    st.markdown("""
    <div class="upload-section">
        <h2>Ready to analyze your quadrats?</h2>
        <p>Click the button below to get started!</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Go to Image Upload", use_container_width=True, key="start_analysis"):
            st.session_state.page = "upload"
            st.rerun()

    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Resources")
        st.markdown("""
        - [GitHub Repository](https://github.com/DolapoSalim/photoquadrats_analysis)
        - [Documentation](https://github.com/DolapoSalim/EcoQuad#ecoquad)
        - [Issues & Discussions](https://github.com/DolapoSalim/EcoQuad/issues)
        """)
    
    with col2:
        st.markdown("### Current Configuration")
        with st.expander("View Settings"):
            st.info(f"""
            **Analysis Settings:**
            - Grid Analysis: {'Enabled' if st.session_state.use_grid else 'Disabled'}
            - Grid Size: {st.session_state.grid_shape[0]}×{st.session_state.grid_shape[1]}
            - Frame Size: {st.session_state.frame_size_cm}×{st.session_state.frame_size_cm} cm
            - Confidence Threshold: {CONFIDENCE_THRESHOLD}
            - Frame Confidence: {FRAME_CONFIDENCE}
            """)


# ========== PAGE: UPLOAD ==========
def page_upload():
    """Upload image page."""
    
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-label">Welcome</div>
        </div>
        <div class="step-item active">
            <div class="step-number">2</div>
            <div class="step-label">Upload</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-label">Analyze</div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-label">Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Go to Welcome", use_container_width=True, key="nav_1_upload"):
            st.session_state.page = "landing"
            st.rerun()
    with col2:
        pass
    with col3:
        pass
    with col4:
        pass
    
    st.header("Upload Your Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box info-box3">
            <strong>Supported Formats:</strong> JPG, JPEG, PNG, TIFF<br>
            <strong>Max File Size:</strong> 200 MB<br>
            <strong>Recommended:</strong> High resolution, well-lit, straight-on photos
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your quadrat image here or click to select",
            type=['jpg', 'jpeg', 'png', 'tiff'],
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            st.session_state.image_array = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            
            st.success("Image uploaded successfully!")
            
            st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)
            
            st.markdown(f"""
            <div class="info-box info-box3">
                <strong>File Info:</strong><br>
                • Name: {uploaded_file.name}<br>
                • Size: {uploaded_file.size / 1024 / 1024:.2f} MB<br>
                • Dimensions: {st.session_state.image_array.shape[1]}×{st.session_state.image_array.shape[0]} px
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Next Steps")
        st.markdown("""
        Your image has been uploaded and is ready for analysis.
        
        Click **Start Analysis** to begin processing.
        """)
        
        st.markdown("### Analysis Mode")
        if st.session_state.use_grid:
            st.success(f"Grid Analysis: {st.session_state.grid_shape[0]}×{st.session_state.grid_shape[1]}")
        else:
            st.info("Whole-frame Analysis")
        
        st.markdown(f"**Frame Size:** {st.session_state.frame_size_cm}×{st.session_state.frame_size_cm} cm")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Home", use_container_width=True, key="back_from_upload"):
            st.session_state.page = "landing"
            st.rerun()
    
    with col3:
        if st.session_state.image_array is not None:
            if st.button("Start Analysis →", use_container_width=True, key="start_analysis_btn"):
                st.session_state.page = "analyze"
                st.rerun()
        else:
            st.button("Start Analysis →", use_container_width=True, disabled=True)


# ========== PAGE: ANALYZE ==========
def page_analyze():
    """Analysis page."""
    
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-label">Welcome</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-label">Upload</div>
        </div>
        <div class="step-item active">
            <div class="step-number">3</div>
            <div class="step-label">Analyze</div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-label">Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Go to Welcome", use_container_width=True, key="nav_1_analyze"):
            st.session_state.page = "landing"
            st.rerun()
    with col2:
        if st.button("Go to Upload", use_container_width=True, key="nav_2_analyze"):
            st.session_state.page = "upload"
            st.rerun()
    with col3:
        pass
    with col4:
        pass
    
    st.header("Analyzing Your Image")
    
    if st.session_state.use_grid:
        st.info(f"Running grid analysis with {st.session_state.grid_shape[0]}×{st.session_state.grid_shape[1]} cells")
    else:
        st.info("Running whole-frame analysis")
    
    with st.spinner("Loading AI models..."):
        frame_model, seg_model = load_models()
    
    if frame_model is None or seg_model is None:
        st.error("Could not load the analysis models. Please check the models folder and try again.")
        
        if st.button("Back to Upload"):
            st.session_state.page = "upload"
            st.rerun()
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.info("Step 1/4: Detecting photoquadrat frame...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        frame_coords, frame_polygon, frame_conf = detect_frame(
            st.session_state.image_array, frame_model, FRAME_CONFIDENCE
        )
        
        status_text.info("Step 2/4: Cropping to frame region...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        cropped_img, scale_factor, crop_offset = crop_to_frame(
            st.session_state.image_array, frame_coords, frame_polygon
        )
        
        status_text.info("Step 3/4: Running species segmentation...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        seg_results = seg_model(cropped_img, conf=CONFIDENCE_THRESHOLD)
        
        status_text.info("Step 4/4: Calculating coverage metrics...")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        class_names = seg_results[0].names
        num_classes = len(class_names)
        class_colors = get_class_colors(num_classes)
        
        grid_areas, grid_detections, class_masks = calculate_segmentation_coverage(
            cropped_img, seg_results[0].masks, seg_results,
            use_grid=st.session_state.use_grid, 
            grid_shape=st.session_state.grid_shape
        )
        
        viz_img = visualize_results(
            cropped_img, grid_areas, grid_detections, class_masks, class_colors,
            use_grid=st.session_state.use_grid, 
            grid_shape=st.session_state.grid_shape
        )
        
        progress_bar.progress(100)
        status_text.success("Analysis complete!")
        
        st.session_state.analysis_results = {
            'grid_areas': grid_areas,
            'grid_detections': grid_detections,
            'class_names': class_names,
            'scale_factor': scale_factor,
            'frame_conf': frame_conf,
            'viz_img': viz_img,
            'cropped_img': cropped_img,
            'use_grid': st.session_state.use_grid,
            'grid_shape': st.session_state.grid_shape
        }
        
        st.session_state.analysis_complete = True
        
        time.sleep(1)
        st.success("Analysis complete! Proceed to view results.")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.info("Please check your image and try again.")
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Back to Upload", use_container_width=True, key="back_from_analyze"):
            st.session_state.page = "upload"
            st.rerun()
    
    with col3:
        if st.session_state.analysis_complete:
            if st.button("View Results", use_container_width=True, key="view_results"):
                st.session_state.page = "results"
                st.rerun()
        else:
            st.button("View Results", use_container_width=True, disabled=True)


# ========== PAGE: RESULTS ==========
def page_results():
    """Results and export page."""
    
    if not st.session_state.analysis_complete:
        st.error("No analysis results available. Please complete the analysis first.")
        if st.button("← Go Back"):
            st.session_state.page = "landing"
            st.rerun()
        return
    
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-label">Welcome</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-label">Upload</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-label">Analyze</div>
        </div>
        <div class="step-item active">
            <div class="step-number">4</div>
            <div class="step-label">Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Go to Welcome", use_container_width=True, key="nav_1_results"):
            st.session_state.page = "landing"
            st.rerun()
    with col2:
        if st.button("Go to Upload", use_container_width=True, key="nav_2_results"):
            st.session_state.page = "upload"
            st.rerun()
    with col3:
        if st.button("Go to Analyze", use_container_width=True, key="nav_3_results"):
            st.session_state.page = "analyze"
            st.rerun()
    with col4:
        pass
    
    st.header("Analysis Results")
    
    results = st.session_state.analysis_results
    
    st.markdown("### Segmentation Visualization")
    st.image(cv.cvtColor(results['viz_img'], cv.COLOR_BGR2RGB), use_container_width=True)
    
    st.divider()
    
    st.markdown("### Coverage Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Frame Detection Confidence",
            f"{results['frame_conf']:.1%}",
            delta="High confidence" if results['frame_conf'] > 0.7 else "Check frame detection"
        )
    
    with col2:
        if results.get('use_grid', True):
            st.metric("Grid Size", f"{results['grid_shape'][0]}×{results['grid_shape'][1]}")
        else:
            st.metric("Analysis Mode", "Whole-frame")
    
    with col3:
        st.metric(
            "Scale Factor",
            f"{results['scale_factor']:.2f} px/cm"
        )
    
    st.divider()
    
    if results.get('use_grid', True):
        st.markdown("### Grid Cell Coverage")
        
        grid_data = []
        for (row, col), coverage_dict in sorted(results['grid_areas'].items()):
            row_data = {'Grid (Row, Col)': f"({row}, {col})"}
            
            if coverage_dict:
                for class_name, coverage in coverage_dict.items():
                    row_data[class_name] = f"{coverage:.1f}%"
                row_data['Total'] = f"{sum(coverage_dict.values()):.1f}%"
            else:
                row_data['Total'] = "Bare"
            
            grid_data.append(row_data)
        
        grid_df = pd.DataFrame(grid_data)
        st.dataframe(grid_df, use_container_width=True)
        
    else:
        st.markdown("### Total Coverage")
        coverage_data = {
            'Species/Class': list(results['grid_areas'].keys()),
            'Coverage (%)': [f"{v:.2f}" for v in results['grid_areas'].values()]
        }
        coverage_df = pd.DataFrame(coverage_data)
        st.dataframe(coverage_df, use_container_width=True)
    
    st.divider()
    
    st.markdown("### Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excel_buffer = BytesIO()
        
        if results.get('use_grid', True):
            export_data = []
            rows, cols = results['grid_shape']
            for i in range(rows):
                for j in range(cols):
                    row_data = {
                        'Grid Row': i,
                        'Grid Column': j,
                    }
                    coverage_dict = results['grid_areas'][(i, j)]
                    for class_name in results['class_names'].values():
                        row_data[f'{class_name} (%)'] = coverage_dict.get(class_name, 0)
                    export_data.append(row_data)
            
            export_df = pd.DataFrame(export_data)
        else:
            export_df = pd.DataFrame({
                'Species': results['grid_areas'].keys(),
                'Coverage (%)': results['grid_areas'].values()
            })
        
        export_df.to_excel(excel_buffer, index=False, sheet_name='Coverage Analysis')
        excel_buffer.seek(0)
        
        st.download_button(
            label="Download Excel Report",
            data=excel_buffer,
            file_name="photoquadrat_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        _, viz_image_bytes = cv.imencode('.png', results['viz_img'])
        
        st.download_button(
            label="Download Visualization",
            data=viz_image_bytes.tobytes(),
            file_name="segmentation_visualization.png",
            mime="image/png",
            use_container_width=True
        )
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Analysis", use_container_width=True, key="back_from_results"):
            st.session_state.page = "analyze"
            st.rerun()
    
    with col3:
        if st.button("New Analysis", use_container_width=True, key="new_analysis"):
            st.session_state.page = "landing"
            st.session_state.analysis_complete = False
            st.session_state.uploaded_file = None
            st.session_state.image_array = None
            st.rerun()


# ========== MAIN APP ==========
def main():
    """Main app router."""
    
    if st.session_state.page == "landing":
        page_landing()
    elif st.session_state.page == "upload":
        page_upload()
    elif st.session_state.page == "analyze":
        page_analyze()
    elif st.session_state.page == "results":
        page_results()


if __name__ == "__main__":
    main()