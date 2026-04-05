import sys
import os
import torch
import cv2
import yaml
import pandas as pd
from glob import glob
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

log_date = datetime.now().strftime("%d_%m-%Y-%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{log_folder}/[FLuxSatellite_TEST]_{log_date}.log"),
        logging.StreamHandler(),   # also print to terminal so you can watch live
    ]
)

# ── Model ─────────────────────────────────────────────────────────────────────
def init_model(conf):
    sys.path.append('.\\yolov5')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load(
        './yolov5', 'custom',
        path="STIM_fluxSatellite_v51_r1.pt",
        source='local'
    ).to(device)
    model.conf = conf
    model.eval()
    if device.type == 'cuda':
        model.half()
        torch.backends.cudnn.benchmark = True
    logging.info(f"Model loaded on {device}")
    return model


# ── Image utilities ───────────────────────────────────────────────────────────
def masking(image, config):
    masked_image = image.copy()
    for rect in config:
        x1, y1, x2, y2 = rect
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return image, masked_image


def check_light(image, dark_threshold=55):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_gray = np.percentile(gray, 50)
    return mean_gray < dark_threshold


def process_result(image, result):
    result_image = image.copy()
    detections = result.pandas().xyxy[0]
    for _, detection in detections.iterrows():
        x1 = int(detection['xmin'])
        y1 = int(detection['ymin'])
        x2 = int(detection['xmax'])
        y2 = int(detection['ymax'])
        confidence = detection['confidence']
        class_name = detection['name']

        box_color  = (0, 0, 255)    # Red (BGR)
        text_color = (255, 255, 255) # White

        cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, thickness=1, lineType=cv2.LINE_AA)

        label = f"{class_name} {confidence:.2f}"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 1
        font_thick  = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        cv2.rectangle(result_image,
                      (x1, y1 - text_h - baseline - 3),
                      (x1 + text_w, y1),
                      box_color, -1)
        cv2.putText(result_image, label,
                    (x1, y1 - baseline - 2),
                    font, font_scale, text_color, font_thick)
    return result_image


def is_meet_threshold(yolo_df_result, config):
    defect_meet = []
    df_result = yolo_df_result.copy()
    required_columns = ['xcenter', 'ycenter', 'width']
    df_result[required_columns] = df_result[required_columns].astype(int)

    for defect_type, defect_config in config.items():
        size_mask = (
            (df_result['width']  > defect_config['size']) |
            (df_result['height'] > defect_config['size'])
        )
        df_matched = df_result[size_mask].copy()

        if 'additional_mask' in defect_config and defect_config['additional_mask']:
            mask_coords  = np.array(defect_config['additional_mask'])
            outside_mask = np.ones(len(df_matched), dtype=bool)
            for x1, y1, x2, y2 in mask_coords:
                inside_rect = (
                    (df_matched['xcenter'] >= x1) &
                    (df_matched['xcenter'] <= x2) &
                    (df_matched['ycenter'] >= y1) &
                    (df_matched['ycenter'] <= y2)
                )
                outside_mask &= ~inside_rect
            df_matched = df_matched[outside_mask]

        if len(df_matched) >= defect_config['count']:
            defect_meet.append(defect_type)

    return defect_meet


# ── Core detect ───────────────────────────────────────────────────────────────
def detect(img_path, model, config):
    image = cv2.imread(img_path)
    if image is None:
        logging.warning(f"Could not read image: {img_path}")
        return None, None

    image, masked_image = masking(image, config=config['mask'])

    if check_light(image):
        logging.info(f"Image too dark, skipped: {os.path.basename(img_path)}")
        return None, None

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            result = model(masked_image, size=1280)

    yolo_df_result = result.pandas().xywh[0]

    if not yolo_df_result.empty:
        logging.info(f"Detections found in: {os.path.basename(img_path)} — checking thresholds.")
        defect_type = is_meet_threshold(yolo_df_result, config['threshold'])
        if len(defect_type) == 0:
            logging.info(f"Detections did not meet threshold: {os.path.basename(img_path)}")
            return None, None
        result_image = process_result(image=image, result=result)
        return result_image, defect_type
    else:
        logging.info(f"No detections: {os.path.basename(img_path)}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del masked_image, result
    return None, None


# ── Summary CSV — same schema as main_KM ─────────────────────────────────────
def summary_record(master_data, file_name, output_folder):
    """
    Mirrors main_KM.py's summary_record().
    Since test images are local (no network path to parse for ww/link/lot/vid),
    those fields are extracted from the filename and folder structure where possible,
    or filled with 'N/A' for fields that only exist on the factory network.

    CSV columns: lot, link, WW, vid, defect_type, result_address
    """
    result_df = pd.DataFrame(columns=['lot', 'link', 'WW', 'vid', 'defect_type', 'result_address'])

    for image_path, defect_type, result_address in master_data:
        basename = os.path.basename(image_path)
        name_no_ext = os.path.splitext(basename)[0]

        # Try to parse lot and vid from the filename (format: ..._<vid>.jpg)
        # Falls back gracefully when the naming convention doesn't apply locally.
        parts = name_no_ext.split('_')
        vid = parts[-1] if len(parts) > 1 else name_no_ext
        lot = os.path.basename(os.path.dirname(image_path))   # parent folder name

        result_df.loc[len(result_df)] = [
            lot,
            'N/A',              # link — factory network concept, N/A locally
            'N/A',              # WW   — work-week folder, N/A locally
            vid,
            ': '.join(defect_type),
            result_address
        ]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_name  = f"{timestamp}_{file_name}"
    out_path  = os.path.join(output_folder, out_name)
    result_df.to_csv(out_path, index=False)
    logging.info(f"Summary CSV saved: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    src_folder      = args.src_folder
    output_folder   = args.output_folder
    result_file_name = args.result_file_name
    conf            = args.conf

    # Create output subfolders
    image_out = os.path.join(output_folder, 'image')
    os.makedirs(image_out, exist_ok=True)

    # Processed-lots record — same mechanism as main_KM so already-run images are skipped
    record_file = os.path.join(output_folder, args.lot_record_file)
    if os.path.exists(record_file):
        record_df = pd.read_csv(record_file)
    else:
        record_df = pd.DataFrame(columns=['processed_lots'])

    # Init model and config
    model = init_model(conf=conf)
    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    # Collect all .jpg images under src_folder (recursively, like main_KM's lot scan)
    image_list = glob(os.path.join(src_folder, '**', '*.jpg'), recursive=True) + \
                 glob(os.path.join(src_folder, '*.jpg'))
    image_list = list(set(image_list))   # deduplicate

    if len(image_list) == 0:
        logging.info(f"No .jpg images found in: {src_folder}")
        return

    logging.info(f"Found {len(image_list)} image(s) to process.")

    master_data = []

    with tqdm(total=len(image_list), desc="Processing") as pbar:
        for image_path in image_list:
            try:
                basename = os.path.basename(image_path)

                # Skip already-processed images (mirrors main_KM lot-record logic)
                if basename in record_df['processed_lots'].values:
                    logging.info(f"Already processed, skipping: {basename}")
                    pbar.update(1)
                    continue

                # Match image to a product config
                matched = [cfg for cfg in configs if cfg in image_path]
                if len(matched) == 0:
                    logging.info(f"No matching config for image: {basename}. Skipping.")
                    pbar.update(1)
                    continue

                logging.info(f"Scan image: {basename}")
                result_image, defect_type = detect(
                    img_path=image_path,
                    model=model,
                    config=configs[matched[0]]
                )

                if defect_type is not None:
                    result_address = os.path.join(image_out, basename)
                    cv2.imwrite(result_address, result_image)
                    master_data.append((image_path, defect_type, result_address))
                    logging.info(f"Defect saved: {basename} → {defect_type}")

                # Mark as processed
                record_df.loc[len(record_df)] = basename

            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")

            pbar.update(1)

    # Write outputs — same as main_KM
    summary_record(master_data, result_file_name, output_folder)
    record_df.to_csv(record_file, index=False)
    logging.info(f"Done. {len(master_data)} defect image(s) saved to: {image_out}")


# ── CLI args ──────────────────────────────────────────────────────────────────
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='FLuxSatellite — Local Test Runner', add_help=add_help)
    parser.add_argument('--src-folder',       default='test_images',          help='Folder containing test .jpg images')
    parser.add_argument('--output-folder',    default='test_results',         help='Folder for annotated images + CSV output')
    parser.add_argument('--result_file_name', default='Flux_satellite.csv',   help='Filename for the summary CSV')
    parser.add_argument('--lot_record_file',  default='processed_lots.csv',   help='CSV to track already-processed images')
    parser.add_argument('--conf',             default=0.3, type=float,        help='YOLOv5 confidence threshold')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
