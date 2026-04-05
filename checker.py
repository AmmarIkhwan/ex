"""
visualize_masks.py
==================
Visual verification tool for config.yaml mask zones.

Shows 3 views side by side for any image:
  LEFT   — Original image (no masks)
  MIDDLE — Main mask applied (what the AI sees)
  RIGHT  — Additional masks overlaid (post-detection ignore zones)

Usage:
    python visualize_masks.py --image "path/to/image.jpg"
    python visualize_masks.py --image "path/to/image.jpg" --config "config.yaml"
    python visualize_masks.py --folder "test_images"

Controls (when image window is open):
    Q — quit / next image
    S — save current visualization to  viz_results/
"""

import cv2
import yaml
import os
import sys
import argparse
import numpy as np
from glob import glob


# ── colour palette (BGR) ─────────────────────────────────────────────────────
COL_MAIN_MASK       = (0,   0,   0  )   # black fill  – main mask (AI never sees)
COL_MAIN_BORDER     = (0,   255, 255)   # yellow border around main mask zones
COL_NORMAL_ADD      = (0,   165, 255)   # orange – normal additional_mask
COL_BIGDOT_ADD      = (255, 0,   255)   # magenta – very_big_dot additional_mask
COL_LABEL_BG        = (30,  30,  30 )   # dark grey label background
COL_WHITE           = (255, 255, 255)
COL_GREEN           = (0,   255, 0  )


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def match_product(image_path, configs):
    """Return (product_key, product_config) whose key appears in the image path."""
    for key in configs:
        if key in image_path:
            return key, configs[key]
    return None, None


def draw_filled_rect(img, x1, y1, x2, y2, color, alpha=1.0):
    """Draw a filled rectangle, optionally semi-transparent."""
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_border_rect(img, x1, y1, x2, y2, color, thickness=3):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def put_label(img, text, x, y, font_scale=0.55, color=COL_WHITE, bg=COL_LABEL_BG):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - bl - 2), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - bl), font, font_scale, color, thickness, cv2.LINE_AA)


def apply_main_mask(image, mask_coords):
    """Return masked image (black rectangles) — what AI actually sees."""
    masked = image.copy()
    for rect in mask_coords:
        x1, y1, x2, y2 = rect
        cv2.rectangle(masked, (x1, y1), (x2, y2), COL_MAIN_MASK, -1)
    return masked


def draw_mask_overlay(image, config, product_key):
    """
    Draw coloured outlines on the ORIGINAL image showing:
      - Yellow border  = main mask zones  (AI is blind here)
      - Orange fill    = normal additional_mask  (post-AI ignore)
      - Magenta fill   = very_big_dot additional_mask  (post-AI ignore)
    """
    overlay = image.copy()

    # ── Main mask zones (yellow border only so you can see what's underneath) ──
    main_masks = config.get("mask", [])
    for i, rect in enumerate(main_masks):
        x1, y1, x2, y2 = rect
        draw_border_rect(overlay, x1, y1, x2, y2, COL_MAIN_BORDER, thickness=4)
        put_label(overlay, f"MASK-{i+1}", x1, y1 + 20, color=COL_MAIN_BORDER)

    # ── Additional masks ──────────────────────────────────────────────────────
    threshold = config.get("threshold", {})

    # normal additional_mask
    normal_add = threshold.get("normal", {}).get("additional_mask", [])
    for i, rect in enumerate(normal_add):
        x1, y1, x2, y2 = [max(0, v) for v in rect]
        draw_filled_rect(overlay, x1, y1, x2, y2, COL_NORMAL_ADD, alpha=0.35)
        draw_border_rect(overlay, x1, y1, x2, y2, COL_NORMAL_ADD, thickness=2)
        put_label(overlay, f"NRM-ADD-{i+1}", x1, max(y1 + 20, 20), color=COL_NORMAL_ADD)

    # very_big_dot additional_mask
    bigdot_add = threshold.get("very_big_dot", {}).get("additional_mask", [])
    for i, rect in enumerate(bigdot_add):
        x1, y1, x2, y2 = [max(0, v) for v in rect]
        draw_filled_rect(overlay, x1, y1, x2, y2, COL_BIGDOT_ADD, alpha=0.35)
        draw_border_rect(overlay, x1, y1, x2, y2, COL_BIGDOT_ADD, thickness=2)
        put_label(overlay, f"BIG-ADD-{i+1}", x1, max(y1 + 40, 40), color=COL_BIGDOT_ADD)

    return overlay


def draw_legend(img, product_key, main_count, normal_add_count, bigdot_add_count):
    """Draw a legend panel on the top-left of the image."""
    lines = [
        f"Product : {product_key}",
        f"Main masks      : {main_count}  zones  [YELLOW border]",
        f"Normal add mask : {normal_add_count}  zones  [ORANGE fill]",
        f"BigDot add mask : {bigdot_add_count}  zones  [MAGENTA fill]",
        "",
        "AI is BLIND inside yellow borders.",
        "Detections inside orange/magenta are IGNORED.",
    ]
    x, y = 10, 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.55, 1
    line_h = 24
    max_w = max(cv2.getTextSize(l, font, fs, th)[0][0] for l in lines if l) + 20
    box_h = len(lines) * line_h + 14
    cv2.rectangle(img, (x, y), (x + max_w, y + box_h), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x + max_w, y + box_h), COL_GREEN, 1)
    for i, line in enumerate(lines):
        col = COL_GREEN if i == 0 else COL_WHITE
        cv2.putText(img, line, (x + 6, y + 20 + i * line_h),
                    font, fs, col, th, cv2.LINE_AA)


def make_three_panel(original, masked_ai, annotated, product_key, config,
                     display_width=1800):
    """
    Stack three images side by side and resize to fit display.
    LEFT  = original, MIDDLE = what AI sees, RIGHT = annotation overlay
    """
    h, w = original.shape[:2]

    # Add title bar to each panel
    def titled(img, title, color):
        bar_h = 44
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        cv2.putText(bar, title, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return np.vstack([bar, img])

    left   = titled(original.copy(),   "ORIGINAL",            COL_WHITE)
    middle = titled(masked_ai.copy(),  "WHAT AI SEES",        (0, 200, 255))
    right  = titled(annotated.copy(),  "MASK OVERLAY",        COL_GREEN)

    combined = np.hstack([left, middle, right])

    # Resize to fit screen
    scale = display_width / combined.shape[1]
    if scale < 1.0:
        new_h = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (display_width, new_h))

    return combined


def process_image(image_path, configs, save=False):
    # ── Load ──────────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image    : {os.path.basename(image_path)}")
    print(f"Size     : {w} x {h} px")

    # ── Match product ─────────────────────────────────────────────────────────
    product_key, config = match_product(image_path, configs)
    if config is None:
        print(f"[WARN] No matching product config found for: {os.path.basename(image_path)}")
        print(f"       Available keys: {list(configs.keys())}")
        return

    print(f"Product  : {product_key}")

    main_masks    = config.get("mask", [])
    normal_add    = config.get("threshold", {}).get("normal", {}).get("additional_mask", [])
    bigdot_add    = config.get("threshold", {}).get("very_big_dot", {}).get("additional_mask", [])

    print(f"Main mask zones      : {len(main_masks)}")
    print(f"Normal add_mask zones: {len(normal_add)}")
    print(f"BigDot add_mask zones: {len(bigdot_add)}")

    # ── Build three panels ────────────────────────────────────────────────────
    masked_ai  = apply_main_mask(image, main_masks)
    annotated  = draw_mask_overlay(image, config, product_key)
    draw_legend(annotated, product_key, len(main_masks), len(normal_add), len(bigdot_add))

    panel = make_three_panel(image, masked_ai, annotated, product_key, config)

    # ── Save if requested ─────────────────────────────────────────────────────
    if save:
        os.makedirs("viz_results", exist_ok=True)
        out_path = os.path.join("viz_results", f"viz_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, panel)
        print(f"[SAVED] {out_path}")

    # ── Show ──────────────────────────────────────────────────────────────────
    win = "Mask Visualizer  |  Q=next/quit  S=save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    ph, pw = panel.shape[:2]
    cv2.resizeWindow(win, pw, ph)
    cv2.imshow(win, panel)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            os.makedirs("viz_results", exist_ok=True)
            out_path = os.path.join("viz_results", f"viz_{os.path.basename(image_path)}")
            cv2.imwrite(out_path, panel)
            print(f"[SAVED] {out_path}")

    cv2.destroyAllWindows()


def main(args):
    configs = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Products found: {list(configs.keys())}")

    # ── Single image mode ─────────────────────────────────────────────────────
    if args.image:
        process_image(args.image, configs, save=args.save)

    # ── Folder mode ───────────────────────────────────────────────────────────
    elif args.folder:
        image_list = glob(os.path.join(args.folder, "*.jpg")) + \
                     glob(os.path.join(args.folder, "*.png"))
        if not image_list:
            print(f"[ERROR] No jpg/png images found in: {args.folder}")
            sys.exit(1)
        print(f"Found {len(image_list)} images in {args.folder}")
        for img_path in sorted(image_list):
            process_image(img_path, configs, save=args.save)

    else:
        print("[ERROR] Provide --image or --folder")
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser(
        description="Visualize config.yaml mask zones on inspection images."
    )
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to a single image file")
    parser.add_argument("--folder", type=str, default=None,
                        help="Path to a folder of images (processes all jpg/png)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--save",   action="store_true",
                        help="Auto-save every visualization to viz_results/")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
