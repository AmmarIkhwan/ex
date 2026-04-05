"""
visualize_masks.py
==================
Visual verification tool for config.yaml mask zones.

Shows ONE panel at a time at FULL resolution (no downscaling = no pixelation).
Switch between panels with 1 / 2 / 3 keys.

  Panel 1 — ORIGINAL      : raw image, no changes
  Panel 2 — AI SEES       : black = areas AI is completely blind to (main mask)
  Panel 3 — MASK OVERLAY  : yellow border = main mask zones
                             cyan fill     = normal additional_mask (post-AI ignore)
                             magenta fill  = very_big_dot additional_mask

Save (S key) → creates a subfolder in viz_results/ with all 3 full-res images.

Usage:
    python visualize_masks.py --image "path/to/image.jpg"
    python visualize_masks.py --folder "test_images"
    python visualize_masks.py --folder "test_images" --save

Controls:
    1 / 2 / 3   — switch panel (Original / AI Sees / Overlay)
    Scroll       — zoom in/out centered on mouse
    + / -        — zoom in/out centered on screen
    Drag         — pan
    Arrow keys   — pan
    R            — reset zoom and pan
    S            — save all 3 full-resolution images to viz_results/<imagename>/
    Q / ESC      — quit / next image
    Hover mouse  — shows pixel coordinate in HUD bar and terminal
"""

import cv2
import yaml
import os
import sys
import argparse
import numpy as np
from glob import glob


# ── colour palette (BGR) ─────────────────────────────────────────────────────
COL_MAIN_MASK   = (0,   0,   0  )   # black fill     – main mask
COL_MAIN_BORDER = (0,   255, 255)   # yellow border  – main mask zones
COL_NORMAL_ADD  = (255, 200, 0  )   # cyan           – normal additional_mask
COL_BIGDOT_ADD  = (255, 0,   255)   # magenta        – very_big_dot additional_mask
COL_WHITE       = (255, 255, 255)
COL_GREEN       = (0,   255, 0  )

PANEL_NAMES  = ["ORIGINAL", "AI SEES  (black = AI blind zone)", "MASK OVERLAY"]
PANEL_COLORS = [COL_WHITE,  (0, 200, 255),                       COL_GREEN]


# ── config helpers ────────────────────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def match_product(image_path, configs):
    for key in configs:
        if key in image_path:
            return key, configs[key]
    return None, None


# ── panel builders (all at FULL resolution) ───────────────────────────────────

def build_original(image):
    return image.copy()


def build_ai_view(image, mask_coords):
    """Paint main mask zones solid black — exactly what the AI model receives."""
    out = image.copy()
    for rect in mask_coords:
        x1, y1, x2, y2 = rect
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_MAIN_MASK, -1)
    return out


def build_overlay(image, config):
    """
    Draw zone borders/fills on the original image at full resolution.
    Font size is scaled to image width so labels are always readable when zoomed.
    """
    out   = image.copy()
    ih, iw = out.shape[:2]

    # Scale thickness and font to full image size
    fs    = max(1.0, iw / 2200)          # font scale
    lth   = max(2,   int(iw / 1500))     # line/text thickness
    bth   = max(5,   int(iw / 600))      # border thickness

    def border(x1, y1, x2, y2, col):
        cv2.rectangle(out, (x1, y1), (x2, y2), col, bth, cv2.LINE_AA)

    def fill(x1, y1, x2, y2, col, alpha=0.28):
        ov = out.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(ov, alpha, out, 1 - alpha, 0, out)

    def label(text, x, y, col):
        (tw, th), bl = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fs, lth)
        bx1 = max(0, x)
        by1 = max(0, y - th - bl - 6)
        bx2 = min(iw, x + tw + 8)
        by2 = min(ih, y + 6)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
        cv2.putText(out, text, (bx1 + 4, by2 - bl - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, col, lth, cv2.LINE_AA)

    # Main mask — yellow border only (so you can see the image underneath)
    for i, rect in enumerate(config.get("mask", [])):
        x1, y1, x2, y2 = rect
        border(x1, y1, x2, y2, COL_MAIN_BORDER)
        label(f"MASK-{i+1}", x1, y1 + int(fs * 40), COL_MAIN_BORDER)

    threshold = config.get("threshold", {})

    # Normal additional_mask — cyan
    for i, rect in enumerate(
            threshold.get("normal", {}).get("additional_mask", [])):
        x1, y1, x2, y2 = [max(0, v) for v in rect]
        fill(x1, y1, x2, y2, COL_NORMAL_ADD)
        border(x1, y1, x2, y2, COL_NORMAL_ADD)
        label(f"NRM-{i+1}", x1, max(y1 + int(fs * 80), int(fs * 40)),
              COL_NORMAL_ADD)

    # BigDot additional_mask — magenta
    for i, rect in enumerate(
            threshold.get("very_big_dot", {}).get("additional_mask", [])):
        x1, y1, x2, y2 = [max(0, v) for v in rect]
        fill(x1, y1, x2, y2, COL_BIGDOT_ADD)
        border(x1, y1, x2, y2, COL_BIGDOT_ADD)
        label(f"BIG-{i+1}", x1, max(y1 + int(fs * 120), int(fs * 60)),
              COL_BIGDOT_ADD)

    return out


# ── HUD bar — drawn post-zoom, always crisp ───────────────────────────────────

def draw_hud(bar, panel_idx, product_key, main_n, normal_n, bigdot_n,
             zoom, coord_text=""):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 0.50
    th   = 1
    pad  = 10
    lh   = 22
    vw   = bar.shape[1]

    cv2.line(bar, (0, 0), (vw, 0), (70, 70, 70), 1)

    # Line 0 — panel tabs + product + zoom
    y0 = pad + 14
    x  = pad
    for i, name in enumerate(["1:ORIGINAL", "2:AI SEES", "3:OVERLAY"]):
        col = (0, 220, 255) if i == panel_idx else (70, 70, 70)
        txt = f"[{name}]  "
        cv2.putText(bar, txt, (x, y0), font, fs, col, th, cv2.LINE_AA)
        x += cv2.getTextSize(txt, font, fs, th)[0][0]

    info = f"   Product: {product_key}   Zoom: {zoom:.2f}x"
    cv2.putText(bar, info, (x, y0), font, fs, COL_WHITE, th, cv2.LINE_AA)

    # Line 1 — legend + controls + coord
    y1 = pad + lh + 14
    x  = pad
    seg = [
        (f"Main:{main_n} ",        COL_WHITE),
        ("[YELLOW] ",               COL_MAIN_BORDER),
        (f"Normal:{normal_n} ",     COL_WHITE),
        ("[CYAN] ",                 COL_NORMAL_ADD),
        (f"BigDot:{bigdot_n} ",     COL_WHITE),
        ("[MAGENTA]",               COL_BIGDOT_ADD),
        ("   |   Scroll=zoom  Drag=pan  R=reset  S=save  Q=quit",
         (120, 120, 120)),
    ]
    if coord_text:
        seg.append((f"   |   {coord_text}", (80, 255, 80)))

    for txt, col in seg:
        cv2.putText(bar, txt, (x, y1), font, fs, col, th, cv2.LINE_AA)
        x += cv2.getTextSize(txt, font, fs, th)[0][0]


# ── Viewer ────────────────────────────────────────────────────────────────────

class ZoomPanViewer:
    ZOOM_STEP = 1.15
    ZOOM_MIN  = 0.02
    ZOOM_MAX  = 40.0
    PAN_STEP  = 50
    BAR_H     = 56
    WIN_W     = 1600

    def __init__(self, panels, win_name, image_name,
                 product_key, main_n, normal_n, bigdot_n):
        self.panels      = panels       # [original, ai_view, overlay] full-res
        self.win         = win_name
        self.image_name  = image_name
        self.product_key = product_key
        self.main_n      = main_n
        self.normal_n    = normal_n
        self.bigdot_n    = bigdot_n
        self.panel_idx   = 0

        self.ph, self.pw = panels[0].shape[:2]
        self._win_w = self.WIN_W
        self._img_h = int(self.ph * self.WIN_W / self.pw)
        self._win_h = self._img_h + self.BAR_H

        self.zoom   = self.WIN_W / self.pw
        self.pan_x  = 0.0
        self.pan_y  = 0.0

        self._drag       = False
        self._drag_start = (0, 0)
        self._pan_start  = (0.0, 0.0)
        self._coord_text = ""

    def _win_to_img(self, wx, wy):
        return wx / self.zoom + self.pan_x, wy / self.zoom + self.pan_y

    def _get_img_view(self):
        vis_w = self._win_w / self.zoom
        vis_h = self._img_h / self.zoom
        self.pan_x = max(0.0, min(self.pan_x, self.pw - vis_w))
        self.pan_y = max(0.0, min(self.pan_y, self.ph - vis_h))
        x1 = int(self.pan_x);  y1 = int(self.pan_y)
        x2 = min(self.pw, x1 + int(vis_w) + 2)
        y2 = min(self.ph, y1 + int(vis_h) + 2)
        crop = self.panels[self.panel_idx][y1:y2, x1:x2]
        return cv2.resize(crop, (self._win_w, self._img_h),
                          interpolation=cv2.INTER_LINEAR)

    def _refresh(self):
        img_view = self._get_img_view()

        # Panel title — drawn post-zoom (always crisp)
        title = f"[{self.panel_idx+1}] {PANEL_NAMES[self.panel_idx]}"
        col   = PANEL_COLORS[self.panel_idx]
        for off, c in [((14, 36), (0,0,0)), ((12, 34), col)]:   # shadow + text
            cv2.putText(img_view, title, off,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, c, 2, cv2.LINE_AA)

        bar = np.zeros((self.BAR_H, self._win_w, 3), dtype=np.uint8)
        draw_hud(bar, self.panel_idx, self.product_key,
                 self.main_n, self.normal_n, self.bigdot_n,
                 self.zoom, self._coord_text)

        cv2.imshow(self.win, np.vstack([img_view, bar]))

    def _zoom_at(self, factor, wx, wy):
        px, py    = self._win_to_img(wx, wy)
        self.zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self.zoom * factor))
        self.pan_x = px - wx / self.zoom
        self.pan_y = py - wy / self.zoom
        self._refresh()

    def _reset(self):
        self.zoom  = self.WIN_W / self.pw
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._refresh()

    def _save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(self.image_name)[0]
        suffixes = ["1_original", "2_ai_sees", "3_overlay"]
        for panel, suffix in zip(self.panels, suffixes):
            path = os.path.join(out_dir, f"{base}_{suffix}.jpg")
            cv2.imwrite(path, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  [SAVED] {path}")
        print(f"  Folder: {out_dir}")

    def mouse_callback(self, event, x, y, flags, param):
        if y > self._img_h:   # ignore clicks on HUD bar
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            f = self.ZOOM_STEP if flags > 0 else 1.0 / self.ZOOM_STEP
            self._zoom_at(f, x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._drag       = True
            self._drag_start = (x, y)
            self._pan_start  = (self.pan_x, self.pan_y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag:
                dx = (x - self._drag_start[0]) / self.zoom
                dy = (y - self._drag_start[1]) / self.zoom
                self.pan_x = self._pan_start[0] - dx
                self.pan_y = self._pan_start[1] - dy
                self._refresh()
            else:
                ix, iy = self._win_to_img(x, y)
                ix, iy = int(ix), int(iy)
                if 0 <= ix < self.pw and 0 <= iy < self.ph:
                    self._coord_text = f"x={ix}  y={iy}  ← copy to config.yaml"
                    print(f"\r  x={ix:5d}  y={iy:5d}    ", end="", flush=True)
                self._refresh()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False

    def run(self, out_dir=None):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self._win_w, self._win_h)
        cv2.setMouseCallback(self.win, self.mouse_callback)
        self._refresh()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key in (ord('1'), ord('2'), ord('3')):
                self.panel_idx = int(chr(key)) - 1
                self._refresh()
            elif key == ord('s'):
                d = out_dir or os.path.join(
                    "viz_results", os.path.splitext(self.image_name)[0])
                self._save(d)
            elif key == ord('r'):
                self._reset()
            elif key in (ord('+'), ord('=')):
                self._zoom_at(self.ZOOM_STEP, self._win_w//2, self._img_h//2)
            elif key == ord('-'):
                self._zoom_at(1/self.ZOOM_STEP, self._win_w//2, self._img_h//2)
            elif key in (81, 2424832):
                self.pan_x -= self.PAN_STEP / self.zoom; self._refresh()
            elif key in (83, 2555904):
                self.pan_x += self.PAN_STEP / self.zoom; self._refresh()
            elif key in (82, 2490368):
                self.pan_y -= self.PAN_STEP / self.zoom; self._refresh()
            elif key in (84, 2621440):
                self.pan_y += self.PAN_STEP / self.zoom; self._refresh()

        cv2.destroyAllWindows()
        print()


# ── process one image ─────────────────────────────────────────────────────────

def process_image(image_path, configs, save=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read: {image_path}"); return

    h, w = image.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image   : {os.path.basename(image_path)}")
    print(f"Size    : {w} x {h} px")

    product_key, config = match_product(image_path, configs)
    if config is None:
        print(f"[WARN] No matching config. Keys: {list(configs.keys())}"); return

    print(f"Product : {product_key}")

    main_masks = config.get("mask", [])
    normal_add = config.get("threshold", {}).get("normal", {}).get("additional_mask", [])
    bigdot_add = config.get("threshold", {}).get("very_big_dot", {}).get("additional_mask", [])

    print(f"Main mask      : {len(main_masks)} zones")
    print(f"Normal add     : {len(normal_add)} zones")
    print(f"BigDot add     : {len(bigdot_add)} zones")
    print(f"Keys: 1=Original  2=AI Sees  3=Overlay  S=save  Q=quit\n")

    # Build all 3 panels at FULL resolution — no quality loss
    panels = [
        build_original(image),
        build_ai_view(image, main_masks),
        build_overlay(image, config),
    ]

    img_name = os.path.basename(image_path)
    out_dir  = os.path.join("viz_results", os.path.splitext(img_name)[0])

    if save:
        os.makedirs(out_dir, exist_ok=True)
        base     = os.path.splitext(img_name)[0]
        suffixes = ["1_original", "2_ai_sees", "3_overlay"]
        for panel, suffix in zip(panels, suffixes):
            path = os.path.join(out_dir, f"{base}_{suffix}.jpg")
            cv2.imwrite(path, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  [SAVED] {path}")

    win_name = f"Mask Visualizer  [{img_name}]"
    viewer   = ZoomPanViewer(
        panels, win_name, img_name,
        product_key=product_key,
        main_n=len(main_masks),
        normal_n=len(normal_add),
        bigdot_n=len(bigdot_add),
    )
    viewer.run(out_dir=out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args):
    configs = load_config(args.config)
    print(f"Config  : {args.config}")
    print(f"Products: {list(configs.keys())}")

    if args.image:
        process_image(args.image, configs, save=args.save)
    elif args.folder:
        image_list = sorted(
            glob(os.path.join(args.folder, "*.jpg")) +
            glob(os.path.join(args.folder, "*.png"))
        )
        if not image_list:
            print(f"[ERROR] No images in: {args.folder}"); sys.exit(1)
        print(f"Found {len(image_list)} images")
        for img_path in image_list:
            process_image(img_path, configs, save=args.save)
    else:
        print("[ERROR] Provide --image or --folder"); sys.exit(1)


def get_args():
    p = argparse.ArgumentParser(description="Visualize config.yaml mask zones.")
    p.add_argument("--image",  type=str, default=None)
    p.add_argument("--folder", type=str, default=None)
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--save",   action="store_true",
                   help="Auto-save 3 full-res images per image to viz_results/")
    return p.parse_args()


if __name__ == "__main__":
    main(get_args())
