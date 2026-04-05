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
    Q / ESC     — quit / next image
    S           — save current visualization to viz_results/
    R           — reset zoom and pan to fit screen

    ZOOM:
      Scroll wheel UP    — zoom in  (centered on mouse position)
      Scroll wheel DOWN  — zoom out
      +  /  =            — zoom in
      -                  — zoom out

    PAN  (while zoomed in):
      Left-click + drag  — pan the view
      Arrow keys         — pan left / right / up / down

    COORDINATES:
      Mouse hover        — shows pixel coordinate in the original image space
                           printed in terminal (useful for finding mask coords!)
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
COL_NORMAL_ADD      = (255, 200, 0  )   # teal/cyan – normal additional_mask
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


def draw_overlay_hud(view, product_key, main_count, normal_add_count, bigdot_add_count,
                     zoom, coord_text=""):
    """
    Draw a crisp HUD bar at the BOTTOM of the window view — never overlaps images.
    Always redrawn after zoom/crop so text is pixel-perfect at any zoom level.
    """
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.50
    th     = 1
    pad    = 10
    line_h = 22
    vw     = view.shape[1]

    # Draw solid dark background (bar is already black from np.zeros)
    cv2.line(view, (0, 0), (vw, 0), (60, 60, 60), 1)  # top border line

    # Colour-coded keyword highlights drawn as separate segments on line 0
    y0 = pad + 14

    segments_line0 = [
        (f"Product: {product_key}   |   Main mask: {main_count} zones ", COL_WHITE),
        ("[YELLOW]", (0, 255, 255)),
        (f"   |   Normal additional: {normal_add_count} zones ", COL_WHITE),
        ("[CYAN]", (255, 200, 0)),
        (f"   |   BigDot additional: {bigdot_add_count} zones ", COL_WHITE),
        ("[MAGENTA]", (255, 0, 255)),
        (f"   |   Zoom: {zoom:.2f}x", COL_WHITE),
    ]
    x_cursor = pad
    for text, col in segments_line0:
        cv2.putText(view, text, (x_cursor, y0), font, fs, col, th, cv2.LINE_AA)
        tw, _ = cv2.getTextSize(text, font, fs, th)[0]
        x_cursor += tw

    # Line 1: controls + coordinate
    y1 = pad + line_h + 14
    ctrl_text = "Scroll=zoom   Drag=pan   R=reset   S=save   Q=quit"
    cv2.putText(view, ctrl_text, (pad, y1), font, fs, (140, 140, 140), th, cv2.LINE_AA)
    if coord_text:
        ctrl_w = cv2.getTextSize(ctrl_text, font, fs, th)[0][0]
        cv2.putText(view, f"   |   {coord_text}", (pad + ctrl_w, y1),
                    font, fs, (100, 255, 100), th, cv2.LINE_AA)


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


class ZoomPanViewer:
    """
    Interactive zoom & pan viewer for the three-panel image.

    Zoom  : scroll wheel  or  +/-  keys
    Pan   : left-click drag  or  arrow keys
    Reset : R key
    Coords: hover mouse → live coordinate shown in the HUD overlay (always crisp)
    """

    ZOOM_STEP   = 1.15
    ZOOM_MIN    = 0.05
    ZOOM_MAX    = 40.0
    PAN_STEP    = 40

    def __init__(self, panel, win_name, orig_image_w,
                 product_key="", main_count=0, normal_add_count=0, bigdot_add_count=0):
        self.panel       = panel
        self.win         = win_name
        self.ph, self.pw = panel.shape[:2]
        self.orig_image_w = orig_image_w

        # Legend data (drawn fresh every frame — never pixelates)
        self.product_key      = product_key
        self.main_count       = main_count
        self.normal_add_count = normal_add_count
        self.bigdot_add_count = bigdot_add_count
        self._coord_text      = ""

        self._win_w  = min(1800, self.pw)
        self._img_h  = int(self.ph * self._win_w / self.pw)  # image area height
        self._bar_h  = 56                                      # HUD bar height
        self._win_h  = self._img_h + self._bar_h              # total window height
        self.zoom    = self._win_w / self.pw

        self.pan_x  = 0.0
        self.pan_y  = 0.0

        self._drag       = False
        self._drag_start = (0, 0)
        self._pan_start  = (0.0, 0.0)
        self._mouse_x    = 0
        self._mouse_y    = 0

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _win_to_panel(self, wx, wy):
        """Convert window pixel → panel pixel (full composite)."""
        px = wx / self.zoom + self.pan_x
        py = wy / self.zoom + self.pan_y
        return px, py

    def _panel_to_orig(self, px, py, panel_w_per_col):
        """
        Convert panel pixel → (panel_index 0/1/2, x_in_orig, y_in_orig).
        panel_w_per_col includes the title bar height offset already in Y.
        """
        title_bar_h = 44
        col = int(px // panel_w_per_col)
        col = max(0, min(2, col))
        x_in = int(px - col * panel_w_per_col)
        y_in = int(py - title_bar_h)
        return col, x_in, y_in

    # ── render ────────────────────────────────────────────────────────────────

    def _get_view(self):
        """Crop + resize the panel to produce the current image view (no HUD bar)."""
        vis_w = self._win_w / self.zoom
        vis_h = self._img_h / self.zoom

        self.pan_x = max(0.0, min(self.pan_x, self.pw - vis_w))
        self.pan_y = max(0.0, min(self.pan_y, self.ph - vis_h))

        x1 = int(self.pan_x)
        y1 = int(self.pan_y)
        x2 = min(self.pw, int(self.pan_x + vis_w) + 1)
        y2 = min(self.ph, int(self.pan_y + vis_h) + 1)

        crop = self.panel[y1:y2, x1:x2]
        view = cv2.resize(crop, (self._win_w, self._img_h),
                          interpolation=cv2.INTER_LINEAR)
        return view

    def _refresh(self):
        # Image area (zoomable/pannable)
        img_view = self._get_view()

        # HUD bar (fixed height, always crisp)
        bar = np.zeros((self._bar_h, self._win_w, 3), dtype=np.uint8)
        draw_overlay_hud(bar,
                         self.product_key,
                         self.main_count,
                         self.normal_add_count,
                         self.bigdot_add_count,
                         self.zoom,
                         self._coord_text)

        # Stack: image on top, HUD bar on bottom
        combined = np.vstack([img_view, bar])
        cv2.imshow(self.win, combined)

    # ── zoom helper ───────────────────────────────────────────────────────────

    def _zoom_at(self, factor, anchor_wx, anchor_wy):
        """Zoom by `factor`, keeping panel point under (anchor_wx, anchor_wy) fixed."""
        px_before, py_before = self._win_to_panel(anchor_wx, anchor_wy)
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self.zoom * factor))
        self.zoom  = new_zoom
        # Recompute pan so the same panel pixel stays under the mouse
        self.pan_x = px_before - anchor_wx / self.zoom
        self.pan_y = py_before - anchor_wy / self.zoom
        self._refresh()

    def _reset(self):
        self.zoom  = self._win_w / self.pw
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._refresh()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def mouse_callback(self, event, x, y, flags, param):
        self._mouse_x, self._mouse_y = x, y

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = self.ZOOM_STEP if flags > 0 else 1.0 / self.ZOOM_STEP
            self._zoom_at(factor, x, y)

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
                # Compute original-image pixel under cursor → show in HUD
                px, py = self._win_to_panel(x, y)
                col_w  = self.pw / 3
                col    = int(px // col_w)
                names  = ["ORIGINAL", "AI-SEES", "OVERLAY"]
                x_in   = int(px - col * col_w)
                y_in   = int(py - 44)   # subtract title bar height
                if 0 <= col <= 2 and y_in >= 0:
                    self._coord_text = (
                        f"[{names[min(col,2)]}]  x={x_in}  y={y_in}"
                        f"  ← copy to config.yaml mask"
                    )
                    # Also print to terminal for easy copy-paste
                    print(f"\r  {self._coord_text}    ", end="", flush=True)
                self._refresh()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, save_path=None):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self._win_w, self._win_h)
        cv2.setMouseCallback(self.win, self.mouse_callback)
        self._refresh()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):       # Q / ESC → exit
                break

            elif key == ord('s'):           # S → save full panel
                if save_path:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    cv2.imwrite(save_path, self.panel)
                    print(f"\n[SAVED] {save_path}")

            elif key == ord('r'):           # R → reset
                self._reset()

            elif key in (ord('+'), ord('=')):   # + → zoom in at centre
                self._zoom_at(self.ZOOM_STEP,
                              self._win_w // 2, self._win_h // 2)

            elif key == ord('-'):               # - → zoom out at centre
                self._zoom_at(1.0 / self.ZOOM_STEP,
                              self._win_w // 2, self._win_h // 2)

            elif key == 81 or key == 2424832:   # ← arrow
                self.pan_x -= self.PAN_STEP / self.zoom
                self._refresh()
            elif key == 83 or key == 2555904:   # → arrow
                self.pan_x += self.PAN_STEP / self.zoom
                self._refresh()
            elif key == 82 or key == 2490368:   # ↑ arrow
                self.pan_y -= self.PAN_STEP / self.zoom
                self._refresh()
            elif key == 84 or key == 2621440:   # ↓ arrow
                self.pan_y += self.PAN_STEP / self.zoom
                self._refresh()

        cv2.destroyAllWindows()
        print()   # newline after coordinate printout


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

    main_masks = config.get("mask", [])
    normal_add = config.get("threshold", {}).get("normal", {}).get("additional_mask", [])
    bigdot_add = config.get("threshold", {}).get("very_big_dot", {}).get("additional_mask", [])

    print(f"Main mask zones      : {len(main_masks)}")
    print(f"Normal add_mask zones: {len(normal_add)}")
    print(f"BigDot add_mask zones: {len(bigdot_add)}")
    print(f"\nHover mouse over window → pixel coordinates printed here.")
    print(f"Use these coordinates to add new mask zones in config.yaml.\n")

    # ── Build three panels ────────────────────────────────────────────────────
    masked_ai = apply_main_mask(image, main_masks)
    annotated = draw_mask_overlay(image, config, product_key)
    # NOTE: legend is no longer baked into the panel — drawn live on the view instead

    panel = make_three_panel(image, masked_ai, annotated, product_key, config)

    # ── Save full panel if requested ──────────────────────────────────────────
    save_path = None
    if save:
        os.makedirs("viz_results", exist_ok=True)
        save_path = os.path.join("viz_results", f"viz_{os.path.basename(image_path)}")
        cv2.imwrite(save_path, panel)
        print(f"[SAVED] {save_path}")

    # ── Launch interactive zoom/pan viewer ────────────────────────────────────
    win_name = f"Mask Visualizer — {os.path.basename(image_path)}"
    viewer   = ZoomPanViewer(
        panel, win_name, orig_image_w=w,
        product_key=product_key,
        main_count=len(main_masks),
        normal_add_count=len(normal_add),
        bigdot_add_count=len(bigdot_add),
    )
    viewer.run(save_path=save_path or
               os.path.join("viz_results", f"viz_{os.path.basename(image_path)}"))


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
