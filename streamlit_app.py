# pip install streamlit opencv-python numpy pillow
# Run with: python -m streamlit run thermal_date_app.py

import datetime
import io
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# =====================================================================
# ✏️ EDITABLE CONFIG
# =====================================================================
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf",
]

OLD_TEXT = "2026-09-12 11 30"      # Stamp text as it appears in the photo
DEFAULT_YEAR = 2026
DEFAULT_MONTH = 10
DEFAULT_DAY = 20
DEFAULT_TIME = datetime.time(11, 30)

SEARCH_TOP_PCT = 0.15              # Search the top 15% of the photo
SEARCH_LEFT_PCT = 0.45             # Search from 45% of the width to the right edge
MERGE_KERNEL_W = 13                # Joins letters into one line for detection
ERASE_LOW_FRAC = 0.12              # Lower = catches fainter edges
ERASE_PAD = 3                      # Extra pixels erased around the old stamp

FONT_HEIGHT_DEFAULT = 0            # Font height in px. 0 = automatic (match original)
WIDTH_ADJ_DEFAULT = 100            # Total width vs original (100 = same width)
HEIGHT_ADJ_DEFAULT = 100           # Used only when font height is 0 (100 = same height)

AUTO_COLOR = True                  # True = copy colour from the original stamp
TEXT_COLOR_HEX = "#C8C8C8"         # Used when AUTO_COLOR is False
SOFTNESS = 0.0                     # Edge blur in px (0 = sharp)
BLARE = 1.0                        # Brightness of text (1.0 = none)
# =====================================================================


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_font(size_px):
    size_px = max(6, int(round(size_px)))
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size=size_px)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def font_size_for_digit_height(target_h):
    """Font size whose digit '0' ink height is about target_h."""
    f = load_font(100)
    t, b = f.getbbox("0")[1], f.getbbox("0")[3]
    ratio = (b - t) / 100.0
    if ratio <= 0:
        return 10
    return max(6, int(round(target_h / ratio)))


def find_stamp(img, top_pct, left_pct, merge_w):
    """Find the bright date & time block in the top-right area."""
    H, W = img.shape[:2]
    band_h = max(10, int(H * top_pct))
    x_start = int(W * left_pct)
    strip = cv2.cvtColor(img[:band_h, x_start:], cv2.COLOR_BGR2GRAY)

    tophat = cv2.morphologyEx(strip, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    _, bw = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    merged = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (merge_w, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged)

    best_idx, best_right = None, None
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if h < 5 or h > band_h * 0.6 or w < 30:
            continue
        if w / float(h) < 2.5:
            continue
        if best_idx is None or (x + w) > best_right:
            best_idx, best_right = i, x + w
    if best_idx is None:
        return None

    comp = labels == best_idx
    strict = comp & (bw > 0)
    if not strict.any():
        return None

    ref = max(float(np.percentile(tophat[strict], 95)), 1.0)
    region = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    visible = region & (tophat >= ERASE_LOW_FRAC * ref)

    cols = np.nonzero(strict.any(axis=0))[0]
    cx0, cx1 = int(cols.min()), int(cols.max() + 1)

    vrows = np.nonzero(visible[:, cx0:cx1].any(axis=1))[0]
    vy0, vy1 = int(vrows.min()), int(vrows.max() + 1)

    return {
        "x_start": x_start, "band_h": band_h, "W": W, "H": H,
        "cx0": cx0, "cx1": cx1, "vy0": vy0, "vy1": vy1,
        "tophat": tophat, "ref": ref, "visible": visible,
        "box": (x_start + cx0, vy0, cx1 - cx0, vy1 - vy0),
    }


def sample_color(img, mask):
    """Colour of the brighter core of the original digits."""
    px = img[mask > 0].astype(np.float32)
    if len(px) == 0:
        return (190, 190, 190)
    lum = px @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    core = px[lum >= np.percentile(lum, 60)]
    if len(core) == 0:
        core = px
    return tuple(int(round(c)) for c in np.median(core, axis=0))


def render_line(text, font):
    """Render the whole line as a tight-cropped float mask (0..1)."""
    pad = 8
    canvas = Image.new("L", (int(font.getlength(text)) + 2 * pad,
                             font.size * 2 + 2 * pad), 0)
    ImageDraw.Draw(canvas).text((pad, pad), text, font=font, fill=255)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(arr > 0.05)
    if len(xs) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste_max(layer, patch, x, y):
    """Blend a patch into the layer at (x, y) using the brighter value."""
    H, W = layer.shape
    ph, pw = patch.shape
    xa, ya = max(0, x), max(0, y)
    xb, yb = min(W, x + pw), min(H, y + ph)
    if xb <= xa or yb <= ya:
        return
    layer[ya:yb, xa:xb] = np.maximum(layer[ya:yb, xa:xb],
                                     patch[ya - y:yb - y, xa - x:xb - x])


# ---------------------------------------------------------------------
# Main replacement
# ---------------------------------------------------------------------
def replace_stamp(img, s):
    """Erase the old date & time and draw the new line with the chosen width and height."""
    H, W = img.shape[:2]
    f = find_stamp(img, s["top_pct"], s["left_pct"], s["merge_w"])
    if f is None:
        return img, {"ok": False, "reason": "No stamp found. Adjust the search settings."}

    x_start, band_h = f["x_start"], f["band_h"]
    cx0, cx1 = f["cx0"], f["cx1"]
    vy0, vy1 = f["vy0"], f["vy1"]
    old_w = cx1 - cx0
    old_h = vy1 - vy0
    left_abs = x_start + cx0

    # 1. Erase the whole old stamp
    erase_full = np.zeros((H, W), np.uint8)
    erase_full[:band_h, x_start:] = f["visible"].astype(np.uint8) * 255
    erase_full[max(0, vy0 - ERASE_PAD):min(H, vy1 + ERASE_PAD),
               max(0, left_abs - ERASE_PAD):min(W, x_start + cx1 + ERASE_PAD)] = 255
    erase_full = cv2.dilate(erase_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            iterations=1)
    cleaned = cv2.inpaint(img, erase_full, 3, cv2.INPAINT_TELEA)

    # 2. Colour
    vis_full = np.zeros((H, W), np.uint8)
    vis_full[:band_h, x_start:] = f["visible"].astype(np.uint8) * 255
    color_bgr = sample_color(img, vis_full) if s["auto_color"] else hex_to_bgr(s["color_hex"])

    # 3. Target height: manual font height if set, otherwise match the original
    if s["font_height"] > 0:
        target_h = int(s["font_height"])
    else:
        target_h = max(1, int(round(old_h * s["height_adj"] / 100.0)))

    font = load_font(font_size_for_digit_height(target_h))
    line = render_line(s["new_text"], font)
    if line is None:
        return cleaned, {"ok": False, "reason": "Could not draw the new text."}

    # 4. Scale the line to the target width and height
    tw = max(1, int(round(old_w * s["width_adj"] / 100.0)))
    th = max(1, target_h)
    line = cv2.resize(line, (tw, th), interpolation=cv2.INTER_AREA)
    lh, lw = line.shape

    # 5. Place it at the old left edge, centred vertically on the old stamp
    top = vy0 + (old_h - lh) // 2
    layer = np.zeros((H, W), np.float32)
    paste_max(layer, line, left_abs, top)

    # 6. Softness, brightness and colour
    if s["softness"] > 0:
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=s["softness"])
    alpha = np.clip(layer * s["blare"], 0.0, 1.0)[..., None]
    col = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = (cleaned.astype(np.float32) * (1 - alpha) + col * alpha).clip(0, 255).astype(np.uint8)

    info = {
        "ok": True, "box": f["box"], "color": color_bgr,
        "old_w": old_w, "old_h": old_h, "new_w": lw, "new_h": lh,
        "font_size": font.size, "manual_height": s["font_height"] > 0,
    }
    return out, info


def draw_detected_box(img, box):
    vis = img.copy()
    x, y, w, h = box
    cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Thermal Date Replacer", layout="wide", page_icon="🌡️")
st.title("🌡️ Thermal Date Replacer (Year, Month, Day & Time)")

st.sidebar.header("⚙️ New Date & Time")
year = st.sidebar.number_input("Year", min_value=1000, max_value=2099, value=DEFAULT_YEAR)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=DEFAULT_MONTH)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=DEFAULT_DAY)
new_time = st.sidebar.time_input("Time", value=DEFAULT_TIME)

try:
    datetime.date(int(year), int(month), int(day))
    date_ok = True
except ValueError:
    date_ok = False
    st.sidebar.error(f"{int(year)}-{int(month):02d} has no day {int(day)}. Please change the day.")

new_text = (f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
            f"{new_time.hour:02d} {new_time.minute:02d}")
st.sidebar.success(f"New stamp: `{new_text}`")

st.sidebar.subheader("🔤 Font size & colour")
font_height = st.sidebar.number_input(
    "Font height (px) — 0 = auto", min_value=0, max_value=200, value=FONT_HEIGHT_DEFAULT,
    help="0 = match the original stamp's height. Any value above 0 sets the exact height in pixels.")
width_adj = st.sidebar.slider("Total width vs original (%)", 80, 120, WIDTH_ADJ_DEFAULT,
                              help="100 = same width as the original stamp.")
height_adj = st.sidebar.slider("Height vs original (%) — auto only", 80, 120, HEIGHT_ADJ_DEFAULT,
                               disabled=font_height > 0,
                               help="Used only when Font height is 0.")
auto_color = st.sidebar.checkbox("Copy colour from original", value=AUTO_COLOR)
color_hex = st.sidebar.color_picker("Font colour", value=TEXT_COLOR_HEX, disabled=auto_color)
softness = st.sidebar.slider("Softness (px)", 0.0, 2.0, float(SOFTNESS), 0.1)
blare = st.sidebar.slider("Glow / Blare (×)", 0.5, 2.0, float(BLARE), 0.05)

with st.sidebar.expander("🔧 Detection (advanced)"):
    top_pct = st.slider("Search height (% of photo from top)", 5, 30,
                        int(SEARCH_TOP_PCT * 100)) / 100
    left_pct = st.slider("Search from left (% of photo width)", 0, 80,
                         int(SEARCH_LEFT_PCT * 100)) / 100
    merge_w = st.slider("Word join width (px)", 5, 30, int(MERGE_KERNEL_W))

settings = {
    "new_text": new_text,
    "auto_color": auto_color, "color_hex": color_hex,
    "softness": float(softness), "blare": float(blare),
    "font_height": int(font_height),
    "width_adj": width_adj, "height_adj": height_adj,
    "top_pct": top_pct, "left_pct": left_pct, "merge_w": int(merge_w),
}

files = st.file_uploader("Upload thermal photos (JPG or PNG)",
                         type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not files:
    st.info("👆 Upload one or more photos to begin.")
    st.stop()

if st.button("🚀 Replace Date & Time", type="primary", disabled=not date_ok):
    results = []
    for f in files:
        img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"Could not read {f.name}.")
            continue
        out, info = replace_stamp(img, settings)
        is_jpg = f.name.lower().endswith((".jpg", ".jpeg"))
        if is_jpg:
            _, data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 100])
            ext = "jpg"
        else:
            _, data = cv2.imencode(".png", out)
            ext = "png"
        results.append({"name": f"{f.name.rsplit('.', 1)[0]}.{ext}", "orig": img,
                        "out": out, "data": data.tobytes(), "ext": ext, "info": info})
    st.session_state["results"] = results

results = st.session_state.get("results", [])
if len(results) > 1:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.writestr(r["name"], r["data"])
    st.download_button("📦 Download All (ZIP)", data=zip_buf.getvalue(),
                       file_name="updated_photos.zip", mime="application/zip")

for idx, r in enumerate(results):
    st.subheader(f"📄 {r['name']}")
    info = r["info"]
    if not info["ok"]:
        st.warning(info["reason"])
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original (green box = detected stamp)")
        shown = draw_detected_box(r["orig"], info["box"]) if info["ok"] else r["orig"]
        st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.caption("Processed")
        st.image(cv2.cvtColor(r["out"], cv2.COLOR_BGR2RGB), use_container_width=True)
    if info["ok"]:
        b, g, rr = info["color"]
        mode = "manual" if info["manual_height"] else "auto"
        st.caption(f"Old stamp: {info['old_w']} × {info['old_h']} px · "
                   f"New stamp: {info['new_w']} × {info['new_h']} px · "
                   f"Font height ({mode}): {info['new_h']} px · "
                   f"Font size: {info['font_size']} px · "
                   f"Colour: RGB({rr}, {g}, {b})")
    st.download_button(f"📥 Download {r['name']}", data=r["data"], file_name=r["name"],
                       mime="image/jpeg" if r["ext"] == "jpg" else "image/png",
                       key=f"dl_{idx}")