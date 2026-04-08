import math
import base64
from pathlib import Path
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Last Mile AI Agent Demo", layout="wide")

ASSETS = Path("assets")
VIDEO_PATH = ASSETS / "demo_delivery_clip.mp4"
REF_LOCATION = ASSETS / "reference_location.jpg"
REF_RECIPIENT = ASSETS / "reference_recipient.jpg"
GENUINE_POD = ASSETS / "genuine_pod.jpg"
NON_GENUINE_POD = ASSETS / "non_genuine_pod.jpg"

FACE_CASCADE = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))

ADDRESS_BOOK = {
    "Pune Customer Drop": {"lat": 18.5204, "lon": 73.8567, "eta": 11.8, "window": "11:30-12:30", "customer": "MRM Carriers Pvt. Ltd."},
    "Ahmedabad Consignee": {"lat": 23.0225, "lon": 72.5714, "eta": 9.2, "window": "09:00-10:00", "customer": "ARS Divine Ltd"},
    "New Delhi Hub": {"lat": 28.6139, "lon": 77.2090, "eta": 16.0, "window": "15:30-17:00", "customer": "North Zone Crossdock"},
}


def load_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bytes_to_rgb(uploaded):
    if uploaded is None:
        return None
    arr = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": float(fps), "frames": frame_count, "duration_s": float(duration), "w": w, "h": h}


def frame_at(video_path: Path, t_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_no = max(0, int(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def variance_of_laplacian(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 2]))


def edge_density(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.mean(edges > 0))


def hist_similarity(a_rgb, b_rgb):
    a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2HSV)
    b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2HSV)
    hist_a = cv2.calcHist([a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        return False
    return True


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def plot_geo(expected_lat, expected_lon, observed_lat, observed_lon, threshold_km):
    fig, ax = plt.subplots(figsize=(5, 4))
    lat_rad = math.radians(expected_lat)
    deg_lat_per_km = 1.0 / 110.574
    deg_lon_per_km = 1.0 / (111.320 * max(math.cos(lat_rad), 1e-6))
    theta = np.linspace(0, 2 * math.pi, 200)
    circle_lat = expected_lat + threshold_km * deg_lat_per_km * np.sin(theta)
    circle_lon = expected_lon + threshold_km * deg_lon_per_km * np.cos(theta)
    ax.plot(circle_lon, circle_lat)
    ax.scatter([expected_lon], [expected_lat], s=90, marker="o", label="Expected drop")
    ax.scatter([observed_lon], [observed_lat], s=90, marker="x", label="Photo / app GPS")
    ax.legend(loc="best")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("POD location validation")
    return fig


def analyze_video(video_path: Path):
    info = get_video_info(video_path)
    if not info:
        return None, None
    duration = max(1.0, info["duration_s"])
    sample_ts = np.linspace(0, duration * 0.95, 8)
    rows = []
    prev_gray = None
    cabin_motion_vals = []
    for ts in sample_ts:
        frame = frame_at(video_path, float(ts))
        if frame is None:
            continue
        sharp = variance_of_laplacian(frame)
        bright = brightness_score(frame)
        edges = edge_density(frame)
        face_seen = detect_face(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            motion = 0.0
        else:
            flow = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(flow))
        prev_gray = gray
        cabin_motion_vals.append(motion)
        rows.append({
            "t_sec": round(float(ts), 1),
            "sharpness": round(sharp, 1),
            "brightness": round(bright, 1),
            "edge_density": round(edges, 3),
            "face_visible": int(face_seen),
            "motion": round(motion, 1),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return info, df
    on_time_prob = 0.92
    pod_capture_readiness = float(np.clip((df["sharpness"].mean() / 250.0) * 50 + (df["brightness"].mean() / 255.0) * 50, 0, 100))
    compliance = float(np.clip(65 + df["face_visible"].mean() * 20 + (1 - min(df["motion"].mean() / 30.0, 1)) * 15, 0, 100))
    df.attrs["summary"] = {
        "stop_detected": True,
        "on_time_prob": on_time_prob,
        "pod_capture_readiness": pod_capture_readiness,
        "route_compliance": compliance,
        "cabin_motion_avg": float(df["motion"].mean()),
    }
    return info, df


@st.cache_data(show_spinner=False)
def analyze_video_cached(video_path_str: str):
    return analyze_video(Path(video_path_str))


@st.cache_data(show_spinner=False)
def video_data_url(video_path_str: str):
    video_bytes = Path(video_path_str).read_bytes()
    encoded = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def pod_quality_score(img_rgb):
    sharp = variance_of_laplacian(img_rgb)
    bright = brightness_score(img_rgb)
    edges = edge_density(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    aspect = w / max(h, 1)
    doc_shape = 1.0 if 0.6 <= aspect <= 1.8 else 0.6
    base = min(sharp / 450.0, 1.0) * 40 + min(bright / 180.0, 1.0) * 25 + min(edges / 0.12, 1.0) * 25 + doc_shape * 10
    return float(np.clip(base, 0, 100)), {"sharpness": sharp, "brightness": bright, "edge_density": edges, "aspect_ratio": aspect}


def classify_pod(candidate_rgb, genuine_rgb, non_genuine_rgb):
    sim_good = hist_similarity(candidate_rgb, genuine_rgb)
    sim_bad = hist_similarity(candidate_rgb, non_genuine_rgb)
    score, metrics = pod_quality_score(candidate_rgb)
    margin = sim_good - sim_bad
    authenticity = float(np.clip(50 + margin * 50 + (score - 60) * 0.35, 0, 100))
    verdict = "Likely genuine" if authenticity >= 60 else "Needs review / likely non-genuine"
    return authenticity, verdict, sim_good, sim_bad, metrics


def validate_epod(order_id, recipient, timestamp_text, lat, lon, address_name, pod_authenticity, file_present):
    required_ok = all([str(order_id).strip(), str(recipient).strip(), str(timestamp_text).strip(), file_present])
    target = ADDRESS_BOOK[address_name]
    distance_km = haversine_km(target["lat"], target["lon"], lat, lon)
    geo_ok = distance_km <= 0.35
    ts_ok = True
    try:
        datetime.fromisoformat(timestamp_text)
    except Exception:
        ts_ok = False
    epod_ok = required_ok and geo_ok and ts_ok and pod_authenticity >= 55
    reasons = []
    if not required_ok:
        reasons.append("missing mandatory fields or POD file")
    if not geo_ok:
        reasons.append("photo/app GPS outside allowed delivery geofence")
    if not ts_ok:
        reasons.append("timestamp is not valid ISO datetime")
    if pod_authenticity < 55:
        reasons.append("POD image authenticity below threshold")
    if not reasons:
        reasons.append("all ePOD checks passed")
    return epod_ok, distance_km, reasons


def build_summary(video_df, epod_ok=None, pod_verdict=None, distance_km=None):
    parts = []
    if video_df is not None and not video_df.empty:
        s = video_df.attrs.get("summary", {})
        parts.append(f"Route compliance proxy is {s.get('route_compliance', 0):.0f}% and POD capture readiness is {s.get('pod_capture_readiness', 0):.0f}%.")
        if s.get("cabin_motion_avg", 0) > 18:
            parts.append("Driver movement near stop points is elevated, so the agent should request a steadier POD capture.")
        else:
            parts.append("Cabin motion is stable enough for clean proof capture.")
    if pod_verdict:
        parts.append(f"POD document check: {pod_verdict}.")
    if distance_km is not None:
        parts.append(f"Drop-location variance is {distance_km*1000:.0f} meters from expected destination.")
    if epod_ok is not None:
        parts.append("System ePOD decision: ACCEPT." if epod_ok else "System ePOD decision: HOLD FOR REVIEW.")
    return " ".join(parts)


def answer_question(question, video_df, epod_ok, pod_verdict, distance_km):
    q = question.lower().strip()
    s = video_df.attrs.get("summary", {}) if video_df is not None and not video_df.empty else {}
    if "how is the delivery" in q or "summary" in q or "last mile" in q:
        return build_summary(video_df, epod_ok, pod_verdict, distance_km)
    if "pod" in q and ("genuine" in q or "authentic" in q or "fake" in q):
        return f"POD validation result: {pod_verdict or 'not run yet'}. The agent checks image quality, compares against genuine/non-genuine examples, and blocks weak uploads."
    if "location" in q or "geofence" in q:
        if distance_km is None:
            return "Location validation has not been run yet. Once GPS is entered, the app computes the distance from the expected drop and decides pass or review."
        return f"Photo/app GPS is {distance_km*1000:.0f} meters from the expected drop. {'Pass' if distance_km <= 0.35 else 'Review'} under the demo geofence rule."
    if "best practice" in q or "best practices" in q:
        return "Best practices shown here: enforce geofenced POD capture, require timestamped ePOD uploads, reject low-quality or suspicious PODs, monitor stop readiness before upload, and keep exceptions in review instead of auto-closing jobs."
    if "on time" in q or "sla" in q:
        return f"On-time delivery proxy is {s.get('on_time_prob', 0)*100:.0f}% in this demo, based on route-stop assumptions and stop-compliance signals."
    return "Ask about delivery quality, POD authenticity, geofence validation, ePOD acceptance, or last-mile best practices."


st.title("Last Mile AI Agent - POD, Location, and ePOD Validation Demo")
st.caption("Streamlit demo for GitHub deployment using the included delivery clip and POD images.")

video_info, video_df = analyze_video_cached(str(VIDEO_PATH))
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []
if "epod_state" not in st.session_state:
    st.session_state.epod_state = {"epod_ok": None, "pod_verdict": None, "distance_km": None}

summary = video_df.attrs.get("summary", {}) if video_df is not None and not video_df.empty else {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Route compliance proxy", f"{summary.get('route_compliance', 0):.0f}%")
col2.metric("POD readiness", f"{summary.get('pod_capture_readiness', 0):.0f}%")
col3.metric("On-time likelihood", f"{summary.get('on_time_prob', 0)*100:.0f}%")
col4.metric("Delivery status", "Ready for validation")

main_tab, pod_tab, epod_tab, qa_tab = st.tabs(["Delivery video", "POD authenticity", "ePOD + location validation", "AI agent Q&A"])

with main_tab:
    st.subheader("Delivery clip and stop-readiness analytics")
    st.markdown(
        f"""
        <video controls loop autoplay muted playsinline style="width: 100%; border-radius: 0.5rem;">
            <source src="{video_data_url(str(VIDEO_PATH))}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """,
        unsafe_allow_html=True,
    )
    if video_info:
        st.write(f"Video: {video_info['w']}x{video_info['h']} | {video_info['duration_s']:.1f}s | {video_info['fps']:.1f} fps")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        if video_df is not None and not video_df.empty:
            st.line_chart(video_df.set_index("t_sec")[["sharpness", "brightness", "motion"]])
            st.dataframe(video_df, use_container_width=True)
    with c2:
        st.image(str(REF_LOCATION), caption="Reference drop-location context", use_column_width=True)
        st.image(str(REF_RECIPIENT), caption="Reference recipient / handoff context", use_column_width=True)
        st.markdown(
            """
            **What the AI agent checks in last mile:**
            - stop readiness before POD capture
            - driver stability / motion during proof capture
            - expected destination vs. observed GPS
            - document quality before ePOD acceptance
            - exception handling instead of auto-closing weak PODs
            """
        )

with pod_tab:
    st.subheader("Genuine vs non-genuine POD demonstration")
    a, b = st.columns(2)
    with a:
        st.image(str(GENUINE_POD), caption="Reference genuine POD", use_column_width=True)
    with b:
        st.image(str(NON_GENUINE_POD), caption="Reference non-genuine / weak POD", use_column_width=True)

    upload = st.file_uploader("Upload a POD image to validate", type=["jpg", "jpeg", "png"], key="pod_upload")
    if upload is not None:
        candidate = bytes_to_rgb(upload)
        genuine = load_rgb(GENUINE_POD)
        non_genuine = load_rgb(NON_GENUINE_POD)
        if candidate is not None and genuine is not None and non_genuine is not None:
            authenticity, verdict, sim_good, sim_bad, m = classify_pod(candidate, genuine, non_genuine)
            st.session_state.epod_state["pod_verdict"] = verdict
            st.image(candidate, caption="Uploaded POD", use_column_width=True)
            x1, x2, x3 = st.columns(3)
            x1.metric("Authenticity score", f"{authenticity:.0f}/100")
            x2.metric("Similarity to genuine ref", f"{sim_good:.2f}")
            x3.metric("Similarity to non-genuine ref", f"{sim_bad:.2f}")
            st.write(
                {
                    "verdict": verdict,
                    "sharpness": round(m["sharpness"], 1),
                    "brightness": round(m["brightness"], 1),
                    "edge_density": round(m["edge_density"], 3),
                    "aspect_ratio": round(m["aspect_ratio"], 2),
                }
            )
            if authenticity >= 60:
                st.success("The AI agent would allow this POD to move to ePOD validation.")
            else:
                st.warning("The AI agent would flag this POD for supervisor review before closure.")
        else:
            st.error("Could not read the uploaded image.")
    else:
        st.info("Upload a POD image to run a live demo classification using the attached references.")

with epod_tab:
    st.subheader("ePOD execution and delivery-location validation")
    address_name = st.selectbox("Expected destination", list(ADDRESS_BOOK.keys()))
    target = ADDRESS_BOOK[address_name]
    left, right = st.columns(2)
    with left:
        order_id = st.text_input("Order / shipment ID", value="AWB-999920979")
        recipient = st.text_input("Recipient / consignee", value=target["customer"])
        ts_text = st.text_input("POD timestamp (ISO)", value="2026-04-08T12:05:00")
        lat = st.number_input("Captured POD latitude", value=float(target["lat"] + 0.0008), format="%.6f")
        lon = st.number_input("Captured POD longitude", value=float(target["lon"] + 0.0012), format="%.6f")
        uploaded_again = st.file_uploader("Upload POD file for ePOD", type=["jpg", "jpeg", "png"], key="epod_upload")
        if st.button("Run ePOD validation"):
            pod_auth = 70.0
            if uploaded_again is not None:
                candidate = bytes_to_rgb(uploaded_again)
                if candidate is not None:
                    auth, verdict, _, _, _ = classify_pod(candidate, load_rgb(GENUINE_POD), load_rgb(NON_GENUINE_POD))
                    pod_auth = auth
                    st.session_state.epod_state["pod_verdict"] = verdict
            epod_ok, distance_km, reasons = validate_epod(order_id, recipient, ts_text, lat, lon, address_name, pod_auth, uploaded_again is not None)
            st.session_state.epod_state["epod_ok"] = epod_ok
            st.session_state.epod_state["distance_km"] = distance_km
            if epod_ok:
                st.success("ePOD accepted into system")
            else:
                st.error("ePOD held for review")
            st.write({
                "order_id": order_id,
                "destination": address_name,
                "distance_from_expected_m": round(distance_km * 1000, 1),
                "pod_authenticity": round(pod_auth, 1),
                "system_decision": "ACCEPT" if epod_ok else "REVIEW",
                "reasons": reasons,
            })
    with right:
        fig = plot_geo(target["lat"], target["lon"], lat, lon, 0.35)
        st.pyplot(fig)
        st.markdown(
            f"""
            **Delivery-policy controls**
            - expected window: **{target['window']}**
            - expected ETA proxy: **{target['eta']:.1f} hrs**
            - geofence threshold: **350 m**
            - upload policy: **mandatory POD image + timestamp + recipient + location**
            """
        )
        current_summary = build_summary(video_df, st.session_state.epod_state["epod_ok"], st.session_state.epod_state["pod_verdict"], st.session_state.epod_state["distance_km"])
        st.info(current_summary)

with qa_tab:
    st.subheader("Ask the AI agent about this last-mile job")
    prompt = st.text_input("Ask a question", placeholder="How is the delivery going? Was the POD genuine? Is the upload acceptable?")
    sample_prompts = [
        "How is the delivery going?",
        "Was the POD genuine?",
        "Is the location valid?",
        "What best practices are enforced?",
    ]
    cols = st.columns(len(sample_prompts))
    for i, p in enumerate(sample_prompts):
        if cols[i].button(p):
            answer = answer_question(p, video_df, st.session_state.epod_state["epod_ok"], st.session_state.epod_state["pod_verdict"], st.session_state.epod_state["distance_km"])
            st.session_state.qa_log.append((p, answer))
    if st.button("Answer question") and prompt:
        answer = answer_question(prompt, video_df, st.session_state.epod_state["epod_ok"], st.session_state.epod_state["pod_verdict"], st.session_state.epod_state["distance_km"])
        st.session_state.qa_log.append((prompt, answer))

    if st.session_state.qa_log:
        for q, a in reversed(st.session_state.qa_log[-8:]):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI agent:** {a}")
            st.divider()
    else:
        st.info("Ask about delivery quality, POD authenticity, geofence validation, or ePOD acceptance.")

st.markdown("---")
st.caption("Demo note: this app uses proxy analytics and image-quality heuristics for demonstration. Production deployments should validate against device GPS, order systems, and approved POD policies.")
