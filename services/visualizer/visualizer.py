import concurrent.futures
import io
import os
import pickle
import shutil
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(page_title="Iris Visualizer", layout="wide")

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN", "postgresql://iris:iris@postgres:5432/iris"
)
SOURCE_DIR = os.getenv("SOURCE_DIR", "/data/source")
FRAMES_DIR = os.getenv("FRAMES_DIR", "/data/frames")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/iris_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


umap_available = False
try:
    import umap

    umap_available = True
except ImportError:
    pass


def compute_and_store_pca(conn, embeddings, df):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    with conn.cursor() as cur:
        for i, (vid, fidx) in enumerate(zip(df["video_id"], df["idx"])):
            cur.execute(
                "UPDATE frames SET pca_x = %s, pca_y = %s WHERE video_id = %s::uuid AND idx = %s",
                (float(coords[i, 0]), float(coords[i, 1]), vid, int(fidx)),
            )
    conn.commit()
    return coords


@st.cache_data(ttl=60)
def load_data():
    conn = psycopg.connect(POSTGRES_DSN)
    query = """
        SELECT
            f.video_id::text AS video_id,
            f.idx,
            f.timestamp_s,
            f.embedding::text AS embedding_str,
            f.pca_x,
            f.pca_y,
            v.filename,
            v.source_path,
            v.duration_s,
            v.width,
            v.height
        FROM frames f
        JOIN videos v ON v.id = f.video_id
        WHERE v.status = 'done'
        ORDER BY v.filename, f.idx
    """
    df = pd.read_sql(query, conn)

    if df.empty:
        conn.close()
        return df

    embeddings = np.array(
        [np.fromstring(e.strip("[]"), sep=",") for e in df["embedding_str"]]
    )
    df["embedding"] = list(embeddings)
    df.drop(columns=["embedding_str"], inplace=True)

    need_pca = df["pca_x"].isna().any() or df["pca_y"].isna().any()
    if need_pca:
        progress_text = "Computing PCA on all frames (one-time)..."
        progress_bar = st.progress(0, text=progress_text)
        coords = compute_and_store_pca(conn, embeddings, df)
        df["pca_x"] = coords[:, 0]
        df["pca_y"] = coords[:, 1]
        progress_bar.empty()

    conn.close()
    return df


@st.cache_data
def reduce_dimensions(method, n_samples, random_state, precomputed_pca=None):
    if method == "PCA" and precomputed_pca is not None:
        coords = precomputed_pca
        if n_samples < len(coords):
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(coords), n_samples, replace=False)
            coords = coords[idx]
        else:
            idx = None
        return coords, idx, "PCA (all frames)"

    sampled = None
    idx = None

    if method == "t-SNE" or method == "UMAP":
        if "cached_coords" not in st.session_state:
            st.session_state.cached_coords = {}
        cache_key = f"{method}_{n_samples}"
        if cache_key in st.session_state.cached_coords:
            return st.session_state.cached_coords[cache_key]

    return None, idx, method


def compute_non_pca(embeddings, method, n_samples, random_state):
    if n_samples < len(embeddings):
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(embeddings), n_samples, replace=False)
        sampled = embeddings[idx]
    else:
        idx = None
        sampled = embeddings

    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(sampled)
        label = f"PCA (explained variance: {reducer.explained_variance_ratio_.sum():.1%})"
    elif method == "t-SNE":
        reducer = PCA(n_components=min(50, sampled.shape[1]))
        pca_init = reducer.fit_transform(sampled)
        tsne = TSNE(n_components=2, random_state=random_state, init="random",
                    perplexity=min(30, max(5, n_samples // 10)))
        coords = tsne.fit_transform(pca_init)
        label = "t-SNE"
    elif method == "UMAP" and umap_available:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(sampled)
        label = "UMAP"
    else:
        coords = np.zeros((len(sampled), 2))
        label = "No data"

    return coords, idx, label


def load_frame(video_id, idx, timestamp_s, source_path):
    path = os.path.join(FRAMES_DIR, video_id, f"{idx:06d}.jpg")
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not os.path.exists(source_path):
        return None
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_s * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def find_similar(video_id, frame_idx, limit=10, exclude_video_id=None):
    conn = psycopg.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM frames WHERE video_id = %s::uuid AND idx = %s",
            (video_id, frame_idx),
        )
        row = cur.fetchone()
        if row is None:
            conn.close()
            return []
        query_emb = row[0]

        if exclude_video_id:
            sql = (
                "SELECT * FROM ("
                "SELECT DISTINCT ON (f.video_id) "
                "f.video_id::text, f.idx, f.timestamp_s, v.filename, v.source_path, "
                "f.embedding <=> %s::vector AS distance "
                "FROM frames f JOIN videos v ON v.id = f.video_id "
                "WHERE v.id != %s::uuid "
                "ORDER BY f.video_id, distance"
                ") sub ORDER BY distance"
            )
            params = [query_emb, exclude_video_id]
        else:
            sql = (
                "SELECT f.video_id::text, f.idx, f.timestamp_s, v.filename, v.source_path, "
                "f.embedding <=> %s::vector AS distance "
                "FROM frames f JOIN videos v ON v.id = f.video_id "
                "ORDER BY distance LIMIT %s"
            )
            params = [query_emb, limit]
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return rows


def extract_clip(video_id, start_idx, clip_frames, source_path):
    out_path = os.path.join(
        CACHE_DIR, f"flow_{video_id}_{start_idx}_{clip_frames}s.mp4"
    )
    if os.path.exists(out_path):
        return out_path
    subprocess.run(
        ["ffmpeg", "-ss", str(float(start_idx)), "-i", source_path,
         "-t", str(clip_frames),
         "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", "-y", out_path],
        capture_output=True, check=True,
    )
    return out_path


def build_flow(video_id, start_idx, clip_frames, steps, first_filename, first_source_path):
    flow = []
    cur_vid, cur_idx = video_id, start_idx
    cur_fname, cur_spath = first_filename, first_source_path
    used_clips = {(video_id, start_idx)}
    for step in range(steps):
        clip_path = extract_clip(cur_vid, cur_idx, clip_frames, cur_spath)
        flow.append((cur_vid, cur_idx, cur_fname, clip_path, None))
        if step == steps - 1:
            break
        last_idx = cur_idx + clip_frames - 1
        neighbors = find_similar(cur_vid, last_idx, limit=10, exclude_video_id=cur_vid)
        next_step = None
        for n in neighbors:
            key = (n[0], n[1])
            if key not in used_clips:
                next_step = n
                break
        if next_step is None:
            break
        n_vid, n_idx, n_ts, n_fname, n_spath, n_dist = next_step
        used_clips.add((n_vid, n_idx))
        flow[-1] = (cur_vid, cur_idx, cur_fname, clip_path, n_dist)
        cur_vid, cur_idx, cur_fname, cur_spath = n_vid, n_idx, n_fname, n_spath
    return flow


def merge_pair(path_a, dur_a, path_b, dur_b, out_path, overlap):
    offset = dur_a - overlap
    subprocess.run(
        ["ffmpeg", "-i", path_a, "-i", path_b,
         "-filter_complex",
         f"[0:v][1:v]xfade=transition=fade:duration={overlap}:offset={offset},"
         f"format=yuv420p",
         "-an", "-y", out_path],
        capture_output=True, check=True,
    )
    real_dur = get_duration(out_path)
    return out_path, real_dur


def concat_clips(clip_paths, output_path):
    list_file = os.path.join(CACHE_DIR, "concat_list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", "-an", "-y", output_path],
        capture_output=True, check=True,
    )
    return output_path


def merge_flow(clip_paths, output_path, clip_duration, overlap, use_xfade=False):
    if not use_xfade:
        return concat_clips(clip_paths, output_path)
    if len(clip_paths) <= 1:
        if clip_paths:
            shutil.copy2(clip_paths[0], output_path)
        return output_path

    clips = [(p, clip_duration) for p in clip_paths]
    level = 0
    while len(clips) > 1:
        next_clips = []
        tasks = []
        for i in range(0, len(clips), 2):
            if i + 1 >= len(clips):
                next_clips.append(clips[i])
                continue
            path_a, dur_a = clips[i]
            path_b, dur_b = clips[i | 1]
            out = os.path.join(CACHE_DIR, f"merge_l{level}_{i // 2}.mp4")
            tasks.append((path_a, dur_a, path_b, dur_b, out, overlap))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            futures = [pool.submit(merge_pair, *t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                out_path, out_dur = f.result()
                next_clips.append((out_path, out_dur))

        consumed = set()
        for i in range(0, len(clips) - 1, 2):
            consumed.add(clips[i][0])
            consumed.add(clips[i + 1][0])
        for path, _ in clips:
            if path.startswith(os.path.join(CACHE_DIR, "merge_l")) and path in consumed:
                try:
                    os.unlink(path)
                except Exception:
                    pass
        clips = next_clips
        level += 1

    shutil.copy2(clips[0][0], output_path)
    try:
        os.unlink(clips[0][0])
    except Exception:
        pass
    return output_path


def st_main():
    st.title("Iris — Embedding Visualizer")

    df = load_data()
    if df.empty:
        st.warning("No processed videos found in the database.")
        st.stop()

    embeddings = np.vstack(df["embedding"].values)
    n_total = len(embeddings)
    precomputed_pca = np.column_stack([df["pca_x"].values, df["pca_y"].values])

    with st.sidebar:
        st.header("Controls")

        with st.expander("Visualization", expanded=True):
            method = st.radio(
                "Dimensionality reduction",
                ["PCA", "t-SNE"] + (["UMAP"] if umap_available else []),
                index=0,
                help="PCA uses pre-computed values (instant). t-SNE/UMAP need explicit computation.",
            )

            max_sample = min(n_total, 2000)
            n_samples = st.slider(
                "Sample size",
                min_value=min(50, n_total),
                max_value=n_total,
                value=min(max_sample, n_total),
                step=1,
                help=f"Total embeddings: {n_total}. Reduce for faster rendering.",
            )

            color_by = st.selectbox("Color by", ["filename", "video_id", None])

        with st.expander("Search", expanded=True):
            cross_video = st.checkbox(
                "Cross-video search only",
                value=True,
                help="Exclude frames from the same video as the selected point.",
            )

        with st.expander("Flow Settings", expanded=False):
            clip_duration = st.slider("Clip duration (s)", 1, 10, 5)
            flow_steps = st.number_input("Flow steps", min_value=0, max_value=500, value=5)
            overlap = st.slider("Transition overlap (s)", 0.0, 3.0, 1.0, 0.5)
            use_xfade = st.checkbox("Crossfade transitions", value=False,
                                    help="Merge clips with crossfade (re-encodes, slow)")

        with st.expander("Advanced", expanded=False):
            compute_clicked = False
            if method in ("t-SNE", "UMAP"):
                compute_clicked = st.button(f"Compute {method}", type="primary")
            if st.button("Refresh data", type="secondary"):
                st.cache_data.clear()
                st.rerun()

    coords, sample_idx, method_label = None, None, ""

    if method == "PCA":
        coords, sample_idx, method_label = reduce_dimensions(
            method, n_samples, random_state=42,
            precomputed_pca=precomputed_pca,
        )
    elif compute_clicked or "cached_coords" in st.session_state:
        cache_key = f"{method}_{n_samples}"
        if compute_clicked or cache_key not in st.session_state.get("cached_coords", {}):
            progress_text = f"Running {method} on {n_samples} points..."
            progress_bar = st.progress(0, text=progress_text)
            coords, sample_idx, method_label = compute_non_pca(
                embeddings, method, n_samples, random_state=42,
            )
            progress_bar.empty()
            if "cached_coords" not in st.session_state:
                st.session_state.cached_coords = {}
            st.session_state.cached_coords[cache_key] = (coords, sample_idx, method_label)
        else:
            coords, sample_idx, method_label = st.session_state.cached_coords[cache_key]

    if coords is None:
        st.info(f"Select **{method}** and click **Compute {method}** to generate the embedding projection.")
        st.stop()

    if sample_idx is not None:
        plot_df = df.iloc[sample_idx].copy()
    else:
        plot_df = df.copy()

    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['filename']} @ {r['timestamp_s']:.1f}s", axis=1
    )

    color_col = color_by if color_by in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_col,
        hover_data={
            "label": True,
            "video_id": True,
            "timestamp_s": ":.1f",
            "filename": True,
        },
        title=f"{method_label} — {n_samples} frames",
        height=700,
    )

    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))

    if "selected_point_index" not in st.session_state:
        st.session_state.selected_point_index = None

    tab1, tab2, tab3 = st.tabs(["Projection", "Neighbors", "Flow"])

    with tab1:
        clicked = st.plotly_chart(fig, width='stretch', on_select="rerun")

        fresh_click = False
        if clicked and "selection" in clicked:
            pts = clicked["selection"].get("points", [])
            if pts and "point_index" in pts[0]:
                idx_in_plot = pts[0]["point_index"]
                if idx_in_plot is not None and idx_in_plot < len(plot_df):
                    fresh_click = True
                    st.session_state.selected_point_index = idx_in_plot

        selection = None
        if st.session_state.selected_point_index is not None:
            selection = plot_df.iloc[st.session_state.selected_point_index]

        if selection is not None:
            with st.expander("Selected Frame", expanded=True):
                st.write(
                    f"**Video:** {selection['filename']}  "
                    f"**Timestamp:** {selection['timestamp_s']:.1f}s  "
                    f"**Video ID:** {selection['video_id']}"
                )

                source_path = selection.get("source_path", "")
                if source_path:
                    frame = load_frame(
                        selection["video_id"],
                        selection["idx"],
                        selection["timestamp_s"],
                        source_path,
                    )
                    if frame is not None:
                        st.image(frame, width='stretch')
                    else:
                        st.caption("Could not load frame.")
                else:
                    st.caption("No source path available.")

    with tab2:
        selection = None
        if st.session_state.selected_point_index is not None:
            selection = plot_df.iloc[st.session_state.selected_point_index]

        if selection is not None:
            st.subheader("Nearest Neighbors")
            neighbors = find_similar(
                selection["video_id"],
                selection["idx"],
                limit=10,
                exclude_video_id=selection["video_id"] if cross_video else None,
            )

            for vid, fidx, ts, fname, spath, dist in neighbors:
                nn_frame = load_frame(vid, fidx, ts, spath)
                st.write(f"**{fname}** @ {ts:.1f}s — distance: {dist:.4f}")
                if nn_frame is not None:
                    st.image(nn_frame, width=150)
                st.divider()
        else:
            st.info("Click a point on the **Projection** tab to see its nearest neighbors.")

    with tab3:
        selection = None
        if st.session_state.selected_point_index is not None:
            selection = plot_df.iloc[st.session_state.selected_point_index]

        if selection is not None:
            st.subheader("Build Flow")
            if st.button("Build Flow", type="primary"):
                with st.spinner("Building flow..."):
                    flow = build_flow(
                        selection["video_id"],
                        selection["idx"],
                        clip_duration,
                        flow_steps,
                        selection["filename"],
                        selection["source_path"],
                    )
                st.session_state.flow = flow
                st.rerun()
        else:
            st.info("Click a point on the **Projection** tab to build a flow.")

        if "flow" in st.session_state and st.session_state.flow:
            st.divider()
            st.subheader("Flow Steps")
            for i, (vid, sidx, fname, clip_path, tdist) in enumerate(st.session_state.flow):
                with st.expander(f"Step {i+1} — {fname} @ {sidx}s", expanded=i==0):
                    st.video(clip_path)
                    if tdist is not None:
                        st.caption(f"→ distance: {tdist:.4f}")

            st.divider()
            if st.button("Merge to single video"):
                with st.spinner("Merging clips..."):
                    merged = merge_flow(
                        [step[3] for step in st.session_state.flow],
                        os.path.join(CACHE_DIR, "flow_merged.mp4"),
                        clip_duration, overlap, use_xfade,
                    )
                st.session_state.flow_merged = merged
                st.rerun()

            if "flow_merged" in st.session_state:
                st.subheader("Merged Flow")
                st.video(st.session_state.flow_merged)


if __name__ == "__main__":
    st_main()
