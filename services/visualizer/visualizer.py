import io
import os
import pickle
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

        st.divider()
        cross_video = st.checkbox(
            "Cross-video search only",
            value=True,
            help="Exclude frames from the same video as the selected point.",
        )

        compute_clicked = False
        if method in ("t-SNE", "UMAP"):
            compute_clicked = st.button(f"Compute {method}", type="primary")

        st.divider()
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
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Selected Frame")
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

        with col2:
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


if __name__ == "__main__":
    st_main()
