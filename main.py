from pathlib import Path
import sys

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "football.csv"

# Metrik kinerja yang bisa dipilih. "contribution" (gol + assist) adalah default.
METRICS = {
    "contribution": "Kontribusi gol (Gol + Assist)",
    "goals": "Gol",
    "assists": "Assist",
    "xG": "xG (peluang gol)",
    "xA": "xA (peluang assist)",
    "shots": "Tembakan",
    "key_passes": "Key passes",
}
DEFAULT_METRIC = "contribution"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the football dataset and remove an exported CSV index column."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data = data.loc[:, ~data.columns.str.match(r"^Unnamed(: .*)?$")]
    if data.empty:
        raise ValueError(f"Dataset tidak memiliki baris data: {DATA_PATH}")

    if {"goals", "assists"}.issubset(data.columns):
        data["contribution"] = data["goals"] + data["assists"]
    return data


def available_metrics(data: pd.DataFrame) -> dict:
    return {key: label for key, label in METRICS.items() if key in data.columns}


# --------------------------------------------------------------------------
# Analisis: tim terbaik, pemain terbaik, top 3 per tim
# --------------------------------------------------------------------------
def team_summary(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Total (sum) dan rata-rata (mean) kinerja pemain per tim."""
    summary = (
        data.groupby("team_title")[metric]
        .agg(total="sum", rata_rata="mean", jumlah_pemain="count")
        .reset_index()
    )
    summary["rata_rata"] = summary["rata_rata"].round(2)
    return summary.sort_values("total", ascending=False).reset_index(drop=True)


def top_players(data: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    """N pemain terbaik secara keseluruhan."""
    columns = [c for c in ("player_name", "team_title", "position", "games",
                           "goals", "assists", metric) if c in data.columns]
    columns = list(dict.fromkeys(columns))
    return data.nlargest(n, metric)[columns].reset_index(drop=True)


def top_players_per_team(data: pd.DataFrame, metric: str, n: int = 3) -> pd.DataFrame:
    """N pemain terbaik di setiap tim."""
    ranked = data.sort_values(metric, ascending=False).groupby("team_title").head(n)
    ranked = ranked.sort_values(["team_title", metric], ascending=[True, False]).copy()
    ranked["peringkat"] = ranked.groupby("team_title").cumcount() + 1
    columns = [c for c in ("team_title", "peringkat", "player_name", "position",
                           "games", "goals", "assists", metric) if c in ranked.columns]
    columns = list(dict.fromkeys(columns))
    return ranked[columns].reset_index(drop=True)


# --------------------------------------------------------------------------
# EDA (terminal)
# --------------------------------------------------------------------------
def print_eda(data: pd.DataFrame) -> None:
    """Print a concise, text-only overview of the dataset to the terminal."""
    print("=" * 72)
    print("EXPLORATORY DATA ANALYSIS - DATA PEMAIN SEPAK BOLA")
    print("=" * 72)
    print(f"Sumber data       : {DATA_PATH}")
    print(f"Jumlah pemain     : {len(data):,}")
    print(f"Jumlah kolom      : {len(data.columns):,}")
    print(f"Jumlah tim        : {data['team_title'].nunique():,}" if "team_title" in data else "Jumlah tim        : tidak tersedia")
    print(f"Jumlah posisi     : {data['position'].nunique():,}" if "position" in data else "Jumlah posisi     : tidak tersedia")

    print("\nGAMBARAN DATA")
    print("Setiap baris merepresentasikan statistik seorang pemain.")
    print("Kolom mencakup identitas, tim/posisi, penampilan, waktu bermain,")
    print("gol, assist, tembakan, peluang (xG/xA), dan kontribusi permainan.")
    print("\nTIPE DATA")
    print(data.dtypes.to_string())

    missing = data.isna().sum()
    missing = missing[missing > 0]
    print("\nKUALITAS DATA")
    print(f"Baris duplikat    : {data.duplicated().sum():,}")
    if "player_name" in data:
        print(f"Nama pemain kosong: {data['player_name'].isna().sum():,}")
        print(f"Pemain duplikat   : {data['player_name'].duplicated().sum():,}")
    if missing.empty:
        print("Nilai kosong      : tidak ditemukan")
    else:
        print("Nilai kosong per kolom:")
        print(missing.to_string())

    numeric_columns = [
        column
        for column in (
            "games", "time", "goals", "xG", "assists", "xA", "shots",
            "key_passes", "yellow_cards", "red_cards",
        )
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column])
    ]
    if numeric_columns:
        print("\nRINGKASAN STATISTIK (mean, median, min, max)")
        summary = data[numeric_columns].agg(["mean", "median", "min", "max"]).T
        print(summary.to_string(float_format=lambda value: f"{value:,.2f}"))

    for column, label in (("team_title", "5 TIM DENGAN PEMAIN TERBANYAK"),
                          ("position", "5 KATEGORI POSISI TERBANYAK")):
        if column in data:
            print(f"\n{label}")
            for category, count in data[column].value_counts().head(5).items():
                print(f"- {category}: {count} pemain")

    # Jawaban pertanyaan analisis
    if {"team_title", "player_name", DEFAULT_METRIC}.issubset(data.columns):
        label = METRICS[DEFAULT_METRIC]
        print("\n" + "-" * 72)
        print(f"KINERJA BERDASARKAN: {label}")
        print("-" * 72)

        teams = team_summary(data, DEFAULT_METRIC)
        print("\nTIM TERBAIK (berdasarkan total)")
        print(teams.head(5).to_string(index=False))
        best_mean = teams.sort_values("rata_rata", ascending=False).iloc[0]
        print(f"\nTim terbaik (total)         : {teams.iloc[0]['team_title']} ({teams.iloc[0]['total']:,.0f})")
        print(f"Tim terbaik (rata-rata/pmn) : {best_mean['team_title']} ({best_mean['rata_rata']:.2f})")

        print("\nTOP 5 PEMAIN TERBAIK")
        print(top_players(data, DEFAULT_METRIC).to_string(index=False))

        print("\nTOP 3 PEMAIN PER TIM")
        per_team = top_players_per_team(data, DEFAULT_METRIC)
        for team, group in per_team.groupby("team_title"):
            names = ", ".join(
                f"{row.player_name} ({row[DEFAULT_METRIC]:,.0f})"
                for _, row in group.iterrows()
            )
            print(f"- {team}: {names}")
    print("=" * 72)


# --------------------------------------------------------------------------
# Dashboard (Streamlit + Plotly)
# --------------------------------------------------------------------------
def run_dashboard(data: pd.DataFrame) -> None:
    """Render the Streamlit dashboard using Plotly visualizations."""
    import plotly.express as px
    import streamlit as st

    st.set_page_config(page_title="Dashboard Statistik Pemain", layout="wide")
    st.title("Dashboard Statistik Pemain Sepak Bola")
    st.caption("Perbandingan tim dan pemain dari football.csv.")

    metrics = available_metrics(data)
    if "team_title" not in data or "player_name" not in data or not metrics:
        st.error("Kolom team_title, player_name, dan metrik kinerja diperlukan.")
        return

    # ---- Sidebar: filter ----
    st.sidebar.header("Filter")
    metric = st.sidebar.selectbox(
        "Metrik kinerja",
        list(metrics),
        index=list(metrics).index(DEFAULT_METRIC) if DEFAULT_METRIC in metrics else 0,
        format_func=lambda key: metrics[key],
    )
    metric_label = metrics[metric]

    filtered = data.copy()
    if "position" in filtered:
        positions = sorted(filtered["position"].dropna().unique())
        selected_positions = st.sidebar.multiselect("Posisi", positions)
        if selected_positions:
            filtered = filtered[filtered["position"].isin(selected_positions)]
    if "time" in filtered and pd.api.types.is_numeric_dtype(filtered["time"]):
        max_time = int(filtered["time"].max())
        min_time = st.sidebar.slider("Minimal menit bermain", 0, max_time, 0, step=90)
        filtered = filtered[filtered["time"] >= min_time]

    if filtered.empty:
        st.warning("Tidak ada data yang cocok dengan filter.")
        return

    teams = team_summary(filtered, metric)
    best_total = teams.iloc[0]
    best_mean = teams.sort_values("rata_rata", ascending=False).iloc[0]
    best_player = top_players(filtered, metric, 1).iloc[0]

    # ---- Ringkasan / jawaban utama ----
    cols = st.columns(4)
    cols[0].metric("Jumlah pemain", f"{len(filtered):,}")
    cols[1].metric("Tim terbaik (total)", best_total["team_title"], f"{best_total['total']:,.0f}")
    cols[2].metric("Tim terbaik (rata-rata)", best_mean["team_title"], f"{best_mean['rata_rata']:.2f}")
    cols[3].metric("Pemain terbaik", best_player["player_name"], f"{best_player[metric]:,.0f}")

    st.info(
        f"Berdasarkan **{metric_label}**: tim terbaik menurut total adalah "
        f"**{best_total['team_title']}**, sedangkan menurut rata-rata per pemain adalah "
        f"**{best_mean['team_title']}**. Pemain terbaik: **{best_player['player_name']}** "
        f"({best_player['team_title']})."
    )

    tab_team, tab_player, tab_top3, tab_data = st.tabs(
        ["Perbandingan tim", "Top 5 pemain", "Top 3 per tim", "Data"]
    )

    # ---- Tab 1: tim ----
    with tab_team:
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                teams.sort_values("total"),
                x="total", y="team_title", orientation="h",
                color="total", color_continuous_scale="Blues",
                title=f"Total {metric_label} per tim",
                labels={"total": "Total", "team_title": "Tim"},
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(
                teams.sort_values("rata_rata"),
                x="rata_rata", y="team_title", orientation="h",
                color="rata_rata", color_continuous_scale="Greens",
                title=f"Rata-rata {metric_label} per pemain",
                labels={"rata_rata": "Rata-rata", "team_title": "Tim"},
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(
            teams, x="rata_rata", y="total", size="jumlah_pemain", text="team_title",
            title="Total vs rata-rata (ukuran titik = jumlah pemain)",
            labels={"rata_rata": "Rata-rata per pemain", "total": "Total tim",
                    "jumlah_pemain": "Jumlah pemain"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Total dipengaruhi ukuran skuad; rata-rata menunjukkan kualitas per pemain. "
            "Gunakan filter menit bermain agar rata-rata tidak terdistorsi pemain cadangan."
        )
        st.dataframe(teams, use_container_width=True, hide_index=True)

    # ---- Tab 2: top 5 pemain ----
    with tab_player:
        top5 = top_players(filtered, metric, 5)
        fig = px.bar(
            top5.sort_values(metric),
            x=metric, y="player_name", orientation="h",
            color="team_title" if "team_title" in top5 else None,
            text=metric,
            title=f"Top 5 pemain - {metric_label}",
            labels={metric: metric_label, "player_name": "Pemain", "team_title": "Tim"},
        )
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top5, use_container_width=True, hide_index=True)

    # ---- Tab 3: top 3 per tim ----
    with tab_top3:
        per_team = top_players_per_team(filtered, metric, 3)
        team_options = sorted(per_team["team_title"].unique())
        selected_team = st.selectbox("Pilih tim", team_options)
        team_top = per_team[per_team["team_title"] == selected_team]
        fig = px.bar(
            team_top.sort_values(metric),
            x=metric, y="player_name", orientation="h", text=metric,
            title=f"Top 3 pemain {selected_team} - {metric_label}",
            labels={metric: metric_label, "player_name": "Pemain"},
        )
        fig.update_layout(yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            per_team, x="team_title", y=metric, color="player_name",
            title="Top 3 pemain di semua tim",
            labels={"team_title": "Tim", metric: metric_label, "player_name": "Pemain"},
        )
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(per_team, use_container_width=True, hide_index=True)

    # ---- Tab 4: data ----
    with tab_data:
        st.dataframe(
            filtered.sort_values(metric, ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    data = load_data()
    if "--eda" in sys.argv[1:]:
        print_eda(data)
        return
    run_dashboard(data)


if __name__ == "__main__":
    main()