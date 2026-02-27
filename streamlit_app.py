"""
Streamlit app for stockout prediction.
"""

import pandas as pd
import streamlit as st

from ingest import REQUIRED_COLUMNS
from predict import load_artifacts, score_latest


def _build_chart_data(output):
    """
    Build chart data for Streamlit visualizations.

    Args:
        output (pd.DataFrame): Recommendations dataframe.

    Returns:
        tuple: (counts_df, top_risk_df)
    """
    counts = output["recommendation"].value_counts().reset_index()
    counts.columns = ["recommendation", "count"]

    top_risk = (
        output.sort_values("stockout_prob_30d", ascending=False)
        .head(10)
        .loc[:, ["Item name", "stockout_prob_30d"]]
    )

    return counts, top_risk


def main():
    """
    Render the Streamlit UI and handle file uploads.
    """
    st.set_page_config(page_title="Stockout Predictor", layout="wide")

    st.markdown(
        """
        <style>
        body { background-color: #0b0f14; }
        .main { background-color: #0b0f14; color: #f8fafc; }
        .card { background: #0f172a; border: 1px solid #1f2937; border-radius: 14px;
                padding: 16px; box-shadow: 0 8px 20px rgba(2,6,23,0.35); }
        .badge { background: #f97316; color: #0b0f14; padding: 6px 12px;
                 border-radius: 999px; font-weight: 700; font-size: 12px; }
        .subtitle { color: #cbd5f5; font-size: 13px; }
        .hero { display: flex; gap: 18px; align-items: center; justify-content: space-between; }
        .hero img { width: 360px; height: 180px; object-fit: cover; border-radius: 12px;
                    border: 1px solid #1f2937; }
        .step { background: #0f172a; border: 1px solid #1f2937; border-radius: 12px;
                padding: 12px; box-shadow: 0 6px 16px rgba(2,6,23,0.35); }
        .kpi { background: #0f172a; border: 1px solid #1f2937; border-radius: 12px;
               padding: 12px; box-shadow: 0 8px 20px rgba(2,6,23,0.35); }
        .kpi-label { color: #cbd5f5; font-size: 12px; text-transform: uppercase; }
        .kpi-value { font-size: 22px; font-weight: 700; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <div style="font-size:26px; font-weight:700;">Stockout Predictor</div>
              <div class="subtitle">
                Upload a CSV with Item name, Transaction Date, Opening Stock,
                QTY transacted, Closing Stock, Type, and Sales value.
              </div>
            </div>
            <div class="badge">BI Dashboard</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card hero" style="margin-top:16px;">
          <div>
            <div style="font-size:22px; font-weight:700;">Smarter Stockout Prevention</div>
            <div class="subtitle">
              This dashboard scores stockout risk for each item and highlights what needs review first.
            </div>
          </div>
          <img src="https://images.unsplash.com/photo-1607619056574-7b8d3ee536b2?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8cGlsbHN8ZW58MHx8MHx8fDA%3D"
               alt="Pills on orange background" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if not uploaded:
        try:
            load_artifacts()
        except Exception:
            st.warning(
                "Model artifacts not found. Run training first:\n"
                "`python train.py --csv \"Data/modeling_dataset.csv\" "
                "--horizon-days 60 --stockout-level 5`"
            )
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns: {missing_list}")

        output = score_latest(df, stockout_level=0)
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    st.success("Success! Showing recommendations.")

    counts_df, top_risk_df = _build_chart_data(output)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">Total Items</div>
              <div class="kpi-value">{int(output.shape[0])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">Reorder / Review</div>
              <div class="kpi-value">{int((output["recommendation"] == "Reorder / Review").sum())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">OK Items</div>
              <div class="kpi-value">{int((output["recommendation"] == "OK").sum())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Summary")
        st.bar_chart(counts_df.set_index("recommendation"))
    with c2:
        st.subheader("Top 10 Stockout Risk")
        st.bar_chart(top_risk_df.set_index("Item name"))

    st.subheader("Recommendations (Top 50)")
    st.dataframe(output.head(50), use_container_width=True)

    csv_bytes = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="stockout_recommendations.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
