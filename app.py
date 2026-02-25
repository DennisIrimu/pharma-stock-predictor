"""
Simple Flask app for uploading CSVs and getting recommendations.
"""

import io
import tempfile

from flask import Flask, request, render_template, send_file

from ingest import read_csv_file
from predict import score_latest


app = Flask(__name__)

LAST_OUTPUT = None


def _build_chart_data(output):
    """
    Build chart data dictionaries for the dashboard.

    Args:
        output (pd.DataFrame): Recommendations dataframe.

    Returns:
        dict: Chart data payloads.
    """
    counts = output["recommendation"].value_counts().to_dict()
    count_labels = list(counts.keys())
    count_values = [counts[k] for k in count_labels]
    total_items = int(output.shape[0])
    reorder_count = int(counts.get("Reorder / Review", 0))
    ok_count = int(counts.get("OK", 0))

    top_risk = output.sort_values("stockout_prob_30d", ascending=False).head(10)
    top_labels = top_risk["Item name"].tolist()
    top_values = top_risk["stockout_prob_30d"].round(3).tolist()

    return {
        "count_labels": count_labels,
        "count_values": count_values,
        "top_labels": top_labels,
        "top_values": top_values,
        "total_items": total_items,
        "reorder_count": reorder_count,
        "ok_count": ok_count,
    }


def _render_page(message="", table_html="", chart_data=None):
    """
    Render the upload page with optional results.

    Args:
        message (str): Status message.
        table_html (str): HTML table of results.
        chart_data (dict | None): Chart data payloads.

    Returns:
        str: Rendered HTML.
    """
    return render_template(
        "index.html",
        message=message,
        table_html=table_html,
        chart_data=chart_data,
    )


@app.route("/", methods=["GET", "POST"])
def upload():
    """
    Upload CSV and return recommendations.
    """
    global LAST_OUTPUT

    if request.method == "GET":
        return _render_page()

    file = request.files.get("file")
    if not file:
        return _render_page(message="No file uploaded.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        file.save(tmp.name)
        try:
            df = read_csv_file(tmp.name)
            output = score_latest(df, stockout_level=0)
            LAST_OUTPUT = output
            table_html = output.head(50).to_html(index=False)
            chart_data = _build_chart_data(output)
            return _render_page(
                message="Success! Showing top 50 rows.",
                table_html=table_html,
                chart_data=chart_data,
            )
        except Exception as exc:
            return _render_page(message=f"Error: {exc}")


@app.route("/download", methods=["GET"])
def download():
    """
    Download the latest recommendations as CSV.
    """
    if LAST_OUTPUT is None:
        return _render_page(message="No output to download yet.")

    output = LAST_OUTPUT.copy()
    buffer = io.StringIO()
    output.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="stockout_recommendations.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
