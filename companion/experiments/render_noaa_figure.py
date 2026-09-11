"""Render an original SVG from verified frozen results, without new dependencies."""

from html import escape
import sys

if __package__ in (None, ""):
    from noaa_study import ROOT, StudyError, run_offline
else:
    from .noaa_study import ROOT, StudyError, run_offline


ASSET_PATH = ROOT / "experiments" / "assets" / "noaa_calibration_scores.svg"


def render_svg(report: dict) -> str:
    primary = report["primary"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="560" viewBox="0 0 1020 560" role="img" aria-labelledby="title desc">',
        '<title id="title">NOAA monthly-rise forecast calibration and Brier scores</title>',
        '<desc id="desc">Original descriptive figure from the frozen final-vintage NOAA study. '
        'Five-bin primary calibration, with counts beside nonempty bins; primary and '
        'refitted sensitivity single-YES Brier scores. No confidence intervals or trading evidence.</desc>',
        '<rect width="1020" height="560" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#192b3b;font-size:14px}'
        '.heading{font-size:20px;font-weight:bold}.subheading{font-size:16px;font-weight:bold}'
        '.small{font-size:12px}.axis{stroke:#506474;stroke-width:1}'
        '.grid{stroke:#e2e7ed;stroke-width:1}</style>',
    ]

    def text(x, y, value, extra=""):
        parts.append(f'<text x="{x}" y="{y}" {extra}>{escape(str(value))}</text>')

    def line(x1, y1, x2, y2, css):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{css}"/>')

    text(40, 35, "Real NOAA observations: seasonal and overall probability forecasts", 'class="heading"')
    text(40, 59, "FINAL-VINTAGE retrospective exercise | 2015-2025 | Not a prediction-market backtest")
    text(65, 100, f"Calibration: primary cohort (n={primary['count']})", 'class="subheading"')
    text(570, 100, "Single-YES Brier score (lower is better)", 'class="subheading"')
    x0, y0, width, height = 80, 420, 350, 280
    for value in (0, 0.2, 0.4, 0.6, 0.8, 1):
        x, y = x0 + width * value, y0 - height * value
        line(x, y0, x, y0 - height, "grid")
        line(x0, y, x0 + width, y, "grid")
        text(x, y0 + 22, f"{value:.1f}", 'text-anchor="middle" class="small"')
        text(x0 - 12, y + 4, f"{value:.1f}", 'text-anchor="end" class="small"')
    line(x0, y0, x0 + width, y0, "axis")
    line(x0, y0, x0, y0 - height, "axis")
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + width}" y2="{y0 - height}" stroke="#8b959e" stroke-dasharray="5 4"/>')
    text(x0 + width / 2, y0 + 47, "Mean predicted YES probability", 'text-anchor="middle"')
    text(25, 280, "Observed YES fraction", 'text-anchor="middle" transform="rotate(-90 25 280)"')
    colors = {"overall": "#b75608", "seasonal": "#086d9b"}
    for model, color in colors.items():
        for item in primary["models"][model]["calibration_bins"]:
            if item["count"] == 0:
                continue
            x = x0 + width * item["mean_probability"]
            y = y0 - height * item["observed_rate"]
            parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{color}" stroke="white"/>')
            offset = 18 if item["observed_rate"] > 0.9 else -10
            text(f"{x:.3f}", f"{y + offset:.3f}", f"n={item['count']}",
                 'text-anchor="middle" class="small"')
    chart_x, chart_width, chart_top, max_score = 700, 230, 148, 0.25
    for value in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
        x = chart_x + chart_width * value / max_score
        line(x, chart_top - 10, x, 410, "grid")
        text(x, 438, f"{value:.2f}", 'text-anchor="middle" class="small"')
    for i, cohort in enumerate(("primary", "sensitivity")):
        section = report[cohort]
        top = chart_top + i * 137
        text(570, top - 15, f"{cohort.title()} (n={section['count']})", 'class="subheading"')
        for j, model in enumerate(("overall", "seasonal")):
            value = section["models"][model]["brier"]
            y = top + j * 43
            text(chart_x - 12, y + 18, model.title(), 'text-anchor="end"')
            parts.append(f'<rect x="{chart_x}" y="{y}" width="{chart_width * value / max_score:.3f}" height="25" fill="{colors[model]}"/>')
            text(chart_x + chart_width * value / max_score + 5, y + 18, f"{value:.4f}", 'class="small"')
    text(570, 467, "Sensitivity: affected target/predecessor months removed,", 'class="small"')
    text(570, 485, "same reduced evaluation cohort for both refitted models.", 'class="small"')
    for i, (model, color) in enumerate(colors.items()):
        x = 90 + i * 160
        parts.append(f'<rect x="{x}" y="482" width="13" height="13" fill="{color}"/>')
        text(x + 20, 494, model.title(), 'class="small"')
    text(40, 524, "Five equal-width calibration bins; empty bins omitted. Counts are monthly events, not independent replicates.", 'class="small"')
    text(40, 544, "Data: NOAA GML, Carbon Cycle and Greenhouse Gases group. Original transformations; no NOAA endorsement.", 'class="small"')
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> int:
    try:
        report = run_offline()
        content = render_svg(report).encode("utf-8")
        ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
        ASSET_PATH.write_bytes(content)
    except (StudyError, OSError) as exc:
        print(f"Figure error: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered verified frozen study: {ASSET_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
