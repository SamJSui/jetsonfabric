#!/usr/bin/env python3

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "benchmarks" / "data"
FIGURES = ROOT / "docs" / "benchmarks" / "figures"

WIDTH = 960
FONT = "system-ui,-apple-system,Segoe UI,sans-serif"
INK = "#17212b"
MUTED = "#5b6672"
GRID = "#d9e0e6"
ONE = "#0f766e"
TWO = "#c2410c"


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attrs(values):
    return " ".join(
        f'{name.replace("_", "-")}="{html.escape(str(value), quote=True)}"'
        for name, value in values.items()
    )


def element(name, content="", **values):
    return f"<{name} {attrs(values)}>{content}</{name}>"


def text(x, y, value, size=13, fill=INK, weight=400, anchor="start"):
    return element(
        "text",
        html.escape(str(value)),
        x=x,
        y=y,
        fill=fill,
        font_family=FONT,
        font_size=size,
        font_weight=weight,
        text_anchor=anchor,
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    values = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "stroke": stroke,
        "stroke_width": width,
    }
    if dash:
        values["stroke_dasharray"] = dash
    return element("line", **values)


def rect(x, y, width, height, fill, radius=0):
    return element(
        "rect",
        x=x,
        y=y,
        width=width,
        height=height,
        fill=fill,
        rx=radius,
    )


def circle(x, y, radius, fill, stroke="#ffffff", stroke_width=2):
    return element(
        "circle",
        cx=x,
        cy=y,
        r=radius,
        fill=fill,
        stroke=stroke,
        stroke_width=stroke_width,
    )


def document(title, description, height, body):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'role="img" aria-labelledby="title desc">\n'
        f"  <title id=\"title\">{html.escape(title)}</title>\n"
        f"  <desc id=\"desc\">{html.escape(description)}</desc>\n"
        f'  <rect width="{WIDTH}" height="{height}" fill="#ffffff"/>\n'
        + "\n".join(f"  {item}" for item in body)
        + "\n</svg>\n"
    )


def write_quality_throughput():
    rows = read_rows("humaneval-summary.csv")
    left, top, plot_width, plot_height = 78, 90, 800, 270
    x_max, y_min, y_max = 40.0, 60.0, 92.0

    def px(value):
        return left + float(value) / x_max * plot_width

    def py(value):
        return top + (y_max - float(value)) / (y_max - y_min) * plot_height

    body = [
        text(48, 34, "Quality / throughput tradeoff", 22, weight=700),
        text(
            48,
            58,
            "Qwen2.5-Coder Q4_K_M on Jetson Orin Nano MAXN_SUPER",
            13,
            MUTED,
        ),
    ]
    for tick in (60, 70, 80, 90):
        y = py(tick)
        body.extend(
            [
                line(left, y, left + plot_width, y),
                text(left - 12, y + 5, f"{tick}%", 12, MUTED, anchor="end"),
            ]
        )
    for tick in (0, 10, 20, 30, 40):
        x = px(tick)
        body.extend(
            [
                line(x, top, x, top + plot_height),
                text(x, top + plot_height + 24, tick, 12, MUTED, anchor="middle"),
            ]
        )

    points = [
        (px(row["fixed_64_tokens_per_second"]), py(row["humaneval_plus_percent"]))
        for row in rows
    ]
    body.append(
        element(
            "polyline",
            points=" ".join(f"{x:.1f},{y:.1f}" for x, y in points),
            fill="none",
            stroke="#9aa6b2",
            stroke_width=2,
        )
    )
    label_offsets = {
        "1.5B": (-12, -14, "end"),
        "3B": (12, -12, "start"),
        "7B": (12, -10, "start"),
        "14B": (12, 22, "start"),
    }
    for row, (x, y) in zip(rows, points):
        low = py(row["humaneval_plus_ci_high"])
        high = py(row["humaneval_plus_ci_low"])
        color = ONE if row["placement"] == "one" else TWO
        body.extend(
            [
                line(x, low, x, high, color, 2),
                line(x - 5, low, x + 5, low, color, 2),
                line(x - 5, high, x + 5, high, color, 2),
                circle(x, y, 7, color),
            ]
        )
        dx, dy, anchor = label_offsets[row["model"]]
        body.append(
            text(
                x + dx,
                y + dy,
                f'{row["model"]}  {float(row["humaneval_plus_percent"]):.1f}%',
                13,
                INK,
                650,
                anchor,
            )
        )

    body.extend(
        [
            text(
                left + plot_width / 2,
                407,
                "Fixed-output throughput (tokens/second)",
                13,
                MUTED,
                anchor="middle",
            ),
            text(18, 230, "HumanEval+ pass@1", 13, MUTED),
            circle(710, 44, 6, ONE),
            text(723, 49, "one node", 12, MUTED),
            circle(808, 44, 6, TWO),
            text(821, 49, "two-node split", 12, MUTED),
            text(48, 438, "Error bars: Wilson 95% confidence interval; 164 tasks.", 11, MUTED),
        ]
    )
    output = document(
        "JetsonFabric quality and throughput tradeoff",
        "HumanEval Plus pass at one increases with model size while fixed-output throughput decreases.",
        456,
        body,
    )
    (FIGURES / "quality-throughput.svg").write_text(output, encoding="utf-8")


def panel(
    body,
    rows,
    metric,
    title,
    unit,
    x,
    y,
    width,
    height,
    maximum,
    decimals,
):
    models = ("1.5B", "3B", "7B", "14B")
    grouped = {(row["model"], row["placement"]): row for row in rows}
    baseline = y + height
    body.append(text(x, y - 22, title, 16, weight=700))
    for tick_index in range(6):
        value = maximum * tick_index / 5
        line_y = baseline - height * tick_index / 5
        body.extend(
            [
                line(x, line_y, x + width, line_y),
                text(x - 10, line_y + 4, f"{value:g}", 11, MUTED, anchor="end"),
            ]
        )

    group_width = width / len(models)
    bar_width = 28
    for index, model in enumerate(models):
        center = x + group_width * (index + 0.5)
        for placement, offset, color in (
            ("one", -bar_width / 2, ONE),
            ("two", bar_width / 2, TWO),
        ):
            row = grouped.get((model, placement))
            if row is None:
                continue
            value = float(row[metric])
            bar_height = value / maximum * height
            bar_x = center + offset - bar_width / 2
            body.extend(
                [
                    rect(bar_x, baseline - bar_height, bar_width, bar_height, color, 3),
                    text(
                        bar_x + bar_width / 2,
                        baseline - bar_height - 7,
                        f"{value:.{decimals}f}",
                        10,
                        INK,
                        600,
                        "middle",
                    ),
                ]
            )
        body.append(text(center, baseline + 23, model, 12, MUTED, 600, "middle"))
    body.append(text(x + width / 2, baseline + 46, unit, 11, MUTED, anchor="middle"))


def write_model_scaling():
    rows = read_rows("model-scaling.csv")
    body = [
        text(48, 34, "Latency, throughput, and energy by placement", 22, weight=700),
        text(
            48,
            58,
            "Three warmups + 20 measured requests; 1,024-token context; 64 output tokens",
            13,
            MUTED,
        ),
        rect(702, 30, 13, 13, ONE, 2),
        text(723, 41, "one node", 12, MUTED),
        rect(800, 30, 13, 13, TWO, 2),
        text(821, 41, "two-node split", 12, MUTED),
    ]
    panel(
        body,
        rows,
        "tokens_per_second",
        "Output throughput",
        "tokens/second",
        72,
        116,
        360,
        190,
        40,
        1,
    )
    panel(
        body,
        rows,
        "ttft_p50_ms",
        "Time to first token (p50)",
        "milliseconds",
        552,
        116,
        336,
        190,
        500,
        0,
    )
    panel(
        body,
        rows,
        "itl_p50_ms",
        "Inter-token latency (p50)",
        "milliseconds",
        72,
        424,
        360,
        190,
        200,
        0,
    )
    panel(
        body,
        rows,
        "energy_j_per_output_token",
        "Energy per output token",
        "joules/token",
        552,
        424,
        336,
        190,
        5,
        2,
    )
    body.append(
        text(
            48,
            690,
            "14B is split-only because 8.98 GB of resident tensors exceed one node's physical RAM.",
            11,
            MUTED,
        )
    )
    output = document(
        "JetsonFabric latency, throughput, and energy by placement",
        "Grouped bars compare one-node and two-node throughput, time to first token, inter-token latency, and energy per token.",
        712,
        body,
    )
    (FIGURES / "model-scaling.svg").write_text(output, encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_quality_throughput()
    write_model_scaling()
    print(f"Wrote benchmark figures to {FIGURES}")


if __name__ == "__main__":
    main()
