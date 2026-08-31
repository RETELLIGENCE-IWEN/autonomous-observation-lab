#!/usr/bin/env python3
"""Build the predictive-gimbal research journey as a print-ready PDF.

The builder is deliberately offline. It translates the repository Markdown,
replaces the two Mermaid source blocks with publication SVG diagrams, embeds
the exact-data SVG plots, and asks a local Chromium-family browser to print the
result. No web fonts, JavaScript packages, or remote rendering services are
required.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "docs/research_tracks/predictive_gimbal_servoing/"
    "predictive_gimbal_servoing_journey_paper.md"
)
DEFAULT_OUTPUT = DEFAULT_SOURCE.with_suffix(".pdf")


def _arrow_marker(color: str = "#506274") -> str:
    return (
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/>'
        "</marker></defs>"
    )


def _svg_box(
    x: int,
    y: int,
    width: int,
    height: int,
    lines: tuple[str, ...],
    *,
    fill: str,
    stroke: str,
    text_color: str = "#1c2936",
    font_size: int = 17,
    dash: bool = False,
) -> str:
    dash_style = ' stroke-dasharray="9 6"' if dash else ""
    line_height = font_size + 5
    first_y = y + height / 2 - (len(lines) - 1) * line_height / 2 + 1
    text = "".join(
        f'<text x="{x + width / 2}" y="{first_y + index * line_height}" '
        f'text-anchor="middle" dominant-baseline="middle" '
        f'font-family="DejaVu Sans, sans-serif" font-size="{font_size}" '
        f'font-weight="{600 if index == 0 else 400}" fill="{text_color}">'
        f"{html.escape(line)}</text>"
        for index, line in enumerate(lines)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2.4"{dash_style}/>'
        + text
    )


def _svg_line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    dash: bool = False,
    arrow: bool = True,
) -> str:
    dash_style = ' stroke-dasharray="9 6"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" '
        f'stroke="#506274" stroke-width="2.7"{dash_style}{marker}/>'
    )


def _architecture_svg() -> str:
    """Return a deterministic SVG rendering of the system-boundary diagram."""

    parts = [
        '<svg class="paper-diagram" xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1320 570" role="img" '
        'aria-label="Predictive gimbal estimator and control system boundary">',
        _arrow_marker(),
        '<rect x="1" y="1" width="1318" height="568" rx="18" '
        'fill="#fbfcfe" stroke="#d8e0e8" stroke-width="2"/>',
        '<text x="34" y="43" font-family="DejaVu Sans, sans-serif" '
        'font-size="18" font-weight="700" fill="#43566a">DEPLOYABLE SIGNALS</text>',
        '<text x="526" y="43" font-family="DejaVu Sans, sans-serif" '
        'font-size="18" font-weight="700" fill="#824396">LEARNED ESTIMATOR</text>',
        '<text x="910" y="43" font-family="DejaVu Sans, sans-serif" '
        'font-size="18" font-weight="700" fill="#2f6c9e">CONVENTIONAL CONTROL + PLANT</text>',
    ]

    # Input fan-in.
    parts.extend(
        [
            _svg_box(
                28,
                75,
                205,
                74,
                ("Visual detection", "bbox • confidence • time"),
                fill="#eef4f9",
                stroke="#6d8ba5",
                font_size=15,
            ),
            _svg_box(
                28,
                171,
                205,
                74,
                ("Gimbal telemetry", "angle • angular rate"),
                fill="#eef4f9",
                stroke="#6d8ba5",
                font_size=15,
            ),
            _svg_box(
                28,
                267,
                205,
                74,
                ("Vehicle + action", "body rate • previous u"),
                fill="#eef4f9",
                stroke="#6d8ba5",
                font_size=15,
            ),
            _svg_box(
                296,
                139,
                190,
                138,
                ("Causal encoder", "fixed-width features", "validity masks"),
                fill="#eef4f9",
                stroke="#3977b8",
                font_size=16,
            ),
            _svg_line(233, 112, 296, 177),
            _svg_line(233, 208, 296, 208),
            _svg_line(233, 304, 296, 239),
        ]
    )

    # Learned state path and conventional command path.
    parts.extend(
        [
            _svg_box(
                533,
                139,
                190,
                138,
                ("O2 causal GRU", "36,240 parameters", "four horizons"),
                fill="#f0e5f5",
                stroke="#8e44ad",
                font_size=16,
            ),
            _svg_box(
                770,
                139,
                190,
                138,
                ("Target state", "bearing • rate", "uncertainty • validity"),
                fill="#f0e5f5",
                stroke="#8e44ad",
                font_size=16,
            ),
            _svg_box(
                1006,
                71,
                270,
                90,
                ("Configured adapter", "desired rate  |  position"),
                fill="#e3eff9",
                stroke="#3977b8",
                font_size=16,
            ),
            _svg_box(
                1006,
                189,
                270,
                90,
                ("Safety boundary", "projection • limits • watchdog"),
                fill="#e3eff9",
                stroke="#3977b8",
                font_size=16,
            ),
            _svg_line(486, 208, 533, 208),
            _svg_line(723, 208, 770, 208),
            _svg_line(960, 183, 1006, 132),
            _svg_line(1141, 161, 1141, 189),
        ]
    )

    # Inner loop and feedback loop.
    parts.extend(
        [
            _svg_box(
                1006,
                339,
                270,
                86,
                ("Bounded servo loop", "motor + configurable dynamics"),
                fill="#e3eff9",
                stroke="#3977b8",
                font_size=16,
            ),
            _svg_box(
                706,
                339,
                230,
                86,
                ("One-axis gimbal", "0° = body forward"),
                fill="#e3eff9",
                stroke="#3977b8",
                font_size=16,
            ),
            _svg_box(
                405,
                339,
                230,
                86,
                ("Camera + detector", "configured timing and FOV"),
                fill="#eef4f9",
                stroke="#6d8ba5",
                font_size=16,
            ),
            _svg_line(1141, 279, 1141, 339),
            _svg_line(1006, 382, 936, 382),
            _svg_line(706, 382, 635, 382),
            '<path d="M 405 382 L 132 382 L 132 149" fill="none" '
            'stroke="#506274" stroke-width="2.7" marker-end="url(#arrow)"/>',
        ]
    )

    # Simulator-only supervision.
    parts.extend(
        [
            _svg_box(
                28,
                468,
                458,
                67,
                ("Simulator-only truth", "LOS • motion • plant • delay — labels only"),
                fill="#faecd7",
                stroke="#c27b2a",
                font_size=15,
                dash=True,
            ),
            '<path d="M 486 502 L 628 502 L 628 277" fill="none" '
            'stroke="#506274" stroke-width="2.7" stroke-dasharray="9 6" '
            'marker-end="url(#arrow)"/>',
            '<text x="649" y="503" font-family="DejaVu Sans, sans-serif" '
            'font-size="14" fill="#7a5a28">never exposed at runtime</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _journey_svg() -> str:
    """Return a compact two-row research-journey SVG."""

    stages = (
        ("01", "Concept", "scope + novelty", "pass"),
        ("02", "Simulator", "rate + position", "pass"),
        ("03", "State interface", "truth visualization", "pass"),
        ("04", "Oracle + data", "leakage-controlled", "pass"),
        ("05", "Causal GRU", "O0 / O1 / O2", "pass"),
        ("06", "Closed loop", "paired testing", "pass"),
        ("07", "Replication", "three seeds", "pass"),
        ("08", "Global scaling", "accepted narrowly", "pass"),
        ("09", "Context table", "held-out reject", "reject"),
        ("10", "Recovery", "fresh-gate reject", "reject"),
        ("11", "Transfer gate", "measure + hardware", "next"),
    )
    palette = {
        "pass": ("#dff2ec", "#188977"),
        "reject": ("#fae4e2", "#cf5c5c"),
        "next": ("#fff1c9", "#c38a18"),
    }
    width = 192
    height = 112
    gap = 22
    left = 28
    top_y = 62
    bottom_y = 292
    positions: list[tuple[int, int]] = []
    for index in range(6):
        positions.append((left + index * (width + gap), top_y))
    for index in range(5):
        positions.append((left + (4 - index) * (width + gap), bottom_y))

    parts = [
        '<svg class="paper-diagram" xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1328 474" role="img" '
        'aria-label="Predictive gimbal research development timeline">',
        _arrow_marker(),
        '<rect x="1" y="1" width="1326" height="472" rx="18" '
        'fill="#fbfcfe" stroke="#d8e0e8" stroke-width="2"/>',
        '<text x="28" y="34" font-family="DejaVu Sans, sans-serif" '
        'font-size="17" font-weight="700" fill="#43566a">FROM SYNTHETIC CONCEPT TO MEASUREMENT-AND-TRANSFER GATE</text>',
    ]

    # Draw arrows first so boxes remain visually dominant.
    for index in range(5):
        x, y = positions[index]
        next_x, next_y = positions[index + 1]
        parts.append(_svg_line(x + width, y + height / 2, next_x, next_y + height / 2))
    x, y = positions[5]
    next_x, next_y = positions[6]
    parts.append(
        f'<path d="M {x + width / 2} {y + height} L {x + width / 2} '
        f'{next_y - 25} L {next_x + width / 2} {next_y - 25} '
        f'L {next_x + width / 2} {next_y}" fill="none" stroke="#506274" '
        'stroke-width="2.7" marker-end="url(#arrow)"/>'
    )
    for index in range(6, 10):
        x, y = positions[index]
        next_x, next_y = positions[index + 1]
        parts.append(_svg_line(x, y + height / 2, next_x + width, next_y + height / 2))

    for stage, (x, y) in zip(stages, positions, strict=True):
        number, title, subtitle, state = stage
        fill, stroke = palette[state]
        parts.append(
            _svg_box(
                x,
                y,
                width,
                height,
                (f"{number}  {title}", subtitle),
                fill=fill,
                stroke=stroke,
                font_size=15,
            )
        )

    parts.extend(
        [
            '<circle cx="34" cy="435" r="8" fill="#188977"/><text x="50" y="440" '
            'font-family="DejaVu Sans, sans-serif" font-size="14" fill="#43566a">accepted evidence</text>',
            '<circle cx="232" cy="435" r="8" fill="#cf5c5c"/><text x="248" y="440" '
            'font-family="DejaVu Sans, sans-serif" font-size="14" fill="#43566a">rejected by fresh test</text>',
            '<circle cx="466" cy="435" r="8" fill="#c38a18"/><text x="482" y="440" '
            'font-family="DejaVu Sans, sans-serif" font-size="14" fill="#43566a">current frontier</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _math_atom(value: str) -> str:
    return value if value.startswith("<") else f"<mi>{value}</mi>"


def _sub(base: str, subscript: str) -> str:
    return f"<msub>{_math_atom(base)}{_math_atom(subscript)}</msub>"


def _hat(base: str) -> str:
    return f'<mover accent="true">{_math_atom(base)}<mo>^</mo></mover>'


def _dot(base: str) -> str:
    return f'<mover accent="true">{_math_atom(base)}<mo>˙</mo></mover>'


def _display_math(latex: str) -> str:
    """Render the paper's three display equations with native MathML."""

    normalized = " ".join(line.strip() for line in latex.splitlines())
    comma = "<mo>,</mo>"
    if normalized.startswith("o_t=(e_t,w_t"):
        terms = (
            _sub("e", "t"),
            _sub("w", "t"),
            _sub("h", "t"),
            _sub("c", "t"),
            _sub("m", "t"),
            "<mrow><mi>Δ</mi><msub><mi>t</mi><mi>t</mi></msub></mrow>",
            _sub("q", "t"),
            f"<msub>{_dot('q')}<mi>t</mi></msub>",
            '<msubsup><mi>ω</mi><mi>t</mi><mtext>body</mtext></msubsup>',
            '<msub><mi>u</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub>',
        )
        body = comma.join(terms)
        mathml = f"{_sub('o', 't')}<mo>=</mo><mo>(</mo>{body}<mo>)</mo>"
    elif normalized.startswith(r"\hat s_t=(\hat\theta_t"):
        target = '<mtext>target/body</mtext>'
        terms = (
            f'<msubsup>{_hat("θ")}<mi>t</mi>{target}</msubsup>',
            f'<msubsup>{_dot(_hat("θ"))}<mi>t</mi>{target}</msubsup>',
            '<msub><mi>σ</mi><mrow><mi>θ</mi><mo>,</mo><mi>t</mi></mrow></msub>',
            '<msub><mi>σ</mi><mrow><mover><mi>θ</mi><mo>˙</mo></mover><mo>,</mo><mi>t</mi></mrow></msub>',
            _sub("m", "t"),
        )
        mathml = (
            f'<msub>{_hat("s")}<mi>t</mi></msub><mo>=</mo><mo>(</mo>'
            + comma.join(terms)
            + "<mo>)</mo>"
        )
    elif normalized.startswith(r"u_t^{rate}=\operatorname{clip}"):
        bearing_error = (
            f'<mo>(</mo><msub>{_hat("θ")}<mi>t</mi></msub>'
            f'<mo>−</mo>{_sub("q", "t")}<mo>)</mo>'
        )
        numerator = (
            f'<msub>{_dot(_hat("θ"))}<mi>t</mi></msub><mo>+</mo>'
            f'<msub><mi>k</mi><mi>p</mi></msub>{bearing_error}'
        )
        mathml = (
            '<msubsup><mi>u</mi><mi>t</mi><mtext>rate</mtext></msubsup>'
            '<mo>=</mo><mi mathvariant="normal">clip</mi><mo>(</mo>'
            f'<mfrac><mrow>{numerator}</mrow><msub>{_dot("q")}<mtext>max</mtext></msub></mfrac>'
            '<mo>,</mo><mo>−</mo><mn>1</mn><mo>,</mo><mn>1</mn><mo>)</mo>'
        )
    else:
        return f'<pre class="equation-fallback">{html.escape(latex)}</pre>'
    return (
        '<div class="equation" aria-label="'
        + html.escape(normalized, quote=True)
        + '"><math display="block" xmlns="http://www.w3.org/1998/Math/MathML"><mrow>'
        + mathml
        + "</mrow></math></div>"
    )


def _slug(text: str) -> str:
    plain = re.sub(r"[*_`]", "", text).lower()
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-")


def _resolve_url(url: str, source_dir: Path) -> str:
    if re.match(r"^[a-z]+://", url) or url.startswith("#"):
        return url
    path = (source_dir / url).resolve()
    return path.as_uri()


def _inline(text: str, source_dir: Path) -> str:
    placeholders: list[str] = []

    def protect(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    # Preserve intentional line breaks before escaping the remaining source.
    text = re.sub(r"<br\s*/?>", lambda _: protect("<br>"), text, flags=re.I)
    text = html.escape(text, quote=False)

    image_pattern = re.compile(r"!\[([^]]*)\]\(([^)\s]+)(?:\s+&quot;.*?&quot;)?\)")
    text = image_pattern.sub(
        lambda match: protect(
            '<img src="'
            + html.escape(_resolve_url(match.group(2), source_dir), quote=True)
            + '" alt="'
            + html.escape(match.group(1), quote=True)
            + '">'
        ),
        text,
    )
    link_pattern = re.compile(r"\[([^]]+)\]\(([^)]+)\)")
    text = link_pattern.sub(
        lambda match: protect(
            '<a href="'
            + html.escape(_resolve_url(match.group(2), source_dir), quote=True)
            + '">'
            + html.escape(html.unescape(match.group(1)))
            + "</a>"
        ),
        text,
    )
    text = re.sub(
        r"`([^`]+)`",
        lambda match: protect(f"<code>{html.escape(html.unescape(match.group(1)))}</code>"),
        text,
    )
    text = re.sub(
        r"\$([^$]+)\$",
        lambda match: protect(
            '<span class="math-inline">'
            + html.escape(html.unescape(match.group(1)))
            .replace(r"\in", "∈")
            .replace(r"\pm", "±")
            .replace("_t", "ₜ")
            + "</span>"
        ),
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _is_special(line: str, next_line: str = "") -> bool:
    return bool(
        not line.strip()
        or re.match(r"^#{1,3}\s+", line)
        or line.startswith("```")
        or line.strip() == "---"
        or line.strip() == r"\["
        or line.startswith(">")
        or re.match(r"^\s*(?:[-*]|\d+\.)\s+", line)
        or (line.lstrip().startswith("|") and _is_table_separator(next_line))
        or re.fullmatch(r"!\[[^]]*\]\([^)]+\)", line.strip())
    )


def _render_markdown(source: str, source_dir: Path) -> tuple[str, list[tuple[str, str]]]:
    lines = source.splitlines()
    output: list[str] = []
    toc: list[tuple[str, str]] = []
    mermaid_index = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        next_line = lines[index + 1] if index + 1 < len(lines) else ""

        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            identifier = _slug(title)
            if level == 1:
                output.append(
                    '<header class="paper-masthead">'
                    '<div class="journal-line">AUTONOMOUS OBSERVATION LAB · RESEARCH REPORT</div>'
                    f'<h1 id="{identifier}">{_inline(title, source_dir)}</h1>'
                    '<div class="title-rule"></div>'
                    "</header>"
                )
            else:
                output.append(
                    f'<h{level} id="{identifier}">{_inline(title, source_dir)}</h{level}>'
                )
                if level == 2:
                    toc.append((identifier, re.sub(r"^\d+\.\s*", "", title)))
            index += 1
            continue

        if line.startswith("```"):
            language = line.removeprefix("```").strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            if language == "mermaid":
                diagram = _architecture_svg() if mermaid_index == 0 else _journey_svg()
                mermaid_index += 1
                output.append(f'<div class="visual-block diagram-block">{diagram}</div>')
            else:
                class_name = f' class="language-{html.escape(language, quote=True)}"' if language else ""
                output.append(
                    f"<pre><code{class_name}>{html.escape(chr(10).join(code_lines))}</code></pre>"
                )
            continue

        if stripped == r"\[":
            math_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                math_lines.append(lines[index])
                index += 1
            index += 1
            output.append(_display_math("\n".join(math_lines)))
            continue

        if stripped == "---":
            output.append('<hr class="section-rule">')
            index += 1
            continue

        if line.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].startswith(">"):
                quote_lines.append(lines[index].lstrip("> "))
                index += 1
            output.append(
                f'<blockquote>{_inline(" ".join(quote_lines), source_dir)}</blockquote>'
            )
            continue

        if line.lstrip().startswith("|") and _is_table_separator(next_line):
            headers = _table_cells(line)
            alignments = _table_cells(next_line)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(_table_cells(lines[index]))
                index += 1
            head = "".join(
                f'<th class="align-{"center" if rule.startswith(":") and rule.endswith(":") else "right" if rule.endswith(":") else "left"}">'
                f"{_inline(value, source_dir)}</th>"
                for value, rule in zip(headers, alignments, strict=False)
            )
            body_rows = []
            for row in rows:
                cells = "".join(
                    f'<td class="align-{"center" if rule.startswith(":") and rule.endswith(":") else "right" if rule.endswith(":") else "left"}">'
                    f"{_inline(value, source_dir)}</td>"
                    for value, rule in zip(row, alignments, strict=False)
                )
                body_rows.append(f"<tr>{cells}</tr>")
            output.append(
                '<div class="table-wrap"><table><thead><tr>'
                + head
                + "</tr></thead><tbody>"
                + "".join(body_rows)
                + "</tbody></table></div>"
            )
            continue

        list_match = re.match(r"^\s*(-|\d+\.)\s+(.+)$", line)
        if list_match:
            ordered = list_match.group(1) != "-"
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while index < len(lines):
                item_match = re.match(
                    r"^\s*(\d+\.|-)\s+(.+)$",
                    lines[index],
                )
                if not item_match or (item_match.group(1) != "-") != ordered:
                    break
                item_lines = [item_match.group(2).strip()]
                index += 1
                while index < len(lines) and re.match(r"^\s{2,}\S", lines[index]):
                    item_lines.append(lines[index].strip())
                    index += 1
                items.append(" ".join(item_lines))
                if index < len(lines) and not lines[index].strip():
                    probe = index + 1
                    if probe < len(lines) and re.match(r"^\s*(\d+\.|-)\s+", lines[probe]):
                        index = probe
                    else:
                        break
            output.append(
                f"<{tag}>"
                + "".join(f"<li>{_inline(item, source_dir)}</li>" for item in items)
                + f"</{tag}>"
            )
            continue

        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            output.append(
                '<div class="visual-block chart-block"><img src="'
                + html.escape(_resolve_url(image_match.group(2), source_dir), quote=True)
                + '" alt="'
                + html.escape(image_match.group(1), quote=True)
                + '"></div>'
            )
            index += 1
            continue

        # Fold an ordinary Markdown paragraph until the next block construct.
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if _is_special(lines[index], following):
                break
            paragraph_lines.append(lines[index].strip())
            index += 1
        paragraph = " ".join(paragraph_lines)
        class_name = ' class="figcaption"' if paragraph.startswith("**Figure ") else ""
        output.append(f"<p{class_name}>{_inline(paragraph, source_dir)}</p>")

    rendered = "\n".join(output)
    # Keep each visual and its following Figure caption on the same page.
    rendered = re.sub(
        r'(<div class="visual-block[^>]*>.*?</div>)\s*(<p class="figcaption">.*?</p>)',
        r'<figure class="paper-figure">\1\2</figure>',
        rendered,
        flags=re.S,
    )
    return rendered, toc


def _stylesheet() -> str:
    return r"""
@page {
  size: A4;
  margin: 17mm 16mm 19mm 16mm;
}

* { box-sizing: border-box; }
html {
  color-scheme: light;
  print-color-adjust: exact;
  -webkit-print-color-adjust: exact;
}
body {
  margin: 0;
  color: #1d2b38;
  background: #fff;
  font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif;
  font-size: 9.35pt;
  line-height: 1.47;
  text-rendering: optimizeLegibility;
}
.paper-masthead {
  margin: -2mm 0 7mm;
  padding: 7mm 8mm 8mm;
  border-top: 2.1mm solid #244c6f;
  background: linear-gradient(135deg, #f5f8fb 0%, #edf4f8 100%);
  break-inside: avoid;
}
.journal-line {
  margin-bottom: 6mm;
  color: #3977b8;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 7.5pt;
  font-weight: 700;
  letter-spacing: 0.12em;
}
h1 {
  margin: 0;
  color: #18334a;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 25pt;
  font-weight: 700;
  line-height: 1.12;
  letter-spacing: -0.025em;
}
.title-rule {
  width: 25mm;
  margin-top: 5mm;
  border-bottom: 1.4mm solid #9b59b6;
}
h2, h3 {
  color: #244c6f;
  font-family: "DejaVu Sans", sans-serif;
  break-after: avoid;
}
h2 {
  margin: 7mm 0 3mm;
  padding-bottom: 1.2mm;
  border-bottom: 0.35mm solid #b8c8d6;
  font-size: 15.2pt;
  line-height: 1.2;
}
h3 {
  margin: 5mm 0 2mm;
  font-size: 11.3pt;
  line-height: 1.25;
}
p {
  margin: 0 0 3.2mm;
  orphans: 3;
  widows: 3;
}
p:first-of-type { margin-top: 0; }
strong { color: #152b3d; }
a {
  color: #2f6f9f;
  text-decoration: none;
  overflow-wrap: anywhere;
}
code {
  padding: 0.15mm 0.75mm;
  border-radius: 1mm;
  background: #eef2f6;
  color: #5d3569;
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.88em;
}
pre {
  margin: 3mm 0 4mm;
  padding: 3.5mm 4mm;
  border-left: 1.1mm solid #3977b8;
  border-radius: 1.5mm;
  background: #f2f5f8;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}
pre code { padding: 0; background: transparent; color: #243748; font-size: 7.7pt; }
blockquote {
  margin: 4mm 7mm 4.5mm;
  padding: 3.5mm 5mm;
  border-left: 1.4mm solid #9b59b6;
  background: #f8f2fa;
  color: #394655;
  font-size: 10pt;
  font-style: italic;
  break-inside: avoid;
}
ul, ol {
  margin: 1.5mm 0 3.5mm 5.5mm;
  padding-left: 4.5mm;
}
li { margin: 0 0 1.2mm; padding-left: 1mm; }
li::marker { color: #3977b8; font-family: "DejaVu Sans", sans-serif; font-weight: 700; }
.section-rule {
  margin: 5mm 0;
  border: 0;
  border-top: 0.35mm solid #c7d2dd;
}
.toc {
  margin: 7mm 0 8mm;
  padding: 5mm 6mm;
  border: 0.3mm solid #c9d6e0;
  border-radius: 2mm;
  background: #f8fafc;
  break-inside: avoid;
}
.toc-title {
  margin-bottom: 3mm;
  color: #244c6f;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.toc ol {
  columns: 2;
  column-gap: 10mm;
  margin: 0;
  padding-left: 5mm;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 8pt;
}
.toc li { break-inside: avoid; margin-bottom: 1.2mm; }
.toc a { color: #38556d; }
.table-wrap {
  margin: 3mm 0 4.5mm;
  break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 7.45pt;
  line-height: 1.3;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  padding: 2.1mm 2mm;
  border: 0.25mm solid #68859d;
  background: #315f81;
  color: #fff;
  font-weight: 700;
  vertical-align: bottom;
}
td {
  padding: 1.7mm 2mm;
  border: 0.23mm solid #c5d0da;
  vertical-align: top;
}
tbody tr:nth-child(even) td { background: #f3f6f8; }
.align-left { text-align: left; }
.align-center { text-align: center; }
.align-right { text-align: right; font-variant-numeric: tabular-nums; }
.paper-figure {
  display: block;
  margin: 5mm 0 5.5mm;
  break-inside: avoid;
}
.visual-block {
  width: 100%;
  break-inside: avoid;
}
.chart-block img {
  display: block;
  width: 100%;
  max-height: 157mm;
  object-fit: contain;
}
.diagram-block {
  padding: 1.5mm;
  border: 0.25mm solid #d7e0e8;
  border-radius: 2mm;
  background: #fbfcfe;
}
.paper-diagram { display: block; width: 100%; height: auto; }
.figcaption {
  margin: 2.2mm 4mm 0;
  color: #455565;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 7.45pt;
  line-height: 1.38;
  text-align: left;
}
.equation {
  margin: 3mm 0 4mm;
  padding: 2.6mm 4mm;
  border-radius: 1.5mm;
  background: #f8fafc;
  text-align: center;
  break-inside: avoid;
}
.equation math { font-size: 13.5pt; }
.math-inline {
  white-space: nowrap;
  font-family: "DejaVu Serif", serif;
  font-style: italic;
}
.equation-fallback { text-align: center; font-family: "DejaVu Serif", serif; }
#results, #recommended-next-phase, #references { break-before: page; }
"""


def _html_document(body: str, toc: list[tuple[str, str]]) -> str:
    toc_items = "".join(
        f'<li><a href="#{identifier}">{html.escape(title)}</a></li>'
        for identifier, title in toc
    )
    toc_html = (
        '<nav class="toc"><div class="toc-title">Contents</div><ol>'
        + toc_items
        + "</ol></nav>"
    )
    # The source has one rule between its front matter and section 1. Place the
    # generated table of contents there without changing the canonical Markdown.
    body = body.replace('<hr class="section-rule">', toc_html, 1)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="author" content="Autonomous Observation Lab">
  <meta name="description" content="Technical retrospective for predictive one-axis gimbal servoing research.">
  <title>Dream-to-Center: Predictive Gimbal Servoing Research Journey</title>
  <style>{_stylesheet()}</style>
</head>
<body>
  <main>{body}</main>
</body>
</html>
"""


def _find_browser(requested: str | None) -> str:
    if requested:
        path = shutil.which(requested) or requested
        if Path(path).exists():
            return str(path)
        raise SystemExit(f"Browser not found: {requested}")
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit(
        "A Chromium-family browser is required (google-chrome or chromium)."
    )


def _run_browser(browser: str, html_path: Path, output: Path, profile: Path) -> None:
    command = [
        browser,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={output}",
        html_path.as_uri(),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(f"PDF rendering failed ({completed.returncode}): {details}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--browser",
        help="Chromium-family executable (auto-detected by default)",
    )
    parser.add_argument(
        "--keep-html",
        type=Path,
        help="Also write the intermediate print-ready HTML to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"Markdown source not found: {source}")
    body, toc = _render_markdown(source.read_text(encoding="utf-8"), source.parent)
    document = _html_document(body, toc)
    if args.keep_html:
        keep_html = args.keep_html.resolve()
        keep_html.parent.mkdir(parents=True, exist_ok=True)
        keep_html.write_text(document, encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    browser = _find_browser(args.browser)
    with tempfile.TemporaryDirectory(prefix="aol-gimbal-paper-") as temp_dir:
        temporary = Path(temp_dir)
        html_path = temporary / "paper.html"
        html_path.write_text(document, encoding="utf-8")
        _run_browser(browser, html_path, output, temporary / "chrome-profile")
    if not output.is_file() or output.stat().st_size < 10_000:
        raise SystemExit(f"Browser did not create a valid-looking PDF: {output}")
    if output.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"Output is not a PDF: {output}")
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
