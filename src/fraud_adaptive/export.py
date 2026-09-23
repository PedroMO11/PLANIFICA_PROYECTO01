"""Exportacion del informe a PDF con un limite de paginas.

El enunciado fija un maximo de ocho paginas, incluidas las referencias. El limite
se verifica sobre el PDF generado y, en modo estricto, la exportacion falla si se
excede.

La conversion va de Markdown a HTML con estilos de impresion y de ahi a PDF con el
motor headless de un navegador, que respeta los saltos de pagina definidos en CSS.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("fraud_adaptive.export")

MAX_PAGES = 8

# Motores headless en orden de preferencia.
BROWSER_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
]

PRINT_CSS = """
@page {
  size: A4;
  margin: 13mm 14mm 14mm 14mm;
  @bottom-right { content: counter(page); }
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 8.1pt;
  line-height: 1.26;
  color: #12110f;
  margin: 0;
}
h1 { font-size: 15pt; margin: 0 0 3pt; color: #0b0b0b; letter-spacing: -0.2pt; }
h2 {
  font-size: 10.2pt; margin: 0 0 5pt; padding-top: 4pt; color: #0b0b0b;
  border-top: 1.4pt solid #2a78d6;
  /* Cada seccion H2 empieza en una pagina nueva. */
  break-before: page; page-break-before: always;
}
h2:first-of-type { break-before: avoid; page-break-before: avoid; }
h3 { font-size: 9pt; margin: 7pt 0 3pt; color: #1a1918; }
p { margin: 0 0 3.8pt; text-align: justify; }
ul, ol { margin: 0 0 4.5pt; padding-left: 13pt; }
li { margin-bottom: 1.5pt; }
strong { color: #0b0b0b; }
code {
  font-family: Consolas, "Courier New", monospace;
  font-size: 7.4pt; background: #f4f4f2; padding: 0 2pt; border-radius: 2pt;
}
pre {
  background: #f7f7f5; border-left: 2.2pt solid #2a78d6;
  padding: 4pt 6pt; margin: 4pt 0; overflow: hidden;
  font-family: Consolas, "Courier New", monospace; font-size: 6.9pt; line-height: 1.24;
  break-inside: avoid; page-break-inside: avoid;
}
table {
  border-collapse: collapse; width: 100%; margin: 4pt 0 6pt;
  font-size: 6.9pt; break-inside: avoid; page-break-inside: avoid;
}
th {
  background: #eef4fb; color: #0b0b0b; font-weight: 700;
  text-align: left; padding: 2.6pt 3.4pt; border-bottom: 1pt solid #2a78d6;
}
td { padding: 2.4pt 3.4pt; border-bottom: 0.5pt solid #e3e2de; vertical-align: top; }
tr:nth-child(even) td { background: #fafaf9; }
blockquote {
  margin: 5pt 0; padding: 5pt 8pt; background: #fff6ee;
  border-left: 2.6pt solid #eb6834; font-size: 7.8pt;
  break-inside: avoid; page-break-inside: avoid;
}
blockquote p { margin-bottom: 3pt; }
/* Las figuras son paneles 2x2 o 1x3; esta altura mantiene legibles sus ejes. */
img { max-width: 100%; max-height: 72mm; display: block; margin: 5pt auto; }
em { color: #52514e; }
hr { border: none; border-top: 0.6pt solid #e3e2de; margin: 6pt 0; }
h1 + p em, p > em:only-child { display: block; text-align: center; font-size: 7pt; color: #6b6a66; }
"""


def find_browser() -> str | None:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def markdown_to_html(markdown_text: str, *, base_dir: Path, title: str) -> str:
    """Convierte Markdown a un HTML autocontenido con estilos de impresion."""
    import markdown as md

    body = md.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
        output_format="html5",
    )
    # Rutas file:// absolutas, porque el motor headless no resuelve rutas relativas.
    body = body.replace('src="figures/', 'src="%s/figures/' % base_dir.resolve().as_uri())
    return (
        "<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n"
        "<meta charset=\"utf-8\">\n<title>%s</title>\n<style>%s</style>\n"
        "</head>\n<body>\n%s\n</body>\n</html>\n" % (title, PRINT_CSS, body)
    )


def html_to_pdf(html_path: Path, pdf_path: Path, *, timeout: int = 180) -> bool:
    browser = find_browser()
    if browser is None:
        LOGGER.error("No se encontro un motor headless para imprimir el PDF")
        return False

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--print-to-pdf=%s" % str(pdf_path.resolve()),
        html_path.resolve().as_uri(),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.error("Fallo al invocar el motor headless: %s", exc)
        return False
    if not pdf_path.exists():
        LOGGER.error("El motor no produjo un PDF: %s", result.stderr[-400:])
        return False
    return True


def count_pdf_pages(pdf_path: Path) -> int | None:
    try:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            return document.page_count
    except ImportError:
        # Sin pymupdf se cuentan los objetos /Type /Page.
        try:
            raw = pdf_path.read_bytes()
            return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
        except OSError:
            return None
    except Exception:  # noqa: BLE001 - un fallo del recuento no detiene la exportacion
        return None


def export_report(
    markdown_path: str | Path,
    pdf_path: str | Path,
    *,
    title: str = "Informe final",
    max_pages: int = MAX_PAGES,
    strict: bool = True,
) -> dict[str, Any]:
    """Exporta el informe y verifica el limite de paginas.

    Con ``strict``, exceder el limite de ocho paginas del enunciado es un error.
    """
    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path)
    text = markdown_path.read_text(encoding="utf-8")

    html = markdown_to_html(text, base_dir=markdown_path.parent, title=title)
    html_path = markdown_path.with_suffix(".print.html")
    html_path.write_text(html, encoding="utf-8")

    produced = html_to_pdf(html_path, pdf_path)
    result: dict[str, Any] = {
        "markdown": markdown_path.as_posix(),
        "html": html_path.as_posix(),
        "pdf": pdf_path.as_posix() if produced else None,
        "generado": produced,
        "motor": find_browser(),
    }
    if not produced:
        result["motivo"] = "no se encontro motor headless o la impresion fallo"
        return result

    pages = count_pdf_pages(pdf_path)
    result["paginas"] = pages
    result["limite"] = max_pages
    result["dentro_del_limite"] = pages is not None and pages <= max_pages

    if pages is not None and pages > max_pages:
        message = "El PDF tiene %d paginas y el maximo es %d" % (pages, max_pages)
        if strict:
            raise ValueError(
                message + ". Hay que reducir texto o figuras antes de entregar."
            )
        LOGGER.warning(message)
    return result
