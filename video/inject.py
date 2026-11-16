"""Inserta la línea de tiempo de la voz en index.html → render.html."""
import sys
from pathlib import Path
work = Path(sys.argv[1])
here = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent  # carpeta con index.html
html = (here / "index.html").read_text().replace("/*TIMELINE*/null", (work / "timeline.json").read_text())
assert "/*" + "TIMELINE" not in html
(work / "render.html").write_text(html)
