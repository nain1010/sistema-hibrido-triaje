import markdown_it
from pathlib import Path

def convertir_md_a_html():
    md_path = Path("INFORME_TECNICO_FINAL_PROYECTO.md")
    if not md_path.exists():
        print("No se encontró el archivo Markdown.")
        return

    md_text = md_path.read_text(encoding="utf-8")
    md = markdown_it.MarkdownIt()
    html_body = md.render(md_text)

    plantilla = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Técnico Final - Sistema Multimodal de Triaje Clínico</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #1f2937;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 30px;
  }
  h1 { color: #1e3a8a; border-bottom: 2px solid #2563eb; padding-bottom: 0.3em; margin-top: 1.5em; font-size: 26px; }
  h2 { color: #1e40af; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.2em; margin-top: 1.3em; font-size: 20px; }
  h3 { color: #334155; margin-top: 1.1em; font-size: 16px; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 12px;
  }
  table, th, td {
    border: 1px solid #cbd5e1;
  }
  th, td {
    padding: 8px 10px;
    text-align: left;
  }
  th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
  }
  tr:nth-child(even) { background-color: #f8fafc; }
  blockquote {
    border-left: 4px solid #3b82f6;
    padding: 10px 15px;
    color: #1e40af;
    background-color: #eff6ff;
    margin: 15px 0;
    border-radius: 0 6px 6px 0;
  }
  code {
    background-color: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 90%;
    color: #0f172a;
  }
  pre {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
  }
  @media print {
    body { max-width: 100%; margin: 12mm; padding: 0; font-size: 10.5pt; }
    h1 { page-break-before: always; }
    h1:first-of-type { page-break-before: avoid; }
    table { page-break-inside: avoid; }
  }
</style>
</head>
<body>
""" + html_body + """
</body>
</html>"""

    Path("INFORME_TECNICO_FINAL_PROYECTO.html").write_text(plantilla, encoding="utf-8")
    print("INFORME_TECNICO_FINAL_PROYECTO.html generado con éxito.")

if __name__ == "__main__":
    convertir_md_a_html()
