from bs4 import BeautifulSoup
from pathlib import Path
import re


def keep_tables_compact_single_line(
    input_path: str,
    output_path: str
) -> None:
    """
    Read an HTML file, keep only <table> elements,
    remove all HTML attributes, normalize whitespace,
    and output a single-line HTML document.

    Args:
        input_path: Path to the input HTML file
        output_path: Path to the cleaned HTML file
    """
    html = Path(input_path).read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    cleaned_soup = BeautifulSoup("<html><body></body></html>", "html.parser")

    for table in tables:
        # Strip all attributes
        for tag in table.find_all(True):
            tag.attrs = {}

        # Normalize text nodes
        for text_node in table.find_all(string=True):
            text = re.sub(r"\s+", " ", text_node).strip()
            if text:
                text_node.replace_with(text)
            else:
                text_node.extract()

        cleaned_soup.body.append(table)

    # Convert to string and remove all newlines and excess spaces between tags
    html_output = str(cleaned_soup)
    html_output = re.sub(r">\s+<", "><", html_output)
    html_output = re.sub(r"\s{2,}", " ", html_output)
    html_output = html_output.strip()

    Path(output_path).write_text(html_output, encoding="utf-8")


def get_clean_prices_markdown(input_path: str, output_path: str) -> None:
    """
    Read an HTML file, extract <table> elements,
    and convert them to Markdown tables.

    Args:
        input_path: Path to the input HTML file
        output_path: Path to the output Markdown file
    """
    html = Path(input_path).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    markdown_tables = []

    for table in soup.find_all("table"):
        rows = []

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [
                re.sub(r"\s+", " ", cell.get_text(strip=True))
                for cell in cells
            ]
            if row:
                rows.append(row)

        if not rows:
            continue

        # Header handling
        header = rows[0]
        separator = ["---"] * len(header)

        md = []
        md.append("| " + " | ".join(header) + " |")
        md.append("| " + " | ".join(separator) + " |")

        for row in rows[1:]:
            # Pad or trim rows to header length
            row = row[:len(header)] + [""] * (len(header) - len(row))
            md.append("| " + " | ".join(row) + " |")

        markdown_tables.append("\n".join(md))

    output_md = "\n\n".join(markdown_tables)
    Path(output_path).write_text(output_md, encoding="utf-8")


if __name__ == "__main__":
    html_tables_to_markdown(
        input_path="scraped_content/fireworks.html",
        output_path="scraped_content/fireworks_clean.md"
    )
