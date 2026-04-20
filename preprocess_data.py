#!/usr/bin/env python3
"""
EnisoData1 Preprocessor (Enhanced)
===================================
Converts all files in EnisoData1 (PDFs, DOC/DOCX, images) into structured
Markdown files with YAML frontmatter, optimised for RAG ingestion.

Enhancements:
- Smart table-to-Markdown-list conversion for timetable PDFs
- Cleaner output for LLM consumption

Usage:
    python preprocess_data.py
"""

import os
import re
import glob
import argparse
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENISO_DATA_DIR = os.path.join(
    BASE_DIR,
    "EnisoData1-20260418T142956Z-3-001",
    "EnisoData1",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

# ---------------------------------------------------------------------------
# Category detection from filename / path
# ---------------------------------------------------------------------------
CATEGORY_RULES = [
    (re.compile(r"emploi|groupes", re.I), "timetable"),
    (re.compile(r"calendrier", re.I), "calendar"),
    (re.compile(r"stage|rapport", re.I), "stage"),
    (re.compile(r"pfe|book", re.I), "pfe"),
    (re.compile(r"inscri|reinscri", re.I), "inscription"),
    (re.compile(r"guide|fiche|lettre", re.I), "guide"),
]

# Also detect from parent folder name
FOLDER_CATEGORY = {
    "EmploieDuTemps": "timetable",
    "Stage": "stage",
    "PFE": "pfe",
    "Inscri Et Reinscri": "inscription",
}


def detect_category(filepath: str) -> str:
    """Return a category string based on filename and parent folder."""
    basename = os.path.basename(filepath)
    parent = os.path.basename(os.path.dirname(filepath))

    # Check parent folder first
    cat = FOLDER_CATEGORY.get(parent)
    if cat:
        return cat

    # Check filename patterns
    for pattern, category in CATEGORY_RULES:
        if pattern.search(basename):
            return category

    return "general"


def extract_metadata(filepath: str) -> dict:
    """Extract metadata from filename heuristics."""
    basename = os.path.basename(filepath)
    meta = {
        "category": detect_category(filepath),
        "source_file": basename,
    }

    # Try to extract year
    year_match = re.search(r"(\d{4})[_-](\d{4})", basename)
    if year_match:
        meta["academic_year"] = f"{year_match.group(1)}-{year_match.group(2)}"

    # Try to extract semester
    sem_match = re.search(r"[_-](S[12])[_-]", basename, re.I)
    if sem_match:
        meta["semester"] = sem_match.group(1).upper()

    # Try to extract version
    ver_match = re.search(r"[_-]V(\d+)", basename, re.I)
    if ver_match:
        meta["version"] = f"V{ver_match.group(1)}"

    # Try to extract year level (1ère, 2ème, 3ème)
    level_match = re.search(r"(\d)[eè](?:re|me|ème)", basename, re.I)
    if level_match:
        meta["year_level"] = int(level_match.group(1))

    return meta


# ---------------------------------------------------------------------------
# Text extraction per file type
# ---------------------------------------------------------------------------
def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF, with OCR fallback for image-based pages."""
    from pypdf import PdfReader

    reader = PdfReader(filepath)
    pages = []
    empty_pages = 0

    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"## Page {i}\n\n{text}")
        else:
            empty_pages += 1

    # If most pages are empty, try OCR fallback
    if empty_pages > len(reader.pages) * 0.5 and not pages:
        print(f"  📷 PDF appears to be image-based, attempting OCR...")
        try:
            import subprocess
            # Try pdftotext (poppler) first
            result = subprocess.run(
                ["pdftotext", filepath, "-"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and result.stdout.strip():
                pages = [f"## Extracted Content (pdftotext)\n\n{result.stdout.strip()}"]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not pages:
        # Last resort: create a metadata-only entry so it's not completely lost
        basename = os.path.basename(filepath)
        num_pages = len(reader.pages)
        pages = [
            f"## {basename}\n\n"
            f"Ce document PDF contient {num_pages} pages mais le texte n'a pas pu être extrait "
            f"(document probablement scanné/image). "
            f"Fichier source: {basename}"
        ]

    return "\n\n---\n\n".join(pages)


def extract_text_from_doc(filepath: str) -> str:
    """Extract text from DOC/DOCX files. Tries python-docx first, then raw fallback."""
    # Try python-docx (works for .docx and some .doc)
    try:
        import docx
        doc = docx.Document(filepath)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                if para.style and para.style.name.startswith("Heading"):
                    level = para.style.name.replace("Heading ", "")
                    try:
                        level = int(level)
                    except ValueError:
                        level = 2
                    paragraphs.append(f"{'#' * (level + 1)} {text}")
                else:
                    paragraphs.append(text)

        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header_sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
                rows.insert(1, header_sep)
                paragraphs.append("\n".join(rows))

        if paragraphs:
            return "\n\n".join(paragraphs)
    except Exception:
        pass

    # Fallback: try antiword or catdoc for old .doc format
    import subprocess
    for cmd in ["antiword", "catdoc"]:
        try:
            result = subprocess.run(
                [cmd, filepath], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  📄 Extracted with {cmd}")
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Last resort: raw binary text extraction
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        # Extract readable ASCII/UTF-8 sequences (min 4 chars)
        text_chunks = re.findall(rb"[\x20-\x7E\xC0-\xFF]{4,}", raw)
        decoded = " ".join(chunk.decode("latin-1", errors="ignore") for chunk in text_chunks)
        # Clean up
        decoded = re.sub(r"\s+", " ", decoded).strip()
        if len(decoded) > 50:  # Only if we got meaningful content
            print(f"  📄 Extracted via raw text extraction")
            return f"(Contenu extrait du fichier Word ancien format)\n\n{decoded}"
    except Exception:
        pass

    print(f"  ⚠ Could not extract text from DOC file: {filepath}")
    basename = os.path.basename(filepath)
    return (
        f"Ce document ({basename}) est au format Word ancien (.doc). "
        f"Le contenu n'a pas pu être extrait automatiquement. "
        f"Fichier source: {basename}"
    )


def extract_text_from_image(filepath: str) -> str:
    """Extract text from images via OCR or return descriptive metadata."""
    basename = os.path.basename(filepath)
    text = f"[Image: {basename}]\n\n"

    try:
        from PIL import Image
        img = Image.open(filepath)
        width, height = img.size
        text += f"Image dimensions: {width}x{height}\n"
    except ImportError:
        pass

    try:
        import pytesseract
        from PIL import Image
        img = Image.open(filepath)
        ocr_text = pytesseract.image_to_string(img, lang="fra+eng")
        if ocr_text.strip():
            text += f"\n### Extracted Text (OCR)\n\n{ocr_text.strip()}"
        else:
            text += "\nNo text could be extracted from this image via OCR."
    except ImportError:
        text += "\nOCR not available (install pytesseract + Pillow)."
    except Exception as e:
        text += f"\nOCR failed: {e}"

    return text


# ---------------------------------------------------------------------------
# Smart Table-to-Markdown Converter
# ---------------------------------------------------------------------------
def _convert_table_blob_to_markdown_list(raw_text: str) -> str:
    """Convert raw PDF table dumps (concatenated cells) into
    LLM-friendly Markdown lists.

    Detects timetable patterns like:
    - Day + time slot + subject + professor + room
    - Attempts to split the blob into structured entries
    """
    # Pattern: time ranges like 08:30-10:00, 10:15-11:45, etc.
    time_pattern = re.compile(r"(\d{2}:\d{2})\s*[-–]\s*(\d{2}:\d{2})")
    # Pattern: days of the week (French)
    day_pattern = re.compile(
        r"(Lundi|Mardi|Mercredi|Jeudi|Vendredi|Samedi|Dimanche)",
        re.IGNORECASE,
    )
    # Pattern: room codes like A03, B22, E13, R02, M11, etc.
    room_pattern = re.compile(r"\b([A-Z]\d{2}(?:\s*\([^)]*\))?)\b")

    # If the text doesn't look like a timetable blob, return as-is
    times = time_pattern.findall(raw_text)
    days = day_pattern.findall(raw_text)

    if len(times) < 2 or len(days) < 2:
        return raw_text  # Not a timetable blob

    # Strategy: split by time slots to create structured entries
    lines = []
    # Split on time patterns
    segments = time_pattern.split(raw_text)

    current_time = None
    for i, seg in enumerate(segments):
        time_match = time_pattern.match(seg) if len(seg) < 15 else None
        if i > 0 and i % 3 == 1:
            # This is the start time
            start = segments[i] if i < len(segments) else ""
            end = segments[i + 1] if i + 1 < len(segments) else ""
            current_time = f"{start}-{end}"
            continue
        elif i > 0 and i % 3 == 2:
            continue  # end time, already captured

        if current_time and seg.strip():
            # Clean up the content
            content = seg.strip()
            # Remove excessive whitespace
            content = re.sub(r"\s{2,}", " | ", content)
            content = re.sub(r"---+", "", content)
            content = content.strip(" |")
            if content:
                lines.append(f"- **{current_time}** : {content}")

    if lines:
        return "\n".join(lines)

    return raw_text


def _clean_raw_timetable_text(raw_text: str) -> str:
    """Post-process raw timetable text to be more LLM-friendly.
    Converts dense table dumps into structured bullet lists."""
    # Split by pages
    page_pattern = re.compile(r"(## Page \d+)")
    parts = page_pattern.split(raw_text)

    cleaned_parts = []
    for part in parts:
        if page_pattern.match(part):
            cleaned_parts.append(part)
        else:
            # Process each page's content
            converted = _convert_table_blob_to_markdown_list(part)
            cleaned_parts.append(converted)

    return "\n\n".join(cleaned_parts)


# ---------------------------------------------------------------------------
# Formatting: category-specific post-processing
# ---------------------------------------------------------------------------
def format_timetable_content(raw_text: str, meta: dict) -> str:
    """Add contextual headers and smart-format timetable documents."""
    title = "# Emploi du Temps"
    if meta.get("semester"):
        title += f" — {meta['semester']}"
    if meta.get("version"):
        title += f" ({meta['version']})"
    if meta.get("academic_year"):
        title += f" — {meta['academic_year']}"

    description = (
        "Ce document contient les emplois du temps des groupes d'étudiants "
        "de l'ENISO (École Nationale d'Ingénieurs de Sousse).\n"
        "Chaque section correspond à une page du document original."
    )

    # Smart conversion: turn raw table blobs into structured lists
    formatted_text = _clean_raw_timetable_text(raw_text)

    return f"{title}\n\n{description}\n\n{formatted_text}"


def format_calendar_content(raw_text: str, meta: dict) -> str:
    """Add contextual headers for calendar/exam schedule documents."""
    title = "# Calendrier des Examens"
    if meta.get("year_level"):
        title += f" — {meta['year_level']}{'ère' if meta['year_level'] == 1 else 'ème'} Année"
    if meta.get("academic_year"):
        title += f" — {meta['academic_year']}"

    description = (
        "Ce document contient le calendrier des devoirs surveillés (DS) "
        "et examens de l'ENISO."
    )

    # Also apply the table-to-list converter for exam calendars
    formatted_text = _clean_raw_timetable_text(raw_text)

    return f"{title}\n\n{description}\n\n{formatted_text}"


def format_stage_content(raw_text: str, meta: dict) -> str:
    """Add contextual headers for stage-related documents."""
    title = "# Stage"
    basename = meta.get("source_file", "")
    if "guide" in basename.lower():
        title = "# Guide de Stage d'Été — ENISO"
    elif "lettre" in basename.lower():
        title = "# Lettre d'Appui pour Stage d'Été — ENISO"
    elif "fiche" in basename.lower():
        title = "# Fiche Entreprise pour Stage — ENISO"
    elif "rapport" in basename.lower() or "modele" in basename.lower():
        title = "# Modèle de Rapport de Stage d'Été — ENISO"

    return f"{title}\n\n{raw_text}"


def format_pfe_content(raw_text: str, meta: dict) -> str:
    """Add contextual headers for PFE documents."""
    title = "# Projet de Fin d'Études (PFE) — ENISO"
    basename = meta.get("source_file", "")
    if "book" in basename.lower():
        title = "# PFE Book — Catalogue des Projets de Fin d'Études — ENISO"
        description = (
            "Ce document contient la liste des sujets de PFE disponibles "
            "pour les étudiants de l'ENISO, avec les descriptions des projets "
            "et les encadrants."
        )
        return f"{title}\n\n{description}\n\n{raw_text}"

    return f"{title}\n\n{raw_text}"


def format_content(raw_text: str, meta: dict) -> str:
    """Apply category-specific formatting."""
    category = meta.get("category", "general")

    formatters = {
        "timetable": format_timetable_content,
        "calendar": format_calendar_content,
        "stage": format_stage_content,
        "pfe": format_pfe_content,
    }

    formatter = formatters.get(category)
    if formatter:
        return formatter(raw_text, meta)

    # Default: just add a title
    return f"# {meta.get('source_file', 'Document')}\n\n{raw_text}"


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def make_frontmatter(meta: dict) -> str:
    """Create YAML frontmatter string."""
    return "---\n" + yaml.dump(meta, allow_unicode=True, default_flow_style=False).strip() + "\n---\n\n"


def process_file(filepath: str) -> tuple[str, dict] | None:
    """Process a single file, return (content, metadata) or None."""
    ext = os.path.splitext(filepath)[1].lower()

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".doc": extract_text_from_doc,
        ".docx": extract_text_from_doc,
        ".png": extract_text_from_image,
        ".jpg": extract_text_from_image,
        ".jpeg": extract_text_from_image,
    }

    extractor = extractors.get(ext)
    if not extractor:
        print(f"  ⏭ Skipping unsupported file type: {filepath}")
        return None

    meta = extract_metadata(filepath)
    raw_text = extractor(filepath)

    if not raw_text or not raw_text.strip():
        print(f"  ⚠ No content extracted from: {filepath}")
        return None

    formatted = format_content(raw_text, meta)
    return formatted, meta


def safe_filename(name: str) -> str:
    """Convert a filename to a safe, lowercase slug."""
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name.lower().strip("_")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw ENISO files to LLM-friendly markdown."
    )
    parser.add_argument(
        "--input-dir",
        default=ENISO_DATA_DIR,
        help="Raw source directory containing PDFs/DOCs/images.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Destination directory for processed markdown files.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing markdown files in output before processing.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    print("=" * 60)
    print("  EnisoData1 Preprocessor (Enhanced)")
    print("=" * 60)

    if not os.path.exists(input_dir):
        print(f"\n❌ EnisoData1 directory not found: {input_dir}")
        print("   Please make sure EnisoData1 is in the expected location.")
        return

    os.makedirs(output_dir, exist_ok=True)

    if args.clean_output:
        deleted = 0
        for md in Path(output_dir).glob("*.md"):
            md.unlink()
            deleted += 1
        print(f"\n🧹 Cleaned output directory: removed {deleted} markdown files")

    # Collect all files recursively
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    print(f"\n📂 Found {len(all_files)} files in EnisoData1\n")

    processed = 0
    skipped = 0
    errors = 0

    for filepath in sorted(all_files):
        rel = os.path.relpath(filepath, input_dir)
        print(f"Processing: {rel}")

        try:
            result = process_file(filepath)
            if result is None:
                skipped += 1
                continue

            content, meta = result
            # Use relative path in output filename for deterministic overwrite behavior.
            # This avoids endless *_1, *_2 duplicates across reruns.
            rel_safe = safe_filename(rel.replace("\\", "_").replace("/", "_"))
            out_name = rel_safe + ".md"
            out_path = os.path.join(output_dir, out_name)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(make_frontmatter(meta))
                f.write(content)

            print(f"  ✅ → {out_name} (category: {meta['category']})")
            processed += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1

    print(f"\n{'=' * 60}")
    print(f"  Done! Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
