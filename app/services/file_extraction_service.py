from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _extension(filename_or_url: str) -> str:
    try:
        return Path(urlparse(filename_or_url).path or filename_or_url).suffix.lower()
    except Exception:
        return ""


def _base_result(
    *,
    input_kind: str,
    filename: str = "",
    url: str = "",
    content_type: str = "",
) -> Dict[str, Any]:
    return {
        "status": "not_started",
        "input_kind": input_kind,
        "url": url,
        "domain": _domain(url) if url else "",
        "title": filename or Path(urlparse(url).path).name or "Uploaded file",
        "author": "",
        "published_date": "",
        "text": "",
        "text_length": 0,
        "limitations": [],
        "metadata": {
            "source_type": input_kind,
            "filename": filename,
            "content_type": content_type,
        },
    }


def _decode_text_bytes(data: bytes) -> Dict[str, Any]:
    limitations: List[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            return {"text": text, "encoding": encoding, "limitations": limitations}
        except Exception:
            continue
    limitations.append("Could not confidently decode text file; used replacement decoding.")
    return {"text": data.decode("utf-8", errors="replace"), "encoding": "utf-8-replace", "limitations": limitations}


def extract_text_file_bytes(
    data: bytes,
    *,
    filename: str = "",
    content_type: str = "",
    input_kind: str = "text_file_upload",
    url: str = "",
) -> Dict[str, Any]:
    result = _base_result(input_kind=input_kind, filename=filename, url=url, content_type=content_type)
    decoded = _decode_text_bytes(data)
    text = _clean(decoded.get("text", ""))
    limitations = list(decoded.get("limitations", []))

    if len(text) < 40:
        limitations.append("Extracted text is very short; verification may be unreliable.")

    result.update(
        {
            "status": "success" if text else "failed",
            "text": text,
            "text_length": len(text),
            "limitations": limitations,
            "metadata": {
                **result["metadata"],
                "encoding": decoded.get("encoding", ""),
            },
        }
    )
    return result


def extract_pdf_bytes(
    data: bytes,
    *,
    filename: str = "",
    content_type: str = "application/pdf",
    input_kind: str = "pdf_upload",
    url: str = "",
    max_pages: int = 25,
) -> Dict[str, Any]:
    result = _base_result(input_kind=input_kind, filename=filename, url=url, content_type=content_type)
    limitations: List[str] = []

    try:
        from pypdf import PdfReader
    except Exception as error:
        result.update(
            {
                "status": "unsupported",
                "limitations": [
                    "PDF extraction requires the optional package `pypdf`.",
                    "Install it with: pip install pypdf",
                    f"Import error: {error}",
                ],
            }
        )
        return result

    try:
        reader = PdfReader(BytesIO(data))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)
        page_texts: List[str] = []
        for index in range(pages_to_read):
            try:
                page_text = reader.pages[index].extract_text() or ""
            except Exception as page_error:
                limitations.append(f"Could not extract text from page {index + 1}: {page_error}")
                page_text = ""
            if page_text.strip():
                page_texts.append(page_text)

        if total_pages > max_pages:
            limitations.append(f"PDF has {total_pages} pages; only first {max_pages} pages were analyzed for speed.")

        text = _clean("\n".join(page_texts))
        if not text:
            limitations.append(
                "No embedded text was found in the PDF. It may be scanned/image-only; use screenshot OCR or paste text."
            )
        elif len(text) < 40:
            limitations.append("Extracted PDF text is very short; verification may be unreliable.")

        result.update(
            {
                "status": "success" if text else "failed",
                "text": text,
                "text_length": len(text),
                "limitations": limitations,
                "metadata": {
                    **result["metadata"],
                    "total_pages": total_pages,
                    "pages_analyzed": pages_to_read,
                    "extractor": "pypdf",
                },
            }
        )
        return result
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "limitations": [f"PDF extraction failed: {error}"],
            }
        )
        return result


def extract_image_bytes(
    data: bytes,
    *,
    filename: str = "",
    content_type: str = "",
    input_kind: str = "image_upload",
    url: str = "",
) -> Dict[str, Any]:
    result = _base_result(input_kind=input_kind, filename=filename, url=url, content_type=content_type)

    try:
        from PIL import Image
    except Exception as error:
        result.update(
            {
                "status": "unsupported",
                "limitations": [
                    "Image OCR requires Pillow.",
                    "Install it with: pip install pillow",
                    f"Import error: {error}",
                ],
            }
        )
        return result

    try:
        import pytesseract
    except Exception as error:
        result.update(
            {
                "status": "unsupported",
                "limitations": [
                    "Image OCR requires pytesseract and the Tesseract OCR engine installed on the system.",
                    "Python package: pip install pytesseract pillow",
                    "Windows system dependency: install Tesseract OCR and ensure tesseract.exe is in PATH.",
                    f"Import error: {error}",
                ],
            }
        )
        return result

    limitations: List[str] = []
    try:
        image = Image.open(BytesIO(data))
        width, height = image.size
        text = _clean(pytesseract.image_to_string(image) or "")
        if not text:
            limitations.append("OCR did not extract readable text from the image.")
        elif len(text) < 40:
            limitations.append("OCR extracted text is very short; verification may be unreliable.")
        result.update(
            {
                "status": "success" if text else "failed",
                "text": text,
                "text_length": len(text),
                "limitations": limitations,
                "metadata": {
                    **result["metadata"],
                    "image_width": width,
                    "image_height": height,
                    "extractor": "pytesseract",
                    "note": "OCR verifies extracted text only; visual manipulation/deepfake detection is not performed.",
                },
            }
        )
        return result
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "limitations": [f"Image OCR failed: {error}"],
            }
        )
        return result


def infer_file_kind(filename: str = "", content_type: str = "", source_type: str = "auto") -> str:
    requested = _clean(source_type).lower()
    ext = _extension(filename)
    ctype = (content_type or "").lower()

    if requested in {"pdf", "pdf_upload", "pdf_url"}:
        return "pdf_upload"
    if requested in {"image", "image_upload", "image_url", "screenshot"}:
        return "image_upload"
    if requested in {"text", "plain_text", "text_file"}:
        return "text_file_upload"

    if ext in PDF_EXTENSIONS or "application/pdf" in ctype:
        return "pdf_upload"
    if ext in IMAGE_EXTENSIONS or ctype.startswith("image/"):
        return "image_upload"
    if ext in TEXT_EXTENSIONS or ctype.startswith("text/") or "json" in ctype or "csv" in ctype:
        return "text_file_upload"
    return "unsupported_file"


def extract_uploaded_file(
    data: bytes,
    *,
    filename: str = "uploaded_file",
    content_type: str = "application/octet-stream",
    source_type: str = "auto",
) -> Dict[str, Any]:
    kind = infer_file_kind(filename, content_type, source_type)

    if kind == "pdf_upload":
        return extract_pdf_bytes(data, filename=filename, content_type=content_type, input_kind="pdf_upload")
    if kind == "image_upload":
        return extract_image_bytes(data, filename=filename, content_type=content_type, input_kind="image_upload")
    if kind == "text_file_upload":
        return extract_text_file_bytes(data, filename=filename, content_type=content_type, input_kind="text_file_upload")

    result = _base_result(input_kind="unsupported_file", filename=filename, content_type=content_type)
    result.update(
        {
            "status": "unsupported",
            "limitations": [
                "Unsupported file type. Supported fast-path formats: PDF with embedded text, TXT/MD/CSV/JSON/log files, and images if OCR dependencies are installed."
            ],
        }
    )
    return result


def extract_url_file(
    url: str,
    input_kind: str = "url",
    *,
    content_bytes: Optional[bytes] = None,
    content_type: str = "",
    final_url: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    bytes_data = content_bytes
    resolved_url = final_url or url
    ctype = content_type

    if bytes_data is None:
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            bytes_data = response.content
            ctype = response.headers.get("content-type", ctype)
            resolved_url = response.url
        except Exception as error:
            result = _base_result(input_kind=input_kind, url=url, content_type=content_type)
            result.update({"status": "failed", "limitations": [f"File URL fetch failed: {error}"]})
            return result

    filename = Path(urlparse(resolved_url).path).name or "downloaded_file"
    inferred = infer_file_kind(filename, ctype, input_kind)

    if inferred == "pdf_upload":
        result = extract_pdf_bytes(bytes_data, filename=filename, content_type=ctype, input_kind="pdf_url", url=resolved_url)
    elif inferred == "image_upload":
        result = extract_image_bytes(bytes_data, filename=filename, content_type=ctype, input_kind="image_url", url=resolved_url)
    elif inferred == "text_file_upload":
        result = extract_text_file_bytes(bytes_data, filename=filename, content_type=ctype, input_kind="url_file", url=resolved_url)
    else:
        result = _base_result(input_kind=input_kind, filename=filename, url=resolved_url, content_type=ctype)
        result.update({"status": "unsupported", "limitations": ["Downloaded URL is not a supported document/image/text file."]})

    result["url"] = resolved_url
    result["domain"] = _domain(resolved_url)
    result["metadata"] = {**result.get("metadata", {}), "final_url": resolved_url, "content_type": ctype}
    return result
