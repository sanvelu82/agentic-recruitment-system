"""
PDF Text Extraction Utility

Efficient PDF text extraction using PyMuPDF (fitz).
Optimized for resume parsing with fallback strategies.
"""

import io
import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    text: str
    page_count: int
    char_count: int
    extraction_method: str
    warnings: list[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if extraction produced meaningful content."""
        return len(self.text.strip()) >= 50


def extract_text_from_pdf(
    pdf_content: bytes,
    max_pages: int = 20,
    min_text_length: int = 50,
) -> PDFExtractionResult:
    """
    Extract text from PDF bytes using PyMuPDF (fastest method).
    
    Args:
        pdf_content: Raw PDF file bytes
        max_pages: Maximum pages to process (for safety)
        min_text_length: Minimum expected text length
    
    Returns:
        PDFExtractionResult with extracted text and metadata
    """
    warnings = []
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return PDFExtractionResult(
            text="",
            page_count=0,
            char_count=0,
            extraction_method="failed",
            warnings=["PyMuPDF not installed. Install with: pip install pymupdf"]
        )
    
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        page_count = min(len(pdf_document), max_pages)
        if len(pdf_document) > max_pages:
            warnings.append(f"PDF has {len(pdf_document)} pages, only processing first {max_pages}")
        
        # Extract text from all pages
        text_blocks = []
        
        for page_num in range(page_count):
            page = pdf_document[page_num]
            
            # Method 1: Standard text extraction (fastest)
            page_text = page.get_text("text")
            
            # Method 2: If standard extraction yields little text, try blocks
            if len(page_text.strip()) < 20:
                blocks = page.get_text("blocks")
                page_text = "\n".join(
                    block[4] for block in blocks 
                    if isinstance(block[4], str)
                )
            
            if page_text.strip():
                text_blocks.append(page_text)
        
        pdf_document.close()
        
        # Combine and clean text
        raw_text = "\n\n".join(text_blocks)
        cleaned_text = _clean_extracted_text(raw_text)
        
        # Validate extraction
        if len(cleaned_text.strip()) < min_text_length:
            warnings.append("Extracted text is very short - PDF may be image-based or encrypted")
        
        return PDFExtractionResult(
            text=cleaned_text,
            page_count=page_count,
            char_count=len(cleaned_text),
            extraction_method="pymupdf",
            warnings=warnings,
        )
        
    except Exception as e:
        return PDFExtractionResult(
            text="",
            page_count=0,
            char_count=0,
            extraction_method="failed",
            warnings=[f"PDF extraction failed: {str(e)}"]
        )


def _clean_extracted_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    
    Handles common PDF extraction artifacts:
    - Multiple spaces/tabs
    - Excessive newlines
    - Unicode artifacts
    - Header/footer repetitions
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = text.replace('\ufeff', '')  # BOM
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\u2022', '•')  # Bullet point
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2018', "'")  # Left single quote
    text = text.replace('\u2019', "'")  # Right single quote
    text = text.replace('\u201c', '"')  # Left double quote
    text = text.replace('\u201d', '"')  # Right double quote
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3+ newlines with 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines at start and end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)


def validate_pdf_file(
    content: bytes,
    max_size_mb: float = 10.0,
) -> Tuple[bool, Optional[str]]:
    """
    Validate PDF file before processing.
    
    Args:
        content: File bytes
        max_size_mb: Maximum file size in MB
    
    Returns:
        (is_valid, error_message)
    """
    # Check file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
    
    # Check PDF magic bytes
    if not content[:4] == b'%PDF':
        return False, "Invalid PDF file: missing PDF header"
    
    # Check for minimum size (a valid PDF should be at least a few hundred bytes)
    if len(content) < 100:
        return False, "Invalid PDF file: file too small"
    
    return True, None


def extract_resume_sections(text: str) -> dict:
    """
    Attempt to identify common resume sections from extracted text.
    
    This provides hints to the resume parser about document structure.
    
    Args:
        text: Extracted resume text
    
    Returns:
        Dict with detected sections and their approximate positions
    """
    sections = {}
    text_lower = text.lower()
    
    # Common section headers
    section_patterns = {
        'contact': r'\b(contact|email|phone|address)\b',
        'summary': r'\b(summary|objective|profile|about)\b',
        'experience': r'\b(experience|employment|work history|professional)\b',
        'education': r'\b(education|academic|degree|university|college)\b',
        'skills': r'\b(skills|technologies|technical|competencies|expertise)\b',
        'certifications': r'\b(certifications?|licenses?|credentials)\b',
        'projects': r'\b(projects?|portfolio)\b',
        'languages': r'\b(languages?)\b',
        'references': r'\b(references?)\b',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            sections[section_name] = {
                'found': True,
                'position': match.start(),
            }
    
    return sections
