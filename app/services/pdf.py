"""
PDF Document Processing Service
Advanced text extraction with structure preservation and citation mapping
"""

import asyncio
import logging
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from io import BytesIO

from app.services.summarization import HierarchicalSummarizer, SummarizationResult
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class PDFPage:
    """PDF page with extracted content"""
    page_number: int
    text: str
    word_count: int
    has_images: bool = False
    has_tables: bool = False

@dataclass
class PDFSection:
    """PDF document section"""
    title: str
    content: str
    start_page: int
    end_page: int
    level: int  # Heading level (1, 2, 3, etc.)

class PDFProcessor:
    """
    Advanced PDF processing with:
    - Multi-library text extraction
    - Structure detection (headings, sections)
    - Page-aware citation mapping
    - Table and image detection
    - Metadata extraction
    """
    
    def __init__(self):
        self.summarizer = HierarchicalSummarizer()
    
    async def process_pdf(
        self,
        pdf_content: bytes,
        filename: str = "document.pdf",
        query: Optional[str] = None,
        mode: str = "researcher",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline
        """
        
        try:
            logger.info(f"Processing PDF: {filename}")
            
            # Step 1: Extract text and metadata
            pages, metadata = await self._extract_pdf_content(pdf_content)
            
            if not pages:
                raise ValueError("No text content found in PDF")
            
            # Step 2: Detect document structure
            sections = self._detect_document_structure(pages)
            
            # Step 3: Combine all text
            full_text = self._combine_page_text(pages)
            
            # Step 4: Generate summaries
            summary_result = await self.summarizer.summarize(
                content=full_text,
                content_type="pdf",
                query=query,
                mode=mode,
                use_cache=use_cache
            )
            
            # Step 5: Generate page-specific summaries
            page_summaries = await self._generate_page_summaries(pages, mode)
            
            # Step 6: Generate section summaries if structure detected
            section_summaries = []
            if sections:
                section_summaries = await self._generate_section_summaries(sections, mode)
            
            # Step 7: Create citations with page references
            citations = self._generate_page_citations(pages, summary_result.key_points)
            
            # Step 8: Build final result
            result = {
                "document_info": {
                    "filename": filename,
                    "total_pages": len(pages),
                    "total_words": sum(page.word_count for page in pages),
                    "has_structure": len(sections) > 0,
                    "metadata": metadata
                },
                "summary_short": summary_result.summary_short,
                "summary_medium": summary_result.summary_medium,
                "summary_detailed": summary_result.summary_detailed,
                "key_points": summary_result.key_points,
                "query_focused_summary": summary_result.query_focused_summary,
                "page_summaries": page_summaries,
                "section_summaries": section_summaries,
                "document_structure": [
                    {
                        "title": section.title,
                        "level": section.level,
                        "pages": f"{section.start_page}-{section.end_page}",
                        "word_count": len(section.content.split())
                    }
                    for section in sections
                ],
                "timestamps": [],  # Not applicable for PDFs
                "confidence_scores": summary_result.confidence_scores,
                "citations": citations,
                "explainability": {
                    **summary_result.explainability,
                    "pdf_processing": {
                        "extraction_method": "pymupdf_primary",
                        "pages_processed": len(pages),
                        "sections_detected": len(sections),
                        "structure_analysis": len(sections) > 0
                    }
                },
                "processing_time": summary_result.processing_time,
                "models_used": summary_result.models_used
            }
            
            logger.info(f"PDF processing completed: {len(pages)} pages, {len(sections)} sections")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    async def _extract_pdf_content(self, pdf_content: bytes) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """Extract text content and metadata from PDF"""
        
        pages = []
        metadata = {}
        
        try:
            # Use PyMuPDF (fitz) for primary extraction
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            # Extract metadata
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": pdf_document.metadata.get("creationDate", ""),
                "modification_date": pdf_document.metadata.get("modDate", ""),
                "page_count": pdf_document.page_count
            }
            
            # Extract text from each page
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Clean and process text
                text = self._clean_extracted_text(text)
                
                if text.strip():  # Only add pages with content
                    pdf_page = PDFPage(
                        page_number=page_num + 1,
                        text=text,
                        word_count=len(text.split()),
                        has_images=len(page.get_images()) > 0,
                        has_tables=self._detect_tables_in_text(text)
                    )
                    pages.append(pdf_page)
            
            pdf_document.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}, trying PyPDF2")
            
            # Fallback to PyPDF2
            try:
                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
                
                # Extract metadata
                if pdf_reader.metadata:
                    metadata = {
                        "title": pdf_reader.metadata.get("/Title", ""),
                        "author": pdf_reader.metadata.get("/Author", ""),
                        "subject": pdf_reader.metadata.get("/Subject", ""),
                        "creator": pdf_reader.metadata.get("/Creator", ""),
                        "producer": pdf_reader.metadata.get("/Producer", ""),
                        "page_count": len(pdf_reader.pages)
                    }
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        text = self._clean_extracted_text(text)
                        
                        if text.strip():
                            pdf_page = PDFPage(
                                page_number=page_num + 1,
                                text=text,
                                word_count=len(text.split())
                            )
                            pages.append(pdf_page)
                    
                    except Exception as page_error:
                        logger.warning(f"Failed to extract page {page_num + 1}: {page_error}")
                        continue
                        
            except Exception as fallback_error:
                logger.error(f"Both PDF extraction methods failed: {fallback_error}")
                raise ValueError("Could not extract text from PDF")
        
        return pages, metadata
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace("\x00", "")  # Null characters
        text = text.replace("\uf0b7", "•")  # Bullet points
        text = text.replace("\uf020", " ")  # Special spaces
        
        # Fix common OCR issues
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("–", "-")
        text = text.replace("—", "-")
        
        return text.strip()
    
    def _detect_tables_in_text(self, text: str) -> bool:
        """Simple table detection in text"""
        
        lines = text.split('\n')
        
        # Look for patterns that suggest tables
        table_indicators = 0
        
        for line in lines:
            # Multiple spaces or tabs (column separation)
            if '  ' in line or '\t' in line:
                table_indicators += 1
            
            # Numbers with consistent spacing
            if len([c for c in line if c.isdigit()]) > len(line) * 0.3:
                table_indicators += 1
        
        return table_indicators > len(lines) * 0.1
    
    def _detect_document_structure(self, pages: List[PDFPage]) -> List[PDFSection]:
        """Detect document structure (headings, sections)"""
        
        sections = []
        current_section = None
        
        all_text = self._combine_page_text(pages)
        lines = all_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if not line:
                continue
            
            # Detect headings based on patterns
            heading_level = self._detect_heading_level(line, i, lines)
            
            if heading_level > 0:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = PDFSection(
                    title=line,
                    content="",
                    start_page=self._find_page_for_line(i, pages),
                    end_page=0,
                    level=heading_level
                )
            
            elif current_section:
                # Add content to current section
                current_section.content += line + " "
        
        # Add final section
        if current_section:
            current_section.end_page = pages[-1].page_number if pages else 1
            sections.append(current_section)
        
        return sections
    
    def _detect_heading_level(self, line: str, line_index: int, all_lines: List[str]) -> int:
        """Detect if line is a heading and its level"""
        
        # Skip very long lines (unlikely to be headings)
        if len(line) > 100:
            return 0
        
        # Skip lines with too many common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_count = len([w for w in line.lower().split() if w in common_words])
        if word_count > len(line.split()) * 0.5:
            return 0
        
        heading_score = 0
        
        # Check for numbering patterns
        if line.startswith(('1.', '2.', '3.', '4.', '5.')):
            heading_score += 3
        elif line.startswith(('1', '2', '3', '4', '5')) and len(line.split()) < 8:
            heading_score += 2
        
        # Check for all caps (but not too long)
        if line.isupper() and len(line) < 50:
            heading_score += 2
        
        # Check for title case
        if line.istitle() and len(line.split()) < 10:
            heading_score += 1
        
        # Check if followed by content
        if line_index < len(all_lines) - 1:
            next_line = all_lines[line_index + 1].strip()
            if next_line and not next_line[0].isupper():
                heading_score += 1
        
        # Determine heading level
        if heading_score >= 4:
            return 1  # Main heading
        elif heading_score >= 2:
            return 2  # Sub heading
        elif heading_score >= 1:
            return 3  # Minor heading
        
        return 0  # Not a heading
    
    def _find_page_for_line(self, line_index: int, pages: List[PDFPage]) -> int:
        """Find which page a line belongs to"""
        
        current_line = 0
        
        for page in pages:
            page_lines = len(page.text.split('\n'))
            
            if current_line <= line_index < current_line + page_lines:
                return page.page_number
            
            current_line += page_lines
        
        return pages[-1].page_number if pages else 1
    
    def _combine_page_text(self, pages: List[PDFPage]) -> str:
        """Combine text from all pages"""
        return "\n\n".join([f"[Page {page.page_number}]\n{page.text}" for page in pages])
    
    async def _generate_page_summaries(self, pages: List[PDFPage], mode: str) -> List[Dict[str, Any]]:
        """Generate summary for each page"""
        
        page_summaries = []
        
        # Only summarize pages with substantial content
        substantial_pages = [page for page in pages if page.word_count > 50]
        
        for page in substantial_pages[:10]:  # Limit to first 10 substantial pages
            try:
                summary_result = await self.summarizer.summarize(
                    content=page.text,
                    content_type="pdf_page",
                    mode=mode,
                    use_cache=False
                )
                
                page_summary = {
                    "page_number": page.page_number,
                    "word_count": page.word_count,
                    "summary": summary_result.summary_short,
                    "key_points": summary_result.key_points[:3],  # Limit key points
                    "has_images": page.has_images,
                    "has_tables": page.has_tables
                }
                
                page_summaries.append(page_summary)
                
            except Exception as e:
                logger.error(f"Failed to summarize page {page.page_number}: {e}")
                continue
        
        return page_summaries
    
    async def _generate_section_summaries(self, sections: List[PDFSection], mode: str) -> List[Dict[str, Any]]:
        """Generate summaries for document sections"""
        
        section_summaries = []
        
        for section in sections:
            if len(section.content.split()) < 20:  # Skip very short sections
                continue
            
            try:
                summary_result = await self.summarizer.summarize(
                    content=section.content,
                    content_type="pdf_section",
                    mode=mode,
                    use_cache=False
                )
                
                section_summary = {
                    "title": section.title,
                    "level": section.level,
                    "pages": f"{section.start_page}-{section.end_page}",
                    "word_count": len(section.content.split()),
                    "summary": summary_result.summary_medium,
                    "key_points": summary_result.key_points
                }
                
                section_summaries.append(section_summary)
                
            except Exception as e:
                logger.error(f"Failed to summarize section '{section.title}': {e}")
                continue
        
        return section_summaries
    
    def _generate_page_citations(self, pages: List[PDFPage], key_points: List[str]) -> List[str]:
        """Generate page-based citations for key points"""
        
        citations = []
        
        for i, key_point in enumerate(key_points):
            # Simple approach: find which page likely contains this key point
            best_page = 1
            best_score = 0
            
            key_words = set(key_point.lower().split())
            
            for page in pages:
                page_words = set(page.text.lower().split())
                overlap = len(key_words.intersection(page_words))
                
                if overlap > best_score:
                    best_score = overlap
                    best_page = page.page_number
            
            citations.append(f"Page {best_page}")
        
        return citations