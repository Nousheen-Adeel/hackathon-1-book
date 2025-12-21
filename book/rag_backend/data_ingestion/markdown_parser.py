import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import yaml


class MarkdownParser:
    """Parse markdown files from the Docusaurus docs directory."""
    
    def __init__(self, docs_path: str = "./docs"):
        self.docs_path = Path(docs_path)
        
    def extract_frontmatter(self, content: str) -> Tuple[str, Dict]:
        """Extract YAML frontmatter from markdown content."""
        if content.startswith("---"):
            try:
                end_marker = content.find("---", 3)
                if end_marker != -1:
                    frontmatter = content[3:end_marker].strip()
                    remaining_content = content[end_marker + 3:].strip()
                    metadata = yaml.safe_load(frontmatter) or {}
                    return remaining_content, metadata
            except yaml.YAMLError:
                pass
        
        return content, {}
    
    def split_by_headings(self, content: str) -> List[Dict[str, str]]:
        """Split content by headings into sections."""
        # Regular expression to match markdown headings (# ## ### ####)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = content.split('\n')
        sections = []
        current_section = {
            'heading': '',
            'content': [],
            'level': 0
        }
        
        for line in lines:
            match = re.match(heading_pattern, line.strip())
            
            if match:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append({
                        'heading': current_section['heading'],
                        'content': '\n'.join(current_section['content']).strip(),
                        'level': current_section['level']
                    })
                
                # Start new section
                heading_level = len(match.group(1))
                heading_text = match.group(2)
                current_section = {
                    'heading': heading_text,
                    'content': [],
                    'level': heading_level
                }
            else:
                current_section['content'].append(line)
        
        # Add the last section
        if current_section['content']:
            sections.append({
                'heading': current_section['heading'],
                'content': '\n'.join(current_section['content']).strip(),
                'level': current_section['level']
            })
        
        return sections
    
    def parse_document(self, file_path: Path) -> List[Dict]:
        """Parse a markdown document into structured chunks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content_without_frontmatter, metadata = self.extract_frontmatter(content)
        
        # Extract filename without extension as title if not in metadata
        if 'title' not in metadata:
            metadata['title'] = file_path.stem.replace('-', ' ').title()
        
        # Add file path info to metadata
        metadata['file_path'] = str(file_path.relative_to(self.docs_path))
        
        sections = self.split_by_headings(content_without_frontmatter)
        
        chunks = []
        for section in sections:
            # Combine heading with content
            chunk_content = f"{section['heading']}\n\n{section['content']}".strip() if section['heading'] else section['content']
            
            if chunk_content.strip():  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_content,
                    'metadata': {
                        **metadata,
                        'heading': section['heading'],
                        'heading_level': section['level']
                    }
                })
                
        return chunks
    
    def parse_all_documents(self) -> List[Dict]:
        """Parse all markdown documents in the docs directory."""
        all_chunks = []
        
        for md_file in self.docs_path.rglob("*.md"):
            try:
                chunks = self.parse_document(md_file)
                all_chunks.extend(chunks)
                print(f"Parsed {len(chunks)} chunks from {md_file}")
            except Exception as e:
                print(f"Error parsing {md_file}: {str(e)}")
                
        return all_chunks