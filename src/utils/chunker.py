from typing import List
import re

class LogChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_semantics(self, text: str) -> List[str]:
        """Chunk text based on semantic boundaries (timestamps, error blocks)"""
        chunks = []
        
        # Split by timestamp lines to keep related logs together
        lines = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line)
            
            # If adding this line would exceed chunk size, save current chunk
            if current_length + line_length > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Keep overlap lines for context
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(l) for l in current_chunk)
            
            current_chunk.append(line)
            current_length += line_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def chunk_by_errors(self, text: str) -> List[str]:
        """Chunk text by error blocks (groups logs with same error code)"""
        error_blocks = re.split(r'(?=Code[=:]\s*\w+)', text)
        return [block.strip() for block in error_blocks if block.strip()]