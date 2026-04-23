"""Responsável pela geração de chunks a partir dos documentos extraídos"""
import os
from typing import List, Optional, Dict
import anthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from src.utils.logger import LoggingService
from src.indexing.processing.chunker.context_generator import ContextGenerator, ContextRequest
from src.core.models import ChildChunk

load_dotenv()

logger = LoggingService.setup_logger(__name__)


class DocumentChunker:
    """Handles document chunking with parent-child hierarchy and context generation."""
    
    PARENT_CHUNK_SIZE = 3000
    PARENT_CHUNK_OVERLAP = 300
    CHILD_CHUNK_SIZE = 1000
    CHILD_CHUNK_OVERLAP = 200
    MIN_CHUNK_SIZE = 200
    
    def __init__(self):
        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self.context_generator = ContextGenerator()
        self.parent_splitter = self._make_splitter(
            self.PARENT_CHUNK_SIZE, self.PARENT_CHUNK_OVERLAP
        )
        self.child_splitter = self._make_splitter(
            self.CHILD_CHUNK_SIZE, self.CHILD_CHUNK_OVERLAP
        )
    
    def get_client(self) -> anthropic.Anthropic:
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        return self._anthropic_client
    
    @staticmethod
    def _make_splitter(
        chunk_size: int, chunk_overlap: int
    ) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk_document(self, doc: Dict, meta: Dict, use_context: bool = False) -> List[ChildChunk]:
        """Main chunking pipeline with optional context generation."""
        # Phase 1: Build all chunks without context
        raw_chunks = self._build_raw_chunks(doc)
        
        # Phase 2: Generate contexts if enabled
        contexts = self._generate_contexts(raw_chunks) if use_context else [""] * len(raw_chunks)
        
        # Phase 3: Assemble final ChildChunk objects
        return self._assemble_chunks(raw_chunks, contexts, meta)
    
    def _build_raw_chunks(self, doc: Dict) -> List[Dict]:
        """Create raw chunks from document pages."""
        raw_chunks = []
        for page in doc.get("pages", []):
            text = page.get("text")
            if not text:
                continue
            parent_texts = self.parent_splitter.split_text(text)
            for p_idx, parent_text in enumerate(parent_texts):
                child_texts = self.child_splitter.split_text(parent_text)
                merged_children = self._merge_small_chunks(child_texts)
                for c_idx, child_text in enumerate(merged_children):
                    raw_chunks.append({
                        "child_text": child_text,
                        "parent_text": parent_text,
                        "page": page["page"],
                        "parent_index": p_idx,
                        "child_index": c_idx,
                    })
        return raw_chunks
        
    def _safely_merge_strings(self, s1: str, s2: str) -> str:
        """Merges s2 into s1, avoiding overlapping duplications."""
        max_overlap = min(len(s1), len(s2))
        for i in range(max_overlap, 0, -1):
            if s1.endswith(s2[:i]):
                return s1 + s2[i:]
        return s1 + " " + s2

    def _merge_small_chunks(self, child_texts: List[str]) -> List[str]:
        if not child_texts:
            return []
            
        merged = []
        for text in child_texts:
            if not merged:
                merged.append(text)
                continue
                
            if len(text) < self.MIN_CHUNK_SIZE:
                merged[-1] = self._safely_merge_strings(merged[-1], text)
            elif len(merged[-1]) < self.MIN_CHUNK_SIZE:
                merged[-1] = self._safely_merge_strings(merged[-1], text)
            else:
                merged.append(text)
        return merged
    
    def _generate_contexts(self, raw_chunks: List[Dict]) -> List[str]:
        """Generate contexts for all chunks in batch."""
        requests = [
            ContextRequest(
                parent_text=c["parent_text"],
                child_text=c["child_text"],
                index=i,
            )
            for i, c in enumerate(raw_chunks)
        ]
        return self.context_generator.generate_contexts(requests)
    
    def _assemble_chunks(
        self, raw_chunks: List[Dict], contexts: List[str], meta: Dict
    ) -> List[ChildChunk]:
        """Assemble final ChildChunk objects."""
        all_chunks = []
        for i, raw in enumerate(raw_chunks):
            context_prefix = contexts[i]
            text_to_embed = (
                f"{context_prefix}\n{raw['child_text']}"
                if context_prefix
                else raw["child_text"]
            )
            
            all_chunks.append(ChildChunk(
                text=raw["child_text"],
                context_prefix=context_prefix,
                text_to_embed=text_to_embed,
                parent_text=raw["parent_text"],
                page=raw["page"],
                parent_index=raw["parent_index"],
                child_index=raw["child_index"],
                **meta,
            ))
        
        return all_chunks