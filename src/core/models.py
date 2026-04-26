from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import date, datetime

@dataclass 
class PdfDocument:
    tipo: str
    url: str
    arquivo: str
    baixado: bool

    @classmethod 
    def from_dict(cls, data: dict) -> 'PdfDocument':
        return cls(
            tipo=data.get("tipo"),
            url=data.get("url"),
            arquivo=data.get("arquivo"),
            baixado=data.get("baixado")
        )

@dataclass
class AneelRecord:
    numeracao_item: Optional[str]
    titulo: Optional[str]
    autor: Optional[str]
    material: Optional[str]
    esfera: Optional[str]
    situacao: Optional[str]
    assinatura: Optional[str]
    publicacao: Optional[str]   
    assunto: Optional[str]
    ementa: Optional[str]
    pdfs: List[PdfDocument] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> 'AneelRecord':
        pdfs = [PdfDocument.from_dict(pdf) for pdf in data.get("pdfs")]
            
        return cls(
            numeracao_item=data.get("numeracao_item"),
            titulo=data.get("titulo"),
            autor=data.get("autor"),
            material=data.get("material"),
            esfera=data.get("esfera"),
            situacao=data.get("situacao"),
            assinatura=data.get("assinatura"),
            publicacao=data.get("publicacao"),
            assunto=data.get("assunto"),
            ementa=data.get("ementa"),
            pdfs=pdfs
        )

@dataclass
class DailyResult:
    data: date
    status: str
    registros: List[AneelRecord] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.registros) == 0

    @classmethod
    def from_dict(cls, data: dict) -> 'DailyResult':
        registros = [AneelRecord.from_dict(record) for record in data.get("registros")]
        return cls(
            data=datetime.strptime(data.get("data"), "%Y-%m-%d").date(),
            status=data.get("status"),
            registros=registros
        )

@dataclass
class ProcessedDocument:
    """Representa o texto extraído de um Documento unido aos seus metadados legais."""
    
    # 1. Rastreabilidade
    arquivo_origem: str
    
    # 2. Conteúdo Extraído
    texto_documento: str
    
    # 3. Metadados para o Payload do Qdrant (achatados do AneelRecord)
    titulo: Optional[str]
    autor: Optional[str]
    material: Optional[str]
    esfera: Optional[str]
    situacao: Optional[str]
    assinatura: Optional[str]
    publicacao: Optional[str]
    assunto: Optional[str]
    ementa: Optional[str]

    @classmethod
    def from_extraction(
        cls, 
        pdf: PdfDocument, 
        record: AneelRecord, 
        document_text: str = "", 
    ) -> 'ProcessedDocument':
        """
        Cria um ProcessedDocument combinando as informações do PDF físico 
        e do registro jurídico associado a ele.
        """
        return cls(
            arquivo_origem=pdf.arquivo,
            texto_documento=document_text,
            titulo=record.titulo,
            autor=record.autor,
            material=record.material,
            esfera=record.esfera,
            situacao=record.situacao,
            assinatura=record.assinatura,
            publicacao=record.publicacao,
            assunto=record.assunto,
            ementa=record.ementa,
        )

    def to_dict(self) -> dict:
        """Converte a classe para dicionário para salvar facilmente como JSON."""
        return asdict(self)
    
    
@dataclass
class ChildChunk:
    # O texto que será indexado = context_prefix + text
    text:             str         # texto bruto do chunk filho
    context_prefix:   str         # descrição contextual gerada por LLM
    text_to_embed:    str         # context_prefix + text (o que realmente indexamos)

    # referência ao pai — retornado para o LLM no momento da query
    parent_text:      str

    # posicionamento
    source_file:      str
    page:             int
    parent_index:     int
    child_index:      int

    # Metadados para o Payload do Qdrant (achatados do AneelRecord)
    titulo: Optional[str]
    autor: Optional[str]
    material: Optional[str]
    esfera: Optional[str]
    situacao: Optional[str]
    assinatura: Optional[str]
    publicacao: Optional[str]
    assunto: Optional[str]
    ementa: Optional[str]


# ── Modelo de resultado de recuperação ─────────────────────────────
from typing import Literal  # noqa: E402
from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """Resultado retornado por todos os componentes de recuperação.

    Os campos de score são preenchidos progressivamente pelo pipeline:
      - score:        score bruto do recuperador (BM25 ou similaridade de cosseno)
      - rrf_score:    score RRF fundido (definido pelo HybridRetriever)
      - rerank_score: score do cross-encoder (definido pelo CrossEncoderReranker)
    """

    chunk_id: str
    text: str
    metadata: dict
    score: float
    source: Literal["bm25", "dense", "hybrid"]
    rrf_score: float | None = None
    rerank_score: float | None = None


# ── Modelos de resposta do Agente (LLM) ──────────────────────────────────


class QueryAnalysis(BaseModel):
    """Resposta estruturada do nó query_analyzer.

    O LLM retorna JSON com este formato; o Pydantic valida e
    converte os dados — campos desconhecidos são ignorados, campos
    opcionais ausentes recebem padrões sensatos.
    """

    query_type: Literal["simple", "comparative", "multi_hop"] = "simple"
    reasoning: str = ""


class FaithfulnessResult(BaseModel):
    """Resposta estruturada do nó faithfulness_check."""

    is_grounded: bool = True
    score: float = 0.0
    reasoning: str = ""
    unsupported_claims: list[str] = []


class MultiHopSubQuery(BaseModel):
    """Resposta estruturada do gerador de sub-queries para multi-hop."""

    sub_query: str
    reasoning: str = ""
