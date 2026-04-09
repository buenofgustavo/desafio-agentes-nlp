from dataclasses import dataclass, field
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
    numeracao_item: str
    titulo: str
    autor: str
    material: str
    esfera: str
    situacao: str
    assinatura: str
    publicacao: str
    assunto: str
    ementa: Optional[str]
    pdfs: List[PdfDocument] = field(default_factory=list)

    @property
    def esfera_value(self) -> str:
        return self.esfera.split(":", 1)[-1].strip()

    @property
    def situacao_value(self) -> str:
        return self.situacao.split(":", 1)[-1].strip()

    @property
    def assunto_value(self) -> str:
        return self.assunto.split(":", 1)[-1].strip()

    @property
    def assinatura_date(self) -> date:
        raw = self.assinatura.split(":", 1)[-1].strip()
        return datetime.strptime(raw, "%d/%m/%Y").date()

    @property
    def publicacao_date(self) -> date:
        raw = self.publicacao.split(":", 1)[-1].strip()
        return datetime.strptime(raw, "%d/%m/%Y").date()

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
    records: List[AneelRecord] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0

    @classmethod
    def from_dict(cls, data: dict) -> 'DailyResult':
        records = [AneelRecord.from_dict(record) for record in data.get("records")]
        return cls(
            data=datetime.strptime(data.get("data"), "%Y-%m-%d").date(),
            status=data.get("status"),
            records=records
        )

