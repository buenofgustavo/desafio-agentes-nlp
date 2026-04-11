"""Responsável por extrair os dados dos arquivos JSON, carregar os PDFs e estruturar as informações."""
import json
from pathlib import Path
from typing import List, Dict, Any
from src.core.models import DailyResult, AneelRecord

class JsonLoader:
    @staticmethod
    def _get_json_paths(json_folder: Path | str) -> List[Path]:
        """Lê a pasta json_folder e retorna a lista de arquivos JSON presentes nela."""
        if isinstance(json_folder, str):
            json_folder = Path(json_folder)
            
        return list(json_folder.glob("*.json"))
    
    @staticmethod
    def _load_json_file_data(json_path: Path) -> List[AneelRecord]:
        """Carrega todos os registros do arquivo JSON e retorna uma lista com os registros."""
        aneel_record_list = []
        
        try:
            with open(json_path, encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)

                for key, value, in data.items():
                    result_dict = {
                        "data": key,
                        "status": value.get("status"),
                        "registros": value.get("registros")
                    }
                    daily_result = DailyResult.from_dict(result_dict)

                    if daily_result.is_empty:
                        print(f"Nenhum registro encontrado em {daily_result.data}")
                        continue
                    
                    aneel_record_list.extend(daily_result.registros)
        except Exception as e:
            print(f"Erro na leitura do arquivo {json_path.name}: {e}")
        
        return aneel_record_list
    
    @classmethod
    def load_json_folder_data(cls, json_folder: Path | str) -> List[AneelRecord]:
        """Lê todos os arquivos JSON da pasta json_folder e retorna uma lista com os registros."""
        json_paths = cls._get_json_paths(json_folder)
        all_records = []
        
        for json_path in json_paths:
            records = cls._load_json_file_data(json_path)
            all_records.extend(records)
        
        return all_records