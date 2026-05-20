from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class DataSource(Protocol):
    def download(self, destination: Path) -> None: ...
    def available(self, destination: Path) -> bool: ...


class KaggleSource:
    def __init__(self, competition: str, force: bool = False) -> None:
        self.competition = competition
        self.force = force

    def available(self, destination: Path) -> bool:
        return bool(list(destination.glob("*"))) and not self.force

    def download(self, destination: Path) -> None:
        if self.available(destination):
            logger.info("Dados já existentes em %s — download ignorado.", destination)
            return

        try:
            from kaggle import KaggleApi
        except ImportError:
            raise ImportError("Instale o pacote kaggle: pip install 'mltemplate[kaggle]'")

        destination.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()

        logger.info("Baixando competição '%s'...", self.competition)
        api.competition_download_files(self.competition, path=str(destination), quiet=False)

        self._extract_zips(destination)
        logger.info("Download concluído. Arquivos em %s", destination)

    def _extract_zips(self, destination: Path) -> None:
        for zip_path in destination.glob("*.zip"):
            logger.info("Extraindo %s...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(destination))
            zip_path.unlink()


class LocalSource:
    """Fonte local — copia arquivos para o destino se necessário."""

    def __init__(self, *file_paths: Path) -> None:
        self.file_paths = [Path(p) for p in file_paths]

    def available(self, destination: Path) -> bool:
        return all((destination / p.name).exists() for p in self.file_paths)

    def download(self, destination: Path) -> None:
        if self.available(destination):
            return
        destination.mkdir(parents=True, exist_ok=True)
        for src in self.file_paths:
            dst = destination / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info("Copiado %s → %s", src, dst)
