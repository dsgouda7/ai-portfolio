"""Spark source-adapter interfaces for governed ADLS and Unity Catalog inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import IngestionSettings


class SourceAdapter(ABC):
    @abstractmethod
    def read(self, spark: Any) -> Any:
        """Return binary source rows without collecting file content on the driver."""
        ...


class SparkBinaryFileSourceAdapter(SourceAdapter):
    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings

    def read(self, spark: Any) -> Any:
        from pyspark.sql import functions as functions

        reader = spark.read.format("binaryFile").option("recursiveFileLookup", "true")
        if self.settings.file_glob:
            reader = reader.option("pathGlobFilter", self.settings.file_glob)
        frame = reader.load(self.settings.source_uri)
        extension = functions.lower(functions.regexp_extract("path", r"(\.[^.\/]+)$", 1))
        media_type = (
            functions.when(extension.isin(".txt", ".text"), "text/plain")
            .when(extension.isin(".md", ".markdown"), "text/markdown")
            .when(extension == ".json", "application/json")
            .when(extension.isin(".html", ".htm", ".xhtml"), "text/html")
            .when(extension == ".pdf", "application/pdf")
            .when(
                extension == ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            .otherwise("application/octet-stream")
        )
        return frame.select(
            functions.col("path").alias("source_uri"),
            functions.col("path").alias("storage_uri"),
            functions.element_at(functions.split("path", "/"), -1).alias("source_name"),
            media_type.alias("media_type"),
            functions.col("length").alias("byte_size"),
            functions.col("modificationTime").alias("modified_at"),
            functions.col("content"),
        )


class AdlsGen2SourceAdapter(SparkBinaryFileSourceAdapter):
    def __init__(self, settings: IngestionSettings) -> None:
        if not settings.source_uri.lower().startswith("abfss://"):
            raise ValueError("ADLS adapter requires an abfss:// URI backed by a UC external location")
        super().__init__(settings)


class UnityCatalogVolumeSourceAdapter(SparkBinaryFileSourceAdapter):
    def __init__(self, settings: IngestionSettings) -> None:
        if not (
            settings.source_uri.startswith("/Volumes/")
            or settings.source_uri.startswith("dbfs:/Volumes/")
        ):
            raise ValueError("volume adapter requires a Unity Catalog /Volumes path")
        super().__init__(settings)


def create_source_adapter(settings: IngestionSettings) -> SourceAdapter:
    adapter = settings.source_adapter
    if adapter == "auto":
        adapter = "adls" if settings.source_uri.lower().startswith("abfss://") else "volume"
    if adapter == "adls":
        return AdlsGen2SourceAdapter(settings)
    return UnityCatalogVolumeSourceAdapter(settings)
