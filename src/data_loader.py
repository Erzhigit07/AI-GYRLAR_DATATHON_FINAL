"""Read local tabular files without modifying sources or skipping malformed rows."""

import warnings
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from charset_normalizer import from_bytes


def _read_csv(path: Path, *, encoding: str, **options):
    with warnings.catch_warnings():
        # index_col=False can otherwise truncate excess fields with only a warning.
        warnings.filterwarnings(
            "error",
            message="Length of header or names does not match length of data",
            category=pd.errors.ParserWarning,
        )
        frame = pd.read_csv(path, encoding=encoding, **options)
    index_col = options.get("index_col")
    if (
        isinstance(frame, pd.DataFrame)
        and (index_col is None or index_col is False)
        and not isinstance(frame.index, pd.RangeIndex)
    ):
        raise ValueError(
            "CSV создал неявный индекс: проверьте число полей и заголовков. "
            "Если индекс предусмотрен форматом, задайте index_col явно."
        )
    return frame


def load_data(path: str | Path, *, encoding: str | None = None, **options) -> pd.DataFrame:
    """Read CSV, XLSX (one sheet) or Parquet; pass dtype/sep/sheet_name as needed.

    CSV: strict UTF-8 first, then a heuristic encoding guess with an explicit warning.
    Encoding detection is not proof; specify encoding for ambiguous source files.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".parquet"}:
        raise ValueError(f"Формат {suffix!r} не поддерживается. Используйте CSV, XLSX или Parquet.")
    if options.get("on_bad_lines", "error") != "error":
        raise ValueError("Пропуск повреждённых строк запрещён: on_bad_lines должен быть 'error'.")
    if options.get("encoding_errors", "strict") != "strict":
        raise ValueError("Декодирование должно быть строгим: encoding_errors='strict'.")
    try:
        if suffix == ".csv":
            try:
                frame = _read_csv(path, encoding=encoding or "utf-8-sig", **options)
            except UnicodeDecodeError:
                if encoding:
                    raise
                with path.open("rb") as stream:
                    guess = from_bytes(stream.read(1_000_000)).best()
                if guess is None:
                    raise ValueError("Кодировка не определена. Укажите encoding явно.") from None
                warnings.warn(
                    f"Предполагаемая кодировка CSV: {guess.encoding}. Проверьте текст или задайте "
                    "encoding явно.",
                    UserWarning,
                    stacklevel=2,
                )
                frame = _read_csv(path, encoding=guess.encoding, **options)
        elif suffix == ".xlsx":
            frame = pd.read_excel(path, engine="openpyxl", **options)
        else:
            frame = pd.read_parquet(path, engine="pyarrow", **options)
    except ImportError as error:
        raise ImportError(
            "Установите зависимости: python -m pip install -r requirements.txt"
        ) from error
    except (
        OSError,
        ValueError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.ParserWarning,
        BadZipFile,
    ) as error:
        raise ValueError(f"Не удалось прочитать {path}: {error}") from error
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            "Ожидается одна таблица: выберите один sheet_name, без chunksize/iterator."
        )
    if frame.empty:
        raise ValueError(f"Файл не содержит строк данных: {path}")
    if not frame.columns.is_unique:
        raise ValueError("Названия столбцов должны быть уникальными.")
    return frame
