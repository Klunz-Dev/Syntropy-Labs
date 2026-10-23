"""
Syntropy — модуль автоматической очистки данных для AI-анализатора.

Функция auto_cleaning_df принимает путь к CSV или готовый DataFrame,
выполняет умную очистку и возвращает пару (X, y).
"""
from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR = Path(__file__).parent / "testing data"


def _parse_money(value) -> Optional[float]:
    """Парсит '1 234,56 ₽' → 1234.56. Возвращает None, если не получилось."""
    if pd.isna(value):
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", str(value))
    # Унифицируем десятичный разделитель
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_code_like(series: pd.Series) -> bool:
    """True, если почти все значения начинаются с нецифрового символа
    (например T02016694, LA-01, NY-02, SAVE10) — такие коды нельзя
    превращать в числа. Знак минус в начале числа не считается кодом."""
    values = series.dropna().astype(str).str.lstrip("-").str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    return values.str.match(r"^\D").mean() > 0.99


def _is_datetime_like(series: pd.Series) -> bool:
    """True, если колонка действительно содержит дату/время (2022-10-06, 11:30:55)."""
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        # pandas ругается, если формат даты не смог определить сразу — это нормально
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.9


def _is_mixed_units(series: pd.Series) -> bool:
    """True, если числа записаны с текстовыми единицами, причём единицы не одинаковые
    (например '4 years' и '2 months') — превращать в числа нельзя, сравнение бессмысленно."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    flagged = values.str.match(r"^\d") & values.str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True)
    return flagged.mean() > 0.8


def _try_parse_numeric(series: pd.Series) -> Optional[pd.Series]:
    """Пытается превратить object-колонку в числовую. None если не вышло."""
    if _is_code_like(series) or _is_datetime_like(series) or _is_mixed_units(series):
        return None
    parsed = series.apply(_parse_money)
    # Если хотя бы 80% значений распарсились — считаем, что колонка числовая
    success_ratio = parsed.notna().sum() / max(series.notna().sum(), 1)
    if success_ratio >= 0.8:
        return parsed
    return None


def auto_cleaning_df(
    data: Union[str, Path, pd.DataFrame],
    target: Optional[str] = None,
    drop_columns: Optional[list[str]] = None,
    smart_parse: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """Принимает путь к CSV (str/Path) либо готовый DataFrame,
    очищает данные и возвращает пару (X, y).

    Шаги очистки:
      1) удаляются колонки из drop_columns (таргет удалить нельзя);
      2) удаляются полные дубликаты строк;
      3) отделяется колонка-таргет (если target передан);
      4) числовые колонки остаются как есть;
      5) object-колонки, похожие на числа/деньги, преобразуются в число
         (если smart_parse=True);
      6) удаляются полностью пустые колонки;
      7) ID-подобные колонки удаляются (uniqueness > 0.99 И len > 50)
         — правило только для целочисленных значений.

    Пропуски НЕ заполняются — это задача модели, чтобы не было утечки данных.

    Если target не передан, y равен None.

    Если путь не найден, файл ищется в папке DATA_DIR (src/data/testing data) —
    поэтому можно писать и просто имя файла, например 'housing.csv'.
    """
    # --- 1. Загрузка данных ---
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            path = DATA_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {data}")
        df = pd.read_csv(path)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("data должен быть путём к CSV (str/Path) или pd.DataFrame")

    # --- 2. drop_columns ---
    if drop_columns:
        valid = [c for c in drop_columns if c in df.columns and c != target]
        missing = [c for c in drop_columns if c not in df.columns]
        if missing:
            logger.warning("Колонки не найдены и пропущены: %s", missing)
        if valid:
            df = df.drop(columns=valid)
            logger.info("Удалено колонок из drop_columns: %s", valid)

    # --- 3. Дубликаты ---
    before = len(df)
    df = df.drop_duplicates()
    if before != len(df):
        logger.info("Удалено дубликатов строк: %d", before - len(df))

    # --- 4. Отделение таргета ---
    y: Optional[pd.Series] = None
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Колонка-таргет не найдена: {target}")
        y = df[target].copy()
        df = df.drop(columns=[target])

    # --- 5. Умное определение типов ---
    # Сначала — числовые как есть
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Потом — пытаемся распарсить object-колонки
    parsed_cols = []
    if smart_parse:
        for col in df.select_dtypes(include=["object", "str"]).columns:
            parsed = _try_parse_numeric(df[col])
            if parsed is not None:
                df[col] = parsed
                parsed_cols.append(col)
        if parsed_cols:
            logger.info("Преобразовано object-колонок в числа: %s", parsed_cols)

    # Обновляем список числовых колонок
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Текстовые колонки, которые не удалось распарсить — отдельно
    text_cols = [c for c in df.columns if c not in numeric_cols]
    if text_cols:
        logger.info("Текстовые колонки оставлены вне X: %s", text_cols)

    # Берём только числовые (для классической ML-модели)
    X = df[numeric_cols].copy()

    # --- 6. Полностью пустые колонки ---
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        X = X.drop(columns=empty)
        logger.info("Удалено пустых колонок: %s", empty)

    # --- 7. ID-подобные колонки ---
    id_cols = []
    if len(X) > 50:
        for col in X.columns:
            values = X[col].dropna()
            if values.empty:
                continue
            # правило только для целочисленных значений: у непрерывных чисел
            # почти полная уникальность — это норма, а не признак ID
            if not (values % 1 == 0).all():
                continue
            if values.nunique() / len(values) > 0.99:
                id_cols.append(col)
        if id_cols:
            X = X.drop(columns=id_cols)
            logger.info("Удалено ID-колонок: %s", id_cols)
            text_cols.extend(id_cols)

    logger.info("Итоговых признаков: %d, строк: %d", X.shape[1], X.shape[0])

    return X, y


if __name__ == "__main__":
    # Пример запуска — подставьте свой CSV-файл в папку testing data/
    test_file = "housing.csv"
    X, y = auto_cleaning_df(
        test_file,
        target="median_house_value",
        smart_parse=True,
    )
    print(f"\nСтрок: {X.shape[0]}, признаков осталось: {X.shape[1]}")
    print(f"Признаки: {list(X.columns)}")
    if y is not None:
        print("\nТаргет (первые 5 значений):")
        print(y.head())
