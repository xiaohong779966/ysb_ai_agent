"""Unit and sample-workbook tests for the ERP FAQ Excel parser."""

from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import Workbook

from backend.app.knowledge.excel_parser import (
    parse_knowledge_bytes,
    parse_knowledge_workbook,
)
from backend.app.knowledge.exceptions import (
    InvalidKnowledgeHeadersError,
    InvalidWorkbookError,
    UnsupportedWorkbookTypeError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WORKBOOK = PROJECT_ROOT / "knowledge" / "erp_faq.xlsx"


def workbook_bytes(
    rows: list[tuple[object, object]],
    headers: tuple[str, ...] = ("问题", "答案"),
) -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return output.getvalue()


def test_parse_supplied_erp_faq_sample(helpers) -> None:
    result = parse_knowledge_workbook(SAMPLE_WORKBOOK)

    assert result.sheet_name == "Sheet1"
    assert result.total_rows == 28
    assert result.valid_rows == 28
    assert result.invalid_rows == 0
    assert result.is_valid is True
    assert result.records[0].row_number == 2
    assert result.records[0].question == "收银台/后台下载"
    assert result.records[-1].row_number == 29
    assert result.records[-1].question == "采购订单，出现采购价＞入库价的现象"
    assert result.issues == ()


def test_parser_normalizes_whitespace_and_line_endings() -> None:
    content = workbook_bytes([("  如何新增客户？  ", " 第一步\r第二步 \r")])

    result = parse_knowledge_bytes(content, filename="faq.XLSX")

    assert result.is_valid is True
    assert result.records[0].question == "如何新增客户？"
    assert result.records[0].answer == "第一步\n第二步"


def test_parser_collects_empty_missing_and_duplicate_rows() -> None:
    content = workbook_bytes(
        [
            ("如何新增客户", "进入客户管理新增"),
            (None, None),
            (None, "只有答案"),
            ("只有问题", None),
            ("  如何新增客户  ", "重复答案"),
        ]
    )

    result = parse_knowledge_bytes(content, filename="faq.xlsx")

    assert result.total_rows == 5
    assert result.valid_rows == 1
    assert result.invalid_rows == 4
    assert [issue.code for issue in result.issues] == [
        "empty_row",
        "required",
        "required",
        "duplicate",
    ]
    assert result.issues[-1].message == "问题与第 2 行重复"


def test_parser_detects_duplicate_after_an_invalid_first_occurrence() -> None:
    content = workbook_bytes(
        [
            ("重复问题", None),
            ("  重复问题  ", "第二行答案"),
        ]
    )

    result = parse_knowledge_bytes(content, filename="faq.xlsx")

    assert result.valid_rows == 0
    assert result.invalid_rows == 2
    assert result.issues[-1].code == "duplicate"
    assert result.issues[-1].message == "问题与第 2 行重复"


def test_parser_rejects_formula_cells() -> None:
    content = workbook_bytes([("公式问题", "=1+1")])

    result = parse_knowledge_bytes(content, filename="faq.xlsx")

    assert result.valid_rows == 0
    assert result.issues[0].code == "formula_not_allowed"
    assert result.issues[0].field == "answer"


@pytest.mark.parametrize(
    "headers",
    [
        ("question", "answer"),
        ("答案", "问题"),
        ("问题", "答案", "备注"),
    ],
)
def test_parser_rejects_invalid_headers(headers: tuple[str, ...]) -> None:
    content = workbook_bytes([], headers=headers)

    with pytest.raises(InvalidKnowledgeHeadersError, match="问题、答案"):
        parse_knowledge_bytes(content, filename="faq.xlsx")


def test_parser_rejects_non_xlsx_extension() -> None:
    with pytest.raises(UnsupportedWorkbookTypeError, match="仅支持 .xlsx"):
        parse_knowledge_bytes(b"not-an-excel-file", filename="faq.xls")


def test_parser_rejects_corrupt_xlsx_file() -> None:
    with pytest.raises(InvalidWorkbookError, match="文件已损坏"):
        parse_knowledge_bytes(b"not-an-excel-file", filename="faq.xlsx")
