from docmind.ingestion.loader import pdf_layout_flags


class TextPage:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def get_text_range(self, index, count):
        return "X"

    def count_chars(self):
        return len(self.boxes)

    def get_charbox(self, index):
        self.calls += 1
        return self.boxes[index]


def test_interleaved_columns_are_flagged_with_bounded_work():
    boxes = []
    for y in range(700, 100, -15):
        boxes.extend([(50, y, 180, y + 10), (350, y, 470, y + 10)])
    page = TextPage(boxes * 20)
    assert pdf_layout_flags(page, 600) == ["ambiguous_column_or_table_order"]
    assert page.calls == 512


def test_single_column_and_column_major_order_do_not_trigger_row_jump_warning():
    boxes = [(50, y, 180, y + 10) for y in range(700, 100, -15)]
    assert not pdf_layout_flags(TextPage(boxes), 600)
    assert not pdf_layout_flags(
        TextPage(boxes + [(350, y, 470, y + 10) for y in range(700, 100, -15)]), 600
    )


def test_actual_interleaved_pdf_is_quarantined_after_worker_postprocessing(tmp_path):
    from docmind.ingestion.loader import load_document

    content = []
    for y in range(740, 540, -18):
        content.extend(
            [
                f"BT /F1 10 Tf 50 {y} Td (The pressure valve opens.) Tj ET",
                f"BT /F1 10 Tf 350 {y} Td (Emergency shutdown follows.) Tj ET",
            ]
        )
    stream = "\n".join(content).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    output, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(output)
    output += b"xref\n0 6\n0000000000 65535 f \n"
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    output += (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )
    path = tmp_path / "columns.pdf"
    path.write_bytes(output)
    unit = load_document(path)[0]
    assert "pressure valve" in unit.content
    assert unit.metadata["extraction_quality"]["quarantined"]
    assert (
        "ambiguous_column_or_table_order"
        in unit.metadata["extraction_quality"]["flags"]
    )
