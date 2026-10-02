"""PowerPoint figure ordering, geometry, and atomic publication."""

from pathlib import Path
import posixpath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image
import pytest

from reaxkit.presentation import powerpoint


NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def make_image(path, size=(600, 400)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def test_category_titles_precede_figures_and_empty_categories_are_skipped(tmp_path):
    first = make_image(tmp_path / "eos" / "material" / "eos_1.png")
    second = make_image(tmp_path / "eos" / "material" / "eos_2.png")
    third = make_image(tmp_path / "angle.png")
    output = powerpoint.write_figure_presentation(
        {"EOS & volume": [first, second], "Empty": [], "Angles α": [third]},
        tmp_path / "output" / "figures.pptx",
    )
    with ZipFile(output) as archive:
        slides = [ET.fromstring(archive.read(f"ppt/slides/slide{index}.xml")) for index in range(1, 6)]
        texts = [[element.text for element in slide.findall(".//a:t", NS)] for slide in slides]
        assert texts == [
            ["EOS & volume", "2 figures"], ["eos_1", "EOS & volume (1/2)"],
            ["eos_2", "EOS & volume (2/2)"], ["Angles α", "1 figure"],
            ["angle", "Angles α (1/1)"],
        ]
        assert [len(slide.findall(".//p:pic", NS)) for slide in slides] == [0, 1, 1, 0, 1]
        assert archive.read("ppt/media/image2.png") == first.read_bytes()
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        assert len(presentation.findall("p:sldIdLst/p:sldId", NS)) == 5
        size = presentation.find("p:sldSz", NS)
        assert int(size.get("cx")) / int(size.get("cy")) == pytest.approx(16 / 9)
        names = set(archive.namelist())
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                root = ET.fromstring(archive.read(name))
                if name.endswith(".rels"):
                    parent = "" if name == "_rels/.rels" else str(Path(name).parent.parent).replace("\\", "/")
                    for relation in root:
                        target = posixpath.normpath(posixpath.join(parent, relation.get("Target")))
                        assert target in names, (name, target)


@pytest.mark.parametrize("size,extension", [((400, 1200), "png"), ((1800, 300), "jpg"), ((500, 500), "png")])
def test_images_fit_without_cropping_or_distortion(tmp_path, size, extension):
    figure = make_image(tmp_path / f"plot.{extension}", size)
    output = powerpoint.write_figure_presentation({"Plots": [figure]}, tmp_path / "deck.pptx")
    with ZipFile(output) as archive:
        slide = ET.fromstring(archive.read("ppt/slides/slide2.xml"))
        transform = slide.find(".//p:pic/p:spPr/a:xfrm", NS)
        offset = transform.find("a:off", NS)
        extent = transform.find("a:ext", NS)
        left, top = int(offset.get("x")), int(offset.get("y"))
        width, height = int(extent.get("cx")), int(extent.get("cy"))
        assert width / height == pytest.approx(size[0] / size[1], rel=1e-5)
        assert 0 <= left < left + width <= 12192000
        assert 0.9 * 914400 <= top < top + height <= 6.8 * 914400
        assert slide.find(".//a:srcRect", NS) is None


def test_empty_collection_is_rejected_without_creating_output(tmp_path):
    output = tmp_path / "deck.pptx"
    with pytest.raises(ValueError, match="No figures"):
        powerpoint.write_figure_presentation({"Empty": []}, output)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "unsupported"])
def test_invalid_image_leaves_existing_deck_intact(tmp_path, kind):
    image = tmp_path / "bad.png"
    if kind == "corrupt":
        image.write_bytes(b"not an image")
    elif kind == "unsupported":
        Image.new("RGB", (10, 10)).save(image, format="GIF")
    output = tmp_path / "deck.pptx"
    output.write_bytes(b"existing deck")
    with pytest.raises(ValueError, match="Cannot include figure"):
        powerpoint.write_figure_presentation({"Plots": [image]}, output)
    assert output.read_bytes() == b"existing deck"


def test_failed_package_write_is_atomic_and_cleans_temporary_file(monkeypatch, tmp_path):
    image = make_image(tmp_path / "plot.png")
    output = tmp_path / "deck.pptx"
    output.write_bytes(b"existing deck")

    def fail(archive, slides, summary_counts, warning):
        archive.writestr("partial.xml", b"partial")
        raise RuntimeError("failed export")

    monkeypatch.setattr(powerpoint, "_write_package", fail)
    with pytest.raises(RuntimeError, match="failed export"):
        powerpoint.write_figure_presentation({"Plots": [image]}, output)
    assert output.read_bytes() == b"existing deck"
    assert not list(tmp_path.glob("*.tmp"))


def test_requires_powerpoint_extension(tmp_path):
    with pytest.raises(ValueError, match=".pptx"):
        powerpoint.write_figure_presentation({}, tmp_path / "deck.pdf")


def test_summary_alignment_and_native_sections(tmp_path):
    image = make_image(tmp_path / "plot.png")
    warning = "[Warning] Not plotted: 12 entries and results/not_plotted_entries.csv"
    output = powerpoint.write_figure_presentation(
        {"EOS": [image], "Empty": [], "Angles": [image]}, tmp_path / "deck.pptx",
        summary_counts={"EOS": 1, "Empty": 0, "Angles": 1}, warning=warning,
    )
    namespaces = {**NS, "p14": powerpoint._NS["p14"]}
    with ZipFile(output) as archive:
        summary = ET.fromstring(archive.read("ppt/slides/slide1.xml"))
        assert summary.find(".//a:graphicData", NS).get("uri") == (
            "http://schemas.openxmlformats.org/drawingml/2006/table"
        )
        rows = summary.findall(".//a:tbl/a:tr", NS)
        widths = [int(column.get("w")) for column in summary.findall(".//a:tblGrid/a:gridCol", NS)]
        assert widths == [int(5.5 * 914400), int(2.5 * 914400)]
        transform = summary.find(".//p:graphicFrame/p:xfrm", NS)
        assert int(transform.find("a:ext", NS).get("cx")) == sum(widths)
        assert int(transform.find("a:off", NS).get("x")) == (12192000 - sum(widths)) // 2
        assert [[cell.text for cell in row.findall(".//a:t", NS)] for row in rows] == [
            ["Category", "Number of figures"], ["EOS", "1"], ["Empty", "0"], ["Angles", "1"],
        ]
        warning_run = summary.findall(".//p:sp/p:txBody/a:p/a:r", NS)[-1]
        assert warning_run.find("a:t", NS).text == warning
        assert warning_run.find("a:rPr", NS).get("b") == "1"
        for index, alignment in ((2, "ctr"), (3, "l"), (4, "ctr"), (5, "l")):
            slide = ET.fromstring(archive.read(f"ppt/slides/slide{index}.xml"))
            assert slide.find(".//p:sp/p:txBody/a:p/a:pPr", NS).get("algn") == alignment
            divider = slide.find(".//p:cxnSp/p:spPr", NS)
            if alignment == "l":
                assert divider is not None
                assert divider.find("a:ln", NS).get("w") == "12700"
                divider_top = int(divider.find("a:xfrm/a:off", NS).get("y"))
                picture_top = int(slide.find(".//p:pic/p:spPr/a:xfrm/a:off", NS).get("y"))
                assert 0.8 * 914400 < divider_top < picture_top
            else:
                assert divider is None
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        sections = presentation.findall(".//p14:section", namespaces)
        assert [section.get("name") for section in sections] == ["Plot summary", "EOS", "Angles"]
        assert [[entry.get("id") for entry in section.findall("p14:sldIdLst/p14:sldId", namespaces)]
                for section in sections] == [["256"], ["257", "258"], ["259", "260"]]
        assert len({section.get("id") for section in sections}) == 3
