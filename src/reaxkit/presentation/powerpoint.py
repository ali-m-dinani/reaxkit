"""Portable PowerPoint export of categorized, already-rendered figures.

Writes a small PresentationML package using the standard library and Pillow.
PowerPoint and presentation-specific runtime dependencies are not required.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image


_NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p14": "http://schemas.microsoft.com/office/powerpoint/2010/main",
}
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_EMU = 914400
_SLIDE_WIDTH = 12192000
_SLIDE_HEIGHT = 6858000


def _tag(name):
    prefix, local = name.split(":")
    return f"{{{_NS[prefix]}}}{local}"


def _node(parent, tag_name, **attributes):
    return ET.SubElement(parent, _tag(tag_name), {
        _tag(key) if ":" in key else key: str(value)
        for key, value in attributes.items()
    })


def _xml(element):
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _relationships(entries):
    root = ET.Element(f"{{{_REL_NS}}}Relationships")
    for identifier, kind, target in entries:
        ET.SubElement(root, f"{{{_REL_NS}}}Relationship", {
            "Id": identifier, "Type": f"{_NS['r']}/{kind}", "Target": target,
        })
    return _xml(root)


def _slide_tree(root, name=""):
    common = _node(root, "p:cSld", name=name)
    background = _node(_node(common, "p:bg"), "p:bgPr")
    _node(_node(background, "a:solidFill"), "a:srgbClr", val="FFFFFF")
    _node(background, "a:effectLst")
    tree = _node(common, "p:spTree")
    visual = _node(tree, "p:nvGrpSpPr")
    _node(visual, "p:cNvPr", id=1, name="")
    _node(visual, "p:cNvGrpSpPr")
    _node(visual, "p:nvPr")
    transform = _node(_node(tree, "p:grpSpPr"), "a:xfrm")
    for element, attributes in (
        ("off", {"x": 0, "y": 0}), ("ext", {"cx": 0, "cy": 0}),
        ("chOff", {"x": 0, "y": 0}), ("chExt", {"cx": 0, "cy": 0}),
    ):
        _node(transform, f"a:{element}", **attributes)
    return tree


def _shape_properties(parent, left, top, width, height):
    properties = _node(parent, "p:spPr")
    transform = _node(properties, "a:xfrm")
    _node(transform, "a:off", x=int(left), y=int(top))
    _node(transform, "a:ext", cx=int(width), cy=int(height))
    _node(_node(properties, "a:prstGeom", prst="rect"), "a:avLst")
    return properties


def _text(tree, identifier, text, *, top, height, size, bold=False, color="172B4D", align="ctr"):
    shape = _node(tree, "p:sp")
    visual = _node(shape, "p:nvSpPr")
    _node(visual, "p:cNvPr", id=identifier, name=f"Text {identifier}")
    _node(visual, "p:cNvSpPr", txBox=1)
    _node(visual, "p:nvPr")
    properties = _shape_properties(shape, 0.55 * _EMU, top * _EMU,
                                   _SLIDE_WIDTH - 1.1 * _EMU, height * _EMU)
    _node(properties, "a:noFill")
    _node(_node(properties, "a:ln"), "a:noFill")
    body = _node(shape, "p:txBody")
    body_properties = _node(body, "a:bodyPr", wrap="square", anchor="ctr",
                            lIns=0, rIns=0, tIns=0, bIns=0)
    _node(body_properties, "a:normAutofit")
    _node(body, "a:lstStyle")
    paragraph = _node(body, "a:p")
    _node(paragraph, "a:pPr", algn=align)
    run = _node(paragraph, "a:r")
    run_properties = _node(run, "a:rPr", lang="en-US", sz=int(size * 100), b=int(bold))
    _node(_node(run_properties, "a:solidFill"), "a:srgbClr", val=color)
    _node(run_properties, "a:latin", typeface="Arial")
    _node(run, "a:t").text = text


def _title_divider(tree):
    shape = _node(tree, "p:cxnSp")
    visual = _node(shape, "p:nvCxnSpPr")
    _node(visual, "p:cNvPr", id=5, name="Title divider")
    _node(visual, "p:cNvCxnSpPr")
    _node(visual, "p:nvPr")
    properties = _node(shape, "p:spPr")
    transform = _node(properties, "a:xfrm")
    _node(transform, "a:off", x=int(0.55 * _EMU), y=int(0.84 * _EMU))
    _node(transform, "a:ext", cx=int(_SLIDE_WIDTH - 1.1 * _EMU), cy=0)
    _node(_node(properties, "a:prstGeom", prst="line"), "a:avLst")
    line = _node(properties, "a:ln", w=12700)
    _node(_node(line, "a:solidFill"), "a:srgbClr", val="172B4D")
    _node(line, "a:prstDash", val="solid")


def _picture(tree, path, width, height):
    available_width = _SLIDE_WIDTH - 0.9 * _EMU
    available_height = 5.9 * _EMU
    scale = min(available_width / width, available_height / height)
    picture_width, picture_height = int(width * scale), int(height * scale)
    picture = _node(tree, "p:pic")
    visual = _node(picture, "p:nvPicPr")
    _node(visual, "p:cNvPr", id=4, name=path.name, descr=path.name)
    _node(_node(visual, "p:cNvPicPr"), "a:picLocks", noChangeAspect=1)
    _node(visual, "p:nvPr")
    fill = _node(picture, "p:blipFill")
    _node(fill, "a:blip", **{"r:embed": "rIdImage"})
    _node(_node(fill, "a:stretch"), "a:fillRect")
    _shape_properties(picture, (_SLIDE_WIDTH - picture_width) // 2,
                      0.9 * _EMU + (available_height - picture_height) / 2,
                      picture_width, picture_height)


def _theme():
    root = ET.Element(_tag("a:theme"), name="ReaxKit")
    elements = _node(root, "a:themeElements")
    colors = _node(elements, "a:clrScheme", name="ReaxKit")
    for name, value in (
        ("dk1", "000000"), ("lt1", "FFFFFF"), ("dk2", "172B4D"), ("lt2", "F4F6FA"),
        ("accent1", "0057B8"), ("accent2", "D62728"), ("accent3", "008060"),
        ("accent4", "7B2CBF"), ("accent5", "D87800"), ("accent6", "555555"),
        ("hlink", "0057B8"), ("folHlink", "7B2CBF"),
    ):
        _node(_node(colors, f"a:{name}"), "a:srgbClr", val=value)
    fonts = _node(elements, "a:fontScheme", name="Arial")
    for kind in ("majorFont", "minorFont"):
        font = _node(fonts, f"a:{kind}")
        for script, face in (("latin", "Arial"), ("ea", ""), ("cs", "")):
            _node(font, f"a:{script}", typeface=face)
    formats = _node(elements, "a:fmtScheme", name="ReaxKit")
    fills = _node(formats, "a:fillStyleLst")
    lines = _node(formats, "a:lnStyleLst")
    effects = _node(formats, "a:effectStyleLst")
    backgrounds = _node(formats, "a:bgFillStyleLst")
    for width in (6350, 12700, 19050):
        for container in (fills, backgrounds):
            _node(_node(container, "a:solidFill"), "a:schemeClr", val="phClr")
        line = _node(lines, "a:ln", w=width)
        _node(_node(line, "a:solidFill"), "a:schemeClr", val="phClr")
        _node(line, "a:prstDash", val="solid")
        _node(_node(effects, "a:effectStyle"), "a:effectLst")
    return _xml(root)


def _summary_table(tree, counts, warning):
    _text(tree, 2, "Plot summary", top=0.15, height=0.65, size=28, bold=True, align="l")
    frame = _node(tree, "p:graphicFrame")
    visual = _node(frame, "p:nvGraphicFramePr")
    _node(visual, "p:cNvPr", id=4, name="Figure counts")
    _node(visual, "p:cNvGraphicFramePr")
    _node(visual, "p:nvPr")
    transform = _node(frame, "p:xfrm")
    column_widths = (5.5 * _EMU, 2.5 * _EMU)
    table_width = sum(column_widths)
    row_height = min(0.4, 4.9 / (len(counts) + 1)) * _EMU
    _node(transform, "a:off", x=int((_SLIDE_WIDTH - table_width) / 2), y=int(0.95 * _EMU))
    _node(transform, "a:ext", cx=int(table_width), cy=int(row_height * (len(counts) + 1)))
    graphic = _node(frame, "a:graphic")
    data = _node(graphic, "a:graphicData", uri="http://schemas.openxmlformats.org/drawingml/2006/table")
    table = _node(data, "a:tbl")
    _node(table, "a:tblPr", firstRow=1, bandRow=1)
    grid = _node(table, "a:tblGrid")
    for width in column_widths:
        _node(grid, "a:gridCol", w=int(width))
    for index, values in enumerate([("Category", "Number of figures"), *counts.items()]):
        row = _node(table, "a:tr", h=int(row_height))
        for column, value in enumerate(values):
            cell = _node(row, "a:tc")
            body = _node(cell, "a:txBody")
            _node(body, "a:bodyPr")
            _node(body, "a:lstStyle")
            paragraph = _node(body, "a:p")
            _node(paragraph, "a:pPr", algn="l" if column == 0 else "ctr")
            run = _node(paragraph, "a:r")
            properties = _node(run, "a:rPr", lang="en-US", sz=1600, b=int(index == 0))
            _node(_node(properties, "a:solidFill"), "a:srgbClr", val="FFFFFF" if index == 0 else "172B4D")
            _node(properties, "a:latin", typeface="Arial")
            _node(run, "a:t").text = str(value)
            properties = _node(cell, "a:tcPr", marL=100000, marR=100000, marT=25000, marB=25000, anchor="ctr")
            _node(_node(properties, "a:solidFill"), "a:srgbClr",
                  val="172B4D" if index == 0 else ("EDF2F7" if index % 2 else "FFFFFF"))
    if warning:
        _text(tree, 3, warning, top=6.05, height=1.05, size=16,
              bold=True, color="B22222", align="l")


def _write_package(archive, slides, summary_counts=None, warning=None):
    content = ET.Element(f"{{{_CONTENT_NS}}}Types")
    for extension, mime in (("rels", "application/vnd.openxmlformats-package.relationships+xml"),
                            ("xml", "application/xml"), ("png", "image/png"), ("jpeg", "image/jpeg")):
        ET.SubElement(content, f"{{{_CONTENT_NS}}}Default", Extension=extension, ContentType=mime)

    def part(path, kind, data):
        ET.SubElement(content, f"{{{_CONTENT_NS}}}Override", PartName=f"/{path}",
                      ContentType=f"application/vnd.openxmlformats-officedocument.{kind}+xml")
        archive.writestr(path, data)

    archive.writestr("_rels/.rels", _relationships([("rId1", "officeDocument", "ppt/presentation.xml")]))
    master = ET.Element(_tag("p:sldMaster"))
    _slide_tree(master)
    _node(master, "p:clrMap", bg1="lt1", tx1="dk1", bg2="lt2", tx2="dk2",
          **{name: name for name in ("accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink")})
    _node(_node(master, "p:sldLayoutIdLst"), "p:sldLayoutId", id=2147483649, **{"r:id": "rIdLayout"})
    part("ppt/slideMasters/slideMaster1.xml", "presentationml.slideMaster", _xml(master))
    archive.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _relationships([
        ("rIdLayout", "slideLayout", "../slideLayouts/slideLayout1.xml"),
        ("rIdTheme", "theme", "../theme/theme1.xml"),
    ]))
    layout = ET.Element(_tag("p:sldLayout"), type="blank", preserve="1")
    _slide_tree(layout, "Blank")
    _node(_node(layout, "p:clrMapOvr"), "a:masterClrMapping")
    part("ppt/slideLayouts/slideLayout1.xml", "presentationml.slideLayout", _xml(layout))
    archive.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _relationships([
        ("rIdMaster", "slideMaster", "../slideMasters/slideMaster1.xml"),
    ]))
    part("ppt/theme/theme1.xml", "theme", _theme())
    part("ppt/presProps.xml", "presentationml.presProps", _xml(ET.Element(_tag("p:presentationPr"))))
    presentation = ET.Element(_tag("p:presentation"))
    _node(_node(presentation, "p:sldMasterIdLst"), "p:sldMasterId", id=2147483648, **{"r:id": "rIdMaster"})
    slide_ids = _node(presentation, "p:sldIdLst")
    links = [("rIdMaster", "slideMaster", "slideMasters/slideMaster1.xml"),
             ("rIdProps", "presProps", "presProps.xml")]
    sections = []
    if summary_counts is not None:
        slides = [("Plot summary", None, 0, 0), *slides]
    for index, (category, figure, ordinal, count) in enumerate(slides, start=1):
        slide = ET.Element(_tag("p:sld"))
        tree = _slide_tree(slide, category)
        relationships = [("rIdLayout", "slideLayout", "../slideLayouts/slideLayout1.xml")]
        if figure is None:
            sections.append((category, []))
        sections[-1][1].append(255 + index)
        if index == 1 and summary_counts is not None:
            _summary_table(tree, summary_counts, warning)
        elif figure is None:
            _text(tree, 2, category, top=2.4, height=1.4, size=40, bold=True)
            _text(tree, 3, f"{count} figure{'s' if count != 1 else ''}", top=4, height=0.5, size=22, color="555555")
        else:
            path, width, height, extension = figure
            _text(tree, 2, path.stem, top=0.15, height=0.65,
                  size=20 if len(path.stem) > 70 else 26, bold=True, align="l")
            _title_divider(tree)
            _picture(tree, path, width, height)
            _text(tree, 3, f"{category} ({ordinal}/{count})", top=7, height=0.3, size=12, color="555555")
            image_name = f"image{index}.{extension}"
            archive.write(path, f"ppt/media/{image_name}")
            relationships.append(("rIdImage", "image", f"../media/{image_name}"))
        _node(_node(slide, "p:clrMapOvr"), "a:masterClrMapping")
        part(f"ppt/slides/slide{index}.xml", "presentationml.slide", _xml(slide))
        archive.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", _relationships(relationships))
        identifier = f"rIdSlide{index}"
        _node(slide_ids, "p:sldId", id=255 + index, **{"r:id": identifier})
        links.append((identifier, "slide", f"slides/slide{index}.xml"))
    _node(presentation, "p:sldSz", cx=_SLIDE_WIDTH, cy=_SLIDE_HEIGHT)
    _node(presentation, "p:notesSz", cx=_SLIDE_HEIGHT, cy=_SLIDE_WIDTH)
    extension = _node(_node(presentation, "p:extLst"), "p:ext",
                      uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    section_list = _node(extension, "p14:sectionLst")
    for name, identifiers in sections:
        section = _node(section_list, "p14:section", name=name, id="{" + str(uuid4()).upper() + "}")
        section_ids = _node(section, "p14:sldIdLst")
        for identifier in identifiers:
            _node(section_ids, "p14:sldId", id=identifier)
    part("ppt/presentation.xml", "presentationml.presentation.main", _xml(presentation))
    archive.writestr("ppt/_rels/presentation.xml.rels", _relationships(links))
    archive.writestr("[Content_Types].xml", _xml(content))


def write_figure_presentation(
    categories: Mapping[str, Sequence[str | Path]], destination: str | Path,
    *, summary_counts: Mapping[str, int] | None = None, warning: str | None = None,
) -> Path:
    """Write category dividers followed by one uncropped figure per 16:9 slide.

    Preserve category and image order. Skip empty categories. Embed PNG/JPEG
    images at their original resolution with editable slide headings. Raise
    ValueError for an empty collection or invalid images. Publish atomically so
    a failed export leaves any existing deck intact.
    Add native PowerPoint sections for each category. Optional summary_counts
    adds an opening editable table (including zero counts), with warning in bold.
    """
    destination = Path(destination).expanduser()
    if destination.suffix.lower() != ".pptx":
        raise ValueError("PowerPoint destination must end in .pptx.")
    slides = []
    for category, paths in categories.items():
        figures = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            try:
                with Image.open(path) as image:
                    if image.format not in {"PNG", "JPEG"}:
                        raise ValueError("Only PNG and JPEG figures are supported.")
                    width, height = image.size
                    extension = "png" if image.format == "PNG" else "jpeg"
                    image.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot include figure {path}: {exc}") from exc
            figures.append((path, width, height, extension))
        if figures:
            slides.append((str(category), None, 0, len(figures)))
            slides.extend((str(category), figure, index, len(figures))
                          for index, figure in enumerate(figures, start=1))
    if not slides:
        raise ValueError("No figures available for a PowerPoint presentation.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}-", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED) as archive:
            _write_package(archive, slides, summary_counts, warning)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
