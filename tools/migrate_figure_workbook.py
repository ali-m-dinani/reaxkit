"""Migrate the packaged legacy figure workbook without losing its source copy."""

from pathlib import Path
from shutil import copy2

from openpyxl import load_workbook
from openpyxl.chart import BarChart, ScatterChart, Reference, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo

from reaxkit.workflows.force_field_opt.plot_from_excel import DATA_HEADERS, SETTINGS_HEADERS, SHEETS, read_figures


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/reaxkit/workflows/force_field_opt/data/template_ffield_opt_figure_generator.xlsx"
BACKUP = TARGET.with_name(TARGET.stem+"_legacy.xlsx")
if BACKUP.exists():
    raise FileExistsError(f"Migration backup already exists: {BACKUP}")
cached = load_workbook(TARGET, data_only=True)
workbook = load_workbook(TARGET)
source_rows = {}
for name in SHEETS:
    source_rows[name] = []
    for row in workbook[name].iter_rows(min_row=2, max_col=4):
        if all(cell.value is None for cell in row):
            continue
        values = []
        for cell in row:
            value = cached[name][cell.coordinate].value
            if cell.data_type == "f" and value is None:
                raise ValueError(f"Recalculate {name}!{cell.coordinate} in Excel before migrating.")
            values.append(value)
        source_rows[name].append(values)
copy2(TARGET, BACKUP)
for name in SHEETS:
    index = workbook.sheetnames.index(name)
    workbook.remove(workbook[name])
    sheet = workbook.create_sheet(name, index)
    sheet.append([*DATA_HEADERS, None, *SETTINGS_HEADERS])
    previous_label = ""
    for point, values in enumerate(source_rows[name], start=1):
        if name == "EOS_plots":
            identifier, coordinate, reaxff, qm = values
            provenance = None
        else:
            provenance, identifier, reaxff, qm = values
            identifier = identifier or previous_label
            coordinate = point if name == "line_plots" else None
        previous_label = identifier
        sheet.append([1, provenance, identifier, coordinate, reaxff, qm])
    axis_labels = {"bar_plots": ("Training entry", "Energy (kcal/mol)"),
                   "line_plots": ("Reaction-coordinate point index", "Energy (source units)"),
                   "EOS_plots": ("Volume (Å³)", "Relative energy (source normalization)")}
    for column, value in enumerate((1, "", *axis_labels[name]), start=8):
        sheet.cell(2, column, value)
    for start, end, table_name in (("A1", f"F{sheet.max_row}", f"{name}_data"),
                                 ("H1", "K2", f"{name}_settings")):
        table = Table(displayName=table_name, ref=f"{start}:{end}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        sheet.add_table(table)
    sheet.freeze_panes = "D2"
    sheet.sheet_view.showGridLines = False
    widths = {"A": 18, "B": 42, "C": 48, "D": 16, "E": 19, "F": 19,
              "G": 3, "H": 18, "I": 30, "J": 32, "K": 40}
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    for row in sheet:
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            cell.font = Font(name="Arial", size=11, color="000000")
            if cell.row == 1 and cell.column != 7:
                cell.fill = PatternFill("solid", fgColor="0057B8")
                cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            if cell.column in (4, 5, 6) and cell.row > 1:
                cell.number_format = "0.0000;-0.0000;0"
        sheet.row_dimensions[row[0].row].height = 45
    validation = DataValidation(type="whole", operator="greaterThanOrEqual", formula1=1, allow_blank=True)
    validation.errorTitle = "Figure number"
    validation.error = "Use a positive integer."
    validation.showErrorMessage = True
    sheet.add_data_validation(validation)
    validation.add("A2:A10000")
    validation.add("H2:H10000")
    chart = BarChart() if name == "bar_plots" else ScatterChart()
    chart.title = "Preview only — ReaxKit uses Figure number for grouping"
    chart.width, chart.height = 24, 13
    chart.x_axis.title, chart.y_axis.title = axis_labels[name]
    if name == "bar_plots":
        chart.type = "col"
        chart.grouping = "clustered"
        chart.add_data(Reference(sheet, min_col=5, max_col=6, min_row=1, max_row=sheet.max_row), titles_from_data=True)
        chart.set_categories(Reference(sheet, min_col=3, min_row=2, max_row=sheet.max_row))
    else:
        for column in (5, 6):
            chart.series.append(Series(Reference(sheet, min_col=column, min_row=1, max_row=sheet.max_row),
                Reference(sheet, min_col=4, min_row=2, max_row=sheet.max_row), title_from_data=True))
    for series, color in zip(chart.series, ("0057B8", "D62728")):
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line.solidFill = color
        series.graphicalProperties.line.width = 33020
    sheet.add_chart(chart, "H6")
instructions = workbook["Sheet1"]
instructions.title = "Instructions"
notes = [
    "plot-from-excel: enter data in A:F and figure settings in H:K on each plot sheet.",
    "Use positive figure numbers; numbers are independent between sheets. Add one settings row per figure.",
    "Line in fort.99 is optional provenance (line number or expression). Values are read from Excel, not fort.99.",
    "Bar order follows rows. Numeric curves sort by X value. EOS requires volume in X value.",
    "For categorical lines, leave ALL X values blank within a figure; identifiers become category labels.",
    "Specify axis units and normalization in settings. ReaxKit never changes signs or normalization.",
    "Formulas require cached results: recalculate and save in Excel before running the command.",
    "Legacy data are preserved as cached numeric values; original formulas/charts remain in the _legacy.xlsx backup.",
    "Legacy line data were assigned sequential X indices because the old table had no numeric X column.",
    "Verify example axis units before manuscript use. Blank titles are allowed.",
    "Excel charts are previews only; they do not dynamically split by figure number. ReaxKit exports do.",
    'reaxkit plot-from-excel --input template_ffield_opt_figure_generator.xlsx --output manuscript_figures --plot-style publication-bold',
    "Use --width 8 --height 5.2 for consistent panel dimensions; --formats png svg pdf; --overwrite to replace outputs.",
]
for row, note in enumerate(notes, start=9):
    instructions.cell(row, 1, note)
    instructions.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    instructions.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="center")
    instructions.row_dimensions[row].height = 34
for column in "ABCDEFGH":
    instructions.column_dimensions[column].width = 18
workbook.save(TARGET)
workbook.close()
cached.close()
figures = read_figures(TARGET)
assert len(figures) == 3
for figure in figures:
    assert len(figure["rows"]) == len(source_rows[figure["sheet"]])
    assert [(row["reaxff"], row["qm"]) for row in figure["rows"]] == [tuple(row[2:4]) for row in source_rows[figure["sheet"]]]
print(f"Migrated {TARGET}; verified every existing numeric value. Original saved as {BACKUP.name}.")
