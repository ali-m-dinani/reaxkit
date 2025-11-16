"""constants for different quantities such as energies, density, etc. across ReaxFF files."""

UNITS = {
    # Common units

    # molfra.out units
    "molecular_mass": "a.u.",
    "total_molecular_mass": "a.u.",

    # fort.78 alias
    "field_x": "V/Å",
    "field_y": "V/Å",
    "field_z": "V/Å",

    "E_field_x": "kcal/mol",
    "E_field_y": "kcal/mol",
    "E_field_z": "kcal/mol",
    "E_field": "kcal/mol",

    # summary.txt units
    "time": "fs",
    "E_pot": "kcal/mol",
    "V": "Å³",
    "T": "K",
    "P": "MPa",
    "D": "kg/dm³",
    "elap_time": "s",

    # eregime.in
    "field": "V/Å",
    "field1": "V/Å",
    "field2": "V/Å",
    "field3": "V/Å",

}
