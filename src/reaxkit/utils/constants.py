"""constants for different quantities such as electron charge, etc. across ReaxFF files."""

CONSTANTS = {
    # Electric field
    "electric_field_VA_to_MVcm": 100.0,  # V/Å → MV/cm

    # Energies
    "energy_kcalmol_to_eV": 0.0433634,

    # Fundamental constants
    "electron_charge_C": 1.602176634e-19,   # Coulomb
    "electron_charge_e": 1.0,               # dimensionless charge

    # Dipole moment
    "ea_to_debye": 4.80320427,              # e·Å → Debye
    "debye_to_ea": 0.20819434,              # Debye → e·Å

    # Polarization (dipole/volume)
    "ea3_to_uC_cm2": 1.602176634e+3,        # (e·Å)/Å³ → μC/cm²
}

