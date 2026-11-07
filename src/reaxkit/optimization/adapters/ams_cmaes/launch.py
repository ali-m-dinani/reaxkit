"""SCM standalone CMA-ES mapping and explicit native restart input."""

import shutil
from pathlib import Path

from ..ams_mcff.launch import prepare as prepare_ams, validate


def prepare(request):
    plan = prepare_ams(request)
    if request["settings"].get("restart"):
        shutil.copyfile(request["settings"]["restart"], Path(request["native_directory"]) / "cmaes_restart.txt")
        plan["warnings"].append("CMA restart uses native last-improvement state, not necessarily the last completed generation.")
    return plan
