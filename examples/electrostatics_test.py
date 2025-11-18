# from reaxkit.io.xmolout_handler import XmoloutHandler
# from reaxkit.io.fort7_handler import Fort7Handler
# from reaxkit.analysis.electrostatics_analyzer import frame_electrostatics, electrostatics_over_frames
#
# # --- Load files ---
# print("=== Loading files ===")
# xh = XmoloutHandler("xmolout")
# f7 = Fort7Handler("fort.7")
#
# print(f"Xmolout frames : {xh.n_frames()}")
# print(f"Fort.7 frames  : {f7.n_frames()}")
#
# # --- Single frame test (frame 0) ---
# test_frame = 0
#
# print(f"\n=== TOTAL polarization for frame {test_frame} ===")
# df_total = frame_electrostatics(
#     xh, f7,
#     frame=test_frame,
#     scope="total",
#     mode="polarization"
# )
# print(df_total.head())
# df_total.to_csv("df_total.csv")
#
# print(f"\n=== LOCAL polarization for frame {test_frame} around Al atoms ===")
# df_local = frame_electrostatics(
#     xh, f7,
#     frame=test_frame,
#     scope="local",
#     core_types=["Al"],
#     mode="polarization"
# )
# print(df_local.head())
# df_local.to_csv("df_local.csv")
#
# # --- Over all frames ---
# print("\n=== TOTAL polarization over all frames ===")
# df_total_all = electrostatics_over_frames(xh, f7, scope="total", mode="polarization")
# print(df_total_all.head())
# df_total_all.to_csv("df_total_all.csv")
#
# print("\n=== LOCAL polarization over all frames (core: Al) ===")
# df_local_all = electrostatics_over_frames(xh, f7, scope="local", core_types=["Al"], mode="polarization")
# print(df_local_all.head())
# df_local_all.to_csv("df_local_all.csv")


from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.io.control_handler import ControlHandler
from reaxkit.analysis.fort78_analyzer import match_electric_field_to_iout2
#
# # --- prepare handlers ---
# f78 = Fort78Handler("fort.78")
# ctrl = ControlHandler("control")
#
# # --- target iterations (simulated xmolout iters) ---
# target_iters = [0, 60, 120]
#
# # --- run matching ---
# matched_E = match_electric_field_to_iout2(
#     f78,
#     ctrl,
#     target_iters=target_iters,
#     field_var="field_z"   # or any valid column alias
# )
#
# print("Target iters:", target_iters)
# print("Matched Ez  :", matched_E.tolist())




from reaxkit.analysis.electrostatics_analyzer import polarization_field_analysis
from reaxkit.io.xmolout_handler import XmoloutHandler
from reaxkit.io.fort7_handler import Fort7Handler

# --- Assume these handlers are already created correctly ---
xh = XmoloutHandler("xmolout")
f7 = Fort7Handler("fort.7")
f78 = Fort78Handler("fort.78")
ctrl = ControlHandler("control")

# Call the function
full_df, agg_df, y_crossings, x_crossings = polarization_field_analysis(
    xh=xh,
    f7=f7,
    f78=f78,
    ctrl=ctrl,
    field_var="field_z",
    aggregate="mean",       # Try "max", "min", "last", or None
    x_variable="field_z",
    y_variable="P_z (uC/cm^2)",
)

# Print results
full_df.to_csv("full_data.csv")

agg_df.to_csv("aggregated_data.csv")

print("\n--- Zero Crossings (y = 0) ---")
print(y_crossings)

print("\n--- Zero Crossings (x = 0) ---")
print(x_crossings)
