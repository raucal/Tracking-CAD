# Filtered and unfiltered comparison

Unfiltered: 34/45. Filtered: 36/45.

03a proposals excluded in the corner: 134. Changed primary masks: 13.

Recovered correct cases: front_far_rot, hmi_rot_side, power_far_rot, power_rot_side

Lost correct cases: sensor_rot_side, sensor_side

RGB, depth, camera, checkpoint hashes and reference keys were verified equal across all 45 cases. Both runs performed fresh inference; only CAD references were reused.

Same data and weights; corner heuristic requested after inspecting these cases. This is not an independent held-out validation.

| Image | Expected | Before | After | Before correct | After correct |
|---|---|---|---|---:|---:|
| cassette_far | 03_option_cassette | 03a_vertical_option_a | 05_control_board | False | False |
| cassette_rot_far | 03_option_cassette | 03_option_cassette | 03_option_cassette | True | True |
| cassette_rot_side | 03_option_cassette | 03_option_cassette | 03_option_cassette | True | True |
| cassette_side | 03_option_cassette | 03a_vertical_option_a | 03a_vertical_option_a | False | False |
| control_far | 05_control_board | 05_control_board | 05_control_board | True | True |
| control_far_ro | 05_control_board | 05_control_board | 05_control_board | True | True |
| control_rot_side | 05_control_board | 05_control_board | 05_control_board | True | True |
| control_side | 05_control_board | 05_control_board | 05_control_board | True | True |
| front_far | 01_front_housing | 01_front_housing | 01_front_housing | True | True |
| front_far_rot | 01_front_housing | 03b_vertical_option_b | 01_front_housing | False | True |
| front_rot_side | 01_front_housing | 03a_vertical_option_a | 03b_vertical_option_b | False | False |
| front_side | 01_front_housing | 01_front_housing | 01_front_housing | True | True |
| hmi_far | 02_control_panel | 02_control_panel | 02_control_panel | True | True |
| hmi_far_rot | 02_control_panel | 02_control_panel | 02_control_panel | True | True |
| hmi_rot_side | 02_control_panel | 03b_vertical_option_b | 02_control_panel | False | True |
| hmi_side | 02_control_panel | 02_control_panel | 02_control_panel | True | True |
| inverse_cassette | 03_option_cassette | 03_option_cassette | 03_option_cassette | True | True |
| inverse_control | 05_control_board | 05_control_board | 05_control_board | True | True |
| inverse_front | 01_front_housing | 01_front_housing | 01_front_housing | True | True |
| inverse_hmi | 02_control_panel | 02_control_panel | 02_control_panel | True | True |
| inverse_io | 04_io_terminals | 04_io_terminals | 04_io_terminals | True | True |
| inverse_power | 06_power_unit | 06_power_unit | 06_power_unit | True | True |
| inverse_rear | 00_rear_mounting_plate | 00_rear_mounting_plate | 00_rear_mounting_plate | True | True |
| inverse_sensor | 03a_vertical_option_a | 03a_vertical_option_a | 03a_vertical_option_a | True | True |
| inverse_termal | 07_thermal_core | 07_thermal_core | 07_thermal_core | True | True |
| io_far | 04_io_terminals | 03b_vertical_option_b | 03b_vertical_option_b | False | False |
| io_rot_far | 04_io_terminals | 03a_vertical_option_a | 06_power_unit | False | False |
| io_rot_side | 04_io_terminals | 04_io_terminals | 04_io_terminals | True | True |
| io_side | 04_io_terminals | 04_io_terminals | 04_io_terminals | True | True |
| power_far | 06_power_unit | 06_power_unit | 06_power_unit | True | True |
| power_far_rot | 06_power_unit | 03a_vertical_option_a | 06_power_unit | False | True |
| power_rot_side | 06_power_unit | 03a_vertical_option_a | 06_power_unit | False | True |
| power_side | 06_power_unit | 06_power_unit | 06_power_unit | True | True |
| rear_far | 00_rear_mounting_plate | 00_rear_mounting_plate | 00_rear_mounting_plate | True | True |
| rear_far_rot | 00_rear_mounting_plate | 00_rear_mounting_plate | 00_rear_mounting_plate | True | True |
| rear_rot_side | 00_rear_mounting_plate | 03a_vertical_option_a | 03a_vertical_option_a | False | False |
| rear_side | 00_rear_mounting_plate | 00_rear_mounting_plate | 00_rear_mounting_plate | True | True |
| sensor_module_far | 03a_vertical_option_a | 03a_vertical_option_a | 03a_vertical_option_a | True | True |
| sensor_rot_far | 03a_vertical_option_a | 03b_vertical_option_b | 03b_vertical_option_b | False | False |
| sensor_rot_side | 03a_vertical_option_a | 03a_vertical_option_a | 05_control_board | True | False |
| sensor_side | 03a_vertical_option_a | 03a_vertical_option_a | 00_rear_mounting_plate | True | False |
| termal_far | 07_thermal_core | 07_thermal_core | 07_thermal_core | True | True |
| termal_far_rot | 07_thermal_core | 07_thermal_core | 07_thermal_core | True | True |
| termal_rot_side | 07_thermal_core | 07_thermal_core | 07_thermal_core | True | True |
| termal_side | 07_thermal_core | 07_thermal_core | 07_thermal_core | True | True |

The old unfiltered bundle was removed under the two-report retention policy. Per-case before/after results and input-equality checks remain in `filter_comparison.json` and `filter_comparison.csv`; recorded paths describe the original execution environment.
