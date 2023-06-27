# graphite-discharge
code for graphite discharge modeling project

| Data file      | Plot | Description |
| ----------- | ----------- | ------- | 
| COMSOL_5block_sweep_hours_x_k.csv      | ideal_vs_realistic_tin_outlet.pdf       | discharge sweep over multiple hours for k=x. Plot was made for hours = 10, and k = 5, 10, 30 to demonstrate thermal energy vs. electrochemical storage. no change in flowrate
| COMSOL_5block_10_k_fastcharge_sweep_basemaxflow.csv   | COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_x_y_y.pdf and COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_x.pdf        | charge sweep over multiple hours x, and multiple maximum flowrate factors y
| COMSOL_5block_10_k_fastcharge_sweep_log_maxflow_SOC.csv | COMSOL_5block_10_k_fastcharge_sweep_log_maxflow_SOC_P_31hrs_comp.pdf | charge sweep as above, also including P vs. SOC curves
| COMSOL_5block_10_k_fastcharge_sweep_basemaxTin_cap_P.csv | COMSOL_5block_10_k_fastcharge_sweep_basemaxTin_x_y_y.png and OMSOL_5block_10_k_fastcharge_sweep_basemaxTin_x.png| charge sweep over multiple hours x, and multiple maximum tin temperatures y
| COMSOL_5block_10_k_fastcharge_sweep_rad_gap_cap_thin(_2).csv | COMSOL_5block_10_k_fastcharge_sweep_rad_gap.pdf | charge sweep for 5 hours case, over multiple spacings between the tin tube and surrounding graphite 
| COMSOL_5block_10_k_discharge_sweep_log_maxflow.csv | COMSOL_5block_10_k_discharge_sweep_maxflow_x_y.png and COMSOL_5block_10_k_discharge_sweep_maxflow_contour.png | discharge sweep over hours x and max flowrate factors y to achieve constant discharge power
| COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC.csv | COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC_P_31hrs_comp.pdf | discharge sweep as above, also including P vs. SOC curves, and TPV area required for constant discharge power
| COMSOL_5block_10_k_charge_discharge_10maxflow(_halfDC)(_reverse).csv | COMSOL_5block_10_k_charge_discharge_10maxflow(_halfDC)(_reverse).pdf | charge discharge dynamics of 10x max flowrate for 24 hours (4 hours charging), and also at half the duty-cycle
| COMSOL_5block_10_k_porousmedia_coarse_rad_x.csv | COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_configs.pdf | discharge data for 20 hours for different configurations x of grid-scale storage blocks (vertical string, horizontal string, 10x10 1 path, 10x10 10 paths, 10x10 100 paths)
| COMSOL_5block_10_k_discharge_10x10grid_10parallel_maxflow_5.0_10.0.csv | COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_10x10_varyflow.pdf | discharge data for 10x10 10 paths grid, employing the varying flowrate method for constant power discharge
| COMSOL_5block_10_k_charge_10x10grid_10parallel_maxflow_1.0_5.0_10.0.csv | COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_10x10_varyflow_charging.pdf | charge data for 10x10 10 paths grid, with various max flowrate factors (1,5,10) for accelerated charging
