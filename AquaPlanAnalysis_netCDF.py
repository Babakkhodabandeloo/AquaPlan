
#
# Assuming a dataset has been prepared and available (xarray/zarr)
# i.e. the CRIMAC preprocessing pipeline has been run (raw->zarr)
# Here we use a test data set (two files) from the annual 2024 NVG herring spawning survey
# The data includes clear herring schools midwater on the shelf outside Vesterålen
#

# Typical steps
# Bottom detection (not performed here yet)
# Remove samples bellow the seafloor (not performed here yet)
# Potential noise reduction (remove spikes etc.) (not performed here yet)
# Resample/average/smooth the two variables
# Calculate dB difference
# Select data that meets the specified "dB difference" criteria (not performed here yet)

#
# Some other important factors include common observation range (ie higher frequencies have
# shorter observation range due to attenuation)
#

import os
import sys
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import holoviews as hv
import hvplot.xarray  # enables hvplot on xarray
hv.extension('matplotlib')  # use Matplotlib backend
# hv.extension('bokeh')
import pandas as pd

start = time.time()

local = 0 # 1: Running tests on local machine, 0-> on IMR server
average_data = 1 # 1->Average data, 0->No averaging
freq1 = 70000.   # Frequency 1
freq2 = 70000.  # Frequency 2 to substract from frequency 1

# Frequency selection (70 kHz)
freq = 70000.
Threshold = -80 # (dB) Filter the data and ignore data below Threshold (dB)

Transducer_Depth = 254 #m

# Time selection
start_time = "2018-02-15 01:40:25" # Select subset of data
end_time   = "2018-02-15 01:43:24" # Select subset of data

# Range selection
start_range = 20 # m
end_range   = 220 # m

# # Example dataset with herring schools, two (crimac-scratch/test_data/dBDiff)
# # No preprocessing in Korona (ie averaging)
# # Assumes regular grid across frequencies
# if local==1:
#     f = '/mnt/z/tmp/test_BlueEco/LoVe/2018/DayExample/out/netcdfLoVe_2018_N1.test.zarr'
# elif local==0:
#     # f = '/data/crimac-scratch/test_data/dBDiff/ACOUSTIC/GRIDDED/out/S2024204001_sv.zarr'
#     f='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/DayExample/out/netcdfLoVe_2018_N1.test.zarr'

# freqs=[freq1, freq2]
# print(f)
# # data=xr.open_dataset(f,engine="zarr")
# # data = xr.open_dataset(f, engine="zarr", chunks={"ping_time": 100, "range": 500})

# # print('type(data)    : ',type(data) )
# # print(data.coords)
# # print(data.data_vars)

# # ping_times = data['ping_time']
# # print(ping_times)
# # print(data['frequency'])

# # 


# # Select sv between times at 70 kHz
# # sv_sel = data['sv'].sel(
# #     ping_time=slice(start_time, end_time),
# #     range=slice(start_range,end_range),
# #     frequency=freq
# # )
# sv_sel = (
#     xr.open_dataset(
#         f,
#         engine="zarr",
#         chunks={"ping_time": 1000, "range": 5000}
#     )['sv']
#     .sel(
#         ping_time=slice(start_time, end_time),
#         range=slice(start_range, end_range),
#         frequency=freq
#     )
# )

#----------------------------------------------------
#                   read NetCDF                     #  
#----------------------------------------------------
dataout = '/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/MonthExample/out/sv/'
nc_files = sorted([os.path.join(dataout, f) for f in os.listdir(dataout) if f.endswith('.nc')])

#||||||||||||||||||||||||||||||||||||||||||||||
#----------------------------------------------
# batch_size = 100
# processed_batches = []

# # ds = xr.open_mfdataset(
# #     nc_files[0],
# #     combine="nested",
# #     concat_dim="ping_time",
# #     engine="netcdf4",
# #     parallel=False
# # )
# for i in range(0, 101+0*len(nc_files), batch_size):
#     batch_files = nc_files[i:i+batch_size]
        
#     print(str(i)+' of '+str(len(nc_files)))
#     ds = xr.open_mfdataset(
#         batch_files,
#         combine="nested",
#         concat_dim="ping_time",
#         engine="netcdf4",
#         parallel=False
#     )

# # print('type(ds)  ', type(ds))
# print(' ds   ',  ds)
# print(' ds["ping_time"].values   ',  ds["ping_time"].values)

# sv_sel = ds['sv'].sel(
#     # ping_time=slice(start_time, end_time),
#     range=slice(start_range, end_range),
#     frequency=freq
# ).chunk({"ping_time": 1000, "range": 5000})

# sv_sel_db = 10 * xr.apply_ufunc(
#     np.log10,
#     sv_sel,
#     dask='allowed'  # <-- this enables lazy computation with dask
# )

# print(sv_sel['ping_time'].values)   # all ping times as numpy datetime64
# print(sv_sel['range'].values)  

# # Set min/max for visualization
# vmin = -82
# vmax = -66

# # Create plot
# plot = sv_sel_db.hvplot(
#     x='ping_time',
#     y='range',
#     cmap='viridis',
#     clim=(vmin, vmax),
#     flip_yaxis=True,
#     width=2000,
#     height=800,
#     # xlabel='Ping Time (d HH:MM)',
#     xlabel='Ping Time (mm-dd HH)',
#     ylabel='Range (m)',
#     title='Sv at 70 kHz (dB)',
#     clabel='Sv (dB)',
#     fontsize={'title': 24, 'labels': 18, 'xticks': 16, 'yticks': 16,'cticks':14, 'clabel':14}  # increase font sizes
# )

# hv.save(plot, 'sv_70kHz.png')  # PNG

#----------------------------------------------
#||||||||||||||||||||||||||||||||||||||||||||||




batch_size = 10
processed_batches = []

for i in range(0, 101+0*len(nc_files), batch_size):
    batch_files = nc_files[i:i+batch_size]
        
    print(str(i)+' of '+str(len(nc_files)))
    ds = xr.open_mfdataset(
        batch_files,
        combine="nested",
        concat_dim="ping_time",
        engine="netcdf4",
        parallel=False
    )
    
    sv_sel = ds['sv'].sel(
        # ping_time=slice(start_time, end_time),
        range=slice(start_range, end_range),
        frequency=freq
    ).chunk({"ping_time": 1000, "range": 5000})
    
    # Optional: compute some stats or save intermediate result
    # e.g., mean over ping_time: sv_sel.mean(dim="ping_time").to_netcdf(f"batch_{i}.nc")
    processed_batches.append(sv_sel)  # or append computed result
#----------------------------------------------------
print('type(processed_batches)  ',type(processed_batches))

# Align batches
processed_batches_aligned = xr.align(*processed_batches, join='outer')  # 'outer' keeps all coordinates

# Then concatenate along ping_time
sv_sel = xr.concat(processed_batches_aligned, dim="ping_time")

print('type(sv_sel)  ',type(sv_sel))

Delta_R = float(sv_sel['range'].diff('range').isel(range=0))
print('DeltaR = ', Delta_R)

# sv_sel_db = 10 * np.log10(sv_sel)
sv_sel_db = 10 * xr.apply_ufunc(
    np.log10,
    sv_sel,
    dask='allowed'  # <-- this enables lazy computation with dask
)

# create mask from sv_sel_db
Mask = sv_sel_db > Threshold

# Apply mask
sv_sel_threshold = sv_sel.where(Mask, 1E-30)  # for linear scale
sv_sel_db = sv_sel_db.where(Mask, -300)   # for dB scale

sv_sel = sv_sel_threshold

# # Just mask, don't drop
# sv_sel_db_masked = sv_sel_db.where(~sv_sel_db.isnull(), other=-300)
# sv_sel_db = sv_sel_db_masked

# # sv_sel_loaded = sv_sel.load()  # converts to in-memory numpy array
# # sv_sel_db = 10 * np.log10(sv_sel_loaded)
# # print('type(sv_sel),  type(sv_sel_threhholds)', type(sv_sel), type(sv_sel_threhholds))
# # print(sv_sel)

# # dataSel=sv_sel.resample(ping_time="1min").mean(dim=["ping_time"]).coarsen(range=10, boundary="trim").mean()


# Set min/max for visualization
vmin = -82
vmax = -66

# Create plot
plot = sv_sel_db.hvplot(
    x='ping_time',
    y='range',
    cmap='viridis',
    clim=(vmin, vmax),
    flip_yaxis=True,
    width=2000,
    height=800,
    # xlabel='Ping Time (d HH:MM)',
    xlabel='Ping Time (mm-dd HH)',
    ylabel='Range (m)',
    title='Sv at 70 kHz (dB)',
    clabel='Sv (dB)',
    fontsize={'title': 24, 'labels': 18, 'xticks': 16, 'yticks': 16,'cticks':14, 'clabel':14}  # increase font sizes
)


# Save as PNG, PDF, or SVG
hv.save(plot, 'sv_70kHz.png')  # PNG
# hv.save(plot, 'sv_70kHz.pdf')  # PDF
# hv.save(plot, 'sv_70kHz.svg')  # SVG

# end = time.time()
# print(f"Runtime before Urmy parameter Calcs: {end - start:.2f} seconds")

# # Urmy parameters: =============================================
# # sv_values = sv_sel.values
# # print(sv_values.shape)  # (40, 76)
# # print(len(sv_values[0]))

# # sum across the range dimension (i.e. collapse depth bins into one value per ping)
# sv_sum_range = sv_sel.sum(dim="range")

# # From Urmy parameters Urmy et al 2012 - ICES J Marine Science
# Integrate_sv_dz = ( Delta_R * sv_sum_range ) # as a function of ping time
# # Abundance = 10*np.log10( Integrate_sv_dz ) # as a function of ping time

# # Abundance in dB using xarray's built-in log10
# # Abundance
# Abundance = xr.apply_ufunc(
#     np.log10, 
#     Integrate_sv_dz,
#     dask='allowed'
# ) * 10

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(sv_sel.ping_time, Abundance)
# ax.set_title('Abundance')
# # ax.set_xlabel('Ping Time (d HH:MM)')
# ax.set_xlabel=('Ping Time (mm-dd HH)')
# fig.savefig('Abundance.png', dpi=150)
# plt.close(fig)


# # Density = 10*np.log10( Integrate_sv_dz/(end_range - start_range) )
# # Density
# Density = xr.apply_ufunc(
#     np.log10, 
#     Integrate_sv_dz / (end_range - start_range),
#     dask='allowed'
# ) * 10


# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(sv_sel.ping_time, Density)
# ax.set_title('Density')
# # ax.set_xlabel('Ping Time (d HH:MM)')
# ax.set_xlabel=('Ping Time (mm-dd HH)')
# fig.savefig('Density.png', dpi=150)
# plt.close(fig)

# # Range_val = sv_sel.coords['range'].values
# # print(sv_sel["range"].shape)
# # print(sv_sel.values.shape)

# z_product_svz = sv_sel * sv_sel["range"]
# z_product_svz_dz = z_product_svz * Delta_R

# # print(z_product_svz)

# CenterofMass = (z_product_svz_dz.sum(dim="range") ) / Integrate_sv_dz

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(sv_sel.ping_time, CenterofMass)
# ax.set_title('CenterofMass')
# # ax.set_xlabel('Ping Time (d HH:MM)')
# ax.set_xlabel=('Ping Time (mm-dd HH)')
# fig.savefig('CenterofMass.png', dpi=150)
# plt.close(fig)


# # Z_minus_CM_mult_svzdz = ( sv_sel * ((sv_sel["range"] - CenterofMass )**2) ) * Delta_R
# # Inertia = (Z_minus_CM_mult_svzdz.sum(dim="range") ) / Integrate_sv_dz
# # Compute the squared deviation from Center of Mass along the range dimension
# squared_dev = (sv_sel["range"] - CenterofMass) ** 2

# # Multiply by Sv
# sv_times_squared_dev = sv_sel * squared_dev

# # Multiply by Delta_R to integrate along the range
# sv_times_squared_dev_dz = sv_times_squared_dev * Delta_R

# # Sum along range and normalize by Integrate_sv_dz
# Inertia = sv_times_squared_dev_dz.sum(dim="range") / Integrate_sv_dz

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(sv_sel.ping_time,Inertia)
# ax.set_title('Inertia')
# # ax.set_xlabel('Ping Time (d HH:MM)')
# ax.set_xlabel=('Ping Time (mm-dd HH)')
# fig.savefig('Inertia.png', dpi=150)
# plt.close(fig)

# end = time.time()
# print(f"Runtime: {end - start:.2f} seconds")


# # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# # Plot as "N x 1" sub-plots: >>>>>>>>>>>>>>

# # --------------------------
# # Downsample data
# # --------------------------
# sv_downsampled = sv_sel_db.isel(
#     range=slice(None, None, 2),   # keep every 2nd range bin
#     ping_time=slice(None, None, 5)  # keep every 5th ping
# )

# x = sv_downsampled.ping_time.values
# y = Transducer_Depth - sv_downsampled.range.values

# # --------------------------
# # Create figure with 2 subplots (N x 1)
# # --------------------------
# fig, axs = plt.subplots(
#     4, 1,
#     figsize=(16, 12),
#     gridspec_kw={'height_ratios': [3, 1, 1, 1]},
#     constrained_layout=True
# )

# # --------------------------
# # Top subplot: Echogram
# # --------------------------
# im = axs[0].imshow(
#     sv_downsampled.values.T,
#     origin='upper',
#     aspect='auto',
#     cmap='viridis', #'viridis', #'gist_ncar_r', 
#     vmin=vmin,
#     vmax=vmax,
#     extent=[mdates.date2num(x[0]), mdates.date2num(x[-1]), y[-1], y[0]]
# )

# axs[0].set_ylabel("Depth (m)", fontsize=20)
# axs[0].invert_yaxis()
# cbar = fig.colorbar(
#     im, 
#     ax=axs[0], 
#     label="Sv (dB)", 
#     fraction=0.04,  # thinner bar
#     pad=0.01        # less spacing
# )
# # Set label size
# cbar.ax.yaxis.label.set_size(16)

# axs[0].xaxis_date()
# axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# axs[0].tick_params(axis='x', labelsize=20)  # x-axis ticks
# axs[0].tick_params(axis='y', labelsize=20)  # y-axis ticks

# # --------------------------
# # Bottom subplot: Center of Mass
# # --------------------------
# com_df = pd.DataFrame({
#     "ping_time": pd.to_datetime(sv_sel.ping_time),
#     "CenterOfMass": CenterofMass
# })

# com_hourly = (
#     com_df.set_index("ping_time")
#     .resample("1h")["CenterOfMass"]
#     .mean()
#     .reset_index()
# )

# axs[1].plot(com_hourly["ping_time"], Transducer_Depth - com_hourly["CenterOfMass"], color="k")
# # axs[1].set_xlabel("Ping Time")
# axs[1].set_ylabel("Center of Mass", fontsize=18)
# axs[1].set_xlim(x[0], x[-1])
# axs[1].invert_yaxis()  # keep aligned with echogram

# axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# axs[1].tick_params(axis='x', labelsize=20)  # x-axis ticks
# axs[1].tick_params(axis='y', labelsize=20)  # y-axis ticks

# # --------------------------
# # Abundance (1-hour bins, dB)
# # --------------------------
# integrated_hourly = Integrate_sv_dz.resample(ping_time="1h").sum()

# Abundance_hourly = xr.apply_ufunc(
#     np.log10,
#     integrated_hourly,
#     dask='allowed'
# ) * 10

# axs[2].plot(Abundance_hourly["ping_time"], Abundance_hourly, color="k")
# # axs[2].set_xlabel("Ping Time", fontsize=18)
# axs[2].set_ylabel("Abundance", fontsize=20)
# axs[2].set_xlim(x[0], x[-1])
# axs[2].tick_params(axis='x', labelsize=20)  # x-axis ticks
# axs[2].tick_params(axis='y', labelsize=20)  # y-axis ticks

# # --------------------------
# # Inertia (1-hour mean, dB)
# # --------------------------
# Inertia_hourly = Inertia.resample(ping_time="1h").mean()
# time_hourly = Abundance_hourly["ping_time"]
# axs[3].plot(time_hourly, Inertia_hourly, color="k")
# axs[3].set_xlabel("Ping Time (mm-dd hh)", fontsize=20)
# axs[3].set_ylabel("Inertia", fontsize=20)
# axs[3].set_xlim(x[0], x[-1])
# axs[3].tick_params(axis='x', labelsize=20)  # x-axis ticks
# axs[3].tick_params(axis='y', labelsize=20)  # y-axis ticks
# # --------------------------
# # Save
# # --------------------------
# plt.savefig("Nx1.png", dpi=300)
# plt.close(fig)



# # # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# # #%% Plot all on top of Echogram |||||||||||||||||||||||||||||||||||||||
# # # hv.extension('bokeh')

# # # --------------------------
# # # Create Echogram
# # # --------------------------
# # # --------------------------
# # # 1. Downsample data
# # # --------------------------
# # sv_downsampled = sv_sel_db.isel(
# #     range=slice(None, None, 2),  # keep range within min/max
# #     ping_time=slice(None, None, 5)
# # )
# # # sv_flipped = sv_downsampled.rename({'range': 'depth'})
# # # sv_flipped = sv_flipped.assign_coords(depth=sv_flipped.depth[::-1])  # flip depth


# # # --------------------------
# # # 2. Echogram
# # # --------------------------
# # echogram = sv_downsampled.hvplot.quadmesh(
# #     x='ping_time',
# #     y='range',
# #     cmap='viridis',
# #     clim=(vmin, vmax),
# #     xlabel='Ping Time (mm-dd HH)',
# #     ylabel='Range (m)',
# #     title='Sv at 70 kHz (dB)',
# #     width=2000,
# #     height=800
# # ).opts(
# #     colorbar=False  # <- disable Holoviews colorbar
# # )

# # # --------------------------
# # # 3. Center of Mass (hourly average)
# # # --------------------------
# # com_df = pd.DataFrame({
# #     "ping_time": sv_sel.ping_time,
# #     "CenterOfMass": CenterofMass
# # })
# # com_hourly = com_df.set_index("ping_time").resample("1h")["CenterOfMass"].mean().reset_index()
# # # max_depth = sv_flipped.depth.max().item()

# # com_line = hv.Curve(
# #     (com_hourly["ping_time"], com_hourly["CenterOfMass"]),
# #     'ping_time', 'range'
# # ).opts(color='red', linewidth=2)

# # # --------------------------
# # # Abundance (1-hour bins, dB)
# # # --------------------------
# # integrated_hourly = Integrate_sv_dz.resample(ping_time="1h").sum()

# # Abundance_hourly = xr.apply_ufunc(
# #     np.log10,
# #     integrated_hourly,
# #     dask='allowed'
# # ) * 10

# # # final_plot = echogram * com_line 
# # # hv.save(final_plot, "All_in_One_without_abundance.png", fmt='png')




# # # final_plot = (echogram * com_line)
# # # # --------------------------
# # # # 5. Render with Matplotlib
# # # # --------------------------
# # # mpl_plot = hv.render(final_plot, backend='matplotlib')

# # # # Set figure size (width, height in inches)
# # # mpl_plot.figure.set_size_inches(15, 8)  # 20in wide, 10in tall
# # # ax = mpl_plot.axes[0]

# # # # Flip y-axis (depth increasing downward)
# # # ax.invert_yaxis()

# # # # Set yticks and labels
# # # ytick_positions = np.linspace(0, max_depth, 10)
# # # ax.set_yticks(ytick_positions)
# # # ax.set_yticklabels([f"{int(d)}" for d in ytick_positions])

# # # # Increase fonts
# # # ax.title.set_fontsize(16)
# # # ax.xaxis.label.set_fontsize(14)
# # # ax.yaxis.label.set_fontsize(14)
# # # ax.tick_params(axis='x', labelsize=12)
# # # ax.tick_params(axis='y', labelsize=12)

# # # # --------------------------
# # # # 6. Save figure
# # # # --------------------------
# # # mpl_plot.figure.savefig("All_in_One_without_abundance.png", dpi=200)



# # # final_plot = (echogram * com_line)
# # final_plot = echogram
# # # --------------------------
# # # 5. Render with Matplotlib
# # # --------------------------
# # mpl_plot = hv.render(final_plot, backend='matplotlib')

# # # # Set figure size (width, height in inches)
# # mpl_plot.figure.set_size_inches(19, 8)  # 20in wide, 10in tall
# # # ax = mpl_plot.axes[0]

# # fig, ax1 = mpl_plot.figure, mpl_plot.axes[0]


# # # Get the QuadMesh image object
# # im = ax1.collections[0]

# # # Add colorbar outside the plot
# # cbar = fig.colorbar(im, ax=ax1, pad=0.1)  # pad increases spacing
# # cbar.set_label("Sv (dB)", fontsize=22, color='black')

# # # Optional: set figure size
# # fig.set_size_inches(20, 10)


# # # Fonts
# # ax1.title.set_fontsize(20)
# # ax1.xaxis.label.set_fontsize(18)
# # ax1.yaxis.label.set_fontsize(18)
# # ax1.tick_params(axis='x', labelsize=18)
# # ax1.tick_params(axis='y', labelsize=18)

# # # Plot as red line on ax1
# # ax1.plot(
# #     com_hourly["ping_time"],
# #     com_hourly["CenterOfMass"],  # plot directly in range coordinates
# #     color='red',
# #     linewidth=2,
# #     label='Center of Mass'
# # )

# # # Show legend
# # ax1.legend(loc='upper right', fontsize=16, frameon=True)
# # # --------------------------
# # # 6. Add Abundance on secondary y-axis
# # # --------------------------
# # ax2 = ax1.twinx()
# # ax2.plot(Abundance_hourly["ping_time"], Abundance_hourly.values, color=[0,0.2,0.9], linewidth=2)
# # ax2.set_ylabel("Abundance (dB)", fontsize=24, color=[0,0.2,0.9])
# # ax2.tick_params(axis='y', labelsize=20, colors=[0,0.2,0.9])

# # # --------------------------
# # # 7. Save figure
# # # --------------------------
# # # fig.tight_layout()
# # fig.savefig("All_in_One_with_abundance_range.png", dpi=200)