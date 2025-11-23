
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
from matplotlib.ticker import MaxNLocator
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
freq1 = 38000.   # Frequency 1
freq2 = 38000.  # Frequency 2 to substract from frequency 1

# Frequency selection (70 kHz)
freq = 38000.
Threshold = -90 # (dB) Filter the data and ignore data below Threshold (dB)

Transducer_Depth = 254 #m

# Time selection
start_time = "2024-11-03 14:44:00" # Select subset of data
end_time   = "2024-11-03 18:44:00" # Select subset of data 

# Range selection
start_range = 5 # m
end_range   = 350 # m

# Example dataset with herring schools, two (crimac-scratch/test_data/dBDiff)
# No preprocessing in Korona (ie averaging)
# Assumes regular grid across frequencies
if local==1:
    # f = '/mnt/z/tmp/test_BlueEco/LoVe/2018/DayExample/out/netcdfLoVe_2018_N1.test.zarr'
    print(' Data in local?')
elif local==0:
    f='/data/prosjekt/16034-AquaPlan/ACOUSTIC/GRIDDED_AquaPlanTest.zarr'
    # f='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/MonthExample2/out/LoVe_2018_N1_2.month_sv.zarr'
    # f='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/MonthExample1/out/LoVe_2018_N1_1.month_sv.zarr'
    # f='/data/crimac-scratch/tmp/test_BlueEco/LoVe/2018/MonthExample/out/LoVe_2018_N1.month_sv.zarr'
     

freqs=[freq1, freq2]
print(f)
# data=xr.open_dataset(f,engine="zarr")
data = xr.open_dataset(f, engine="zarr", chunks={"ping_time": 100, "range": 500})

print('type(data)    : ',type(data) )
print(data.coords)
print(data.data_vars)

ping_times = data['ping_time']
print(ping_times)
print(data['frequency'])

# Excel file:
ExcelFolder = '/data/prosjekt/16034-AquaPlan/EXPERIMENTS/Herring_Experiments/'
df_excel = pd.read_excel(os.path.join(ExcelFolder, 'treatments_herring.xlsx'))
df_excel['Start'] = pd.to_datetime(df_excel['Start'], format='%m/%d/%Y %H:%M')
df_excel['End']   = pd.to_datetime(df_excel['End'],   format='%m/%d/%Y %H:%M')
print('df_excel.head(): ', df_excel.head())
print(df_excel['Start'])
print(df_excel['End'])
# select Block No "BlockNum":
BlockNum = 1
df_block1 = df_excel[df_excel['BlockNo'] == BlockNum]
print(df_block1)
print(df_block1['Start'])
print(df_block1['End'])
# Select sv between times at 70 kHz
# sv_sel = data['sv'].sel(
#     ping_time=slice(start_time, end_time),
#     range=slice(start_range,end_range),
#     frequency=freq
# )
sv_sel = (
    xr.open_dataset(
        f,
        engine="zarr",
        chunks={"ping_time": 500, "range": 2000}
    )['sv']
    .sel(
        ping_time=slice(start_time, end_time),
        range=slice(start_range, end_range),
        frequency=freq
    )
)

Delta_R = float(sv_sel['range'].diff('range').isel(range=0))
print('Full resolution DeltaR = ', Delta_R)

# ===== Resampling for have coarser time and range resolution ==========
Range_resolution = 0.05 # range resolution of 1 m
range_bins = int(Range_resolution / Delta_R) #  

dataSel=sv_sel.resample(ping_time="1min").mean(dim=["ping_time"]).coarsen(range=range_bins, boundary="trim").mean()
# dataSel=sv_sel.resample(ping_time="30s").mean(dim=["ping_time"]).coarsen(range=range_bins, boundary="trim").mean()

# Compute new range coordinates as the mean of each coarsened bin
new_range = sv_sel['range'].coarsen(range=range_bins, boundary="trim").mean().values

# Assign the new range coordinate (length matches coarsened dimension)
dataSel = dataSel.assign_coords(range=("range", new_range))
sv_sel = dataSel

Delta_R = float(sv_sel['range'].diff('range').isel(range=0))
print('DeltaR for resamples = ', Delta_R)
# =====================================================================


# Compute dB lazily
sv_sel_db = 10 * xr.apply_ufunc(
    np.log10,
    sv_sel,
    dask='allowed'
)

# Ensure coordinates are not NaN
ping_time_valid = sv_sel_db['ping_time'].dropna(dim='ping_time', how='any')
range_valid = sv_sel_db['range'].dropna(dim='range', how='any')

sv_sel_db = sv_sel_db.sel(
    ping_time=ping_time_valid,
    range=range_valid
)

# Mask low values
Mask = sv_sel_db > Threshold
sv_sel_db = sv_sel_db.where(Mask, -300)
sv_sel = sv_sel.where(Mask, 1E-30)



# Optional: sort coordinates just in case
sv_sel_db = sv_sel_db.sortby('ping_time').sortby('range')

# sv_sel_loaded = sv_sel.load()  # converts to in-memory numpy array
# sv_sel_db = 10 * np.log10(sv_sel_loaded)
# print('type(sv_sel),  type(sv_sel_threhholds)', type(sv_sel), type(sv_sel_threhholds))
# print(sv_sel)




# Set min/max for visualization
vmin = -82
vmax = -50

# fig, ax = plt.subplots(figsize=(12, 6))

# # xarray plotting
# sv_sel_db.plot.pcolormesh(
#     x='ping_time', 
#     y='range', 
#     ax=ax, 
#     cmap='viridis', 
#     vmin = vmin,
#     vmax = vmax
# )

# ax.set_title('Sv at 70 kHz (dB)')
# ax.set_xlabel('Time')
# ax.set_ylabel('Range (m)')
# # ax.set_ylim(0, 250)       # optional: limit range
# ax.invert_yaxis()         # optional: depth increasing downwards

# # plt.show()
# fig.savefig('sv_70kHz.png', dpi=150)
# plt.close(fig)


#==== Bottom detection ==================
sv_threshold = -35
grad_threshold = 1  # adjust

# Compute gradient (dSv/dr)
dsv = sv_sel_db.differentiate("range")

# Boolean mask for bottom
cond = (sv_sel_db > sv_threshold) & (dsv > grad_threshold)

# Find bottom index per ping safely
bottom_index = cond.argmax(dim="range")

# Check if a ping has no True values
no_bottom = ~cond.any(dim="range")

# Assign NaN for those pings
bottom_index = bottom_index.where(~no_bottom, other=np.nan)

# Convert index to depth
# bottom_index has NaNs where no bottom detected
valid_idx = ~np.isnan(bottom_index)

# Create an empty array to store bottom depths
bottom_depth_values = np.full_like(bottom_index.values, np.nan, dtype=float)

# Assign depths only for valid indices
bottom_depth_values[valid_idx] = sv_sel_db.range.values[bottom_index.values[valid_idx].astype(int)]

# Convert back to DataArray
bottom_depth = xr.DataArray(
    bottom_depth_values,
    coords={'ping_time': sv_sel_db.ping_time},
    dims=['ping_time'],
    name='bottom_depth'
)
#================================================

# ====== Plot Bottom depth values ==============
bottom_plot = bottom_depth.hvplot(
    x='ping_time',
    y='bottom_depth',
    color='red',
    linewidth=2,
    xlabel='Ping Time',
    ylabel='Depth (m)',
    title='Detected Bottom Depth',
    height=400,
    width=1200
).opts(ylim=(bottom_depth.max(), bottom_depth.min()))
hv.save(bottom_plot, './OutputData/bottom_depth.png')   # PNG
#================================================

# # Remove NaNs in coordinates to avoid hvplot errors
# sv_sel_db = sv_sel_db.dropna(dim='ping_time', how='any')
# sv_sel_db = sv_sel_db.dropna(dim='range', how='any')
print('sv_sel_db.values.min(), sv_sel_db.values.max() >>', 
      sv_sel_db.values.min(), sv_sel_db.values.max())

# compute numeric min/max of the coordinate
ymin = float(sv_sel_db['range'].min())
ymax = float(sv_sel_db['range'].max())

plot = sv_sel_db.hvplot(
    x='ping_time',
    y='range',
    cmap='viridis',
    clim=(vmin, vmax),
    width=1200,
    height=600,
    xlabel='Ping Time',
    ylabel='Range (m)',
    title='Sv at 38 kHz (dB)',
)

# set ylim with max first to invert axis
plot = plot.opts(ylim=(ymax, ymin))
plot
# Save as PNG, PDF, or SVG
hv.save(plot, 'OutputData/sv_38kHz.png')  # PNG
# hv.save(plot, 'sv_70kHz.pdf')  # PDF
# hv.save(plot, 'sv_70kHz.svg')  # SVG


# Detected bottom curve
# bottom_depth is a DataArray indexed by ping_time
bottom_curve = hv.Curve(
    (sv_sel_db['ping_time'].values, bottom_depth.values),
    'Ping Time', 'Range (m)',
    label='Detected Bottom'
).opts(color=[1,1,1], linewidth=4)

# Overlay
final_plot = plot * bottom_curve
final_plot

# Save
hv.save(final_plot, 'OutputData/sv_38kHz_with_bottom.png')


end = time.time()
print(f"Runtime before Urmy parameter Calcs: {end - start:.2f} seconds")

# Plot Sv values for a single ping, i.e. sv(range)
one_ping = sv_sel_db.sel(ping_time=sv_sel_db.ping_time[100])
OnePing = one_ping.hvplot(
    x='sv',       # value on x-axis
    y='range',
    flip_yaxis=True,
    title='Sv profile (single ping)',
    xlabel='Sv (dB)',
    ylabel='Range (m)',
    width=600,
    height=600
)

hv.save(OnePing , 'OutputData/single_ping.png')



# ||||||||||| Keep Sv only where range <= bottom_depth |||||||||||||||||||||||
# ||||||||||  Mask values below bottom  |||||||||||||||||||||||||||||||||||||

# 1) Make mask
Shifted_Bottom = 3 # m
mask = sv_sel_db.range <= (bottom_depth - Shifted_Bottom)

# 2) Apply mask
sv_above_bottom = sv_sel_db.where(mask)

# |||||||||||||||||||||||||||||||||||||||||||||||||

plot_above_bottom = sv_above_bottom.hvplot(
    x='ping_time',
    y='range',
    cmap='viridis',
    clim=(vmin, vmax),
    width=1200,
    height=600,
    xlabel='Ping Time',
    ylabel='Range (m)',
    title='Sv above detected bottom',
)

plot_above_bottom = plot_above_bottom.opts(ylim=(ymax, ymin))

hv.save(plot_above_bottom, "OutputData/sv_above_bottom_38kHz.png")

# Plot Sv values for a single ping, i.e. sv(range)
one_ping = sv_above_bottom.sel(ping_time=sv_above_bottom.ping_time[100])
OnePing = one_ping.hvplot(
    x='sv',       # value on x-axis
    y='range',
    flip_yaxis=True,
    title='Sv profile (single ping)',
    xlabel='Sv (dB)',
    ylabel='Range (m)',
    width=600,
    height=600
)

hv.save(OnePing , 'OutputData/single_ping_above_bottom.png')


#|||||||||||||||||||||||||||||||||||
# Sv_linear = 10^(Sv_dB / 10)

sv_linear = xr.apply_ufunc(
    lambda x: 10 ** (x / 10),
    sv_above_bottom,
    dask='allowed'
)
sv_sel = sv_linear # sv with bottom excluded

# Urmy parameters: =============================================
sv_values = sv_sel.values
print(sv_values.shape)  # (40, 76)
print(len(sv_values[0]))

# sum across the range dimension (i.e. collapse depth bins into one value per ping)
sv_sum_range = sv_sel.sum(dim="range")

# From Urmy parameters Urmy et al 2012 - ICES J Marine Science
Integrate_sv_dz = ( Delta_R * sv_sum_range ) # as a function of ping time
# Abundance = 10*np.log10( Integrate_sv_dz ) # as a function of ping time

# Abundance in dB using xarray's built-in log10
# Abundance
Abundance = xr.apply_ufunc(
    np.log10, 
    Integrate_sv_dz,
    dask='allowed'
) * 10

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, Abundance)
ax.set_title('Abundance')
# ax.set_xlabel('Ping Time (d HH:MM)')
ax.set_xlabel=('Ping Time (mm-dd HH)')
fig.savefig('OutputData/Abundance.png', dpi=150)
plt.close(fig)


# Density = 10*np.log10( Integrate_sv_dz/(end_range - start_range) )
# Density
Density = xr.apply_ufunc(
    np.log10, 
    Integrate_sv_dz / (end_range - start_range),
    dask='allowed'
) * 10


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, Density)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
ax.set_title('Density')
# ax.set_xlabel('Ping Time (d HH:MM)')
ax.set_xlabel=('Ping Time (mm-dd HH)')
fig.savefig('OutputData/Density.png', dpi=150)
plt.close(fig)

# Range_val = sv_sel.coords['range'].values
# print(sv_sel["range"].shape)
# print(sv_sel.values.shape)

z_product_svz = sv_sel * sv_sel["range"]
z_product_svz_dz = z_product_svz * Delta_R

# print(z_product_svz)

CenterofMass = (z_product_svz_dz.sum(dim="range") ) / Integrate_sv_dz

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sv_sel.ping_time, CenterofMass)
ax.set_title('CenterofMass')
ax.invert_yaxis()   # This inverts the y-axis
# ax.set_xlabel('Ping Time (d HH:MM)')
ax.set_xlabel=('Ping Time (mm-dd HH)')
fig.savefig('OutputData/CenterofMass.png', dpi=150)
plt.close(fig)


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

# |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# |||||||| Plot Abundance and Center of Mass on the same plot |||||||||||
fig, ax1 = plt.subplots(figsize=(12, 6))

# --- Left axis: Abundance ---
color1 = [0, 0, 1]
ax1.plot(sv_sel.ping_time, Abundance, color=color1, label='Abundance')
ax1.set_xlabel('Ping Time', fontsize=14)            # increase xlabel font
ax1.set_ylabel('Abundance (dB)', fontsize=14)           # increase ylabel font
ax1.tick_params(axis='y', colors=color1, labelsize=12, width=2, length=6)  # tick size
ax1.tick_params(axis='x', labelsize=12, width=2, length=6)                # x-axis ticks
ax1.yaxis.label.set_color(color1)                  # set y-label color

# --- Right axis: Center of Mass ---
color2 = [1, 0, 0]
ax2 = ax1.twinx()
ax2.plot(sv_sel.ping_time, CenterofMass, color=color2, label='Center of Mass')
ax2.set_ylabel('Center of Mass (m)', fontsize=14)
ax2.tick_params(axis='y', colors=color2, labelsize=12, width=2, length=6)
ax2.yaxis.label.set_color(color2)
# Reverse right y-axis
ax2.invert_yaxis()

# --- Add treatment intervals using df_block1 ---
for _, row in df_block1.iterrows():
    start = row['Start']
    end = row['End']
    treatment = row['Treatment']

    ax1.axvspan(start, end, alpha=0.3, color=[1,0.8,0.5])  # light shading
    ax1.text(
        x=start + (end - start) / 2,   # middle of interval
        y=ax1.get_ylim()[1] * 1.02,    # near top of plot
        s=treatment,
        ha='center',
        va='top',
        fontsize=10,
        rotation=90
    )


# --- Title ---
plt.title('Abundance and Center of Mass vs Time', fontsize=16)

# --- Combined legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)

# --- Save ---
fig.savefig('OutputData/Abundance_CoM.png', dpi=150)


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
# print(sv_downsampled.values.min(), sv_downsampled.values.max())

# x = sv_downsampled.ping_time.values
# y = Transducer_Depth - sv_downsampled.range.values

# # -------------------------- # 
# # Create figure with 2 subplots (N x 1) 
# # # -------------------------- 

# fig, axs = plt.subplots(
#     5, 1, figsize=(16, 12), 
#     gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]}, 
#     constrained_layout=True
# )

# # --------------------------
# # Top subplot: Echogram
# # --------------------------
# X, Y = np.meshgrid(mdates.date2num(x), y)

# im = axs[0].pcolormesh(
#     X, Y, sv_downsampled.values.T,
#     cmap='viridis',
#     vmin=vmin,
#     vmax=vmax,
#     shading='auto'   # important: aligns with hvplot/QuadMesh
# )

# axs[0].set_ylabel("Depth (m)", fontsize=20)
# axs[0].invert_yaxis()

# cbar = fig.colorbar(
#     im,
#     ax=axs[0],
#     label="Sv (dB)",
#     fraction=0.04,
#     pad=0.01
# )
# cbar.ax.yaxis.label.set_size(16)

# axs[0].xaxis_date()
# axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# # axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
# axs[0].tick_params(axis='x', labelsize=20)
# axs[0].tick_params(axis='y', labelsize=20)

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
# axs[1].set_ylabel("CM", fontsize=18)
# axs[1].set_xlim(x[0], x[-1])
# axs[1].invert_yaxis()  # keep aligned with echogram

# axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# # axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
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
# # axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
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
# # axs[3].set_xlabel("Ping Time (mm-dd hh)", fontsize=20)
# axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# # axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
# axs[3].set_ylabel("Inertia", fontsize=20)
# axs[3].set_xlim(x[0], x[-1])
# axs[3].tick_params(axis='x', labelsize=20)  # x-axis ticks
# axs[3].tick_params(axis='y', labelsize=20)  # y-axis ticks

# # --------------------------------
# #     Include Hydrophone Data
# # --------------------------------
# cwd = os.getcwd()
# # print("Current working directory:", cwd)
# Hyd_Dir = os.path.join(cwd, 'Hyd_data')

# csv_file = "LoVe_hourly.csv"
# # csv_file = "March_4th_hourly.csv"
# csv_path = os.path.join(Hyd_Dir, csv_file)

# # Load CSV into a DataFrame
# df = pd.read_csv(csv_path)
# df['time'] = pd.to_datetime(df['TIME'])
# df['time_num'] = mdates.date2num(df['time'])

# # 🔹 Filter to echogram time range
# tmin = pd.to_datetime(x[0])
# tmax = pd.to_datetime(x[-1])
# df = df[(df['time'] >= tmin) & (df['time'] <= tmax)]

# # Bar width (≈ 1 hour in days → 1/24 ≈ 0.0417)
# width = 0.02

# # Side-by-side bars
# axs[4].bar(df['time_num'] - width/3, df['Arithmean_63_dB'], width=width, color='b', label='TOL 63 Hz')
# axs[4].bar(df['time_num'] + width/3, df['Arithmean_125_dB'], width=width, color='r', label='TOL 125 Hz')

# axs[4].set_xlabel("Ping Time (mm-dd hh)", fontsize=20)
# # axs[4].set_xlabel("Ping Time", fontsize=20)
# axs[4].set_ylabel("SPL", fontsize=20)

# top_ticks = axs[0].get_xticks()

# # Apply same ticks to bottom subplot
# axs[4].set_xticks(top_ticks)

# # Keep same x-axis range as echogram
# axs[4].set_xlim(mdates.date2num(x[0]), mdates.date2num(x[-1]))
# axs[4].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
# # axs[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
# axs[4].tick_params(axis='x', labelsize=20)
# axs[4].tick_params(axis='y', labelsize=20)
# axs[4].legend()

# # --------------------------
# # Save
# # --------------------------
# plt.savefig("Nx1_month.png", dpi=300)
# plt.savefig("Nx1_month.tif", dpi=300)
# plt.close(fig)

# # Save values in a file
# # Convert Abundance and Inertia to pandas
# abundance_df = Abundance_hourly.to_dataframe(name="Abundance")
# inertia_df = Inertia_hourly.to_dataframe(name="Inertia")

# # Center of Mass (already DataFrame but align on time index)
# com_hourly_df = com_hourly.set_index("ping_time").rename(columns={"CenterOfMass": "CenterOfMass"})

# # Adjust depth for CoM
# com_hourly_df["CenterOfMass"] = Transducer_Depth - com_hourly_df["CenterOfMass"]

# # Merge all into one DataFrame on ping_time
# # Create a clean DataFrame
# merged = pd.DataFrame({
#     "ping_time": time_hourly.values,  # hourly ping times
#     "Abundance": Abundance_hourly.values,
#     "Inertia": Inertia_hourly.values,
#     "CenterOfMass": (Transducer_Depth - com_hourly["CenterOfMass"]).values
# })

# # Save to CSV
# outdir = "OutputData"
# os.makedirs(outdir, exist_ok=True)  # make sure folder exists

# outfile = os.path.join(outdir, "acoustic_summary_month.csv")
# merged.to_csv(outfile, index=False)

# print(f"Saved file to: {outfile}")

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