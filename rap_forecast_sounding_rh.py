import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt
from siphon.catalog import TDSCatalog
import xarray as xr
import metpy
from metpy.plots import SkewT
from metpy.units import units
import pandas as pd

def time_dimensions(ds, var): 
    possible_time_dims = ['time', 'time1', 'time2', 'time3']
    time_dim = None
    for dim in possible_time_dims:
        if dim in ds[var].dims:
            time_dim = dim
            break
    if time_dim is None:
        raise ValueError('Could not find the time dimension')
    return time_dim

def fetch_data(ds, target_lat, target_lon):
    latitudes = ds['latitude'].values
    longitudes = ds['longitude'].values
    distances = np.sqrt((latitudes - target_lat)**2 + (longitudes - target_lon)**2)
    min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
    y_idx, x_idx = min_distance_idx
    ds = ds.isel(x=x_idx, y=y_idx)
    return ds

tds_rap = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/latest.html')

rap_ds = tds_rap.datasets[0]
ds = xr.open_dataset(rap_ds.access_urls['OPENDAP'])
ds = ds.metpy.parse_cf()
ds =  ds.metpy.assign_latitude_longitude()

target_lat, target_lon =  42.66, -83.41

ds_point = fetch_data(ds, target_lat, target_lon)

ds_timedim = time_dimensions(ds_point, 'Temperature_isobaric')

max_forecasthours = 21
extended_run_times_utc = [3, 9, 15, 21]

if ds[ds_timedim][0].dt.hour in extended_run_times_utc:
    max_forecasthours = 51  

for i in range(0, max_forecasthours + 1):
    temperature = (ds_point['Temperature_isobaric'][i,:] - 273.15) * units.degC
    relative_humidity = ds_point['Relative_humidity_isobaric'][i,:]
    u_wind = ds_point['u-component_of_wind_isobaric'][i,:]
    v_wind = ds_point['v-component_of_wind_isobaric'][i,:]
    pressure = ds_point['isobaric'][:] / 100 * units.hPa

    dewpoint = mpcalc.dewpoint_from_relative_humidity(temperature, relative_humidity)

    df = pd.DataFrame({
    'Temperature': temperature,
    'Dewpoint': dewpoint,  
    'Pressure': pressure,  
    'u_wind': u_wind,
    'v_wind': v_wind
})
    p = df['Pressure'].values * units.hPa
    p_decrease = p[::-1]
    T = df['Temperature'].values * units.degC
    T_1 = df['Temperature'][::-1].values * units.degC
    Td = df['Dewpoint'].values * units.degC
    Td_1 = df['Dewpoint'][::-1].values * units.degC
    u = df['u_wind'].values * 1.94384* units.knots
    v = df['v_wind'].values * 1.94384* units.knots

    wnd_drct = mpcalc.wind_direction(u, v)
    wnd_spd = mpcalc.wind_speed(u, v)

    prof = mpcalc.parcel_profile(p_decrease, T_1[0], Td_1[0]).to('degC')
    #ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
    #ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
    #mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=50 * units.hPa)

    fig = plt.figure(figsize=(9, 9))
    skew = SkewT(fig, rotation=45)

    skew.plot_dry_adiabats(alpha=0.25, linewidth=1)
    skew.plot_moist_adiabats(alpha=0.25, linewidth=1)
    skew.plot_mixing_lines(alpha=0.25, linewidth=1)

    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    skew.plot_barbs(p, u, v)
    skew.plot(p_decrease, prof, 'k', linewidth=2, label='SBCAPE PARCEL PATH', linestyle='-', dashes=(3, 1))
    plt.ylabel('Pressure (hPa)')
    plt.xlabel('Temperature (C)')
    
    plt.title('{} RAP: Forecast Sounding | {} | FH: {}'.format(ds[ds_timedim][0].dt.strftime('%H00 UTC').item(), ds[ds_timedim][i].dt.strftime('%Y-%m-%d %H00 UTC').item(), i))
    plt.savefig('plots/models/rap/soundings/rh/sounding_{}.png'.format(i), dpi=450)
