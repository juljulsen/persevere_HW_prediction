### Util funcs

## great circle distance (central angle)
def _circ_dist(lon1, lat1, lon2, lat2):
    delta_lat = lat2-lat1
    return(np.arccos(np.sin(lon1)*np.sin(lon2) + np.cos(lon1)*np.cos(lon2)*np.cos(delta_lat)))

## slower since _circ_dist is calculated for each file
def _preprocess(x, lon, lat):
    _idx = _circ_dist(x.longitude, x.latitude, lon, lat).argmin()     
    return x.isel( values = _idx,drop=True )

## prob. faster to first load the idx
def _preprocess2(x, idx):
    return x.isel( values = idx,drop=True )

def _extract_idx(path, lon, lat):
    ds_idx = xr.open_dataset(path, indexpath='')
    return _circ_dist(ds_idx.longitude, ds_idx.latitude, lon, lat).argmin() 

def count_cons(tru_arr):
    N = len(tru_arr)
    _counter = 0
    _out = np.zeros(N, dtype = "int")
    for i in range(N):
        if tru_arr[i]:
            _counter += 1
        else:
            _out[i-_counter:i] = _counter 
            _counter = 0
    #if not _counter == 0:
    _out[N-_counter:] = _counter
    return _out




### opening data
def open_h_data(paths_Th,lon, lat):
    ds = xr.open_mfdataset(paths_Th , preprocess = lambda x: _preprocess(x, lon, lat) , indexpath='');
    T_max=ds.drop(
        ['step', 'surface', 'number', 'valid_time']).resample(
    time="1D").max(dim="time").rename({"t2m": "t2m_max"})
    T_min=ds.drop(
        ['step', 'surface', 'number', 'valid_time']).resample(
        time="1D").min(dim="time").rename({"t2m": "t2m_min"})
    T_avg=ds.drop(
        ['step', 'surface', 'number', 'valid_time']).resample(
        time="1D").mean(dim="time").rename({"t2m": "t2m_avg"})
    return T_max, T_min, T_avg


def open_d_data(paths_dew, paths_sf_pr, paths_msl_pr, paths_tot_prec, lon, lat):
    T_dew = xr.open_mfdataset(paths_dew,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step', 'surface', 'valid_time']).resample(time='1D').first();
    Pr_sf = xr.open_mfdataset(paths_sf_pr,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step', 'surface', 'valid_time']).resample(time='1D').first();
    P_sl = xr.open_mfdataset(paths_msl_pr,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step', 'surface', 'valid_time']).resample(time='1D').first();
    Prec = xr.open_mfdataset(paths_tot_prec,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step', 'surface', 'valid_time']).resample(time='1D').first();
    return T_dew, Pr_sf, P_sl, Prec

def open_pl_data(paths_u, paths_v, lon, lat):
    u_wind = xr.open_mfdataset(paths_u,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step',  'valid_time']).resample(time='1D').first();
    v_wind = xr.open_mfdataset(paths_u,
                              preprocess = lambda x: _preprocess(x, lon, lat),
                              indexpath='').drop(['step',  'valid_time']).resample(time='1D').first();
    
    bot_lay = u_wind.isobaricInhPa==1000
    U_sq = (v_wind.isel(isobaricInhPa=bot_lay, drop=True)**2+u_wind.isel(isobaricInhPa=bot_lay, drop=True)**2).drop("isobaricInhPa").squeeze()
    return U_sq

def combine_ds(T_max, T_min, T_dew, T_avg, Prec, Pr_sf, P_sl, U_sq):
    ds_Feat = xr.merge([T_max, T_min, T_dew, T_avg, Prec, Pr_sf, P_sl, U_sq])
    return ds_Feat.to_dataframe()
