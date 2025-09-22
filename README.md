## An improved spatiotemporal savitzky-golay (iSTSG) method to improve the quality of vegetation index time-series data  
This Python module is an implementation of the improved spatial-temporal Savitzky–Golay filter  
Original code can be found here: https://github.com/wyWang365/iSTSG/blob/main/PYcode  

### Usage  
The input data is a numpy-array-like object in the shape of y, x, time.  
Make sure there is no NAN in the data.  
  
Here's an example:  
```
import xarray
import istsg


# Load some data (assume it's a NetCDF time series file)
with xarray.open_dataset(<path-to-data>) as Dataset:
    data = Dataset[<variable name>].load()

# Fill data gap
data_interp = data.interpolate_na(dim='time', method='linear').bfill(dim='time').ffill(dim='time')

# Smooth the data with iSTSG method
data_smooth = xarray.apply_ufunc(istsg.run_istsg, data_interp.fillna(-99),
                                 input_core_dims=[['y', 'x', 'time']], 
                                 output_core_dims=[['y', 'x', 'time']]).where(data_interp.notnull())

# Transpose dimension back to time, y, x (standard raster)
data_smooth = data_smooth.transpose('time', 'y', 'x')
```

### Reference of this study:  
Wang, W., Cao, R., Liu, L., Zhou, J., Shen, M., Zhu, X., Chen, J., 2025. 
    An Improved Spatiotemporal Savitzky–Golay (iSTSG) Method to Improve the Quality of Vegetation Index 
    Time-Series Data on the Google Earth Engine. 
    IEEE Transactions on Geoscience and Remote Sensing 63, 1–17. https://doi.org/10.1109/TGRS.2025.3528988
