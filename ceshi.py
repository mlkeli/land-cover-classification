import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import shape

def change_shape(input_raster):
    with rasterio.open(input_raster) as src:
        data = src.read(1)

        # Convert to binary image
        _, binary = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY)

        # Find contours
        contours = shapes(binary, mask=None, connectivity=4, transform=src.transform)

        # Create shapefile
        output_shapefile = input_raster[:-4] + ".shp"
        schema = {'geometry': 'Polygon', 'properties': {'value': 'int'}}
        with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema, src.crs) as output:
            for geom, _ in contours:
                output.write({'properties': {'value': 1}, 'geometry': shape(geom)})

change_shape(r"D:\change_shape\tif4.tif")
