import ee
# import openet.core.landsat


def evi2(landsat_image):
    """Two band enhanced vegetation index

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Jiang2008] Jiang, Z., Huete, A., Didan, K., Miura, T. (2008).
        Development of a two-band enhanced vegetation index without a blue band.
        Remote Sensing of Environment, 112, 3833–3845,
        https://doi.org/10.1016/j.rse.2008.06.006


    """
    return landsat_image.expression(
        "2.5 * (nir - red) / (nir + red * 2.4 + 1)",
        {"red": landsat_image.select("red"), "nir": landsat_image.select("nir")}
    ).rename("evi2").clamp(0, 1.25)


def ndmi_scaled(landsat_image):
    """Normalized difference moisture index

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Gao1996] Gao B. (1996). NDWI—A normalized difference water index for
        remote sensing of vegetation liquid water from space.
        Remote Sensing of Environment, 58, 257-266,
        https://doi.org/10.1016/S0034-4257(96)00067-3

    """
    return (
        ee.Image(landsat_image).normalizedDifference(["nir", "swir1"])
        .unmask(0).add(0.3).divide(0.3).clamp(0, 1)
        .rename("ndmi_scaled")
    )


def lai(landsat_image):
    """Leaf area index (Kang et al., 2016 + NDMI correction)

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Kang2016] Kang et. al. (2016). How Universal Is the Relationship
        between Remotely Sensed Vegetation Indices and Crop Leaf Area Index?
        A Global Assessment. Remote Sensing, https://doi.org/10.3390/rs8070597

    """
    return ee.Image(landsat_image).expression(
        "NDMI_scaled * (2.92 * sqrt(EVI2) - 0.43) ** 2",
        {"EVI2": evi2(ee.Image(landsat_image)), "NDMI_scaled": ndmi_scaled(ee.Image(landsat_image))}
    ).clamp(0, 8).rename("lai")


def ndvi(landsat_image):
    """Normalized difference vegetation index

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Gao1996] Gao B. (1996). NDWI—A normalized difference water index for
        remote sensing of vegetation liquid water from space.
        Remote Sensing of Environment, 58, 257-266,
        https://doi.org/10.1016/S0034-4257(96)00067-3

    """
    return ee.Image(landsat_image).normalizedDifference(["nir", "red"]).rename(["ndvi"]).unmask(0)
    

def ndwi(landsat_image):
    """Normalized difference water index

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------


    """
    return ee.Image(landsat_image).normalizedDifference(["green", "nir"]).rename("ndwi").unmask(0)


def landsat_c2_qa_water_mask(landsat_image):
    """Extract water mask from the Landsat Collection 2 SR QA_PIXEL band.

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    """
    return (
        ee.Image(landsat_image).select(["QA_PIXEL"])
        .rightShift(7).bitwiseAnd(1).neq(0)
        .rename(["qa_water"])
    )


def albedo_disalexi(landsat_image):
    """Total shortwave broadband albedo following [Liang2001]

    Parameters
    ----------
    landsat_image : ee.Image
        "Prepped" Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    Notes
    -----
    [Liang2001] indicate that the green band is intentionally not used.
    Coefficients were derived for Landsat 7 ETM+, but were found to be
        "suitable" to Landsat 4/5 TM also.

    References
    ----------
    .. [Liang2001] Liang S. (2001). Narrowband to broadband conversions
        of land surface albedo - I Algorithms.
        Remote Sensing of Environment, Volume 76, Issue2, Pages 213-238,
        http://doi.org/10.1016/S0034-4257(00)00205-4

    """
    albedo_img = (
        landsat_image
        .select(["blue", "red", "nir", "swir1", "swir2"])
        .multiply([0.356, 0.130, 0.373, 0.085, 0.072])
    )
    return (
        albedo_img.select([0])
        .add(albedo_img.select([1])).add(albedo_img.select([2]))
        .add(albedo_img.select([3])).add(albedo_img.select([4]))
        .subtract(0.0018)
        .rename(["albedo"])
    )


def albedo_l457(landsat_image):
    """Albedo (Landsat 4/5/7)

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Tasumi2008] Tasumi M., Allen R., Trezza R. (2008).
       At-Surface Reflectance and Albedo from Satellite for Operational
       Calculation of Land Surface Energy Balance. Journal of Hydrology,
       https://doi.org/10.1061/(ASCE)1084-0699(2008)13:2(51)

    """
    return landsat_image.expression(
        "(0.254 * B1) + (0.149 * B2) + (0.147 * B3) + (0.311 * B4) + (0.103 * B5) + (0.036 * B7)",
        {
            "B1": landsat_image.select(["blue"]),
            "B2": landsat_image.select(["green"]),
            "B3": landsat_image.select(["red"]),
            "B4": landsat_image.select(["nir"]),
            "B5": landsat_image.select(["swir1"]),
            "B7": landsat_image.select(["swir2"]),
        }
    ).rename("albedo")


def albedo_l89(landsat_image):
    """Albedo (Landsat 8/9)

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    References
    ----------
    .. [Ke2016] Y. Ke, J. Im, S. Park, H. Gong (2016).
       Downscaling of MODIS One Kilometer Evapotranspiration
       Using Landsat-8 Data and Machine Learning Approaches.
       Remote Sensing, https://doi.org/10.3390/rs8030215

    """

    # CGM - These coefficients don't sum to 1.0?
    return landsat_image.expression(
        "(0.130 * B1) + (0.115 * B2) + (0.143 * B3) + (0.180 * B4) + "
        "(0.281 * B5) + (0.108 * B6) + (0.042 * B7)", {
            "B1": landsat_image.select(["ultra_blue"]),
            "B2": landsat_image.select(["blue"]),
            "B3": landsat_image.select(["green"]),
            "B4": landsat_image.select(["red"]),
            "B5": landsat_image.select(["nir"]),
            "B6": landsat_image.select(["swir1"]),
            "B7": landsat_image.select(["swir2"]),
        }
    ).rename("albedo")


def cloud_mask_C2_l457(landsat_image):
    """Cloud mask (Landsat 4/5/7)

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    Notes
    -----
    clear sky value = 5440  (0001010101000000)
    water value     = 5504  (0001010110000000)

    """
    quality = landsat_image.select("QA_PIXEL")
    c01 = quality.eq(5440)
    c02 = quality.eq(5504)
    mask = c01.Or(c02)

    return mask


def cloud_mask_C2_l89(landsat_image):
    """Cloud mask (Landsat 8/9)

    Parameters
    ----------
    landsat_image : ee.Image
        Landsat image with standardized band names.

    Returns
    -------
    ee.Image

    Notes
    -----
    clear sky value = 21824  (0101010101000000)
    water value     = 21952  (0001010110000000)

    """
    quality = landsat_image.select("QA_PIXEL")
    c01 = quality.eq(21824)
    c02 = quality.eq(21952)
    mask = c01.Or(c02)

    return mask


# TODO: Consider renaming this function to "ocean_mask"
#   since the two supported products are not general water masks
def water_mask(product="GLO"):
    """Water mask for precomputing

    Parameters
    ----------
    product : {'GLO', 'OSM'}
        Product name. 
            GLO: Water mask from the Copernicus GLO DEM.
            OSM: Water mask from OpenStreetMap.

    Returns
    -------
    ee.Image

    Notes
    -----
    These water masks are primarily intended for masking ocean and the Great Lakes

    References
    ----------
    GLO 30 DGED: https://doi.org/10.5270/ESA-c5d3d65 
    OSM: https://gee-community-catalog.org/projects/osm_water/

    """
    if product.upper() == "GLO":
        water_mask = ee.Image("projects/openet/assets/features/water_mask_glo30_0p001")
    elif product.upper() == "OSM":
        water_mask = ee.Image("projects/openet/assets/features/water_mask_osm_0p001")
    else:
        raise ValueError(f'unsupported water mask product type: {product}')

    return water_mask.rename('water_mask')


# TODO: Move to openet.core
# TODO: Consider adding support for setting some sort of "lst_source_index"
#   parameter to allow for joining on a property other than "scene_id"
def lst_source_image(lst_source, mask_image):
    """Select the LST image from a source collection based on the scene_id

    Parameters
    ----------
    lst_source : str
        LST source collection ID.
        Images in LST source collection must be single band.
        Images must have a "scene_id" property with an upper case ID in the
            system:index format (LXSS_PPPRRR_YYYYMMDD)
        If a "scale_factor" property is present, it will be applied by multiplying
    mask_image : ee.Image
        Single Landsat image band with "system:index" property that will be
        used for selecting target image in source collection.
        Using a full image instead of the system:index itself so that the image
        footprint/mask can be applied to the matched image.

    Returns
    -------
    ee.Image

    """
    # Merging with an empty base image is needed so that an empty/masked image is
    #   returned if the target scene_id is not present in the source collection
    base_img = mask_image.multiply(0).selfMask().set({'lst_source_id': 'None'})

    lst_image = ee.Image(
        ee.ImageCollection(lst_source)
        .filter(ee.Filter.eq('scene_id', mask_image.get('system:index')))
        .select([0], ['lst'])
        .map(lambda img: img.set({'lst_source_id': img.get('system:id')}))
        .merge(ee.ImageCollection([base_img]))
        .first()
        .updateMask(mask_image.mask())
    )
    # # Could use the input LST as a fallback if the scene is missing from LST source
    # #   instead of returning a masked image
    # .merge(ee.ImageCollection([lst_image.set({'lst_source_id': self._id})]))

    # Save the lst_source_id property since tscale_factor multiply call below drops it
    lst_source_id = lst_image.get('lst_source_id')

    # The OpenET LST images are scaled, so assume the image need to be unscaled,
    #   but set a default scale_factor to 1
    lst_scale_factor = (
        ee.Dictionary({'scale_factor': lst_image.get('scale_factor')})
        .combine({'scale_factor': 1.0}, overwrite=False)
    )

    return (
        lst_image.multiply(ee.Number(lst_scale_factor.get('scale_factor')))
        .rename(['lst'])
        .set({
            'system:index': mask_image.get('system:index'),
            'lst_source_id': lst_source_id,
            # 'scene_id': mask_image.get('system:index'),
        })
    )
