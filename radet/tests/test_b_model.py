from datetime import datetime
import math

import ee
import pytest

import radet.model as model
import radet.utils as utils
# TODO: import utils from openet.core
# import openet.core.utils as utils


# TODO: Try moving to conftest and/or make a fixture
SCENE_ID = 'LC08_044033_20170716'
SCENE_DT = datetime.strptime(SCENE_ID[-8:], '%Y%m%d')
SCENE_DATE = SCENE_DT.strftime('%Y-%m-%d')
SCENE_TIME = utils.millis(SCENE_DT)


@pytest.mark.parametrize(
    'ta, elev, meteo_elev, expected',
    [
        [300, 10, 10, 300],
        [300, 100, 10, 299.415],
        [300, 1000, 10, 293.565],
    ]
)
def test_add_lapse_correction(ta, elev, meteo_elev, expected):
    output = utils.constant_image_value(
        model.add_lapse_correction(ee.Image(ta), ee.Image(elev), meteo_elev)
    )
    assert abs(output['t_corrected'] - expected) < 0.0001


@pytest.mark.parametrize(
    'lai, fc, tauS, tauL',
    [
        [0.0, 0.0100, 1.0000, 1.0000],
        [0.1, 0.0392, 0.9455, 0.9094],  #
        [0.5, 0.1813, 0.7558, 0.6219],  #
        [2.0, 0.5507, 0.3263, 0.1496],  #
        [3.5, 0.7534, 0.1409, 0.0360],  #
        [1.0, 0.3297, 0.5712, 0.3867],
        [4.0, 0.7981, 0.1065, 0.0224],
    ]
)
def test_transmissivities(lai, fc, tauS, tauL, tol=0.0001):
    output = utils.constant_image_value(model.transmissivities(ee.Image.constant(lai)))
    assert abs(output['tauL'] - tauL) < tol
    assert abs(output['tauS'] - tauS) < tol
    assert abs(output['fc'] - fc) < tol


@pytest.mark.parametrize(
    'landcover, lai, wet, water',
    [
        [11, 0, 1, 1],
        [81, 0, 1, 0],
        [82, 0, 1, 0],
        [95, 0, 1, 0],
        [90, 0, 1, 0],
        [90, 0.9, 1, 0],
        [90, 1.0, 0, 0],
        [71, 0, 0, 0],
        [71, 4, 0, 0],
    ]
)
def test_wet_mask(landcover, lai, wet, water):
    output = utils.constant_image_value(model.wet_mask(
        landcover=ee.Image.constant(landcover), lai=ee.Image.constant(lai)
    ))
    assert output['remapped'] == wet
    assert output['remapped_1'] == water


@pytest.mark.parametrize(
    'time_start, lat, elev, srad, Rs_MJ, Ra_MJ, fcd, sunrise_ts',
    [
        [SCENE_TIME, 38.905, 10.0, 353.7, 30.56, 40.73, 1.0000, 4.7775],
        [SCENE_TIME, 38.905, 10.0, 200.0, 17.28, 40.73, 0.4135, 4.7775],
        [SCENE_TIME, 38.905, 10.0, 100.0,  8.64, 40.73, 0.0550, 4.7775],
        [SCENE_TIME, 38.905, 1000, 353.7, 30.56, 40.73, 0.9654, 4.7775],
    ]
)
def test_clear_sky_terms(time_start, lat, elev, srad, Rs_MJ, Ra_MJ, fcd, sunrise_ts):
    # Outputs: Rs_MJ, Ra_MJ, fcd, sunrise_ts
    output = utils.constant_image_value(
        model.clear_sky_terms(time_start, ee.Image(lat), ee.Image(elev), ee.Image(srad))
    )
    assert abs(output['Rs_MJ'] - Rs_MJ) <= 0.01
    assert abs(output['Ra_MJ'] - Ra_MJ) <= 0.01
    assert abs(output['fcd'] - fcd) <= 0.0001
    assert abs(output['sunrise_ts'] - sunrise_ts) <= 0.0001


# TODO: Setup test values that need shade correction
@pytest.mark.parametrize(
    'Rs_MJ, Ra_MJ, elev, time_start, albedo, latitude, longitude, expected',
    [
        [30.56, 40.73, 10, SCENE_TIME, 0.18, 38.905, -121.668, 30.56],
        [17.28, 40.73, 10, SCENE_TIME, 0.19, 38.905, -121.668, 17.28],
        [ 8.64, 40.73, 10, SCENE_TIME, 0.19, 38.905, -121.668,  8.64],
        [30.56, 40.73, 10, SCENE_TIME, 0.15, 38.905, -121.668, 30.56],
    ]
)
def test_terrain_shade_correct_srad(Rs_MJ, Ra_MJ, elev, time_start, albedo, latitude, longitude, expected):
    output = utils.constant_image_value(model.terrain_shade_correct_srad(
        ee.Image(Rs_MJ), ee.Image(Ra_MJ), elev, time_start, albedo, latitude, longitude
    ))
    assert abs(output['Rs_MJ_shade'] - expected) <= 0.0001


@pytest.mark.parametrize(
    'elev, expected',
    [
        [  10, 101.1818],
        [ 100, 100.1235],
        [1000,  90.0246],
    ]
)
def test_air_pressure(elev, expected):
    # Testing the equations directly to build test values since these are not separate function
    output = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26
    assert abs(output - expected) <= 0.0001


@pytest.mark.parametrize(
    'u10, expected',
    [
        [0.0, 0.0],
        [1.8, 1.3463],
        #[3.0, 2.2439],
    ]
)
def test_u2(u10, expected):
    # Testing the equations directly to build test values since these are not separate function
    output = u10 * 4.87 / math.log(67.8 * 10 - 5.42)
    assert abs(output - expected) <= 0.0001


@pytest.mark.parametrize(
    'qa, pair, expected',
    [
        [0.00937, 101.1818, 1.5156],
        [0.00937, 100.1235, 1.4998],
        [0.00937,  90.0246, 1.3485],
    ]
)
def test_vapor_pressure(qa, pair, expected):
    # Testing the equations directly to build test values since these are not separate function
    output = qa * pair / ((1 - 0.622) * qa + 0.622)
    assert abs(output - expected) <= 0.0001


@pytest.mark.parametrize(
    't, expected',
    [
        [290, 1.9194],
        [301, 3.7470],
        [312, 6.9354],
    ]
)
def test_esat(t, expected):
    # Testing the equations directly to build test values since these are not separate function
    output = 0.6108 * math.exp(17.27 * (t - 273.15) / ((t - 273.15) + 237.3))
    assert abs(output - expected) <= 0.0001


@pytest.mark.parametrize(
    't, expected',
    [
        [290, 0.1218],
        [301, 0.2184],
        [312, 0.3727],
    ]
)
def test_delta(t, expected):
    # Testing the equations directly to build test values since these are not separate function
    output = 2503 * math.exp(17.27 * (t - 273.15) / ((t - 273.15) + 237.3)) / ((t - 273.15) + 237.3) ** 2
    assert abs(output - expected) <= 0.0001


@pytest.mark.parametrize(
    'emissivity, fcd, ea, t_avg, expected',
    [
        # 10m elevation
        [0.98, 1.0, 1.5156, 301.0, 32.8159],
        [0.97, 1.0, 1.5156, 301.0, 32.4811],
        [0.96, 1.0, 1.5156, 301.0, 32.1462],
        # 1000m elevation
        [0.98, 1.0, 1.3485, 301.0, 32.4304],
        [0.97, 1.0, 1.3485, 301.0, 32.0995],
        [0.96, 1.0, 1.3485, 301.0, 31.7685],
        # fcd = 0.5
        [0.97, 0.5, 1.5156, 301.0, 35.7521],
    ]
)
def test_Rld_atm_ASCE(emissivity, fcd, ea, t_avg, expected):
    output = utils.constant_image_value(model.Rld_atm_ASCE(emissivity, fcd, ea, t_avg))
    assert abs(output['Rld'] - expected) <= 0.0001


@pytest.mark.parametrize(
    'tminK, lst, sunrise_ts, t_avg, expected',
    [
        # t_avg: (312.4 + 289.6) / 2 = 301
        [289.6, 327, 4.7775, 301, 310.5813],
        [289.6, 323, 4.7775, 301, 308.2916],
        [289.6, 313, 4.7775, 301, 302.5673],
        [289.6, 304, 4.7775, 301, 301.0000],
    ]
)
def test_daily_avg_lst(tminK, lst, sunrise_ts, t_avg, expected):
    output = utils.constant_image_value(
        model.daily_avg_lst(ee.Image(tminK), ee.Image(lst), sunrise_ts, t_avg)
    )
    assert abs(output['constant'] - expected) <= 0.0001


################################################################################
# First Run
################################################################################

@pytest.mark.parametrize(
    'ea, esat, delta, LST_soil, t_avg, mu_s, water, RHs',
    [
        # First run, mu_s=1, set RHs to RH (ea/esat)
        [1.5156, 3.7470, 0.2184, 300.0, 301, 1, 0, 0.4045],
        [1.5156, 3.7470, 0.2184, 310.0, 301, 1, 0, 0.4045],
        # If water is 1, set RHs to 1
        [1.5156, 3.7470, 0.2184, 300.0, 301, 1, 1, 1],
        [1.5156, 3.7470, 0.2184, 310.0, 301, 1, 1, 1],
    ]
)
def test_RHs_model_a(ea, esat, delta, LST_soil, t_avg, mu_s, water, RHs):
    output = utils.constant_image_value(model.RHs_model(ea, esat, delta, LST_soil, t_avg, mu_s, water))
    # print('\n{:6.4f}'.format(output['RHs']))
    assert abs(output['RHs'] - RHs) <= 0.0001


@pytest.mark.parametrize(
    'LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma, LST_canopy',
    [
        # RHs: 1.5156 / 3.7470 = 0.404484
        # gamma: 101.1818 * 0.000665 = 0.067286
        # Tests correspond to LAI: 0.1, 0.5, 2.0, 3.5
        [310.58, 301, 0.0392, 1, 1, 0.4045, 0.2184, 0.0673, 301.21],
        [308.29, 301, 0.1813, 1, 1, 0.4045, 0.2184, 0.0673, 301.78],
        [302.57, 301, 0.5507, 1, 1, 0.4045, 0.2184, 0.0673, 301.63],
        [301.00, 301, 0.7534, 1, 1, 0.4045, 0.2184, 0.0673, 301.00],
    ]
)
def test_canopy_LST_a(LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma, LST_canopy):
    output = utils.constant_image_value(
        model.canopy_LST(LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma)
    )
    # print('\n{:6.2f}'.format(output['LST_canopy']))
    assert abs(output['LST_canopy'] - LST_canopy) <= 0.01


@pytest.mark.parametrize(
    'LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo, LST_soil',
    [
        # Tests correspond to LAI: 0.1, 0.5, 2.0, 3.5
        [310.58, 301.21, 30.56, 32.8159, 0.9455, 0.9094, 0.98, 0.18, 311.47],
        [308.29, 301.78, 30.56, 32.4811, 0.7558, 0.6219, 0.97, 0.19, 312.05],
        [302.57, 301.63, 30.56, 32.4811, 0.3263, 0.1496, 0.97, 0.19, 307.75],
        [301.00, 301.00, 30.56, 32.4811, 0.1409, 0.0360, 0.97, 0.15, 301.00],
    ]
)
def test_soil_LST_a(LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo, LST_soil):
    output = utils.constant_image_value(
        model.soil_LST(LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo)
    )
    # print('\n{:6.2f}'.format(output['LST_soil']))
    assert abs(output['LST_soil'] - LST_soil) <= 0.01


@pytest.mark.parametrize(
    'emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs',
    [
        [0.98, 301.21, 311.47, 30.56, 32.8159, 0.18, 0.9455, 0.9094, 13.1848,  1.2705, 11.9143,  2.6700,  9.2443],
        [0.97, 301.78, 312.05, 30.56, 32.4811, 0.19, 0.7558, 0.6219, 14.2932,  5.5531,  8.7401,  1.5591,  7.1811],
        [0.97, 301.63, 307.75, 30.56, 32.4811, 0.19, 0.3263, 0.1496, 17.3913, 13.6341,  3.7572, -0.1850,  3.9422],
        [0.97, 301.00, 301.00, 30.56, 32.4811, 0.15, 0.1409, 0.0360, 19.4339, 16.0094,  3.4245, -0.3014,  3.7259],
    ]
)
def test_net_radiation_a(emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs):
    output = utils.constant_image_value(
        model.net_radiation(emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL)
    )
    # print('\n{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}'.format(
    #     output['Rn'], output['Rnc'], output['Rns'], output['G'], output['AEs']))
    assert abs(output['Rn'] - Rn) <= 0.0001
    assert abs(output['Rnc'] - Rnc) <= 0.0001
    assert abs(output['Rns'] - Rns) <= 0.0001
    assert abs(output['G'] - G) <= 0.0001
    assert abs(output['AEs'] - AEs) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci',
    [
        [ 1.2705, 0.9094, 0.98, 301, 301.21,  1.2904],
        [ 5.5531, 0.6219, 0.97, 301, 301.78,  5.8590],
        [13.6341, 0.1496, 0.97, 301, 301.63, 14.1898],
        [16.0094, 0.0360, 0.97, 301, 301.00, 16.0094],
    ]
)
def test_isothermal_canopy_net_radiation_a(Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci):
    output = utils.constant_image_value(
        model.isothermal_canopy_net_radiation(Rnc, tauL, emissivity, t_avg, LST_canopy)
    )
    # print('\n{:7.4f}'.format(output['Rnci']))
    assert abs(output['Rnci'] - Rnci) <= 0.0001


@pytest.mark.parametrize(
    'AEs, emissivity, t_avg, LST_soil, AEsi',
    [
        [9.2443, 0.98, 301, 311.47, 20.1846],
        [7.1811, 0.97, 301, 312.05, 18.6684],
        [3.9422, 0.97, 301, 307.75, 10.9593],
        [3.7259, 0.97, 301, 301.00,  3.7259],
    ]
)
def test_isothermal_soil_available_energy_a(AEs, emissivity, t_avg, LST_soil, AEsi):
    output = utils.constant_image_value(
        model.isothermal_soil_available_energy(AEs, emissivity, t_avg, LST_soil)
    )
    # print('\n{:7.4f}'.format(output['AEsi']))
    assert abs(output['AEsi'] - AEsi) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, Rnci, delta, gamma, mu_c',
    [
        [ 1.2705,  1.2904, 0.2184, 0.0673, 1.0635],
        [ 5.5531,  5.8590, 0.2184, 0.0673, 1.2036],
        [13.6341, 14.1898, 0.2184, 0.0673, 1.1553],
        [16.0094, 16.0094, 0.2184, 0.0673, 1.0000],
    ]
)
def test_mu_canopy_a(Rnc, Rnci, delta, gamma, mu_c):
    output = utils.constant_image_value(
        model.mu_canopy(ee.Image(Rnc), ee.Image(Rnci), delta, gamma)
    )
    # print('\n{:6.4f}'.format(output['mu_c']))
    assert abs(output['mu_c'] - mu_c) <= 0.0001


@pytest.mark.parametrize(
    'AEs, AEsi, RHs, delta, gamma, mu_s',
    [
        [9.2443, 20.1846, 0.4045, 0.2184, 0.0673, 2.7487],
        [7.1811, 18.6684, 0.4045, 0.2184, 0.0673, 3.2465],
        [3.9422, 10.9593, 0.4045, 0.2184, 0.0673, 3.4561],
        [3.7259,  3.7259, 0.4045, 0.2184, 0.0673, 1.0000],
    ]
)
def test_mu_soil_a(AEs, AEsi, RHs, delta, gamma, mu_s):
    output = utils.constant_image_value(
        model.mu_soil(ee.Image(AEs), ee.Image(AEsi), RHs, delta, gamma)
    )
    # print('\n{:6.4f}'.format(output['mu_s']))
    assert abs(output['mu_s'] - mu_s) <= 0.0001


################################################################################
# Second Run
################################################################################

@pytest.mark.parametrize(
    'ea, esat, delta, LST_soil, t_avg, mu_s, water, RHs',
    [
        [1.5156, 3.7470, 0.2184, 311.47, 301, 2.7487, 0, 0.2914],
        [1.5156, 3.7470, 0.2184, 312.05, 301, 3.2465, 0, 0.2798],
        [1.5156, 3.7470, 0.2184, 307.75, 301, 3.4561, 0, 0.3161],
        [1.5156, 3.7470, 0.2184, 301.00, 301, 1.0000, 0, 0.4045],
    ]
)
def test_RHs_model_b(ea, esat, delta, LST_soil, t_avg, mu_s, water, RHs):
    output = utils.constant_image_value(model.RHs_model(ea, esat, delta, LST_soil, t_avg, mu_s, water))
    # print('\n{:6.4f}'.format(output['RHs']))
    assert abs(output['RHs'] - RHs) <= 0.0001


@pytest.mark.parametrize(
    'LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma, LST_canopy',
    [
        [310.58, 301, 0.0392, 1.0635, 2.7487, 0.2914, 0.2184, 0.0673, 301.13],
        [308.29, 301, 0.1813, 1.2036, 3.2465, 0.2798, 0.2184, 0.0673, 301.52],
        [302.57, 301, 0.5507, 1.1553, 3.4561, 0.3161, 0.2184, 0.0673, 301.47],
        [301.00, 301, 0.7534, 1.0000, 1.0000, 0.4045, 0.2184, 0.0673, 301.00],
    ]
)
def test_canopy_LST_b(LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma, LST_canopy):
    output = utils.constant_image_value(
        model.canopy_LST(LST_avg, t_avg, fc, mu_c, mu_s, RHs, delta, gamma)
    )
    # print('\n{:5.1f}'.format(output['LST_canopy']))
    assert abs(output['LST_canopy'] - LST_canopy) <= 0.1


@pytest.mark.parametrize(
    'LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo, LST_soil',
    [
        [310.58, 301.13, 30.56, 32.8159, 0.9455, 0.9094, 0.98, 0.18, 311.50],
        [308.29, 301.52, 30.56, 32.4811, 0.7558, 0.6219, 0.97, 0.19, 312.21],
        [302.57, 301.47, 30.56, 32.4811, 0.3263, 0.1496, 0.97, 0.19, 308.60],
        [301.00, 301.00, 30.56, 32.4811, 0.1409, 0.0360, 0.97, 0.15, 301.00],
    ]
)
def test_soil_LST_b(LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo, LST_soil):
    output = utils.constant_image_value(
        model.soil_LST(LST_avg, LST_canopy, Rs_MJ, Rld_MJ, tauS, tauL, emissivity, albedo)
    )
    # print('\n{:5.1f}'.format(output['LST_soil']))
    assert abs(output['LST_soil'] - LST_soil) <= 0.1


@pytest.mark.parametrize(
    'emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs',
    [
        [0.98, 301.13, 311.50, 30.56, 32.8159, 0.18, 0.9455, 0.9094, 13.1727,  1.2797, 11.8931,  2.6626,  9.2305],
        [0.97, 301.52, 312.21, 30.56, 32.4811, 0.19, 0.7558, 0.6219, 14.2870,  5.6907,  8.5963,  1.5087,  7.0876],
        [0.97, 301.47, 308.60, 30.56, 32.4811, 0.19, 0.3263, 0.1496, 17.3915, 14.1783,  3.2132, -0.3754,  3.5886],
        [0.97, 301.00, 301.00, 30.56, 32.4811, 0.15, 0.1409, 0.0360, 19.4339, 16.0094,  3.4245, -0.3014,  3.7259],
    ]
)
def test_net_radiation_b(emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs):
    output = utils.constant_image_value(
        model.net_radiation(emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL)
    )
    # print('\n{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}'.format(
    #     output['Rn'], output['Rnc'], output['Rns'], output['G'], output['AEs']))
    assert abs(output['Rn'] - Rn) <= 0.0001
    assert abs(output['Rnc'] - Rnc) <= 0.0001
    assert abs(output['Rns'] - Rns) <= 0.0001
    assert abs(output['G'] - G) <= 0.0001
    assert abs(output['AEs'] - AEs) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci',
    [
        [ 1.2797, 0.9094, 0.98, 301, 301.13,  1.2920],
        [ 5.6907, 0.6219, 0.97, 301, 301.52,  5.8946],
        [14.1783, 0.1496, 0.97, 301, 301.47, 14.5928],
        [16.0094, 0.0360, 0.97, 301, 301.00, 16.0094],
    ]
)
def test_isothermal_canopy_net_radiation_b(Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci):
    output = utils.constant_image_value(
        model.isothermal_canopy_net_radiation(Rnc, tauL, emissivity, t_avg, LST_canopy)
    )
    # print('\n{:7.4f}'.format(output['Rnci']))
    assert abs(output['Rnci'] - Rnci) <= 0.0001


@pytest.mark.parametrize(
    'AEs, emissivity, t_avg, LST_soil, AEsi',
    [
        [9.2305, 0.98, 301, 311.50, 20.2022],
        [7.0876, 0.97, 301, 312.21, 18.7412],
        [3.5886, 0.97, 301, 308.60, 11.4894],
        [3.7259, 0.97, 301, 301.00,  3.7259],
    ]
)
def test_isothermal_soil_available_energy_b(AEs, emissivity, t_avg, LST_soil, AEsi):
    output = utils.constant_image_value(
        model.isothermal_soil_available_energy(AEs, emissivity, t_avg, LST_soil)
    )
    # print('\n{:7.4f}'.format(output['AEsi']))
    assert abs(output['AEsi'] - AEsi) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, Rnci, delta, gamma, mu_c',
    [
        [ 1.2797,  1.2920, 0.2184, 0.0673, 1.0396],
        [ 5.6907,  5.8946, 0.2184, 0.0673, 1.1380],
        [14.1783, 14.5928, 0.2184, 0.0673, 1.1144],
        [16.0094, 16.0094, 0.2184, 0.0673, 1.0000],
    ]
)
def test_mu_canopy_b(Rnc, Rnci, delta, gamma, mu_c):
    output = utils.constant_image_value(
        model.mu_canopy(ee.Image(Rnc), ee.Image(Rnci), delta, gamma)
    )
    # print('\n{:6.4f}'.format(output['mu_c']))
    assert abs(output['mu_c'] - mu_c) <= 0.0001


@pytest.mark.parametrize(
    'AEs, AEsi, RHs, delta, gamma, mu_s',
    [
        [9.2305, 20.2022, 0.2914, 0.2184, 0.0673, 2.6180],
        [7.0876, 18.7412, 0.2798, 0.2184, 0.0673, 3.1224],
        [3.5886, 11.4894, 0.3161, 0.2184, 0.0673, 3.7965],
        [3.7259,  3.7259, 0.4045, 0.2184, 0.0673, 1.0000],
    ]
)
def test_mu_soil_b(AEs, AEsi, RHs, delta, gamma, mu_s):
    output = utils.constant_image_value(
        model.mu_soil(ee.Image(AEs), ee.Image(AEsi), RHs, delta, gamma)
    )
    # print('\n{:6.4f}'.format(output['mu_s']))
    assert abs(output['mu_s'] - mu_s) <= 0.0001


################################################################################
# ET
################################################################################

@pytest.mark.parametrize(
    'Rnc, AEs, RHs, mu_c, mu_s, delta, gamma, expected',
    [
        [  1.2797, 9.2305, 0.2914, 1.0396, 2.6180, 0.2184, 0.0673, 1.3953],
        [  5.6907, 7.0876, 0.2798, 1.1380, 3.1224, 0.2184, 0.0673, 2.3714],
        [ 14.1783, 3.5886, 0.3161, 1.1144, 3.7965, 0.2184, 0.0673, 4.6193],
        [ 16.0094, 3.7259, 0.4045, 1.0000, 1.0000, 0.2184, 0.0673, 5.8584],
    ]
)
def test_DIF_model(Rnc, AEs, RHs, mu_c, mu_s, delta, gamma, expected):
    output = utils.constant_image_value(model.DIF_model(
        Rnc=ee.Image.constant(Rnc),
        AEs=ee.Image.constant(AEs),
        RHs=ee.Image.constant(RHs),
        mu_c=ee.Image.constant(mu_c),
        mu_s=ee.Image.constant(mu_s),
        delta=ee.Image.constant(delta),
        gamma=ee.Image.constant(gamma)
    ))
    # print('\n{:6.4f}'.format(output['ET_DIF']))
    assert abs(output['ET_DIF'] - expected) <= 0.0001


@pytest.mark.parametrize(
    'del_LC, fc, LST_soil, RHs, delta, gamma, u2, ea, esat, expected',
    [
        # Non-wet landcover
        [0, 0.0392, 311.50, 0.2914, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.1813, 312.21, 0.2798, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.5507, 308.60, 0.3161, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.7534, 301.00, 0.4045, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.0],
        # Wet landcover
        [1, 0.0392, 311.50, 0.2914, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.0983],
        [1, 0.1813, 312.21, 0.2798, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 0.4305],
        [1, 0.5507, 308.60, 0.3161, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 1.3156],
        [1, 0.7534, 301.00, 0.4045, 0.2184, 0.0673, 1.3463, 1.5156, 3.7470, 1.8554],
    ]
)
def test_aerodynamic_term(del_LC, fc, LST_soil, RHs, delta, gamma, u2, ea, esat, expected):
    output = utils.constant_image_value(
        model.aerodynamic_term(del_LC, fc, LST_soil, RHs, delta, gamma, u2, esat, ea)
    )
    # print('\n{:6.4f}'.format(output['ET_Aero']))
    assert abs(output['ET_Aero'] - expected) <= 0.1


@pytest.mark.parametrize(
    'lai, lst, albedo, emissivity, landcover, elevation, tmin, tmax, qa, u10, srad, meteo_elev, time_start, lat, lon, expected',
    [
        # Non-agricultural landcover
        [0.1, 327, 0.18, 0.98, 71, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 1.3970],
        [0.5, 323, 0.19, 0.97, 71, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 2.3718],
        [2.0, 313, 0.19, 0.97, 71, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 4.6267],
        [3.5, 304, 0.15, 0.97, 71, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 5.8588],
        # Agricultural landcover (to trigger aerodynamic adjustment)
        [0.1, 327, 0.18, 0.98, 81, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 1.4958],
        [0.5, 323, 0.19, 0.97, 81, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 2.8026],
        [2.0, 313, 0.19, 0.97, 81, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 5.9374],
        [3.5, 304, 0.15, 0.97, 81, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 7.7139],
        # Check with NLCD=90 since it is LAI dependent and should switch from wet to non-wet values
        [0.1, 327, 0.18, 0.98, 90, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 1.4958],
        [0.5, 323, 0.19, 0.97, 90, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 2.8026],
        [2.0, 313, 0.19, 0.97, 90, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 4.6267],
        [3.5, 304, 0.15, 0.97, 90, 10, 289.6, 312.4, 0.00937, 1.8, 353.7, 10, SCENE_TIME, 38.905, -121.668, 5.8588],
    ]
)
def test_et(
        lai, lst, albedo, emissivity, landcover, elevation,
        tmin, tmax, qa, u10, srad, meteo_elev, time_start, lat, lon, expected
):
    output = utils.constant_image_value(model.et(
        lai=ee.Image(lai),
        lst=ee.Image(lst),
        albedo=ee.Image(albedo),
        emissivity=ee.Image(emissivity),
        landcover=ee.Image(landcover),
        elevation=ee.Image(elevation),
        tmin=ee.Image(tmin),
        tmax=ee.Image(tmax),
        qa=ee.Image(qa),
        u10=ee.Image(u10),
        srad=ee.Image(srad),
        meteo_elevation=ee.Image(meteo_elev),
        time_start=time_start,
        latitude=ee.Image(lat),
        longitude=ee.Image(lon),
    ))
    # print('\n{:6.4f}'.format(output['ET']))
    assert abs(output['ET'] - expected) <= 0.0001
