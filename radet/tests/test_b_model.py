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
        [289.6, 327, 4.7775, 301, 310.6],
        [289.6, 323, 4.7775, 301, 308.3],
        [289.6, 313, 4.7775, 301, 302.6],
        [289.6, 304, 4.7775, 301, 301.0],
    ]
)
def test_daily_avg_lst(tminK, lst, sunrise_ts, t_avg, expected):
    output = utils.constant_image_value(
        model.daily_avg_lst(ee.Image(tminK), ee.Image(lst), sunrise_ts, t_avg)
    )
    assert abs(output['constant'] - expected) <= 0.1


################################################################################
# First Run
################################################################################

@pytest.mark.parametrize(
    'ea, esat, DELTA, LST_soil, t_avg, mu_s, water, RHs',
    [
        # First run, mu_s=1, set RHs to RH (ea/esat)
        [1.5156, 3.7470, 0.2184, 300.0, 301, 1, 0, 0.4045],
        [1.5156, 3.7470, 0.2184, 310.0, 301, 1, 0, 0.4045],
        # If water is 1, set RHs to 1
        [1.5156, 3.7470, 0.2184, 300.0, 301, 1, 1, 1],
        [1.5156, 3.7470, 0.2184, 310.0, 301, 1, 1, 1],
    ]
)
def test_RHs_model(ea, esat, DELTA, LST_soil, t_avg, mu_s, water, RHs):
    output = utils.constant_image_value(model.RHs_model(ea, esat, DELTA, LST_soil, t_avg, mu_s, water))
    # print('\n{:6.4f}'.format(output['RHs']))
    assert abs(output['RHs'] - RHs) <= 0.0001


@pytest.mark.parametrize(
    'LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo, LST_canopy, LST_soil',
    [
        # RHs: 1.5156 / 3.7470 = 0.404484
        # gamma: 101.1818 * 0.000665 = 0.067286
        # First run, LAI=0.1,0.5,2.0,3.5
        [310.6, 301, 30.5, 32.5, 0.0392, 0.9455, 0.9094, 1, 1, 0.4045, 0.2184, 0.0673, 0.98, 0.18, 301.2, 311.5],
        [308.3, 301, 30.5, 32.5, 0.1813, 0.7558, 0.6219, 1, 1, 0.4045, 0.2184, 0.0673, 0.97, 0.19, 301.8, 312.1],
        [302.6, 301, 30.5, 32.5, 0.5507, 0.3263, 0.1496, 1, 1, 0.4045, 0.2184, 0.0673, 0.97, 0.19, 301.6, 307.9],
        [301.0, 301, 30.5, 32.5, 0.7534, 0.1409, 0.0360, 1, 1, 0.4045, 0.2184, 0.0673, 0.97, 0.15, 301.0, 301.0],
    ]
)
def test_canopy_and_soil_LST(LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo, LST_canopy, LST_soil):
    output = utils.constant_image_value(model.canopy_and_soil_LST(
        LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo
    ))
    # print('\n{:5.1f}, {:5.1f}'.format(output['LST_canopy'], output['LST_soil']))
    assert abs(output['LST_canopy'] - LST_canopy) <= 0.1
    assert abs(output['LST_soil'] - LST_soil) <= 0.1


@pytest.mark.parametrize(
    'emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs',
    [
        [0.98, 301.2, 311.5, 30.5, 32.5, 0.18, 0.9455, 0.9094, 12.8043,  1.2417, 11.5626,  2.5469, 9.0157],
        [0.97, 301.8, 312.1, 30.5, 32.5, 0.19, 0.7558, 0.6219, 14.2416,  5.5514,  8.6902,  1.5416, 7.1486],
        [0.97, 301.6, 307.9, 30.5, 32.5, 0.19, 0.3263, 0.1496, 17.3625, 13.7148,  3.6476, -0.2233, 3.8710],
        [0.97, 301.0, 301.0, 30.5, 32.5, 0.15, 0.1409, 0.0360, 19.4018, 15.9838,  3.4180, -0.3037, 3.7217],
    ]
)
def test_net_radiation(emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs):
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
    'Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil, Rnci, AEsi',
    [
        [ 1.2417, 9.0157, 0.9094, 0.98, 301, 307.0, 312.4,  1.8113, 20.9278],
        [ 5.5514, 7.1486, 0.6219, 0.97, 301, 303.9, 316.2,  6.6886, 22.9501],
        [13.7148, 3.8710, 0.1496, 0.97, 301, 301.2, 303.5, 13.8912,  6.4699],
        [15.9838, 3.7217, 0.0360, 0.97, 301, 301.0, 301.0, 15.9838,  3.7217],
    ]
)
def test_isothermal_net_radiation(Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil, Rnci, AEsi):
    output = utils.constant_image_value(
        model.isothermal_net_radiation(Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil)
    )
    # print('\n{:7.4f}, {:7.4f}'.format(output['Rnci'], output['AEsi']))
    assert abs(output['Rnci'] - Rnci) <= 0.0001
    assert abs(output['AEsi'] - AEsi) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, Rnci, DELTA, gamma, AEs, AEsi, RHs, mu_c, mu_s',
    [
        [ 1.2417,  1.8113, 0.2184, 0.0673, 9.0157, 20.9278, 0.4045, 2.1508, 2.9160],
        [ 5.5514,  6.6886, 0.2184, 0.0673, 7.1486, 22.9501, 0.4045, 1.6162, 3.9458],
        [13.7148, 13.8912, 0.2184, 0.0673, 3.8710,  6.4699, 0.4045, 1.0525, 2.0925],
        [15.9838, 15.9838, 0.2184, 0.0673, 3.7217,  3.7217, 0.4045, 1.0000, 1.0000],
    ]
)
def test_mu_terms(Rnc, Rnci, DELTA, gamma, AEs, AEsi, RHs, mu_c, mu_s):
    output = utils.constant_image_value(
        model.mu_terms(ee.Image(Rnc), ee.Image(Rnci), DELTA, gamma, ee.Image(AEs), ee.Image(AEsi), RHs)
    )
    # print('\n{:6.4f}, {:6.4f}'.format(output['mu_c'], output['mu_s']))
    assert abs(output['mu_c'] - mu_c) <= 0.0001
    assert abs(output['mu_s'] - mu_s) <= 0.0001


################################################################################
# Second Run
################################################################################

@pytest.mark.parametrize(
    'ea, esat, DELTA, LST_soil, t_avg, mu_s, water, RHs',
    [
        [1.5156, 3.7470, 0.2184, 311.5, 301, 2.9160, 0, 0.2885],
        [1.5156, 3.7470, 0.2184, 312.1, 301, 3.9458, 0, 0.2727],
        [1.5156, 3.7470, 0.2184, 307.9, 301, 2.0925, 0, 0.3343],
        [1.5156, 3.7470, 0.2184, 301.0, 301, 1.0000, 0, 0.4045],
    ]
)
def test_RHs_model_b(ea, esat, DELTA, LST_soil, t_avg, mu_s, water, RHs):
    output = utils.constant_image_value(model.RHs_model(ea, esat, DELTA, LST_soil, t_avg, mu_s, water))
    # print('\n{:6.4f}'.format(output['RHs']))
    assert abs(output['RHs'] - RHs) <= 0.0001


@pytest.mark.parametrize(
    'LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo, LST_canopy, LST_soil',
    [
        # RHs: 1.5156 / 3.7470 = 0.404484
        # gamma: 101.1818 * 0.000665 = 0.067286
        [310.6, 301, 30.5, 32.5, 0.0392, 0.9455, 0.9094, 2.1508, 2.9160, 0.2885, 0.2184, 0.0673, 0.98, 0.18, 301.2, 311.5],
        [308.3, 301, 30.5, 32.5, 0.1813, 0.7558, 0.6219, 1.6162, 3.9458, 0.2727, 0.2184, 0.0673, 0.97, 0.19, 301.6, 312.2],
        [302.6, 301, 30.5, 32.5, 0.5507, 0.3263, 0.1496, 1.0525, 2.0925, 0.3343, 0.2184, 0.0673, 0.97, 0.19, 301.5, 308.6],
        [301.0, 301, 30.5, 32.5, 0.7534, 0.1409, 0.0360, 1.0000, 1.0000, 0.4045, 0.2184, 0.0673, 0.97, 0.15, 301.0, 301.0],
    ]
)
def test_canopy_and_soil_LST_b(LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo, LST_canopy, LST_soil):
    output = utils.constant_image_value(model.canopy_and_soil_LST(
        LST_avg, t_avg, Rs_MJ, Rld_MJ, fc, tauS, tauL, mu_c, mu_s, RHs, DELTA, gamma, emissivity, albedo
    ))
    # print('\n{:5.1f}, {:5.1f}'.format(output['LST_canopy'], output['LST_soil']))
    assert abs(output['LST_canopy'] - LST_canopy) <= 0.1
    assert abs(output['LST_soil'] - LST_soil) <= 0.1


@pytest.mark.parametrize(
    'emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs',
    [
        [0.98, 301.2, 311.5, 30.5, 32.5, 0.18, 0.9455, 0.9094, 12.8043,  1.2417, 11.5626,  2.5469,  9.0157],
        [0.97, 301.6, 312.2, 30.5, 32.5, 0.19, 0.7558, 0.6219, 14.2451,  5.6522,  8.5929,  1.5075,  7.0854],
        [0.97, 301.5, 308.6, 30.5, 32.5, 0.19, 0.3263, 0.1496, 17.3485, 14.1351,  3.2134, -0.3753,  3.5887],
        [0.97, 301.0, 301.0, 30.5, 32.5, 0.15, 0.1409, 0.0360, 19.4018, 15.9838,  3.4180, -0.3037,  3.7217],
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
    'Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil, Rnci, AEsi',
    [
        [ 1.2417, 9.0157, 0.9094, 0.98, 301, 301.2, 311.5,  1.2607, 19.9874],
        [ 5.6522, 7.0854, 0.6219, 0.97, 301, 301.6, 312.2,  5.8875, 18.7286],
        [14.1351, 3.5887, 0.1496, 0.97, 301, 301.5, 308.6, 14.5761, 11.4895],
        [15.9838, 3.7217, 0.0360, 0.97, 301, 301.0, 301.0, 15.9838,  3.7217],
    ]
)
def test_isothermal_net_radiation_b(Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil, Rnci, AEsi):
    output = utils.constant_image_value(
        model.isothermal_net_radiation(Rnc, AEs, tauL, emissivity, t_avg, LST_canopy, LST_soil)
    )
    # print('\n{:7.4f}, {:7.4f}'.format(output['Rnci'], output['AEsi']))
    assert abs(output['Rnci'] - Rnci) <= 0.0001
    assert abs(output['AEsi'] - AEsi) <= 0.0001


@pytest.mark.parametrize(
    'Rnc, Rnci, DELTA, gamma, AEs, AEsi, RHs, mu_c, mu_s',
    [
        [ 1.2417,  1.2607, 0.2184, 0.0673, 9.0157, 19.9874, 0.2885, 1.0621, 2.6473],
        [ 5.6522,  5.8875, 0.2184, 0.0673, 7.0854, 18.7286, 0.2727, 1.1583, 3.1107],
        [14.1351, 14.5761, 0.2184, 0.0673, 3.5887, 11.4895, 0.3343, 1.1215, 3.8259],
        [15.9838, 15.9838, 0.2184, 0.0673, 3.7217,  3.7217, 0.4045, 1.0000, 1.0000],
    ]
)
def test_mu_terms_b(Rnc, Rnci, DELTA, gamma, AEs, AEsi, RHs, mu_c, mu_s):
    output = utils.constant_image_value(
        model.mu_terms(ee.Image(Rnc), ee.Image(Rnci), DELTA, gamma, ee.Image(AEs), ee.Image(AEsi), RHs)
    )
    # print('\n{:6.4f}, {:6.4f}'.format(output['mu_c'], output['mu_s']))
    assert abs(output['mu_c'] - mu_c) <= 0.0001
    assert abs(output['mu_s'] - mu_s) <= 0.0001


################################################################################
# ET
################################################################################

@pytest.mark.parametrize(
    'gamma, DELTA, Rnc, AEs, RHs, mu_c, mu_s, expected',
    [
        [0.0673, 0.2184,  1.2417, 9.0157, 0.2885, 1.0621, 2.6473, 1.3432],
        [0.0673, 0.2184,  5.6522, 7.0854, 0.2727, 1.1583, 3.1107, 2.3407],
        [0.0673, 0.2184, 14.1351, 3.5887, 0.3343, 1.1215, 3.8259, 4.6112],
        [0.0673, 0.2184, 15.9838, 3.7217, 0.4045, 1.0000, 1.0000, 5.8494],
    ]
)
def test_DIF_model(gamma, DELTA, Rnc, AEs, RHs, mu_c, mu_s, expected):
    # gamma, DELTA, Rnc, AEs, RHs, mu_c, mu_s
    output = utils.constant_image_value(model.DIF_model(
        ee.Image.constant(gamma), ee.Image.constant(DELTA), ee.Image.constant(Rnc),
        ee.Image.constant(AEs), ee.Image.constant(RHs), ee.Image.constant(mu_c),
        ee.Image.constant(mu_s)
    ))
    # print('\n{:6.4f}'.format(output['ET_DIF']))
    assert abs(output['ET_DIF'] - expected) <= 0.0001


@pytest.mark.parametrize(
    'del_LC, fc, LST_soil, RHs, gamma, DELTA, u2, ea, esat, expected',
    [
        # Non-wet landcover
        [0, 0.0392, 311.5, 0.2885, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.1813, 312.2, 0.2727, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.5507, 308.6, 0.3343, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.0],
        [0, 0.7534, 301.0, 0.4045, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.0],
        # Wet landcover
        [1, 0.0392, 311.5, 0.2885, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.0983],
        [1, 0.1813, 312.2, 0.2727, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 0.4305],
        [1, 0.5507, 308.6, 0.3343, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 1.3156],
        [1, 0.7534, 301.0, 0.4045, 0.0673, 0.2184, 1.3463, 1.5156, 3.7470, 1.8554],
    ]
)
def test_aerodynamic_term(del_LC, fc, LST_soil, RHs, gamma, DELTA, u2, ea, esat, expected):
    output = utils.constant_image_value(
        model.aerodynamic_term(del_LC, fc, LST_soil, RHs, gamma, DELTA, u2, esat, ea)
    )
    # print('\n{:6.4f}'.format(output['Ea']))
    assert abs(output['Ea'] - expected) <= 0.1


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
