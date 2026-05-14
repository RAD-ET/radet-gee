from datetime import datetime
import math
import pprint

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
SCENE_TIME = 1500230731090
TEST_POINT = [-121.668, 38.905]

# Using the same meteorology and site locations for the two conditions
#   but the image values are based on fields near the test point
site_et = {
    'tmin': 290,
    'tmax': 312,
    'qa': 0.007,
    'u10': 2,
    'srad': 350,
    'lat': 38.910,
    'lon': -121.66,
    'elev': 10,
    'nlcd': 82,
    'time': SCENE_TIME,
    'meteo_elev': 10,
}
site_et['pair'] = 101.3 * ((293 - 0.0065 * site_et['elev']) / 293) ** 5.26
site_et['gamma'] = (101.3 * ((293 - 0.0065 * site_et['elev']) / 293) ** 5.26) * 0.000665
site_et['u2'] = site_et['u10'] * 4.87 / math.log(67.8 * 10 - 5.42)
site_et['ea'] = site_et['qa'] * site_et['pair'] / ((1 - 0.622) * site_et['qa'] + 0.622)
site_et['t_avg'] = (site_et['tmin'] + site_et['tmax']) / 2
site_et['esat'] = 0.6108 * math.exp(17.27 * (site_et['t_avg'] - 273.15) / ((site_et['t_avg'] - 273.15) + 237.3))
site_et['delta'] = (
    2503 * math.exp(17.27 * (site_et['t_avg'] - 273.15) / ((site_et['t_avg'] - 273.15) + 237.3))
    / ((site_et['t_avg'] - 273.15) + 237.3) ** 2
)
site_et['rhs_1'] = (
    site_et['ea'] / (0.6108 * math.exp(17.27 * (site_et['t_avg'] - 273.15) / ((site_et['t_avg'] - 273.15) + 237.3)))
)
het = site_et | {
    'lai': 3.5,
    'lst': 304,
    'albedo': 0.15,
    'emissivity': 0.97,
    # Outputs
    'Rs_MJ': 30.2400,
    'Ra_MJ': 40.7331,
    'fcd': 0.9860,
    'sunrise_ts': 4.7773,
    'tauL': 0.035973,
    'tauS': 0.140858,
    'fc': 0.753403,
    'Rs_MJ_cor': 30.2400,
    'Rld_MJ': 31.6771,
    'LST_avg': 301.0000,
    'mu_c_0': 1,
    'mu_s_0': 1,
    # First run
    'LST_c_1': 301.0000,
    'LST_s_1': 301.0000,
    'AEs_1': 3.6816,
    'G_1': -0.325276,
    'Rn_1': 18.3580,
    'Rnc_1': 15.0016,
    'Rns_1': 3.3564,
    'Rnci_1': 15.0016,
    'AEsi_1': 3.6816,
    'mu_c_1': 1,
    'mu_s_1': 1,
    'Ea_1': 2.2108,
    'et_dif_1': 5.4256,
    # Second run
    'rhs_2': 0.3026,
    'LST_c_2': 301.0000,
    'LST_s_2': 301.0000,
    'AEs_2': 3.6816,
    'G_2': -0.325276,
    'Rn_2': 18.3580,
    'Rnc_2': 15.0016,
    'Rns_2': 3.3564,
    'Rnci_2': 15.0016,
    'AEsi_2': 3.6816,
    'mu_c_2': 1,
    'mu_s_2': 1,
    'Ea_2': 2.2108,
    'et_dif_2': 5.7539,
    'et_2': 0,
}
let = site_et | {
    'lai': 0.1,
    'lst': 326,
    'albedo': 0.175,
    'emissivity': 0.97,
    # Outputs
    'Rs_MJ': 30.2400,
    'Ra_MJ': 40.7331,
    'fcd': 0.9860,
    'sunrise_ts': 4.7773,
    'tauL': 0.909373,
    'tauS': 0.945539,
    'fc': 0.039211,
    'Rs_MJ_cor': 30.2400,
    'Rld_MJ': 31.6771,
    'LST_avg': 310.179748,
    'mu_c_0': 1,
    'mu_s_0': 1,
    # First run
    'LST_c_1': 301.1716,
    'LST_s_1': 311.0356,
    'AEs_1': 8.9404,
    'G_1': 2.5063,
    'Rn_1': 12.6193,
    'Rnc_1': 1.1726,
    'Rns_1': 11.4467,
    'Rnci_1': 1.1887,
    'AEsi_1': 19.3731,
    'mu_c_1': 1.0559,
    'mu_s_1': 2.2160,
    'Ea_1': 0.1249,
    'et_dif_1': 1.4819,
    # Second run
    'rhs_2': 0.2291,
    'LST_c_2': 301.1212,
    'LST_s_2': 311.0401,
    'AEs_2': 8.9371,
    'G_2': 2.5046,
    'Rn_2': 12.6193,
    'Rnc_2': 1.1775,
    'Rns_2': 11.4418,
    'Rnci_2': 1.1888,
    'AEsi_2': 19.3745,
    'mu_c_2': 1.0396,
    'mu_s_2': 2.2053,
    'Ea_2': 0.1150,
    'et_dif_2': 1.2881,
    'et_2': 0,
}


@pytest.mark.parametrize(
    'time_start, lat, elev, srad, Rs_MJ, Ra_MJ, fcd, sunrise_ts',
    [
        [
            het['time'], het['lat'], het['elev'], het['srad'],
            het['Rs_MJ'], het['Ra_MJ'], het['fcd'], het['sunrise_ts']
        ],
        [
            let['time'], let['lat'], let['elev'], let['srad'],
            let['Rs_MJ'], let['Ra_MJ'], let['fcd'], let['sunrise_ts']
        ],
    ]
)
def test_clear_sky_terms(time_start, lat, elev, srad, Rs_MJ, Ra_MJ, fcd, sunrise_ts, tol=0.0001):
    output = utils.constant_image_value(model.clear_sky_terms(
        ee.Number(time_start),
        ee.Number(lat),
        ee.Number(elev),
        ee.Image.constant(srad)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['Rs_MJ'] - Rs_MJ) < tol
    assert abs(output['Ra_MJ'] - Ra_MJ) < tol
    assert abs(output['fcd'] - fcd) < tol
    assert abs(output['constant'] - sunrise_ts) < tol


@pytest.mark.parametrize(
    'Rs_MJ, Ra_MJ, elev, time_start, albedo, latitude, longitude, expected',
    [
        [
            het['Rs_MJ'], het['Ra_MJ'], het['elev'], het['time'], het['albedo'], het['lat'], het['lon'],
            het['Rs_MJ_cor']
        ],
        # Changing albedo does not change the value
        [
            let['Rs_MJ'], let['Ra_MJ'], let['elev'], let['time'], let['albedo'], let['lat'], let['lon'],
            let['Rs_MJ_cor']
        ],
        # TODO: Identify conditions that should cause a correction to be applied
    ]
)
def test_terrain_shade_correct_srad(
        Rs_MJ, Ra_MJ, elev, time_start, albedo, latitude, longitude, expected, tol=0.0001
):
    output = utils.constant_image_value(model.terrain_shade_correct_srad(
        ee.Image.constant(Rs_MJ),
        ee.Image.constant(Ra_MJ),
        ee.Number(elev),
        ee.Number(time_start),
        ee.Number(albedo),
        ee.Number(latitude),
        ee.Number(longitude)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['Rs_MJ_shade'] - expected) < tol


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
        landcover=ee.Image.constant(landcover), lai=ee.Number(lai)
    ))
    assert output['remapped'] == wet
    assert output['remapped_1'] == water


@pytest.mark.parametrize(
    'lai, tauL, tauS, fc',
    [
        [let['lai'], let['tauL'], let['tauS'], let['fc']],
        [het['lai'], het['tauL'], het['tauS'], het['fc']],
        # Check for low value clamping at LAI values above clamped range
        [0.0, 1.0, 1.0, 0.01],
        [8.0, 0.01, 0.0113334, 0.959238],
        [10, 0.01, 0.01, 0.981684],
    ]
)
def test_transmissivities(lai, tauL, tauS, fc, tol=0.000001):
    output = utils.constant_image_value(model.transmissivities(ee.Image.constant(lai)))
    print('')
    pprint.pprint(output)
    assert abs(output['tauL'] - tauL) < tol
    assert abs(output['tauS'] - tauS) < tol
    assert abs(output['fc'] - fc) < tol


@pytest.mark.parametrize(
    'emissivity, fcd, ea, t_avg, expected',
    [
        [het['emissivity'], het['fcd'], het['ea'], het['t_avg'], het['Rld_MJ']],
        [let['emissivity'], let['fcd'], let['ea'], let['t_avg'], let['Rld_MJ']],
    ]
)
def test_Rld_atm_ASCE(emissivity, fcd, ea, t_avg, expected, tol=0.0001):
    output = utils.constant_image_value(model.Rld_atm_ASCE(
        ee.Number(emissivity),
        ee.Number(fcd),
        ee.Number(ea),
        ee.Number(t_avg)
    ))
    assert abs(output['Rld'] - expected) < tol


@pytest.mark.parametrize(
    'tmin, lst, sunrise_ts, t_avg, expected',
    [
        # Note that average LST is forced to be >= t_avg
        [het['tmin'], het['lst'], het['sunrise_ts'], het['t_avg'], het['LST_avg']],
        [let['tmin'], let['lst'], let['sunrise_ts'], let['t_avg'], let['LST_avg']],
    ]
)
def test_daily_avg_lst(tmin, lst, sunrise_ts, t_avg, expected, tol=0.0001):
    output = utils.constant_image_value(model.daily_avg_lst(
        ee.Number(tmin),
        ee.Image.constant(lst),
        ee.Number(sunrise_ts),
        ee.Number(t_avg)
    ))
    assert abs(output['LST_avg'] - expected) < tol


@pytest.mark.parametrize(
    'LST_avg, t_avg, fc, mu_c, mu_s, RHs, DELTA, gamma, LST_canopy',
    [
        # First run mu terms set to 1
        [
            het['LST_avg'], het['t_avg'], het['fc'], het['mu_c_0'], het['mu_s_0'],
            het['rhs_1'], het['delta'], het['gamma'], het['LST_c_1']
        ],
        [
            let['LST_avg'], let['t_avg'], let['fc'], let['mu_c_0'], let['mu_s_0'],
            let['rhs_1'], let['delta'], let['gamma'], let['LST_c_1']
        ],
        # Second run with updated RHs and mu terms
        [
            het['LST_avg'], het['t_avg'], het['fc'], het['mu_c_1'], het['mu_s_1'],
            het['rhs_2'], het['delta'], het['gamma'], het['LST_c_2']
        ],
        [
            let['LST_avg'], let['t_avg'], let['fc'], let['mu_c_1'], let['mu_s_1'],
            let['rhs_2'], let['delta'], let['gamma'], let['LST_c_2']
        ],
    ]
)
def test_canopy_LST(LST_avg, t_avg, fc, mu_c, mu_s, RHs, DELTA, gamma, LST_canopy, tol=0.0001):
    output = utils.constant_image_value(model.canopy_LST(
        ee.Number(LST_avg),
        ee.Number(t_avg),
        ee.Number(fc),
        ee.Number(mu_c),
        ee.Number(mu_s),
        ee.Number(RHs),
        ee.Number(DELTA),
        ee.Number(gamma),
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['LST_canopy'] - LST_canopy) < tol


@pytest.mark.parametrize(
    'LST_avg, LST_canopy, Rs_MJ_cor, Rld_MJ, tauS, tauL, emissivity, albedo, LST_soil',
    [
        # First run mu terms set to 1
        [
            het['LST_avg'], het['LST_c_1'], het['Rs_MJ_cor'], het['Rld_MJ'],
            het['tauS'], het['tauL'], het['emissivity'], het['albedo'], het['LST_s_1']
        ],
        [
            let['LST_avg'], let['LST_c_1'], let['Rs_MJ_cor'], let['Rld_MJ'],
            let['tauS'], let['tauL'], let['emissivity'], let['albedo'], let['LST_s_1']
        ],
        # Second run with updated RHs and mu terms
        [
            het['LST_avg'], het['LST_c_2'], het['Rs_MJ_cor'], het['Rld_MJ'],
            het['tauS'], het['tauL'], het['emissivity'], het['albedo'], het['LST_s_2']
        ],
        [
            let['LST_avg'], let['LST_c_2'], let['Rs_MJ_cor'], let['Rld_MJ'],
            let['tauS'], let['tauL'], let['emissivity'], let['albedo'], let['LST_s_2']
        ],
    ]
)
def test_soil_LST(
        LST_avg, LST_canopy, Rs_MJ_cor, Rld_MJ, tauS, tauL, emissivity, albedo,
        LST_soil, tol=0.0001
):
    output = utils.constant_image_value(model.soil_LST(
        ee.Number(LST_avg),
        ee.Number(LST_canopy),
        ee.Number(Rs_MJ_cor),
        ee.Number(Rld_MJ),
        ee.Number(tauS),
        ee.Number(tauL),
        ee.Number(emissivity),
        ee.Number(albedo)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['LST_soil'] - LST_soil) < tol


@pytest.mark.parametrize(
    'emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL, Rn, Rnc, Rns, G, AEs',
    [
        # First run
        [
            het['emissivity'], het['LST_c_1'], het['LST_s_1'], het['Rs_MJ'], het['Rld_MJ'],
            het['albedo'], het['tauS'], het['tauL'],
            het['Rn_1'], het['Rnc_1'], het['Rns_1'], het['G_1'], het['AEs_1']
        ],
        [
            let['emissivity'], let['LST_c_1'], let['LST_s_1'], let['Rs_MJ'], let['Rld_MJ'],
            let['albedo'], let['tauS'], let['tauL'],
            let['Rn_1'], let['Rnc_1'], let['Rns_1'], let['G_1'], let['AEs_1']
        ],
        # Second run
        [
            het['emissivity'], het['LST_c_2'], het['LST_s_2'], het['Rs_MJ'], het['Rld_MJ'],
            het['albedo'], het['tauS'], het['tauL'],
            het['Rn_2'], het['Rnc_2'], het['Rns_2'], het['G_2'], het['AEs_2']
        ],
        [
            let['emissivity'], let['LST_c_2'], let['LST_s_2'], let['Rs_MJ'], let['Rld_MJ'],
            let['albedo'], let['tauS'], let['tauL'],
            let['Rn_2'], let['Rnc_2'], let['Rns_2'], let['G_2'], let['AEs_2']
        ],
    ]
)
def test_net_radiation(
        emissivity, LST_canopy, LST_soil, Rs_MJ, Rld_MJ, albedo, tauS, tauL,
        Rn, Rnc, Rns, G, AEs, tol=0.0001
):
    output = utils.constant_image_value(model.net_radiation(
        ee.Number(emissivity),
        ee.Number(LST_canopy),
        ee.Number(LST_soil),
        ee.Number(Rs_MJ),
        ee.Number(Rld_MJ),
        ee.Number(albedo),
        ee.Number(tauS),
        ee.Number(tauL)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['Rn'] - Rn) < tol
    assert abs(output['Rnc'] - Rnc) < tol
    assert abs(output['Rns'] - Rns) < tol
    assert abs(output['G'] - G) < tol
    assert abs(output['AEs'] - AEs) < tol


@pytest.mark.parametrize(
    'Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci',
    [
        # First run
        [het['Rnc_1'], het['tauL'], het['emissivity'], het['t_avg'], het['LST_c_1'], het['Rnci_1']],
        [let['Rnc_1'], let['tauL'], let['emissivity'], let['t_avg'], let['LST_c_1'], let['Rnci_1']],
        # Second run
        [het['Rnc_2'], het['tauL'], het['emissivity'], het['t_avg'], het['LST_c_2'], het['Rnci_2']],
        [let['Rnc_2'], let['tauL'], let['emissivity'], let['t_avg'], let['LST_c_2'], let['Rnci_2']],
    ]
)
def test_isothermal_net_radiation(Rnc, tauL, emissivity, t_avg, LST_canopy, Rnci, tol=0.0001):
    output = utils.constant_image_value(model.isothermal_net_radiation(
        ee.Image.constant(Rnc),
        ee.Number(tauL),
        ee.Number(emissivity),
        ee.Number(t_avg),
        ee.Number(LST_canopy),
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['Rnci'] - Rnci) < tol


@pytest.mark.parametrize(
    'AEs, emissivity, t_avg, LST_soil, AEsi',
    [
        # First run
        [het['AEs_1'], het['emissivity'], het['t_avg'], het['LST_s_1'], het['AEsi_1']],
        [let['AEs_1'], let['emissivity'], let['t_avg'], let['LST_s_1'], let['AEsi_1']],
        # Second run
        [het['AEs_2'], het['emissivity'], het['t_avg'], het['LST_s_2'], het['AEsi_2']],
        [let['AEs_2'], let['emissivity'], let['t_avg'], let['LST_s_2'], let['AEsi_2']],
    ]
)
def test_isothermal_soil_available_energy(AEs, emissivity, t_avg, LST_soil, AEsi, tol=0.0001):
    output = utils.constant_image_value(model.isothermal_soil_available_energy(
        ee.Number(AEs),
        ee.Number(emissivity),
        ee.Number(t_avg),
        ee.Number(LST_soil),
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['AEsi'] - AEsi) < tol


@pytest.mark.parametrize(
    'Rnc, Rnci, DELTA, gamma, mu_c',
    [
        # First run mu terms (that will be used in second run)
        [het['Rnc_1'], het['Rnci_1'], het['delta'], het['gamma'], het['mu_c_1']],
        [let['Rnc_1'], let['Rnci_1'], let['delta'], let['gamma'], let['mu_c_1']],
        # Second run mu terms
        [het['Rnc_2'], het['Rnci_2'], het['delta'], het['gamma'], het['mu_c_2']],
        [let['Rnc_2'], let['Rnci_2'], let['delta'], let['gamma'], let['mu_c_2']],
    ]
)
def test_mu_canopy(Rnc, Rnci, DELTA, gamma, mu_c, tol=0.0001):
    output = utils.constant_image_value(model.mu_canopy(
        ee.Number(Rnc),
        ee.Number(Rnci),
        ee.Number(DELTA),
        ee.Number(gamma),
    ))
    # print('')
    # pprint.pprint(output)
    assert abs(output['mu_c'] - mu_c) < tol


@pytest.mark.parametrize(
    'AEs, AEsi, DELTA, gamma, RHs, mu_s',
    [
        # First run mu terms
        [het['AEs_1'], het['AEsi_1'], het['rhs_1'], het['delta'], het['gamma'], het['mu_s_1']],
        [let['AEs_1'], let['AEsi_1'], let['rhs_1'], let['delta'], let['gamma'], let['mu_s_1']],
        # Second run mu terms
        [het['AEs_2'], het['AEsi_2'], het['rhs_2'], het['delta'], het['gamma'], het['mu_s_2']],
        [let['AEs_2'], let['AEsi_2'], let['rhs_2'], let['delta'], let['gamma'], let['mu_s_2']],
    ]
)
def test_mu_soil(AEs, AEsi, RHs, DELTA, gamma, mu_s, tol=0.0001):
    output = utils.constant_image_value(model.mu_soil(
        ee.Number(AEs),
        ee.Number(AEsi),
        ee.Number(RHs),
        ee.Number(DELTA),
        ee.Number(gamma),
    ))
    # print('')
    # pprint.pprint(output)
    assert abs(output['mu_s'] - mu_s) < tol


@pytest.mark.parametrize(
    'ea, esat, DELTA, LST_soil, t_avg, mu_s, water, expected',
    [
        # RHs for second run is using the mu_s term generated at the end of the first run
        [het['ea'], het['esat'], het['delta'], het['LST_s_1'], het['t_avg'], het['mu_s_1'], 0, het['rhs_2']],
        [let['ea'], let['esat'], let['delta'], let['LST_s_1'], let['t_avg'], let['mu_s_1'], 0, let['rhs_2']],
    ]
)
def test_RHs_model(ea, esat, DELTA, LST_soil, t_avg, mu_s, water, expected, tol=0.0001):
    output = utils.constant_image_value(model.RHs_model(
        ee.Number(ea),
        ee.Number(esat),
        ee.Number(DELTA),
        ee.Number(LST_soil),
        ee.Number(t_avg),
        ee.Number(mu_s),
        ee.Number(water)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['RHs'] - expected) < tol


@pytest.mark.parametrize(
    'del_LC, fc, LST_soil, RHs, DELTA, gamma, u2, esat, ea, expected',
    [
        # Aerodynamic term is not calculated during the first run, but including for testing
        [
            1, het['fc'], het['LST_s_1'], het['rhs_1'], het['delta'], het['gamma'],
            het['u2'], het['esat'], het['ea'], het['Ea_1']
        ],
        [
            1, let['fc'], let['LST_s_1'], let['rhs_1'], let['delta'], let['gamma'],
            let['u2'], let['esat'], let['ea'], let['Ea_1']
        ],
        # Second run using update LST_soil and RHs
        [
            1, het['fc'], het['LST_s_2'], het['rhs_2'], het['delta'], het['gamma'],
            het['u2'], het['esat'], het['ea'], het['Ea_2']
        ],
        [
            1, let['fc'], let['LST_s_2'], let['rhs_2'], let['delta'], let['gamma'],
            let['u2'], let['esat'], let['ea'], let['Ea_2']
        ],
    ]
)
def test_aerodynamic_term(del_LC, fc, LST_soil, RHs, DELTA, gamma, u2, esat, ea, expected, tol=0.0001):
    output = utils.constant_image_value(model.aerodynamic_term(
        ee.Number(del_LC),
        ee.Number(fc),
        ee.Number(LST_soil),
        ee.Number(RHs),
        ee.Number(DELTA),
        ee.Number(gamma),
        ee.Number(u2),
        ee.Number(esat),
        ee.Number(ea)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['Ea'] - expected) < tol


@pytest.mark.parametrize(
    'Rnc, AEs, RHs, gamma, DELTA, mu_c, mu_s, expected',
    [
        # DIF model is not computed using first run values but including for testing
        [
            het['Rnc_1'], het['AEs_1'], het['rhs_1'], het['gamma'], het['delta'],
            het['mu_c_1'], het['mu_s_1'], het['et_dif_1']
        ],
        [
            let['Rnc_1'], let['AEs_1'], let['rhs_1'], let['gamma'], let['delta'],
            let['mu_c_1'], let['mu_s_1'], let['et_dif_1']
        ],
        # Second run values
        [
            het['Rnc_1'], het['AEs_1'], het['delta'], het['gamma'], het['rhs_2'],
            het['mu_c_2'], het['mu_s_2'], het['et_dif_2']
        ],
        [
            let['Rnc_2'], let['AEs_2'], let['delta'], let['gamma'], let['rhs_2'],
            let['mu_c_2'], let['mu_s_2'], let['et_dif_2']
        ],
    ]
)
def test_DIF_model(Rnc, AEs, RHs, DELTA, gamma, mu_c, mu_s, expected, tol=0.0001):
    output = utils.constant_image_value(model.DIF_model(
        ee.Image.constant(Rnc),
        ee.Number(AEs),
        ee.Number(RHs),
        ee.Number(DELTA),
        ee.Number(gamma),
        ee.Number(mu_c),
        ee.Number(mu_s)
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['ET_DIF'] - expected) <= tol


# @pytest.mark.parametrize(
#     'ta, elev, meteo_elev, expected',
#     [
#         [0, 0, 0, 0],
#     ]
# )
# def test_add_lapse_correction(ta, elev, meteo_elev, expected, tol=0.0001):
#     output = utils.constant_image_value(model.add_lapse_correction(
#         ee.Image.constant(ta),
#         ee.Number(elev),
#         ee.Number(meteo_elev)
#     ))
#     print('')
#     pprint.pprint(output)
#     assert abs(output['t_corrected']) >= 0
#     # assert abs(output['t_corrected'] - mu_c) < tol


@pytest.mark.parametrize(
    'lai, lst, albedo, emissivity, landcover, elevation, tmin, tmax, qa, u10, srad, '
    'meteo_elevation, time_start, latitude, longitude, expected',
    [
        # High ET (-121.668, 38.905)
        [
            het['lai'], het['lst'], het['albedo'], het['emissivity'], het['nlcd'], het['elev'],
            het['tmin'], het['tmax'], het['qa'], het['u10'], het['srad'],
            het['meteo_elev'], het['time'], het['lat'], het['lon'], 7.6365
        ],
        # Low ET (-121.650, 38.913)
        [
            let['lai'], let['lst'], let['albedo'], let['emissivity'], let['nlcd'], let['elev'],
            let['tmin'], let['tmax'], let['qa'], let['u10'], let['srad'],
            let['meteo_elev'], let['time'], let['lat'], let['lon'], 1.2952
        ],
        # Check ET if tmin=tmax to mimic test_c_image.py defaults
        [
            het['lai'], het['lst'], het['albedo'], het['emissivity'], het['nlcd'], het['elev'],
            het['t_avg'], het['t_avg'], het['qa'], het['u10'], het['srad'],
            het['meteo_elev'], het['time'], het['lat'], het['lon'], 7.6685
        ],
    ]
)
def test_et(
        lai, lst, albedo, emissivity, landcover, elevation, tmin, tmax, qa, u10, srad,
        meteo_elevation, time_start, latitude, longitude, expected, tol=0.0001
):
    output = utils.constant_image_value(model.et(
        lai=ee.Image.constant(lai),
        lst=ee.Image.constant(lst),
        albedo=ee.Number(albedo),
        emissivity=ee.Number(emissivity),
        landcover=ee.Image.constant(landcover),
        elevation=ee.Number(elevation),
        tmin=ee.Image.constant(tmin),
        tmax=ee.Image.constant(tmax),
        qa=ee.Number(qa),
        u10=ee.Number(u10),
        srad=ee.Image.constant(srad),
        meteo_elevation=ee.Number(meteo_elevation),
        time_start=ee.Number(time_start),
        latitude=ee.Number(latitude),
        longitude=ee.Number(longitude),
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['ET'] - expected) < tol


@pytest.mark.parametrize(
    'lai, lst, albedo, emissivity, landcover, elevation, tmin, tmax, qa, u10, srad, '
    'meteo_elevation, time_start, latitude, longitude, expected',
    [
        # High ET site
        [
            het['lai'], het['lst'], het['albedo'], het['emissivity'], het['nlcd'], het['elev'],
            het['tmin'], het['tmax'], het['qa'], het['u10'], het['srad'],
            het['meteo_elev'], het['time'], het['lat'], het['lon'], 7.6365
        ],
        # Low ET site
        [
            let['lai'], let['lst'], let['albedo'], let['emissivity'], let['nlcd'], let['elev'],
            let['tmin'], let['tmax'], let['qa'], let['u10'], let['srad'],
            let['meteo_elev'], let['time'], let['lat'], let['lon'], 1.2952
        ],
    ]
)
def test_et_positional_arguments(
        lai, lst, albedo, emissivity, tmin, tmax, qa, u10, srad, landcover, elevation,
        meteo_elevation, time_start, latitude, longitude, expected, tol=0.01
):
    output = utils.constant_image_value(model.et(
        ee.Image.constant(lai),
        ee.Image.constant(lst),
        ee.Number(albedo),
        ee.Number(emissivity),
        ee.Image.constant(landcover),
        ee.Number(elevation),
        ee.Image.constant(tmin),
        ee.Image.constant(tmax),
        ee.Number(qa),
        ee.Number(u10),
        ee.Image.constant(srad),
        ee.Number(meteo_elevation),
        ee.Number(time_start),
        ee.Number(latitude),
        ee.Number(longitude),
    ))
    print('')
    pprint.pprint(output)
    assert abs(output['ET'] - expected) < tol
