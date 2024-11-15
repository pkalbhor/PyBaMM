import pybamm
import numpy as np
from numpy.polynomial import Polynomial as P


def graphite_LGM50_ocp_Chen2020(sto):
    """
    LG M50 Graphite open-circuit potential as a function of stochiometry, fit taken
    from [1]. Prada2013 doesn't give an OCP for graphite, so we use this instead.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )

    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12 = (
        1.9793,
        39.3631,
        0.2482,
        0.0909,
        29.8538,
        0.1234,
        0.04478,
        14.9159,
        0.2769,
        0.0205,
        30.4444,
        0.6103,
    )
    # Modifications ------------------------------------------------
    g2 = 160  # Stiffness at the left end
    g3 = 0.2482
    # Higher g5 more bulge on top (left side of flat region)
    # Higer g4 shifts overall curve to the top
    g4, g5, g6 = 0.09509, 13.8538, 0.02534
    g7, g8, g9 = 0.04478, 4.9159, 0.1769
    # Higher g10 shifts right region to top (makes slop more stiffer)
    # Higher the g11 stiffer the middle region, g12 is location of start of middle region
    g10, g11, g12 = 0.0095, 75, 0.49
    # --------------------------------------------------------------
    u_eq = (
        g1 * np.exp(-g2 * sto)
        + g3
        - g4 * np.tanh(g5 * (sto - g6))
        - g7 * np.tanh(g8 * (sto - g9))
        - g10 * np.tanh(g11 * (sto - g12))
    )
    return u_eq


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 1.60e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def costomized_LFP_ocp(sto):
    """
    Used graphite ocp to modify LFP ocp in order to match total ocv
    (Not used due to issue of solver not able handle it)
    """
    params = get_parameter_values()
    Ly = params["Electrode height [m]"]
    Lz = params["Electrode width [m]"]
    F = pybamm.constants.F.value

    epsilon = params["Negative electrode active material volume fraction"]
    cmax = params["Maximum concentration in negative electrode [mol.m-3]"]
    Lx = params["Negative electrode thickness [m]"]
    Qn = (epsilon * Lx * cmax * F * Ly * Lz) / 3600.0

    epsilon_p = params["Positive electrode active material volume fraction"]
    cmax_p = params["Maximum concentration in positive electrode [mol.m-3]"]
    Lx_p = params["Positive electrode thickness [m]"]
    Qp = (epsilon_p * Lx_p * cmax_p * F * Ly * Lz) / 3600.0

    Q = 15.60
    n_range = Q / Qn
    p_range = Q / Qp
    x0 = 0.22987485742344696
    y100 = 0.07175679228303475
    x100 = x0 + n_range
    y0 = y100 + p_range

    def x(soc):
        return x0 + soc * (x100 - x0)

    soc = (y0 - sto) / (y0 - y100)

    params = np.array(
        [
            3.284631984354847,
            -0.014272896024910355,
            0.059241012581277945,
            3.2963956508891594,
            10.48078177034622,
            -35.050703790540744,
            -182.28677411972143,
            -40.221752330856084,
            1341.1571905125331,
            3147.745211154691,
            -5385.093985908603,
            -25799.706715768334,
            12285.074355226732,
            110021.91946497475,
            -13852.683803353008,
            -289411.6810352433,
            -893.6065121716749,
            494406.29669876007,
            24246.64169206978,
            -551172.7256783927,
            -31575.185053443267,
            387511.82145705464,
            18043.289893441222,
            -156118.57807581738,
            -4038.162191300156,
            27487.34233488234,
        ]
    )
    domain = [0.0044, 0.994654]

    full_cell_ocp = P(coef=list(params), domain=domain)

    return full_cell_ocp(soc) + graphite_LGM50_ocp_Chen2020(x(soc))


def LFP_ocp_Afshar2017(sto):
    """
    Open-circuit potential for LFP.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    # Same old function with modified parameters
    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    k = 3.4077 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)

    l1, l2, l3, l4, l5, l6 = 150, 30, 3.4077, 0.020269, 0.5, 0.9
    l3 = 3.4277  # Decides verticle shift of the plot
    l1 = 290  # decies slope of a right peak
    c1 = -l1 * sto
    c2 = -l2 * (1 - sto)
    k = l3 - l4 * sto + l5 * np.exp(c1) - l6 * np.exp(c2)
    return k


def LFP_electrolyte_exchange_current_density_kashkooli2017(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between LFP and electrolyte

    References
    ----------
    .. [1] Kashkooli, A. G., Amirfazli, A., Farhad, S., Lee, D. U., Felicelli, S., Park,
    H. W., ... & Chen, Z. (2017). Representative volume element model of lithium-ion
    battery electrodes based on X-ray nano-tomography. Journal of Applied
    Electrochemistry, 47(3), 281-293.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    m_ref = 3.15 * 10 ** (-5)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_conductivity_Prada2013(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from :footcite:`Prada2013`.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid conductivity
    """
    # convert c_e from mol/m3 to mol/L
    c_e = c_e / 1e6

    sigma_e = (
        4.1253e-4
        + 5.007 * c_e
        - 4721.2 * c_e**2
        + 1.5094e6 * c_e**3
        - 1.6018e8 * c_e**4
    ) * 1e3

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an LFP cell, from the paper :footcite:t:`Prada2013`
    """

    conc_max_neg = 80300
    conc_max_pos = 41283.63
    return {
        "chemistry": "lithium_ion",
        # cell
        "Negative electrode thickness [m]": 135e-06,
        "Separator thickness [m]": 12e-06,
        "Positive electrode thickness [m]": 190e-06,
        "Electrode height [m]": 133e-03,  # to give an area of 0.18 m2
        "Electrode width [m]": 2.350,  # to give an area of 0.18 m2
        "Nominal cell capacity [A.h]": 15,
        "Current function [A]": 15,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 18000,
        "Maximum concentration in negative electrode [mol.m-3]": conc_max_neg,
        "Negative particle diffusivity [m2.s-1]": 5e-8 * 1.9e-6,
        "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
        "Negative electrode porosity": 0.35,
        "Negative electrode active material volume fraction": 0.21,
        "Negative particle radius [m]": 7e-6,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Negative electrode OCP entropic change [V.K-1]": 0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 1e-9,
        "Maximum concentration in positive electrode [mol.m-3]": conc_max_pos,
        "Positive particle diffusivity [m2.s-1]": 5e-11 * 5e-6,
        "Positive electrode OCP [V]": LFP_ocp_Afshar2017,
        "Positive electrode porosity": 0.48,
        "Positive electrode active material volume fraction": 0.27,
        "Positive particle radius [m]": 0.6e-06,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": LFP_electrolyte_exchange_current_density_kashkooli2017,
        "Positive electrode OCP entropic change [V.K-1]": 0,
        # separator
        "Separator porosity": 0.40,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1300.0,
        "Cation transference number": 0.36,
        "Thermodynamic factor": 1.0,
        "Electrolyte diffusivity [m2.s-1]": 2e-10,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Prada2013,
        # experiment
        "Reference temperature [K]": 298,
        "Ambient temperature [K]": 298,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 0.0,
        "Upper voltage cut-off [V]": 3.6,
        "Open-circuit voltage at 0% SOC [V]": 2.0,
        "Open-circuit voltage at 100% SOC [V]": 3.6,
        # initial concentrations adjusted to give 2.3 Ah cell with 3.6 V OCV at 100% SOC
        # and 2.0 V OCV at 0% SOC
        "Initial concentration in negative electrode [mol.m-3]": 0.0065779239014477275
        * conc_max_neg,
        "Initial concentration in positive electrode [mol.m-3]": 0.8724372600038954
        * conc_max_pos,
        "Initial temperature [K]": 298,
        # citations
        "citations": ["Chen2020", "Prada2013"],
    }
