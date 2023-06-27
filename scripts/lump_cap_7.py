import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp, cumtrapz
from scipy.optimize import curve_fit
import pysindy as ps
import mph
from tqdm import tqdm


plt.style.use(['science', 'grid'])


def flowrate_from_hours(hours):
    power = 20e6  # W
    cp_Sn = 240
    delT = 500
    mdot_Sn = power / (cp_Sn * delT)
    energy = power * hours * 3600  # J
    V_block = 0.1**2 * np.pi * 2 - 0.01**2 * np.pi * 2
    rho_C = 1700
    cp_C = 2000
    num_blocks = energy / (V_block * rho_C * cp_C * delT)
    series_blocks = 5
    scaling_factor = series_blocks / num_blocks
    scaled_mdot_Sn = mdot_Sn * scaling_factor
    # print('scaled mdot_Sn: {:.6f} kg/s'.format(scaled_mdot_Sn))
    return scaled_mdot_Sn


def plot_COMSOL():
    # df_all = pd.DataFrame()
    df_all = None
    for i in range(6):
        df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_outlet_T_{}.csv'.format(i),
                         names=['hours', 'time', 'outlet_T_{}'.format(i)],
                         skiprows=5)
        if df_all is None:
            df_all = df
        else:
            df_all['outlet_T_{}'.format(i)] = df['outlet_T_{}'.format(i)]
    # print(df_all.info())
    # return
    del df
    df = df_all
    hours = np.arange(5, 105, 5)
    scaled_mdots = flowrate_from_hours(hours)
    for hour, mdot in zip(hours, scaled_mdots):
        plt.figure(num=1, clear=True)
        if hour not in df['hours'].unique():
            continue
        df_new = df[df['hours'] == hour]
        for i in range(5):
            plt.plot(df_new['time'], df_new['outlet_T_{}'.format(i+1)],
                     label='block {}'.format(i+1))
        # plt.plot(df_new['time'], df_new['outlet_T_5'], '-', label='%s hours' % hour)
        plt.title('hour = %s h, mdot = %0.6f kg/s' % (hour, mdot))
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('outlet temperature (K)')
        plt.savefig('5block_plots/COMSOL_5block_%s_hours.png' % hour)
        df_new.to_csv('5block_data/COMSOL_5block_%s_hours.csv' % hour, index=False)
        # plt.show()
        plt.figure(num=2)
        if hour % 10 == 0:
            plt.plot(df_new['time']/max(df_new['time']), df_new['outlet_T_5'], label='%s hours' % hour)
    plt.legend()
    plt.xlabel('normalized time')
    plt.ylabel('outlet temperature (K)')
    plt.tight_layout()
    plt.show()
    # print(df.info())


def LC_model(hours):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    L = 2
    num_blocks = 5

    # heat transfer
    Nu = 3.66
    Tc = 300
    T1 = 1900 + 273
    Tb_i = 2400 + 273

    tend = 3600*hours*2
    mdot = flowrate_from_hours(hours)
    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C
    hA = Nu * k_Sn / L * np.pi * D_Sn * L * 1  # last 2 terms correction from comsol
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)

    # old LC model

    C1 = np.exp(hA / (mdot * cp_Sn))
    # C1 = hA / (mdot * cp_Sn)
    # C1 = np.exp(hA / (mdot * cp_Sn))
    A1 = mdot * cp_Sn / (m_C * cp_C) * (1/C1 - 1) * 1.3

    def F(t, T):
        dT = []
        for i in range(num_blocks):
            if i == 0:
                dT.append(A1 * (T[i * 2] - T1))
                dT.append(dT[i * 2] * (1 - 1 / C1))
            else:
                dT.append(A1 * (T[i * 2] - T[i * 2 - 1]))
                dT.append(dT[i * 2] * (1 - 1 / C1) + 1 / C1 * dT[i * 2 - 1])
        return dT

    # if os.path.isfile('data/outlet_T_LC_%0.2f_%0.0f.npy' % (mdot, k)):
    # outlet_Ts = np.load('data/outlet_T_LC_%0.2f_%0.0f.npy' % (mdot, k))
    # else:
    Tis = []
    T2 = T1
    for i in range(num_blocks):
        Tis.append(Tb_i)
        T2 = (Tb_i - (Tb_i - T2) / C1)
        Tis.append(T2)

    sol = solve_ivp(F, [0, tend], Tis, t_eval=t_eval)

    # if os.path.isfile('data/outlet_T_ser_COMSOL_%0.2f_%0.0f.csv' % (mdot, k)):
    try:
        df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double.csv',
                         names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
        T_COMSOL = df[df['hours'] == hours]
    except FileNotFoundError:
        T_COMSOL = None
    # T_COMSOL.rename({'% k_C (W/(m*K))': 'k_C (W/(m*K))'}, axis='columns', inplace=True)
    # print(T_COMSOL.info())
    cs = ['c', 'm', 'r', 'g', 'b']
    count = 0
    outlet_Ts = []
    plt.figure(num=1, clear=True)
    for i in range(num_blocks):
        outlet_Ts.append(sol.y[i * 2 + 1])
        # if i == 0 or i == num_blocks - 1 or i % 5 == 0:
        if T_COMSOL is not None:
            plt.plot(T_COMSOL['time'], T_COMSOL['block_' + str(i + 1)], '--',
                     color=cs[count], label='CS ' + str(i + 1))
        plt.plot(sol.t, sol.y[i * 2+1], color=cs[count], label='LC ' + str(i + 1))
        count += 1
    plt.plot(sol.t, np.ones(len(sol.t)) * T1, 'k--')
    plt.plot(sol.t, np.ones(len(sol.t)) * Tb_i, 'k--')
    plt.xlabel('t (s)')
    plt.ylabel('T (K)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.grid()
    plt.savefig('5block_plots/Tout_LC_%s_test.png' % hours)
    pass


def new_LC_model(hours):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    num_blocks = 27
    L = 2/(num_blocks/5)

    # heat transfer
    Nu = 3.66
    Tc = 1900+273
    T1 = 1905+273
    Tb_i = 2400+273

    tend = 3600*hours*2
    mdot = flowrate_from_hours(hours)
    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C
    hA = Nu * k_Sn / L * np.pi * D_Sn * L * 1.1  # last 2 terms correction from comsol
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    C1 = np.exp(-hA/(mdot*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours
    C2 = hA / (m_C * cp_C) * (0.9 + 10/(hours-4))

    # C2 = hA / (m_C * cp_C) * 1.5

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1*(Tb[i] - T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1 - T2) / np.log((T1-Tb[i])/(T2-Tb[i])) * C2
            else:
                new_T1 = Tb[i-1] - C1
                T2 = Tb[i] - C1*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2
            dTbs.append(dTb)
        return dTbs

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double_normalmesh_modtimestep.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    T_COMSOL = df[df['hours'] == hours]

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC 5')
    # plt.plot(sol.t, sol.y[-6], label='LC 4')
    # plt.plot(sol.t, sol.y[-11], label='LC 3')
    # plt.plot(sol.t, sol.y[-16], label='LC 2')
    # plt.plot(sol.t, sol.y[-21], label='LC 1')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s, k = %s' % (hours, 30))
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_27_blocks_hours_%s.png' % hours)
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC.csv', index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS.csv', index=False)
    # plt.figure(num=2, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, np.gradient(sol.y[i], sol.t), '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], np.gradient(T_COMSOL['outlet_T_'+str(i+1)],
    #                                            T_COMSOL['time']), color=cs[i], label='CS')
    # plt.title('hours = %s' % hours)
    # plt.xlabel('time')
    # plt.ylabel('dT/dt (K/s)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('5block_plots/new_LC_issues_dT_hours_%s.png' % hours)
    # plt.show()


def new_LC_model_Pe(hours, k):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    # num_blocks = int((-0.034*k**2 + 1.91*k + 0.3) * (-0.00185*hours**2 + 0.1037*hours + -0.444))
    Pe = flowrate_from_hours(hours)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
    num_blocks = max(int(962/(Pe**(2/3))), 1)
    print(Pe, num_blocks)
    # num_blocks = 1
    L = 2/(num_blocks/5)
    # L = 2

    # heat transfer
    Nu = 3.66
    Tc = 1900+273
    T1 = 1905+273
    Tb_i = 2400+273

    tend = 3600*hours*2
    mdot = flowrate_from_hours(hours)
    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)
    # correction_factor = (mdot / flowrate_from_hours(30))**(1/12) * 1.1
    correction_factor = 1.2

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C
    hA = Nu * k_Sn / L * np.pi * D_Sn * L * correction_factor  # last 2 terms correction from comsol
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    C1 = np.exp(-hA/(mdot*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours
    if hours > 3:
        C2 = hA / (m_C * cp_C) * (0.9 + 10/(hours-3))
    else:
        C2 = hA / (m_C * cp_C) * (1 + 40/hours)
    # C2 = hA / (m_C * cp_C) * (0.8 + abs(15/(hours-0.9)))
    # C2 = hA / (m_C * cp_C) * 4

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1*(Tb[i] - T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1 - T2) / np.log((T1-Tb[i])/(T2-Tb[i])) * C2
            else:
                new_T1 = Tb[i-1] - C1
                T2 = Tb[i] - C1*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2
            dTbs.append(dTb)
        return dTbs

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_'+str(k)+'_k.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    T_COMSOL = df[df['hours'] == hours]

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC 5')
    # plt.plot(sol.t, sol.y[-6], label='LC 4')
    # plt.plot(sol.t, sol.y[-11], label='LC 3')
    # plt.plot(sol.t, sol.y[-16], label='LC 2')
    # plt.plot(sol.t, sol.y[-21], label='LC 1')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s, k = %s' % (hours, k))
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_Pe_hours_%s_k_%s.png' % (hours, k))
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC_k_%s.csv' % k, index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS_k_%s.csv' % k, index=False)

    return sol.t, sol.y[-1]
    # plt.figure(num=2, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, np.gradient(sol.y[i], sol.t), '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], np.gradient(T_COMSOL['outlet_T_'+str(i+1)],
    #                                            T_COMSOL['time']), color=cs[i], label='CS')
    # plt.title('hours = %s' % hours)
    # plt.xlabel('time')
    # plt.ylabel('dT/dt (K/s)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('5block_plots/new_LC_issues_dT_hours_%s.png' % hours)
    # plt.show()


def new_LC_model_Pe_outlet(hours, k):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    # num_blocks = int((-0.034*k**2 + 1.91*k + 0.3) * (-0.00185*hours**2 + 0.1037*hours + -0.444))
    Pe = flowrate_from_hours(hours)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
    num_blocks = max(int(962/(Pe**(2/3))), 1)
    print(Pe, num_blocks)
    # num_blocks = 2
    L = 2/(num_blocks/5)
    # L = 2

    # heat transfer
    Nu = 3.66
    Tc = 1900+273
    T1 = 1905+273
    Tb_i = 2400+273

    tend = 3600*hours*2
    mdot = flowrate_from_hours(hours)
    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)
    # correction_factor = (mdot / flowrate_from_hours(30))**(1/12) * 1.1
    correction_factor = 1.1

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C
    hA = Nu * k_Sn / L * np.pi * D_Sn * L * correction_factor  # last 2 terms correction from comsol
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    C1 = np.exp(-hA/(mdot*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours
    if hours > 3:
        C2 = hA / (m_C * cp_C) * (0.9 + 10/(hours-3))
    else:
        C2 = hA / (m_C * cp_C) * (1 + 40/hours)
    # C2 = hA / (m_C * cp_C) * (0.8 + abs(15/(hours-0.9)))
    # C2 = hA / (m_C * cp_C) * 4

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1*(Tb[i] - T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1 - T2) / np.log((T1-Tb[i])/(T2-Tb[i])) * C2
            else:
                new_T1 = Tb[i-1] - C1
                T2 = Tb[i] - C1*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2
            dTbs.append(dTb)
        return dTbs

    def T2_calc(Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1*(Tb[i] - T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1 - T2) / np.log((T1-Tb[i])/(T2-Tb[i])) * C2
            else:
                new_T1 = Tb[i-1] - C1
                T2 = Tb[i] - C1*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2
            dTbs.append(dTb)
        return T2

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    Tout = []
    for t in range(len(sol.t)):
        Tbs = []
        for i in range(num_blocks):
            Tbs.append(sol.y[i][t])
        Tout.append(T2_calc(Tbs))
    df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_'+str(k)+'_k.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    T_COMSOL = df[df['hours'] == hours]

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC 5')
    plt.plot(sol.t, Tout, label='Tout LC 5')
    # plt.plot(sol.t, sol.y[-6], label='LC 4')
    # plt.plot(sol.t, sol.y[-11], label='LC 3')
    # plt.plot(sol.t, sol.y[-16], label='LC 2')
    # plt.plot(sol.t, sol.y[-21], label='LC 1')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s, k = %s' % (hours, k))
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_Pe_hours_%s_k_%s.png' % (hours, k))
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC_k_%s.csv' % k, index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS_k_%s.csv' % k, index=False)

    return sol.t, sol.y[-1]


def new_LC_model_step(hours):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    num_blocks = 25
    L = 2/(num_blocks/5)

    # heat transfer
    Nu = 3.66
    Tc = 300
    # T1 = 1905 + 273
    Tb_i = 2400 + 273

    tend = 3600*hours*2
    mdot = flowrate_from_hours(hours)
    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C
    hA = Nu * k_Sn / L * np.pi * D_Sn * L * 1.1  # last 2 terms correction from comsol
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    C1 = np.exp(-hA/(mdot*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours
    C2 = hA / (m_C * cp_C) * (0.9 + 10/(hours-4))

    # C2 = hA / (m_C * cp_C) * 1.5

    def T1(t):
        T1_c = 1900+273
        T1_h = 2400+273
        if t < 3600*hours/2:
            return T1_c
        elif 3600*hours/2 <= t < 3600*hours:
            return T1_h
        elif 3600*hours <= t < 3600*hours*1.5:
            return T1_c
        else:
            return T1_h

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1*(Tb[i] - T1(t))
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1(t)-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1(t) - T2) / np.log((T1(t)-Tb[i])/(T2-Tb[i])) * C2
            else:
                new_T1 = Tb[i-1] - C1
                T2 = Tb[i] - C1*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2
            dTbs.append(dTb)
        return dTbs

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    df = pd.read_csv('5block_data/COMSOL_5block_30_hours_4_step_transient.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    T_COMSOL = df[df['hours'] == hours]

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    plt.plot(sol.t, [T1(t) for t in sol.t], '--', color=(0.5, 0.5, 0.5), label='inlet T')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s' % hours)
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_20_blocks_hours_%s_4_step.png' % hours)
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC_4_step.csv', index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS_4_step.csv', index=False)
    # plt.figure(num=2, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, np.gradient(sol.y[i], sol.t), '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], np.gradient(T_COMSOL['outlet_T_'+str(i+1)],
    #                                            T_COMSOL['time']), color=cs[i], label='CS')
    # plt.title('hours = %s' % hours)
    # plt.xlabel('time')
    # plt.ylabel('dT/dt (K/s)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('5block_plots/new_LC_issues_dT_hours_%s.png' % hours)
    # plt.show()


def new_LC_model_opt_P(hours):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    num_blocks = 25
    L = 2/(num_blocks/5)

    # heat transfer
    Nu = 3.66
    Tc = 300
    # T1 = 1905 + 273
    Tb_i = 2400 + 273

    tend = 3600*hours*2
    max_flowrate_factor = 4
    mdot_df = pd.read_csv('5block_data/automation_%d/mdot_iteration_3.csv' % max_flowrate_factor)

    def mdot(t):
        t_index = np.argmin(np.abs(mdot_df['time'] - t))
        return mdot_df['new_mdot'][t_index]

    t_eval = np.arange(0, tend, 1)

    # conversions
    Q_Sn = mdot(0) / rho_Sn
    U_Sn = Q_Sn / (np.pi * D_Sn ** 2 / 4)
    Re = rho_Sn * U_Sn * D_Sn / mu_Sn
    # print(U_Sn)
    del_P = 64 / Re * 1 / 2 * rho_Sn * U_Sn ** 2 * L * num_blocks / D_Sn * 4.3  # 4.3 correction from comsol
    # print(del_P)
    P_pump = del_P * Q_Sn
    # print()
    # print(P_pump)

    m_C = (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L) * rho_C

    def correction_factor(t):
        return (mdot(t) / mdot(0))**3 * 1.1

    def hA(t):
        hA = Nu * k_Sn / L * np.pi * D_Sn * L * correction_factor(t)
        return hA
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    def C1(t):
        return np.exp(-hA(t)/(mdot(t)*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours

    def C2(t):
        return hA(t) / (m_C * cp_C) * (0.9 + 10/(hours-4))

    # C2 = hA / (m_C * cp_C) * 1.5

    def T1(t):
        T1_c = 1900+273
        # T1_h = 2400+273
        # if t < 3600*hours/2:
        #     return T1_c
        # elif 3600*hours/2 <= t < 3600*hours:
        #     return T1_h
        # elif 3600*hours <= t < 3600*hours*1.5:
        #     return T1_c
        # else:
        #     return T1_h
        return T1_c

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1(t)*(Tb[i] - T1(t))
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1(t)-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1(t) - T2) / np.log((T1(t)-Tb[i])/(T2-Tb[i])) * C2(t)
            else:
                new_T1 = Tb[i-1] - C1(t)
                T2 = Tb[i] - C1(t)*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2(t)
            dTbs.append(dTb)
        return dTbs

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    df = pd.read_csv('5block_data/automation_%d/outlet_T_iteration_2.csv' % max_flowrate_factor)
    T_COMSOL = df

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    plt.plot(sol.t, [T1(t) for t in sol.t], '--', color=(0.5, 0.5, 0.5), label='inlet T')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s' % hours)
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_20_blocks_hours_%s_opt_P.png' % hours)
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC_opt_P.csv', index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS_opt_P.csv', index=False)
    # plt.figure(num=2, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, np.gradient(sol.y[i], sol.t), '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], np.gradient(T_COMSOL['outlet_T_'+str(i+1)],
    #                                            T_COMSOL['time']), color=cs[i], label='CS')
    # plt.title('hours = %s' % hours)
    # plt.xlabel('time')
    # plt.ylabel('dT/dt (K/s)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('5block_plots/new_LC_issues_dT_hours_%s.png' % hours)
    # plt.show()


def new_LC_model_opt_P_Pe(hours, k):
    # tin material props
    cp_Sn = 240
    rho_Sn = 6200  # kg/m^3
    mu_Sn = 0.0012  # Pa s
    k_Sn = 62

    # graphite material props
    cp_C = 2000
    rho_C = 1700
    k_C = 30

    # geometry
    D_C = 0.2  # m
    D_Sn = 0.02  # Sn, m
    Pe = flowrate_from_hours(hours)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
    num_blocks = max(int(962/(Pe**(2/3))), 1)
    # num_blocks = 25
    L = 2/(num_blocks/5)

    # heat transfer
    Nu = 3.66
    Tc = 300
    # T1 = 1905 + 273
    Tb_i = 2400 + 273

    tend = 3600*hours*2
    max_flowrate_factor = 4
    # mdot_df = pd.read_csv('5block_data/automation_%d/mdot_iteration_3.csv' % max_flowrate_factor)
    mdot_df = pd.read_csv('5block_data/COMSOL_5block_5_k_fastcharge_sweep_maxflow.csv', names=['hours', 'max_flowrate_factor', 'time', 'inlet_T',
                                                                                               'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                                                                               'block_1', 'block_2', 'block_3', 'block_4', 'block_5',
                                                                                               'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5',
                                                                                               'new_mdot'], skiprows=5)
    mdot_df = mdot_df[mdot_df['hours'] == hours]
    mdot_df = mdot_df[mdot_df['max_flowrate_factor'] == max_flowrate_factor]

    def mdot(t):
        t_index = np.argmin(np.abs(mdot_df['time'] - t))
        return mdot_df['new_mdot'][t_index]

    t_eval = np.arange(0, tend, 1)

    def m_C(t):
        return (((D_C / 2) ** 2 - (D_Sn/2)**2) * np.pi * L(t)) * rho_C

    def correction_factor(t):
        # return (mdot(t) / mdot(0))**3 * 1.1
        return 1.2

    def L(t):
        Pe = mdot(t)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
        num_blocks = max(int(962/(Pe**(2/3))), 1)
        # num_blocks = 25
        L = 2/(num_blocks/5)
        return L

    def hA(t):
        hA = Nu * k_Sn / L(t) * np.pi * D_Sn * L(t) * correction_factor(t)
        return hA
    # print(hA)
    # hA = 3.66 * k * np.pi * 2 * 0.22
    # print(hA)
    # new LC model

    # T2(t) = Tb(t) - np.exp(-hA/(mdot*cp_Sn)*L) * (Tb - T1)
    # implies dT2/dt = dTb/dt
    def C1(t):
        return np.exp(-hA(t)/(mdot(t)*cp_Sn))
    # m_C * cp_C / hA * dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb))
    # implies
    # dTb/dt = (T1 - T2) / ln((T1-Tb)/(T2-Tb)) * hA / (m_C * cp_C)
    # where T2 is given above
    # C2 = hA / (m_C * cp_C) * 2.57  # for 10 hours
    # C2 = hA / (m_C * cp_C) * 1.3 # for 30 hours
    # C2 = hA / (m_C * cp_C) * 1.1  # for 50 hours

    def C2(t):
        # TODO: define hours from flowrate
        return hA(t) / (m_C(t) * cp_C) * (0.9 + 10/(hours-4))

    # C2 = hA / (m_C * cp_C) * 1.5

    def T1(t):
        T1_c = 1900+273
        # T1_h = 2400+273
        # if t < 3600*hours/2:
        #     return T1_c
        # elif 3600*hours/2 <= t < 3600*hours:
        #     return T1_h
        # elif 3600*hours <= t < 3600*hours*1.5:
        #     return T1_c
        # else:
        #     return T1_h
        return T1_c

    def F(t, Tb):
        dTbs = []
        for i in range(num_blocks):
            if i == 0:
                T2 = Tb[i] - C1(t)*(Tb[i] - T1(t))
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((T1(t)-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (T1(t) - T2) / np.log((T1(t)-Tb[i])/(T2-Tb[i])) * C2(t)
            else:
                new_T1 = Tb[i-1] - C1(t)
                T2 = Tb[i] - C1(t)*(Tb[i] - new_T1)
                if (T2-Tb[i]) == 0:
                    dTb = 0
                elif np.log((new_T1-Tb[i])/(T2-Tb[i])) == 0:
                    dTb = 0
                else:
                    dTb = (new_T1 - T2) / np.log((new_T1-Tb[i])/(T2-Tb[i])) * C2(t)
            dTbs.append(dTb)
        return dTbs

    Tb_is = np.ones(num_blocks)*Tb_i
    sol = solve_ivp(F, [0, tend], Tb_is, t_eval=t_eval, method='LSODA')
    df = pd.read_csv('5block_data/automation_%d/outlet_T_iteration_2.csv' % max_flowrate_factor)
    T_COMSOL = df

    cs = ['c', 'm', 'r', 'g', 'b']
    plt.figure(num=1, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, sol.y[i], '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_'+str(i+1)], color=cs[i], label='CS')

    plt.plot(sol.t, sol.y[-1], label='LC')
    plt.plot(T_COMSOL['time'], T_COMSOL['outlet_T_5'], '--', label='CS')
    plt.plot(sol.t, [T1(t) for t in sol.t], '--', color=(0.5, 0.5, 0.5), label='inlet T')
    # plt.plot(sol.t, sol.y[1], label='LC')
    # plt.plot(T_COMSOL['time'], T_COMSOL['Sn_block_2'], label='CS')
    plt.legend()
    plt.title('hours = %s' % hours)
    plt.xlabel('time')
    plt.ylabel('T (K)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/new_LC_20_blocks_hours_%s_opt_P.png' % hours)
    plt.show()
    data = {'LC_time': sol.t, 'LC_T': sol.y[-1]}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_LC_opt_P.csv', index=False)
    data = {'CS_time': T_COMSOL['time'], 'CS_T': T_COMSOL['outlet_T_5']}
    df = pd.DataFrame(data)
    df.to_csv('5block_data/outlet_T_time_CS_opt_P.csv', index=False)
    # plt.figure(num=2, clear=True)
    # for i in range(num_blocks):
    #     plt.plot(sol.t, np.gradient(sol.y[i], sol.t), '--', color=cs[i], label='LC ' + str(i + 1))
    #     plt.plot(T_COMSOL['time'], np.gradient(T_COMSOL['outlet_T_'+str(i+1)],
    #                                            T_COMSOL['time']), color=cs[i], label='CS')
    # plt.title('hours = %s' % hours)
    # plt.xlabel('time')
    # plt.ylabel('dT/dt (K/s)')
    # plt.legend()
    # plt.grid()
    # plt.savefig('5block_plots/new_LC_issues_dT_hours_%s.png' % hours)
    # plt.show()


def PDE_find():
    df_all = None
    for i in range(6):
        df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_outlet_T_{}.csv'.format(i),
                         names=['hours', 'time', 'outlet_T_{}'.format(i)],
                         skiprows=5)
        if df_all is None:
            df_all = df
        else:
            df_all['outlet_T_{}'.format(i)] = df['outlet_T_{}'.format(i)]
    for i in range(5):
        try:
            df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_block_{}.csv'.format(i+1),
                             names=['hours', 'time', 'block_{}'.format(i+1)],
                             skiprows=5)
        except FileNotFoundError:
            continue
        df_all['block_{}'.format(i+1)] = df['block_{}'.format(i+1)]
    # print(df_all.info())
    df_all.to_csv('5block_data/COMSOL_5block_sweep_hours_all.csv', index=False)
    df = df_all[df_all['hours'] == 10]
    columnName = 'outlet_T_3'
    df_PDE = df[[columnName]]
    df_PDE.index = df['time']
    # print(df_PDE)
    # fourier_library = ps.FourierLibrary()
    library_functions = [
        # lambda x: -np.exp(-x),
        lambda x: 1. / x,
        lambda x: x,
        lambda x: x**2,
        # lambda x: x**3,
        lambda x: 1
    ]
    library_function_names = [
        # lambda x: 'exp(' + x + ')',
        lambda x: '1/' + x,
        lambda x: x,
        lambda x: x + '^2',
        # lambda x: x + '^3',
        lambda x: '1'
    ]
    custom_library = ps.CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )

    stlsq_optimizer = ps.STLSQ(threshold=1e-20, alpha=.5)
    model = ps.SINDy(feature_names=df_PDE.columns, feature_library=custom_library, optimizer=stlsq_optimizer)
    model.fit(df_PDE.values, t=df_PDE.index.values)
    model.print()
    plt.plot(df_PDE.index, df_PDE[columnName])
    plt.show()
    diff_predict = model.predict(df_PDE[columnName].values)
    diff_calc = model.differentiate(df_PDE[columnName].values, t=df_PDE.index.values)
    plt.plot(df_PDE.index.values, diff_predict, 'r--')
    plt.plot(df_PDE.index.values, diff_calc, 'k')
    plt.show()
    sim_predict = model.simulate([2400+273], df_PDE.index.values)
    plt.plot(df_PDE.index.values, sim_predict, 'r--')
    plt.plot(df_PDE.index.values, df_PDE[columnName].values, 'k')
    plt.show()
    # df_temp = None
    # for i in range(6):
    #     df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_outlet_T_{}.csv'.format(i),
    #                      names=['hours', 'time', 'outlet_T_{}'.format(i)],
    #                      skiprows=5)
    #     if df_all is None:
    #         df_all = df
    #     else:
    #         df_all['outlet_T_{}'.format(i)] = df['outlet_T_{}'.format(i)]
    pass


def logistic_fun(t, a, b):
    return 500/(1+(1-1/500)*np.exp(a*t-b))+1900+273


def curve_fit_temps():
    df_All = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_all.csv')
    df_all = df_All[df_All['hours'] == 20]
    x = df_all['time']/max(df_all['time'])
    columnName = 'outlet_T_5'
    y = df_all[columnName]

    curve_fit_params, curve_fit_cov = curve_fit(
        logistic_fun, x, y, p0=[7, 7])
    print(curve_fit_params)
    # plt.plot(x, y, 'k')
    hours = np.arange(10, 100, 10)
    plt.figure(num=1, clear=True)
    for hour in hours:
        df = df_All[df_All['hours'] == hour]
        plt.plot(df['time']/max(df['time']), df[columnName], label='{} hours'.format(hour))
    plt.plot(x, logistic_fun(x, *curve_fit_params), 'r--')
    plt.savefig('5block_plots/outlet_T_5_curve_fit.png')
    plt.show()
    return curve_fit_params
    # print(df_all.info())


def curve_fit_temps_2():
    df_All = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double_normalmesh_modtimestep.csv',
                         names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    hour = 20
    df_all = df_All[df_All['hours'] == hour]
    x = df_all['time']/(hour*3600)
    columnName = 'outlet_T_5'
    y = df_all[columnName]

    curve_fit_params, curve_fit_cov = curve_fit(
        logistic_fun, x, y, p0=[7, 7])
    print(curve_fit_params)
    # plt.plot(x, y, 'k')
    hours = np.arange(10, 60, 10)
    plt.figure(num=1, clear=True)
    for hour in hours:
        df = df_All[df_All['hours'] == hour]
        plt.plot(df['time']/(hour*3600), df[columnName], label='{} hours'.format(hour))
    plt.plot(x, logistic_fun(x, *curve_fit_params), 'r--', label='fit')
    plt.plot(x, np.ones(len(x))*(1900+273), 'k--')
    plt.plot(x, np.ones(len(x))*(2400+273), 'k--')
    plt.grid()
    plt.xlabel('normalized time')
    plt.ylabel('outlet temperature (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('5block_plots/outlet_T_5_curve_fit.png')
    plt.show()

    plt.figure(num=1, clear=True)
    for hour in hours:
        df = df_All[df_All['hours'] == hour]
        plt.plot(df['time'], df[columnName], label='{} hours'.format(hour))
    plt.xlabel('time')
    plt.ylabel('outlet temperature (K)')
    plt.legend()
    plt.grid()
    plt.plot(df['time'], np.ones(len(df['time']))*(1900+273), 'k--')
    plt.plot(df['time'], np.ones(len(df['time']))*(2400+273), 'k--')
    plt.tight_layout()
    plt.savefig('5block_plots/outlet_T_5_non_norm.png')
    plt.show()

    return curve_fit_params
    # print(df_all.info())


def ODE_find_fit(curve_fit_params):
    times = np.linspace(0, 1, 100)
    temps = logistic_fun(times, *curve_fit_params)
    # plt.plot(times, temps)
    # plt.show()

    library_functions = [
        lambda x: x,
        lambda x: x**2,
    ]
    library_function_names = [
        lambda x: x,
        lambda x: x + '^2',
    ]
    custom_library = ps.CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )

    optimizer = ps.ConstrainedSR3(normalize_columns=True)
    model = ps.SINDy(optimizer=optimizer)
    model.fit(temps, t=times)
    model.print()
    diff_predict = model.predict(temps)
    diff_calc = model.differentiate(temps, t=times)
    plt.figure(num=2, clear=True)
    plt.plot(times, diff_calc, 'k')
    plt.plot(times, diff_predict, 'r--')
    plt.figure(num=3, clear=True)
    sim_predict = model.simulate([2650], times)
    plt.plot(times, sim_predict, 'r--')
    plt.plot(times, temps, 'k')
    plt.show()


def sweep_initial_T():
    df_all = pd.read_csv('5block_data/COMSOL_5block_sweep_initialT_all.csv',
                         names=['initialT', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    plt.figure(num=1, clear=True)
    for initial_T in np.arange(2000, 2400, 100):
        df = df_all[df_all['initialT'] == initial_T]
        shift_outlet = df['outlet_T_5'] - (1900+273)
        norm_outlet = shift_outlet/max(shift_outlet)
        plt.plot(df['time']/max(df['time']), norm_outlet, label='{} K'.format(initial_T))
    plt.legend()
    plt.show()


def sweep_initial_T_2():
    df_all = pd.read_csv('5block_data/COMSOL_5block_sweep_initialT_double.csv',
                         names=['initialT', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    df_2 = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double.csv',
                       names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                              'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                       skiprows=5)
    df_2400 = df_2[df_2['hours'] == 20]
    plt.figure(num=1, clear=True)
    for initial_T in np.arange(2000, 2400, 100):
        df = df_all[df_all['initialT'] == initial_T]
        print(df.info())
        shift_outlet = df['outlet_T_5'] - (1900+273)
        norm_outlet = shift_outlet/max(shift_outlet)
        plt.plot(df['time']/(20*3600), norm_outlet, label='{} K'.format(initial_T))
    plt.plot(df_2400['time']/(20*3600), (df_2400['outlet_T_5']-(1900+273)) /
             max(df_2400['outlet_T_5']-(1900+273)), label='2400 K')
    plt.grid()
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('5block_plots/sweep_initialT_2.png')
    plt.show()

    plt.figure(num=1, clear=True)
    for initial_T in np.arange(2000, 2400, 100):
        df = df_all[df_all['initialT'] == initial_T]
        print(df.info())
        # shift_outlet = df['outlet_T_5'] - (1900+273)
        # norm_outlet = shift_outlet/max(shift_outlet)
        plt.plot(df['time']/(20*3600), df['outlet_T_5'], label='{} K'.format(initial_T))
    plt.plot(df_2400['time']/(20*3600), df_2400['outlet_T_5'], label='2400 K')
    plt.plot(df['time']/(20*3600), np.ones(len(df['time']))*(1900+273), 'k--')
    plt.plot(df['time']/(20*3600), np.ones(len(df['time']))*(2400+273), 'k--')
    plt.plot()
    plt.grid()
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('5block_plots/sweep_initialT_2_non_norm.png')
    plt.show()


def logistic_fun_2(t, a, b, c):
    # return np.exp(-a*t)
    return 1/(1+(a)*np.exp(b*t-c))


def sweep_both_norm():
    df_All = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_all.csv')
    df_all = df_All[df_All['hours'] == 20]
    hours = np.arange(10, 50, 10)
    plt.figure(num=1, clear=True)
    for hour in hours:
        df = df_All[df_All['hours'] == hour]
        shift_temp = df['outlet_T_5'] - (1900+273)
        norm_temp = shift_temp/max(shift_temp)
        plt.plot(df['time']/max(df['time']), norm_temp, label='{} hours'.format(hour))

    df_all = pd.read_csv('5block_data/COMSOL_5block_sweep_initialT_all.csv',
                         names=['initialT', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    for initial_T in np.arange(2000, 2400, 100):
        df = df_all[df_all['initialT'] == initial_T]
        shift_outlet = df['outlet_T_5'] - (1900+273)
        norm_outlet = shift_outlet/max(shift_outlet)
        plt.plot(df['time']/max(df['time']), norm_outlet, label='{} K'.format(initial_T))

    df = df_all[df_all['initialT'] == 2300]
    # df = df_All[df_All['hours'] == 20]
    x = df['time']/max(df['time'])
    columnName = 'outlet_T_5'
    shift_temp = df[columnName] - (1900+273)
    y = shift_temp/max(shift_temp)

    curve_fit_params, curve_fit_cov = curve_fit(
        logistic_fun_2, x, y, p0=[1e-4, 7, 1e-1])
    print(curve_fit_params)
    plt.plot(x, logistic_fun_2(x, *curve_fit_params), 'r--', label='fit')
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/outlet_T_5_curve_fit_2.png')
    plt.show()


def sweep_both_norm_2():
    df_All = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double.csv',
                         names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    hour = 20
    df_all = df_All[df_All['hours'] == hour]
    hours = np.arange(10, 60, 10)
    plt.figure(num=1, clear=True)
    for hour in hours:
        df = df_All[df_All['hours'] == hour]
        shift_temp = df['outlet_T_5'] - (1900+273)
        norm_temp = shift_temp/max(shift_temp)
        plt.plot(df['time']/(hour*3600), norm_temp, label='{} hours'.format(hour))
    plt.xlabel('normalized time')
    plt.ylabel('temperature (K)')
    # plt.savefig('5block_plots/outlet_T_5_curve_fit.png')
    # plt.show()

    df_all = pd.read_csv('5block_data/COMSOL_5block_sweep_initialT_double.csv',
                         names=['initialT', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'],
                         skiprows=5)
    for initial_T in np.arange(2000, 2400, 100):
        df = df_all[df_all['initialT'] == initial_T]
        shift_outlet = df['outlet_T_5'] - (1900+273)
        norm_outlet = shift_outlet/max(shift_outlet)
        plt.plot(df['time']/(20*3600), norm_outlet, label='{} K'.format(initial_T))

    df = df_all[df_all['initialT'] == 2300]
    # df = df_All[df_All['hours'] == 20]
    x = df['time']/(20*3600)
    columnName = 'outlet_T_5'
    shift_temp = df[columnName] - (1900+273)
    y = shift_temp/max(shift_temp)

    curve_fit_params, curve_fit_cov = curve_fit(
        logistic_fun_2, x, y, p0=[1e-4, 7, 1e-4])
    print(curve_fit_params)
    plt.plot(x, logistic_fun_2(x, *curve_fit_params), 'r--', label='fit')
    df = df_all[df_all['initialT'] == 2100]
    # df = df_All[df_All['hours'] == 20]
    x = df['time']/(20*3600)
    columnName = 'outlet_T_5'
    shift_temp = df[columnName] - (1900+273)
    y = shift_temp/max(shift_temp)

    curve_fit_params, curve_fit_cov = curve_fit(
        logistic_fun_2, x, y, p0=[1e-4, 7, 1e-4])
    print(curve_fit_params)
    plt.plot(x, logistic_fun_2(x, *curve_fit_params), 'r--', label='fit')
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('5block_plots/outlet_T_5_curve_fit_2.png')
    plt.show()
    pass


def compare_longblock():
    df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_double_normalmesh_modtimestep.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    block5_data = df[df['hours'] == 30]
    df = pd.read_csv('5block_data/COMSOL_longblock_sweep_hours_double_normalmesh_modtimestep.csv',
                     names=['hours', 'time', 'inlet_T', 'outlet_T_5',
                            'block', 'Sn_block'], skiprows=5)
    longblock_data = df[df['hours'] == 30]
    plt.figure(num=1, clear=True)
    plt.plot(longblock_data['time']/(3600), longblock_data['outlet_T_5'], label='longblock')
    plt.plot(block5_data['time']/(3600), block5_data['outlet_T_5'], '--', label='5block')
    plt.xlabel('time (hours)')
    plt.ylabel('temperature (K)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    pass


def plot_normalized_time():
    # hours = np.append([1, 5], np.arange(10, 60, 10))
    hours = [10]
    # hours = np.arange(10,60,10)
    # for k in [5, 10, 30]:
    #     df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_%s_k.csv' % k,
    #                      names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
    #                             'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
    #     plt.figure(num=2, clear=True)
    #     for hour in hours:
    #         if hour not in df['hours'].unique():
    #             continue
    #         df_new = df[df['hours'] == hour]
    #         print(k, hour)
    #         # if hour % 10 == 0:
    #         plt.plot(df_new['time']/max(df_new['time'])*2, df_new['outlet_T_5'], label='%s hours' % hour)
    #     plt.legend()
    #     plt.title('k = %s' % k)
    #     plt.xlabel('normalized time')
    #     plt.ylabel('outlet temperature (K)')
    #     plt.tight_layout()
    #     plt.grid()
    #     plt.savefig('5block_plots/outlet_T_5_normalized_time_%s_old.png' % k)

    Pes = []
    times = []
    outlets = []
    for k in [5, 10, 30]:
        df = pd.read_csv('../data/COMSOL_5block_sweep_hours_%s_k.csv' % k,
                         names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
        for hour in hours:
            if hour not in df['hours'].unique():
                continue
            df_new = df[df['hours'] == hour]
            # Pe = flowrate_from_hours(hour)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
            Pe = k
            Pes.append(Pe)
            # print(k, hour, Pe)
            times.append(df_new['time']/max(df_new['time'])*2)
            outlets.append(df_new['outlet_T_5'])
            # if hour % 10 == 0:
            # plt.plot(df_new['time']/max(df_new['time'])*2, df_new['outlet_T_5'], label='%s ' % Pe)
    plt.figure(num=3, dpi=300, clear=True)
    df_all = pd.DataFrame({'Pe': Pes, 'time': times, 'outlet_T_5': outlets})
    df_all.sort_values(by='Pe', inplace=True)
    cmap = mpl.cm.get_cmap('nipy_spectral')
    plt.plot(np.linspace(0, 2, 1000), (np.linspace(0, 2, 1000) < 1), 'k--', label='ideal')
    for row in df_all.iterrows():
        # plt.plot(row[1]['time'], (row[1]['outlet_T_5']-2173)/500, label='%0.0f' % row[1]['Pe'],
        #         color=cmap((np.log(row[1]['Pe']) - np.log(min(Pes)))/(np.log(max(Pes))-np.log(min(Pes)))*0.9))
        plt.plot(row[1]['time']*1.05, ((row[1]['outlet_T_5']-273)-1900)/500, label='k = %0.0f' % row[1]['Pe'],
                 color=cmap(np.sqrt(row[1]['Pe'])/np.sqrt(max(Pes))*0.9))

    plt.legend()
    plt.title('tin outlet temperature')
    plt.xlabel('$t^* = \\frac{t}{\\tau}$')
    plt.ylabel('$\\Theta = \\frac{T-T_{min}}{T_{max}-T_{min}}$')
    plt.tight_layout()
    # plt.grid()
    plt.savefig('../plots/ideal_vs_realistic_tin_outlet.pdf')

    # plt.show()


def plot_nondim_time_CS():
    plt.figure(num=1, clear=True)
    df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_Pe_k30_2.csv',
                     names=['index', 'hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                            'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5', 'bottom_right_corner'], skiprows=5)
    cmap = mpl.cm.get_cmap('nipy_spectral')
    for hour in df['hours'].unique():
        df_new = df[df['hours'] == hour]
        new_row = df_new.iloc[0]
        new_row['time'] = -1
        new_row['outlet_T_5'] = 2673
        df_new.loc[-1] = new_row
        df_new.sort_index(inplace=True)
        plt.figure(num=1)
        plt.plot(df_new['time']/max(df_new['time'])*2, (df_new['outlet_T_5']-2173)/500,
                 label='%0.1f' % (hour*30), color=cmap(np.sqrt(hour*30)/np.sqrt(1500)*0.9))
        plt.figure(num=2)
        plt.plot(df_new['time']/max(df_new['time'])*2, (df_new['bottom_right_corner']-2173) /
                 500, label='%0.1f' % (hour*30), color=cmap(np.sqrt(hour*30)/np.sqrt(1500)*0.9))
        plt.figure(num=3)
        plt.plot(df_new['time']/max(df_new['time'])*2, (df_new['block_5']-2173)/500, label='%0.1f' %
                 (hour*30), color=cmap(np.sqrt(hour*30)/np.sqrt(1500)*0.9))
    plt.figure(num=1)
    plt.title('tin outlet T')
    plt.legend(title='hours*k')
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')
    plt.tight_layout()
    plt.grid()
    plt.savefig('5block_plots/outlet_T_5_nondim_time.png')
    plt.figure(num=2)
    plt.title('last block corner T')
    plt.legend(title='hours*k')
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')
    plt.tight_layout()
    plt.grid()
    plt.savefig('5block_plots/bottom_right_corner_nondim_time.png')
    plt.figure(num=3)
    plt.title('last block T')
    plt.legend(title='hours*k')
    plt.xlabel('normalized time')
    plt.ylabel('normalized temperature')
    plt.tight_layout()
    plt.grid()
    plt.savefig('5block_plots/block_5_nondim_time.png')
    plt.show()
    pass


def plot_normalized_time_CS_LC():
    Pes = []
    cmap = mpl.cm.get_cmap('nipy_spectral')
    plt.figure(num=3, clear=True)
    for k in [5, 10, 30]:
        for hour in [1, 5, 10, 20, 30, 40, 50]:
            df = pd.read_csv('5block_data/COMSOL_5block_sweep_hours_%s_k.csv' % k,
                             names=['hours', 'time', 'inlet_T', 'outlet_T_1', 'outlet_T_2', 'outlet_T_3', 'outlet_T_4', 'outlet_T_5',
                                    'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'Sn_block_1', 'Sn_block_2', 'Sn_block_3', 'Sn_block_4', 'Sn_block_5'], skiprows=5)
            if hour not in df['hours'].unique():
                continue
            df_new = df[df['hours'] == hour]
            Pe = int(flowrate_from_hours(hour)/(6200*.01**2*np.pi)*.09/(k/1700/2000))
            if Pe in Pes:
                continue
            Pes.append(Pe)
            LC_t, LC_y = new_LC_model_Pe(hour, k)
            plt.figure(num=3)
            plt.plot(df_new['time']/max(df_new['time'])*2, df_new['outlet_T_5'],
                     '-', label='%s ' % Pe, color=cmap(Pe/4000))
            plt.plot(LC_t/max(LC_t)*2, LC_y, '--', color=cmap(Pe/4000))
    plt.legend()
    plt.xlabel('normalized time')
    plt.ylabel('outlet temperature (K)')
    plt.tight_layout()
    plt.grid()
    plt.savefig('5block_plots/outlet_T_5_normalized_time_CS_LC.png')
    plt.show()
    pass


def fastcharge_CS_mdot():
    cp_Sn = 244
    df = pd.read_csv('../data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow.csv', names=[
                     'hours', 'base_ff', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
    df['time'] = df['time']/3600
    base_ffs = np.array(sorted(df['base_ff'].unique()))
    base_ffs = base_ffs[:-3]
    max_ffs = np.array(sorted(df['max_ff'].unique()))
    max_ffs = max_ffs[:-3]
    all_hours = np.array(sorted(df['hours'].unique()))
    for hours in all_hours:
        P_FOMS = []
        for base_ff in base_ffs:
            for max_ff in max_ffs:
                if base_ff != max_ff:
                    # P_FOMS.append(np.nan)
                    continue
                print(base_ff, max_ff)
                # df_data = df[df['base_ff'] == base_ff and df['max_ff'] == max_ff]
                df_data = df[(df['base_ff'] == base_ff) & (df['max_ff'] == max_ff) & (df['hours'] == hours)]
                print(df_data['mass_flow'].min(), df_data['mass_flow'].max())
                delT = np.abs(df_data['outlet_T'] - df_data['inlet_T'])
                Pout = df_data['mass_flow']*delT*cp_Sn
                Poutnp = np.array(df_data['mass_flow']*delT*cp_Sn)
                nom_flow = flowrate_from_hours(hours)
                P_FOM = np.trapz(Poutnp[df_data['time'] <= hours], df_data['time']
                                 [df_data['time'] <= hours]) / (nom_flow*500*cp_Sn*hours)
                print(nom_flow)
                print(P_FOM)
                P_FOMS.append(P_FOM)
                fig = plt.figure(num=1, figsize=(5, 4), clear=True)
                ax = fig.add_subplot(221)
                plt.plot(df_data['time'], df_data['bottom_right_T'])
                plt.xlabel('time (hr)')
                plt.ylabel('T (K)')
                # plt.grid()
                plt.title('k = 10, hours = %0.1f, max$_{ff}$ = %0.1f' % (hours, max_ff))
                plt.tight_layout()
                ax = fig.add_subplot(222)
                plt.plot(df_data['time'], df_data['mass_flow'])
                plt.ylim(0, 20)
                # plt.grid()
                plt.xlabel('time (hr)')
                plt.ylabel('mass flow (kg/s)')
                plt.tight_layout()
                ax = fig.add_subplot(223)
                plt.plot(df_data['time'], Poutnp)
                plt.plot(df_data['time'], np.ones(len(df_data['time']))*nom_flow*500*cp_Sn)
                plt.annotate('FOM$_P$ = %0.2f' % P_FOM, (0.5, 0.7), xycoords='axes fraction')
                plt.xlabel('time (hr)')
                plt.ylabel('Power (W)')
                # plt.grid()
                plt.tight_layout()
                plt.savefig('../plots/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_%0.1f_%0.1f_%0.1f.pdf' %
                            (hours, base_ff, max_ff))

                # plt.show()
        plt.figure(num=2, clear=True)
        # plt.contourf(max_ffs, base_ffs, np.array(P_FOMS).reshape(len(base_ffs),len(max_ffs)), 100)
        plt.plot(base_ffs, P_FOMS)
        plt.plot(base_ffs, np.ones_like(base_ffs), 'k--')
        plt.plot(base_ffs, np.ones_like(base_ffs)*0, 'k--')
        plt.title('hours = %0.1f' % hours)
        plt.xlabel('flow multiplier')
        plt.ylabel('FOM$_{P,charge}$')
        # plt.grid()
        plt.tight_layout()
        # plt.colorbar(label='P FOM')
        plt.savefig('../plots/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_%0.1f.pdf' % hours)
        # plt.show()


def fastcharge_CS_mdot_SOC():
    cp_Sn = 240
    df_data = pd.read_csv('data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow.csv', names=[
        'hours', 'base_ff', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
    _, _, energy_max = energy_from_flowrate_hours(df_data['mass_flow'][0], df_data['hours'][0])
    SOCs = []
    powers = []
    for hours in tqdm(df_data['hours'].unique()):
        for max_ff in df_data['max_ff'].unique():
            for base_ff in df_data['base_ff'].unique():
                df_temp = df_data[df_data['hours'] == hours][df_data['max_ff'] == max_ff][df_data['base_ff'] == base_ff]
                power = df_temp['mass_flow']*cp_Sn*(df_temp['inlet_T']-df_temp['outlet_T'])
                try:
                    energy = cumtrapz(power, df_temp['time'], initial=0)
                except:
                    continue
                SOC = energy/energy_max
                print(SOC[0])
                # SOC = SOC - SOC[0]
                df_temp['SOC'] = SOC
                SOCs = SOCs + list(df_temp['SOC'])
                powers = powers + list(power)
                # fig = plt.figure(num=1, clear=True)
                # plt.plot(SOC, power)
                # plt.xlabel('SOC')
                # plt.ylabel('Power (W)')
                # plt.tight_layout()
                # plt.show()
                pass
    df_data['SOC'] = SOCs
    df_data['power'] = powers
    df_data.to_csv('data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_SOC.csv', index=False)


def test_SOC_P_plots_charge():
    df_data = pd.read_csv('data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow_SOC.csv')
    cp_Sn = 240
    df_data = df_data[df_data['hours'] == 5]
    fig = plt.figure(num=1, clear=True)
    fig = plt.figure(num=2, clear=True)
    fig = plt.figure(num=3, clear=True)
    df_maxff_1 = df_data[df_data['max_ff'] == 1.0][df_data['base_ff'] == 1.0]
    df_maxff_1 = df_maxff_1.reset_index(drop=True)
    df_maxff_1['delT'] = df_maxff_1['outlet_T'] - df_maxff_1['inlet_T']
    colors_len = len(df_data['max_ff'].unique())
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for max_ff in df_data['max_ff'].unique():
        df_temp = df_data[df_data['max_ff'] == max_ff][df_data['base_ff'] == max_ff]
        df_temp['delT'] = df_temp['inlet_T'] - df_temp['outlet_T']
        df_temp = df_temp.reset_index(drop=True)
        # df_constP_basecase = df_maxff_1[df_maxff_1['delT'] >= df_maxff_1['delT'][0]/max_ff]
        # df_not_const_SOC = df_temp[df_temp['delT'] < df_temp['delT'][0]/max_ff]
        # df_not_constP_basecase = df_maxff_1[df_maxff_1['delT'] <= df_maxff_1['delT'][0]/max_ff]
        # df_not_constP = df_not_constP[df_not_constP['SOC'] < df_constP['SOC'].values[-1]]
        # basecase_SOC = df_constP_basecase['SOC'].append(df_not_constP_basecase['SOC'])
        color = colors.pop(0)
        plt.figure(1)
        # plt.plot(df_constP_basecase['SOC'], df_constP_basecase['power'], '--', color=color)
        # plt.plot(df_not_constP_basecase['SOC'], df_not_constP_basecase['power']*max_ff, '--', color=color)
        plt.plot(df_temp['SOC'], df_temp['power'], label='max ff = %0.1f' % max_ff, color=color)
        plt.figure(3)
        plt.plot(df_temp['SOC'], df_temp['power'], label='max ff = %0.1f' % max_ff, color=color)
        plt.figure(2)
        plt.plot(df_temp['SOC'], df_temp['delT'], label='max ff = %0.1f' % max_ff, color=color)
    plt.figure(1)
    plt.xlabel('SOC')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_fastcharge_sweep_log_maxflow_SOC_P_31hrs_comp.pdf')
    plt.figure(2)
    plt.xlabel('SOC')
    plt.ylabel('Delta T (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_fastcharge_sweep_log_maxflow_SOC_T_31hrs.pdf')
    plt.figure(3)
    plt.xlabel('SOC')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_fastcharge_sweep_log_maxflow_SOC_P_31hrs.pdf')
    plt.show()
    pass
    pass


def fastcharge_CS_Tin():
    cp_Sn = 244
    # df_mdot = pd.read_csv('5block_data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow.csv', names=['hours','base_ff', 'max_ff','time','inlet_T','outlet_T','block_T','Sn_T','bottom_right_T','mass_flow'], skiprows=5)
    df = pd.read_csv('../data/COMSOL_5block_10_k_fastcharge_sweep_basemaxTin_cap_P.csv', names=[
                     'hours', 'base_Tadd', 'max_Tadd', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'E_Sn', 'E_C'], skiprows=5)
    df['time'] = df['time']/3600
    # df_mdot['time'] = df_mdot['time']/3600
    base_Tadds = np.array(sorted(df['base_Tadd'].unique()))
    print(base_Tadds)
    max_Tadds = np.array(sorted(df['max_Tadd'].unique()))
    all_hours = np.array(sorted(df['hours'].unique()))
    for hours in all_hours:
        P_FOMS = []
        # df_data = df_mdot[(df_mdot['base_ff'] == 1) & (df_mdot['max_ff'] == 1) & (df_mdot['hours'] == hours) & (df_mdot['time'] <= hours)]
        # delT = np.abs(df_data['outlet_T'] - df_data['inlet_T'])
        # Pout = df_data['mass_flow']*delT*cp_Sn
        # Poutnp = np.array(df_data['mass_flow']*delT*cp_Sn)
        # nom_flow = flowrate_from_hours(hours)
        # P_FOM = np.trapz(Poutnp, df_data['time']) / (nom_flow*500*cp_Sn*hours)
        # T_FOM = df_data['bottom_right_T'].values[-1] / (2400+273)
        # print(P_FOM, T_FOM)
        # P_FOMS.append(P_FOM)
        index = 1
        for base_Tadd in base_Tadds:
            for max_Tadd in max_Tadds:
                if base_Tadd != max_Tadd:
                    # P_FOMS.append(np.nan)
                    continue
                print(base_Tadd, max_Tadd)
                # df_data = df[df['base_Tadd'] == base_Tadd and df['max_Tadd'] == max_Tadd]
                df_data = df[(df['base_Tadd'] == base_Tadd) & (df['max_Tadd'] == max_Tadd) & (df['hours'] == hours)]
                print(df_data['mass_flow'].min(), df_data['mass_flow'].max())
                delT = np.abs(df_data['outlet_T'] - df_data['inlet_T'])
                Pout = df_data['mass_flow']*delT*cp_Sn
                Poutnp = np.array(df_data['mass_flow']*delT*cp_Sn)
                nom_flow = flowrate_from_hours(hours)
                P_FOM = np.trapz(Poutnp[df_data['time'] <= hours], df_data['time']
                                 [df_data['time'] <= hours]) / (nom_flow*500*cp_Sn*hours)
                print(nom_flow)

                fig = plt.figure(num=index, figsize=(7.5, 2), clear=True)
                index += 1
                ax = fig.add_subplot(131)
                plt.plot(df_data['time'], df_data['outlet_T'], label='outlet')
                plt.plot(df_data['time'], df_data['inlet_T'], label='inlet')
                # plt.plot(df_data['time'], df_data['bottom_right_T'])
                plt.legend(loc='center left')
                T_FOM = df_data['bottom_right_T'].values[-1] / (2400+273)
                print(P_FOM, T_FOM)
                P_FOMS.append(P_FOM)
                # plt.annotate('T FOM = %0.2f' % T_FOM, (0.1, 0.8), xycoords='axes fraction')
                plt.xlabel('time (hr)')
                plt.ylabel('T (K)')
                # plt.grid()
                # plt.title('k = 10, hours = %0.1f, base$_{Tadd}$ = %0.1f, max$_{Tadd}$ = %0.1f' % (hours, base_Tadd, max_Tadd))
                plt.tight_layout()
                ax = fig.add_subplot(132)
                plt.plot(df_data['time'], df_data['bottom_right_T'])
                # plt.ylim(0, 20)
                # plt.grid()
                plt.xlabel('time (hr)')
                plt.ylabel('bottom graphite T (K)')
                plt.tight_layout()
                ax = fig.add_subplot(133)
                plt.plot(df_data['time'], Poutnp)
                plt.plot(df_data['time'], np.ones(len(df_data['time']))*nom_flow*500*cp_Sn)
                plt.annotate('P FOM = %0.2f' % P_FOM, (0.1, 0.1), xycoords='axes fraction')
                plt.xlabel('time (hr)')
                plt.ylabel('Power (W)')
                # plt.grid()
                plt.tight_layout()
                plt.savefig('../plots/COMSOL_5block_10_k_fastcharge_sweep_basemaxTin_%0.1f_%0.1f_%0.1f.png' %
                            (hours, base_Tadd, max_Tadd), dpi=300)

                # plt.show()
        plt.figure(num=index, clear=True)
        # plt.contourf(max_Tadds, base_Tadds, np.array(P_FOMS).reshape(len(base_Tadds),len(max_Tadds)), 100)
        plt.plot(base_Tadds, P_FOMS)
        plt.plot(base_Tadds, np.ones_like(base_Tadds), 'k--')
        plt.plot(base_Tadds, np.ones_like(base_Tadds)*0, 'k--')
        plt.title('hours = %0.1f' % hours)
        plt.xlabel('added T (K)')
        plt.ylabel('P FOM')
        # plt.grid()
        plt.tight_layout()
        # plt.colorbar(label='P FOM')
        plt.savefig('../plots/COMSOL_5block_10_k_fastcharge_sweep_basemaxTin_%0.1f.png' % hours, dpi=300)
        # plt.show()
        plt.close()


def fastcharge_CS_rad():
    cp_Sn = 244
    hours = 5
    df_orig = pd.read_csv('../data/COMSOL_5block_10_k_fastcharge_sweep_basemaxflow.csv', names=[
                          'hours', 'base_ff', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
    df_orig['time'] = df_orig['time'] / 3600
    df_orig = df_orig[(df_orig['base_ff'] == 1) & (df_orig['max_ff'] == 1)
                      & (df_orig['hours'] == 5) & (df_orig['time'] <= 5)]
    df_rad2 = pd.read_csv('../data/COMSOL_5block_10_k_fastcharge_sweep_rad_gap_cap_thin.csv', names=[
                          'hours', 'spac', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_block_T', 'bottom_right_T', 'mass_flow', 'C_energy', 'G_block_T'], skiprows=5)
    df_rad2['time'] = df_rad2['time'] / 3600
    df_rad3 = pd.read_csv('../data/COMSOL_5block_10_k_fastcharge_sweep_rad_gap_cap_thin_2.csv', names=[
                          'hours', 'spac', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_block_T', 'bottom_right_T', 'mass_flow', 'C_energy', 'G_block_T'], skiprows=5)
    df_rad3['time'] = df_rad3['time'] / 3600
    df_rad2 = pd.concat([df_rad2, df_rad3])
    P_FOMs = []
    T_FOMs = []
    for spec in sorted(df_rad2['spac'].unique()):
        plt.figure(num=1, clear=True)
        df2 = df_rad2[df_rad2['spac'] == spec]
        T_FOM = (df2['bottom_right_T'].values[-1] / (2400+273))
        T_FOMs.append(T_FOM)
        delT = np.abs(df2['outlet_T'] - df2['inlet_T'])
        Pout = df2['mass_flow']*delT*cp_Sn
        Poutnp = np.array(df2['mass_flow']*delT*cp_Sn)
        nom_flow = flowrate_from_hours(hours)
        P_FOM = np.trapz(Poutnp[df2['time'] <= hours], df2['time'][df2['time'] <= hours]) / (nom_flow*500*cp_Sn*hours)
        P_FOMs.append(P_FOM)
        # plt.plot(df_orig['time'], df_orig['bottom_right_T'], label='no rad')
        # # plt.plot(df['time'], df['bottom_right_T'], label='amb T = T_C')
        # plt.plot(df2['time'], df2['bottom_right_T'], label='rad w/ cap')
        plt.plot(df2['time'], Pout, label='P$_{in}$')
        plt.plot(df2['time'], np.ones(len(df2['time']))*nom_flow*500*cp_Sn, label='P$_{nom}$')
        plt.legend()
        plt.title('spac = %0.1e' % spec)
        plt.xlabel('time (hr)')
        plt.ylabel('block T (K)')
        # plt.grid()
        plt.tight_layout()
        # plt.show()
    plt.figure(num=2, clear=True)
    rad_spaces = [0] + list(sorted(df_rad2['spac'].unique()))
    delT = np.abs(df_orig['outlet_T'] - df_orig['inlet_T'])
    Pout = df_orig['mass_flow']*delT*cp_Sn
    Poutnp = np.array(df_orig['mass_flow']*delT*cp_Sn)
    nom_flow = flowrate_from_hours(hours)
    P_FOM = np.trapz(Poutnp[df_orig['time'] <= hours], df_orig['time']
                     [df_orig['time'] <= hours]) / (nom_flow*500*cp_Sn*hours)
    P_FOMs = [P_FOM] + list(P_FOMs)
    T_FOMs = [df_orig['bottom_right_T'].values[-1] / (2400+273)] + list(T_FOMs)
    # print(rad_spaces, P_FOMs)
    plt.plot(rad_spaces, P_FOMs)
    # plt.plot(rad_spaces, T_FOMs, label='T')
    plt.xlabel('Gap size (m)')
    plt.ylabel('P FOM')
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_fastcharge_sweep_rad_gap.pdf')
    # plt.show()


def power_block_ODE_height(T_data, mdot_data):
    # solve ODE for power block height
    Tin = T_data
    Trad = np.array([2173, 2223, 2273, 2323, 2373, 2423, 2473, 2523, 2573, 2623, 2673])
    P_net = np.array([22.13450812, 25.29930801, 28.8144618, 32.70539011, 36.99823388, 41.71983159,
                      46.89769773, 52.56000227, 58.73555122, 65.45376811, 72.74467579])
    heat_flux_T = np.polyfit(Trad, P_net*100**2, 2)
    mdot = mdot_data
    # print(mdot, Tin, (1900+273))
    cp_Sn = 240
    D_Sn = 0.01*2  # m
    dx = 0.001  # m

    def reached_outlet(t, y):
        return y[0]-2173
    reached_outlet.terminal = True

    def F(x, T):
        return -(0.9152*T**2 + -3.4293e+03*T**1 + 3.3536e+06*T**0) * np.pi * D_Sn / (mdot*cp_Sn)
    x_max = 100
    sol = solve_ivp(F, [0, x_max], [Tin], method='LSODA', t_eval=np.linspace(
        0, x_max, 10000), events=reached_outlet, dense_output=True)
    return sol.t_events[0][0]


def vary_mdot_const_P_CS():
    cp_Sn = 240
    df_data = pd.read_csv('../data/COMSOL_5block_10_k_discharge_sweep_log_maxflow.csv', names=[
                          'hours', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
    # hours = 30
    # max_ff = 2
    all_hours = df_data['hours'].unique()
    all_max_ff = df_data['max_ff'].unique()
    P_FOMs = []
    P_heights_add = []
    index = 1
    for hours in tqdm(np.array(sorted(df_data['hours'].unique()))):
        for max_ff in tqdm(np.array(sorted(df_data['max_ff'].unique()))):
            df = df_data[df_data['hours'] == hours]
            df = df[df['max_ff'] == max_ff]
            df['time'] = df['time']/3600
            delT = df['outlet_T'] - df['inlet_T']
            Pout = df['mass_flow']*delT*cp_Sn
            Poutnp = np.array(df['mass_flow']*delT*cp_Sn)
            # find index of Pout near max Pout value
            max_Pout = max(Pout)
            max_Pout_index = Pout[np.abs(Pout-max_Pout) < 10].index[-1]
            max_Pout_time = df.loc[max_Pout_index, 'time']
            P_FOM = max_Pout_time / hours
            # P_FOM = np.trapz(Poutnp[df['time'] < hours], df['time'][df['time'] < hours]) / (Poutnp[0]*hours)
            P_heights = []
            for row in df.iterrows():
                Tout = row[1]['outlet_T']
                mdot = row[1]['mass_flow']
                P_height = power_block_ODE_height(Tout, mdot)
                P_heights.append(P_height)
            P_heights = np.array(P_heights)/P_heights[0]
            P_heights_add.append((max(P_heights)/P_heights[0] - 1)*100)
            P_FOMs.append(P_FOM)
            fig = plt.figure(num=index, clear=True, figsize=(5, 4))
            index += 1
            fig.suptitle('hours = %0.1f, max$_{ff}$ = %0.1f' % (hours, max_ff))
            ax = fig.add_subplot(221)
            plt.plot(df['time'], delT)
            plt.xlabel('time (hr)')
            plt.ylabel('delta T (K)')
            # plt.grid()
            plt.tight_layout()
            ax = fig.add_subplot(222)
            plt.plot(df['time'], df['mass_flow'])
            plt.xlabel('time (hr)')
            plt.ylabel('mass flow (kg/s)')
            # plt.grid()
            plt.tight_layout()
            ax = fig.add_subplot(223)
            plt.plot(df['time'], Pout)
            plt.xlabel('time (hr)')
            plt.ylabel('Power (W)')
            # plt.grid()
            plt.tight_layout()
            ax = fig.add_subplot(224)
            plt.plot(df['time'], P_heights)
            plt.xlabel('time (hr)')
            plt.ylabel('TPV area (norm)')
            # plt.grid()
            plt.tight_layout()
            plt.savefig('../plots/COMSOL_5block_10_k_discharge_sweep_maxflow_%0.1f_%0.1f.png' %
                        (hours, max_ff), dpi=300)
            plt.close()
    # plt.show()
    fig = plt.figure(num=index, clear=True, figsize=(8, 3.2))
    ax = fig.add_subplot(121)
    plt.contourf(all_max_ff, all_hours, np.array(P_FOMs).reshape(len(all_hours), len(all_max_ff)), 100)
    plt.ylabel('hours')
    plt.xlabel('max flow factor')
    plt.colorbar(label='FOM$_P$')
    plt.contour(all_max_ff, all_hours, np.array(P_FOMs).reshape(
        len(all_hours), len(all_max_ff)),  levels=[0.9], colors='k')
    plt.tight_layout()
    ax = fig.add_subplot(122)
    plt.contourf(all_max_ff, all_hours, np.array(P_heights_add).reshape(len(all_hours), len(all_max_ff)), 100)
    plt.ylabel('hours')
    plt.xlabel('max flow factor')
    plt.colorbar(label='TPV area added ($\%$)')
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_discharge_sweep_maxflow_contour.png', dpi=300)
    # plt.show()

    pass


def SOC_discharge_vary_mdot():
    cp_Sn = 240
    df_data = pd.read_csv('data/COMSOL_5block_10_k_discharge_sweep_log_maxflow.csv', names=[
                          'hours', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
    energy_max, _, _ = energy_from_flowrate_hours(df_data['mass_flow'][0], df_data['hours'][0])
    SOCs = []
    powers = []
    P_densities = []
    P_heights_all = []
    for hours in tqdm(df_data['hours'].unique()):
        for max_ff in df_data['max_ff'].unique():
            df_temp = df_data[df_data['hours'] == hours][df_data['max_ff'] == max_ff]
            power = df_temp['mass_flow']*cp_Sn*(df_temp['outlet_T']-df_temp['inlet_T'])
            energy = energy_max - cumtrapz(power, df_temp['time'], initial=0)
            SOC = energy/energy_max
            df_temp['SOC'] = SOC
            SOCs = SOCs + list(df_temp['SOC'])
            powers = powers + list(power)
            P_heights = []
            for row in df_temp.iterrows():
                Tout = row[1]['outlet_T']
                mdot = row[1]['mass_flow']
                P_height = power_block_ODE_height(Tout, mdot)
                P_heights.append(P_height)
            P_area = np.array(P_heights) * np.pi * (0.01*2)
            P_density = np.array(power) / P_area
            P_heights = np.array(P_heights)/P_heights[0]
            P_heights_all = P_heights_all + list(P_heights)
            P_densities = P_densities + list(P_density)
            # fig = plt.figure(num=1, clear=True)
            # plt.plot(SOC, power)
            # plt.xlabel('SOC')
            # plt.ylabel('Power (W)')
            # plt.tight_layout()
            # plt.show()
            pass
    df_data['SOC'] = SOCs
    df_data['power'] = powers
    df_data['TPV_height'] = P_heights_all
    df_data['TPV_density'] = P_densities
    df_data.to_csv('data/COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC.csv', index=False)
    pass


def test_SOC_P_plots_discharge():
    cp_Sn = 240
    df_data = pd.read_csv('data/COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC.csv', names=[
                          'hours', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'SOC', 'power'], skiprows=5)
    df_data = df_data[df_data['hours'] == df_data['hours'].unique()[6]]
    fig = plt.figure(num=1, clear=True)
    fig = plt.figure(num=2, clear=True)
    fig = plt.figure(num=3, clear=True)
    df_maxff_1 = df_data[df_data['max_ff'] == 1.0]
    df_maxff_1 = df_maxff_1.reset_index(drop=True)
    df_maxff_1['delT'] = df_maxff_1['outlet_T'] - df_maxff_1['inlet_T']
    colors = ['r', 'b', 'g', 'k', 'm', 'c']
    for max_ff in df_data['max_ff'].unique():
        df_temp = df_data[df_data['max_ff'] == max_ff]
        df_temp['delT'] = df_temp['outlet_T'] - df_temp['inlet_T']
        df_temp = df_temp.reset_index(drop=True)
        df_constP_basecase = df_maxff_1[df_maxff_1['delT'] >= df_maxff_1['delT'][0]/max_ff]
        # df_not_const_SOC = df_temp[df_temp['delT'] < df_temp['delT'][0]/max_ff]
        df_not_constP_basecase = df_maxff_1[df_maxff_1['delT'] <= df_maxff_1['delT'][0]/max_ff]
        # df_not_constP = df_not_constP[df_not_constP['SOC'] < df_constP['SOC'].values[-1]]
        basecase_SOC = df_constP_basecase['SOC'].append(df_not_constP_basecase['SOC'])
        # basecase_P
        color = colors.pop(0)
        plt.figure(1)
        plt.plot(df_constP_basecase['SOC'], df_constP_basecase['power'], '--', color=color)
        plt.plot(df_not_constP_basecase['SOC'], df_not_constP_basecase['power']*max_ff, '--', color=color)
        plt.plot(df_temp['SOC'], df_temp['power'], label='max ff = %0.1f' % max_ff, color=color)
        plt.figure(3)
        plt.plot(df_temp['SOC'], df_temp['power'], label='max ff = %0.1f' % max_ff, color=color)
        plt.figure(2)
        plt.plot(df_temp['SOC'], df_temp['delT'], label='max ff = %0.1f' % max_ff, color=color)
    plt.figure(1)
    plt.xlabel('SOC')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC_P_31hrs_comp.pdf')
    plt.figure(2)
    plt.xlabel('SOC')
    plt.ylabel('Delta T (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC_T_31hrs.pdf')
    plt.figure(3)
    plt.xlabel('SOC')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/COMSOL_5block_10_k_discharge_sweep_log_maxflow_SOC_P_31hrs.pdf')
    plt.show()
    pass


def energy_from_flowrate_hours(mdot, hours):
    # power = 20e6  # W
    cp_Sn = 240
    delT = 500
    power = mdot*cp_Sn*delT  # W
    # mdot_Sn = power / (cp_Sn * delT)
    energy = power * hours * 3600  # J
    energy_MWh = energy / 3600e6
    V_block = 0.1**2 * np.pi * 10 - 0.01**2 * np.pi * 10
    rho_C = 1700
    cp_C = 2000
    num_blocks = energy / (V_block * rho_C * cp_C * delT)
    energy_1_block = (V_block * rho_C * cp_C * delT)
    # series_blocks = 5
    # scaling_factor = series_blocks / num_blocks
    # scaled_mdot_Sn = mdot_Sn * scaling_factor
    # print('scaled mdot_Sn: {:.6f} kg/s'.format(scaled_mdot_Sn))
    return energy, num_blocks, energy_1_block


def calc_view_factor(R1, R2, H):
    A1 = 2*np.pi*R1*H
    A2 = 2*np.pi*R2*H
    h = H/R1
    R = R2/R1
    f1 = h**2 + R**2 - 1
    f2 = h**2 - R**2 + 1
    f3 = np.sqrt((f1+2)**2 - 4*R**2)
    f4 = f3*np.arccos(f2/(R*f1)) + f2*np.arcsin(1/R) - np.pi*f1/2
    F12 = 1 - 1/np.pi*(np.arccos(f2/f1)-f4/(2*h))
    F21 = A1*F12/A2
    # print(F12, F21)
    return F12


def sweep_total_resistance():
    R_Sn = 0.01
    R_C = 0.005
    R_gap = 0
    R_o = 0.1
    H = 2
    k_C = 10

    Nu = 4
    k_Sn = 62
    h = Nu*k_Sn / (R_Sn*2)
    R_conv = 1/(h*(2*np.pi*R_Sn*2))
    R_cond_1 = np.log((R_C+R_Sn)/R_Sn)/(2*np.pi*2*50)
    R_rad = 1/(1*5.67e-8*(2673**2 + 2173**2)*(2673+2173)*2*np.pi *
               (R_Sn+R_C)*H*calc_view_factor(R_Sn+R_C, R_Sn+R_C+R_gap, H))
    R_rad_2 = 1/(1*5.67e-8*(2173**2 + 2173**2)*(2173+2173)*2*np.pi *
                 (R_Sn+R_C)*H*calc_view_factor(R_Sn+R_C, R_Sn+R_C+R_gap, H))
    R_cond_2 = np.log(R_o/(R_Sn+R_C+R_gap))/(2*np.pi*2*k_C)

    print(R_conv, R_cond_1, R_rad, R_rad_2, R_cond_2)

    A_graphite = np.pi*(R_o**2 - (R_Sn+R_C+R_gap)**2)
    # print(A_graphite)
    R_gap = np.linspace(0.00, 0.2, 1000)
    R_o = np.sqrt(A_graphite/np.pi + (R_Sn + R_C + R_gap)**2)
    # print(R_o[0])
    # print(0.1)
    V_perc_inc = (R_o**2 - 0.1**2)/0.1**2*100
    R_cond_2 = np.log(R_o/(R_Sn+R_C+R_gap))/(2*np.pi*2*k_C)
    R_rad = 1/(1*5.67e-8*(2673**2 + 2173**2)*(2673+2173)*2*np.pi *
               (R_Sn+R_C)*H*calc_view_factor(R_Sn+R_C, R_Sn+R_C+R_gap, H))
    R_rad[0] = 0
    R_tot = R_conv + R_cond_1 + R_rad + R_cond_2
    # R_tot[0] = R_conv + R_cond_1 + R_cond_2[0]
    # print(R_tot)
    plt.figure(num=1, clear=True)
    plt.plot(R_gap, R_tot, label='R$_{tot}$')
    plt.plot(R_gap, R_rad, label='R$_{rad}$')
    plt.plot(R_gap, R_cond_2, label='R$_{cond}$')
    # plt.semilogy(R_gap, np.ones(len(R_gap))*R_rad,'k--',label='R_rad')
    plt.xlabel('Gap size (m)')
    plt.ylabel('Resistance')
    plt.ylim(0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/R_tot_vs_gap.pdf')

    plt.figure(num=2, clear=True)
    plt.xlabel('Gap size (m)')
    plt.plot(R_gap, V_perc_inc, 'r')
    plt.ylabel('$\%$ increase in volume')
    plt.tight_layout()
    plt.savefig('../plots/Rgap_vs_size.pdf')


def charge_discharge_CS():
    cp_Sn = 244
    df = pd.read_csv('../data/COMSOL_5block_10_k_charge_discharge_10maxflow.csv', names=[
                     'hours', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'P'], skiprows=5)
    df['time'] = df['time'] / 3600
    fig = plt.figure(num=1, clear=True, figsize=(6, 6))
    ax = fig.add_subplot(311)
    plt.plot(df['time'], df['inlet_T'], label='inlet Sn')
    plt.plot(df['time'], df['bottom_right_T'], label='outlet Sn')
    # plt.grid()
    plt.legend(loc='center left')
    plt.xlabel('time (hr)')
    plt.ylabel('T (K)')
    plt.tight_layout()
    ax = fig.add_subplot(312)
    plt.plot(df['time'], df['mass_flow'])
    # plt.grid()
    # plt.legend()
    plt.xlabel('time (hr)')
    plt.ylabel('mass flow (kg/s)')
    plt.tight_layout()
    ax = fig.add_subplot(313)
    plt.plot(df['time'], df['P']*cp_Sn)
    plt.xlabel('time (hr)')
    plt.ylabel('P (W)')
    # plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_charge_discharge_10maxflow.pdf')

    # df_orig = pd.read_csv('5block_data/COMSOL_5block_10_k_charge_discharge_10maxflow_reverse.csv', names=['hours','max_flowrate_factor', 'time','inlet_T','outlet_T','block_T','Sn_T','bottom_right_T','mass_flow','P'], skiprows=5)
    df = pd.read_csv('../data/COMSOL_5block_10_k_charge_discharge_10maxflow_reverse.csv', names=[
                     'hours', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'P'], skiprows=5)
    df['time'] = df['time'] / 3600
    # df_orig['time'] = df_orig['time'] / 3600
    fig = plt.figure(num=2, clear=True, figsize=(6, 6))
    ax = fig.add_subplot(311)
    # Tval is inlet T if P is positive, outlet T if P is negative
    Tval = df['inlet_T'].to_numpy(copy=True)
    # Tval2 is outlet T if P is positive, inlet T if P is negative
    Tval2 = df['outlet_T'].to_numpy(copy=True)
    Tval2[df['P'] < 0] = df['inlet_T'][df['P'] < 0]
    Tval[df['P'] < 0] = df['outlet_T'][df['P'] < 0]
    plt.plot(df['time'], Tval, label='inlet Sn')
    # plt.plot(df_orig['time'], df_orig['inlet_T'], label='inlet Sn')
    plt.plot(df['time'], Tval2, label='outlet Sn')
    # plt.plot(df_orig['time'], df_orig['outlet_T'], label='outlet Sn')
    # plt.grid()
    plt.legend(loc='center left')
    plt.xlabel('time (hr)')
    plt.ylabel('T (K)')
    plt.tight_layout()
    ax = fig.add_subplot(312)
    plt.plot(df['time'], df['mass_flow'])
    # plt.grid()
    # plt.legend()
    plt.xlabel('time (hr)')
    plt.ylabel('mass flow (kg/s)')
    plt.tight_layout()
    ax = fig.add_subplot(313)
    plt.plot(df['time'], df['P']*cp_Sn)
    plt.xlabel('time (hr)')
    plt.ylabel('P (W)')
    # plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_charge_discharge_10maxflow_reverse.pdf')

    df = pd.read_csv('../data/COMSOL_5block_10_k_charge_discharge_10maxflow_halfDC.csv', names=[
                     'hours', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'P'], skiprows=5)
    df['time'] = df['time'] / 3600
    fig = plt.figure(num=3, clear=True, figsize=(6, 6))
    ax = fig.add_subplot(311)
    Tval = df['inlet_T'].to_numpy(copy=True)
    Tval2 = df['bottom_right_T'].to_numpy(copy=True)
    plt.plot(df['time'], Tval, label='inlet Sn')
    plt.plot(df['time'], Tval2, label='outlet Sn')
    # plt.grid()
    plt.legend(loc='center left')
    plt.xlabel('time (hr)')
    plt.ylabel('T (K)')
    plt.tight_layout()
    ax = fig.add_subplot(312)
    plt.plot(df['time'], df['mass_flow'])
    # plt.grid()
    plt.xlabel('time (hr)')
    plt.ylabel('mass flow (kg/s)')
    plt.tight_layout()
    ax = fig.add_subplot(313)
    plt.plot(df['time'], df['P']*cp_Sn)
    plt.xlabel('time (hr)')
    plt.ylabel('P (W)')
    # plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_charge_discharge_10maxflow_halfDC.pdf')

    df = pd.read_csv('../data/COMSOL_5block_10_k_charge_discharge_10maxflow_reverse_halfDC.csv', names=[
                     'hours', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow', 'P'], skiprows=5)
    df['time'] = df['time'] / 3600
    fig = plt.figure(num=4, clear=True, figsize=(6, 6))
    ax = fig.add_subplot(311)
    Tval = df['inlet_T'].to_numpy(copy=True)
    Tval2 = df['outlet_T'].to_numpy(copy=True)
    Tval2[df['P'] < 0] = df['inlet_T'][df['P'] < 0]
    Tval[df['P'] < 0] = df['outlet_T'][df['P'] < 0]
    plt.plot(df['time'], Tval, label='inlet Sn')
    plt.plot(df['time'], Tval2, label='outlet Sn')
    # plt.grid()
    plt.legend(loc='center left')
    plt.xlabel('time (hr)')
    plt.ylabel('T (K)')
    plt.tight_layout()
    ax = fig.add_subplot(312)
    plt.plot(df['time'], df['mass_flow'])
    # plt.grid()
    plt.xlabel('time (hr)')
    plt.ylabel('mass flow (kg/s)')
    plt.tight_layout()
    ax = fig.add_subplot(313)
    plt.plot(df['time'], df['P']*cp_Sn)
    plt.xlabel('time (hr)')
    plt.ylabel('P (W)')
    # plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_charge_discharge_10maxflow_reverse_halfDC.pdf')
    # plt.show()


def sq_block_sweep_hrs():
    # df = pd.read_csv('../data/COMSOL_5block_10_k_sqblock_coarse_sweep_hours.csv',
    #                  names=['hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'perm', 'inlet_P', 'outlet_P'], skiprows=5)
    # df_pm = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_tuneh_sweep_hrs.csv',
    #                     names=['hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'perm', 'inlet_P', 'outlet_P'], skiprows=5)
    # df_pm = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_tuneh_sweep_hrs_longblock.csv',
    #                     names=['hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'perm', 'inlet_P', 'outlet_P'], skiprows=5)
    # df_pm2 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_adiab.csv', names=[
    #                      'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3'], skiprows=5)
    df_pm3 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_longblock.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3'], skiprows=5)
    df_pm4 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_horizstring.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    df_pm5 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_10x10grid.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    df_pm6 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_10x10grid_parallel_2.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    df_pm7 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_10x10grid_parallel_all.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    # df_pm8 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_10x10grid_winding_2.csv', names=[
    #                      'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    cs = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # plt.figure(num=1, clear=True)
    # for index, hours in enumerate(df.hours.unique()):
    #     df2 = df[df.hours == hours]
    #     df2_pm = df_pm[df_pm.hours == hours]
    #     plt.plot(df2['time'], df2['outlet_T'], label='{} hrs CS'.format(hours), c=cs[index])
    #     plt.plot(df2_pm['time'], df2_pm['outlet_T'], '--', label='{} hrs PM'.format(hours), c=cs[index])
    # plt.legend()
    # # plt.grid()
    # plt.xlabel('time (s)')
    # plt.ylabel('outlet T (K)')
    # plt.tight_layout()
    # plt.savefig('../plots/COMSOL_5block_10_k_sqblock_porous_sweep_hrs_comp.png')
    # plt.figure(num=2, clear=True)
    # hours = 20
    # df2 = df[df.hours == hours]
    # df2_pm = df_pm[df_pm.hours == hours]
    # plt.plot(df2['time'], df2['outlet_T'], label='{} hrs CS'.format(hours))
    # plt.plot(df2_pm['time'], df2_pm['outlet_T'], '--', label='{} hrs PM'.format(hours))
    # plt.plot(df_pm2['time'], df_pm2['outlet_T'], '-.', label='{} hrs rad 10m'.format(hours))
    # # plt.plot(df_pm3['time'], df_pm3['outlet_T'], '-.', label='{} hrs rad 400m'.format(hours))
    # plt.legend()
    # # plt.grid()
    # plt.xlabel('time (s)')
    # plt.ylabel('outlet T (K)')
    # plt.tight_layout()
    # plt.savefig('../plots/COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp.png')
    plt.figure(num=3, clear=True, figsize=(2.5, 2.5))
    hours = 20
    df_pm3['time'] = df_pm3['time'] / 3600
    df_pm4['time'] = df_pm4['time'] / 3600
    df_pm5['time'] = df_pm5['time'] / 3600
    df_pm6['time'] = df_pm6['time'] / 3600
    df_pm7['time'] = df_pm7['time'] / 3600
    plt.plot(df_pm3['time'], df_pm3['outlet_T'], label='(1) 1x100 vertical blocks'.format(hours))
    plt.plot(df_pm4['time'], df_pm4['outlet_T'], label='(2) 1x100 string of blocks'.format(hours))
    plt.plot(df_pm5['time'], df_pm5['outlet_T'], label='(3) 10x10 grid, all series'.format(hours))
    plt.plot(df_pm6['time'], df_pm6['outlet_T'], label='(4) 10x10 grid, 10 parallel paths'.format(hours))
    plt.plot(df_pm7['time'], df_pm7['outlet_T'], label='(5) 10x10 grid, 100 parallel paths'.format(hours))
    # plt.plot(df_pm8['time'], df_pm8['outlet_T'], label='{} hrs rad 400m 10x10 wind'.format(hours))
    # put legend below figure
    # plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2))
    # plt.grid()
    plt.xlabel('time (hrs)')
    plt.ylabel('outlet T (K)')
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_configs.pdf')
    plt.show()


def grid_varyflow():
    cp_Sn = 244
    df = pd.read_csv('../data/COMSOL_5block_10_k_discharge_10x10grid_10parallel_maxflow_5.0_10.0.csv', names=[
                     'hours', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block_T_2', 'block_T_3', 'block_T_4', 'mass_flowrate', 'P_out'], skiprows=5)
    df_pm4 = pd.read_csv('../data/COMSOL_5block_10_k_porousmedia_coarse_rad_horizstring.csv', names=[
                         'hours', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block2_T', 'block3_T', 'block4_T'], skiprows=5)
    df_5 = df[df.max_flowrate_factor == 5]
    df_10 = df[df.max_flowrate_factor == 10]
    df_5['time'] = df_5['time'] / 3600
    df_5['P_out'] = df_5['P_out']/1e6
    df_10['time'] = df_10['time'] / 3600
    df_pm4['time'] = df_pm4['time'] / 3600
    df_pm4['mass_flowrate'] = df_5['mass_flowrate'].values[0]
    df_pm4['P_out'] = (df_pm4['outlet_T'] - df_pm4['inlet_T'])*df_pm4['mass_flowrate']*cp_Sn
    df_pm4['P_out'] = df_pm4['P_out']/1e6
    # df_5 = df[df['time'] < 20]
    # P_out = (df['outlet_T'] - df['inlet_T'])*df['mass_flowrate']*cp_Sn
    plt.figure(num=1, clear=True, figsize=(2.5, 3))
    # get default colors from matplotlib
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(df_pm4['time'], df_pm4['P_out'], label='1x100 string', color=colors[1])
    plt.plot(df_5['time'], df_5['P_out'], label='10x10 grid, 5x flow', color=colors[3])
    plt.xlabel('time (hrs)')
    plt.ylabel('Power (MW)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_10x10_varyflow.pdf')
    plt.show()


def grid_varyflow_charging():
    hours = 4
    cp_Sn = 244
    df = pd.read_csv('../data/COMSOL_5block_10_k_charge_10x10grid_10parallel_maxflow_1.0_5.0_10.0.csv', names=[
                     'hours', 'base_flowrate_factor', 'max_flowrate_factor', 'time', 'inlet_T', 'outlet_T', 'block_T', 'inlet_T_2', 'outlet_T_2', 'inlet_T_3', 'outlet_T_3', 'block_T_2', 'block_T_3', 'block_T_4', 'mass_flowrate', 'P_out'], skiprows=5)
    nom_flow = df[df['base_flowrate_factor'] == 1]['mass_flowrate'].values[0]
    base_max_flows = [1, 5, 10]
    P_FOMs = []
    for base_max_flow in base_max_flows:
        df_data = df[df['base_flowrate_factor'] == base_max_flow]
        df_data['time'] = df_data['time']/3600
        Pout = np.abs(df_data['P_out'])
        P_FOM = np.trapz(Pout[df_data['time'] <= hours], df_data['time']
                         [df_data['time'] <= hours]) / (nom_flow*500*cp_Sn*hours)
        print(P_FOM)
        P_FOMs.append(P_FOM)
    plt.figure(num=2, clear=True, figsize=(2.5, 3))
    plt.plot(base_max_flows, P_FOMs, label='10x10 grid')
    plt.plot(base_max_flows, [1, 1, 1], 'k--')
    plt.plot(base_max_flows, [0, 0, 0], 'k--')
    plt.xlabel('flow multiplier')
    plt.ylabel('FOM$_{P,charge}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/COMSOL_5block_10_k_sqblock_porous_rad_PBC_comp_10x10_varyflow_charging.pdf')
    plt.show()


# flowrate_from_hours(30)
# plot_COMSOL()
# new_LC_model(10)
# new_LC_model(30)
# new_LC_model_Pe(20,5)
# new_LC_model_Pe_outlet(1,5)
# new_LC_model_step(30)
# new_LC_model_opt_P_Pe(20, 5)
# new_LC_model(50)
# PDE_find()
# curve_fit_params = curve_fit_temps_2()
# ODE_find_fit(curve_fit_params)
# sweep_initial_T_2()
# sweep_both_norm()
# sweep_both_norm_2()
# compare_longblock()
# plot_normalized_time()
# fastcharge_CS_mdot()
# plot_normalized_time_CS_LC()
# plot_nondim_time_CS()
# vary_mdot_const_P_CS()
# fastcharge_CS_Tin()
# sweep_total_resistance()
# fastcharge_CS_rad()
# charge_discharge_CS()
# sq_block_sweep_hrs()
# grid_varyflow()
# grid_varyflow_charging()
SOC_discharge_vary_mdot()
# test_SOC_P_plots_discharge()
# fastcharge_CS_mdot_SOC()
# test_SOC_P_plots_charge()
