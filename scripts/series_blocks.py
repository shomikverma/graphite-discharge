import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use(['science','grid'])

def flowrate_from_num_blocks(i):
    return 0.00825 * i

def import_data():
    data1 = pd.read_csv('5series_discharge/1_block.csv', names=['time', 'outlet_T'], skiprows=5)
    data1['num_blocks'] = 1
    data1['inlet_T'] = 1900+273.15
    data = pd.read_csv('5series_discharge/all_blocks.csv',
                       names=['num_blocks', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data = data[data['num_blocks'] != 1]
    data = data1.append(data)
    data2 = pd.read_csv('5series_discharge/6_to_8_blocks.csv',
                        names=['num_blocks', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data_30 = data.append(data2)
    data_30['k'] = 30
    data_10_1 = pd.read_csv('5series_discharge/series_blocks_k_10_1to4.csv', names=['num_blocks', 'k', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data_10_2 = pd.read_csv('5series_discharge/series_blocks_k_10_5to8.csv', names=['num_blocks', 'k', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data_10 = data_10_1.append(data_10_2)
    data_5_1 = pd.read_csv('5series_discharge/series_blocks_k_5_1to4.csv', names=['num_blocks', 'k', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data_5_2 = pd.read_csv('5series_discharge/series_blocks_k_5_5to8.csv', names=['num_blocks', 'k', 'time', 'inlet_T', 'outlet_T'], skiprows=5)
    data_5 = data_5_1.append(data_5_2)
    data = [data_5, data_10, data_30]
    return data


def plot_data(df_data):
    for data in df_data:
        plt.figure(num=1,clear=True)
        k = data['k'].unique()[0]
        for i in range(1, 9):
            df = data[data['num_blocks'] == i]
            plt.plot(df['time'], df['outlet_T'], label=str(i))
        plt.grid()
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Outlet Temperature (K)')
        plt.title('k = ' + str(k))
        plt.tight_layout()
        plt.savefig('5series_discharge/series_blocks_' + str(k) + '.png')
    
    plt.figure(num=3,figsize=(9,6),clear=True)
    cmap = mpl.cm.get_cmap('nipy_spectral')
    Pes = []
    times = []
    Touts = []
    for data in df_data:
        k = data['k'].unique()[0]
        for i in range(1, 9):
            df = data[data['num_blocks'] == i]
            # Pe = flowrate_from_hours(hour)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
            Pe = np.sqrt(k/1700/2000/(i*2*flowrate_from_num_blocks(i)/(6200*.01**2*np.pi)))
            Pe = k/flowrate_from_num_blocks(i)/i
            Pes.append(Pe)
            times.append(df['time']/max(df['time'])*2)
            Touts.append((df['outlet_T']-2173)/500)
    df = pd.DataFrame({'Pe': Pes, 'time': times, 'Tout': Touts})
    df = df.sort_values(by=['Pe'])
    for row in df.iterrows():
        plt.plot(row[1]['time'], row[1]['Tout'], label='%0.2e' % row[1]['Pe'], color=cmap(row[1]['Pe']/max(Pes)*0.9))
    plt.grid()
    plt.legend()
    plt.xlabel('Time (norm)')
    plt.ylabel('Outlet Temperature (norm)')
    # plt.title('k = ' + str(k))
    plt.tight_layout()
    plt.savefig('5series_discharge/series_blocks_nondim_deviation.png')


def analyze_data(df_data):
    plt.figure(num=5,clear=True)
    Pes = []
    times = []
    Touts = []
    
    for data in df_data:
        outlet_nonideals = []
        ks = []
        all_i = range(1, 9)
        mdots = []
        k = data['k'].unique()[0]
        for i in all_i:
            df = data[data['num_blocks'] == i]
            # Pe = flowrate_from_hours(hour)/(6200*.01**2*np.pi)*.09/(k/1700/2000)
            Pe = np.sqrt(k/1700/2000/(i*2*flowrate_from_num_blocks(i)/(6200*.01**2*np.pi)))
            Pe = k/flowrate_from_num_blocks(i)/i
            Pes.append(Pe)
            time = df['time']/max(df['time'])*2
            times.append(time)
            Tout = (df['outlet_T']-2173)/500
            Touts.append(Tout)
            outlet_nonideal = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
            outlet_nonideals.append(outlet_nonideal)
            ks.append(k)
            mdots.append(flowrate_from_num_blocks(i))
            # all_i.append(i)
            # print(outlet_nonideal, k)
        plt.figure(num=5)
        plt.plot(all_i, outlet_nonideals, label=str(k))
        plt.figure(num=6)
        plt.plot(np.array(all_i) * np.array(ks), outlet_nonideals, '.')
    plt.figure(num=5)
    plt.legend()
    plt.grid()
    plt.xlabel('Number of Blocks')
    plt.ylabel('T FOM')
    plt.tight_layout()


def sweep_AR(hours=30):
    plt.figure(num=2,clear=True)
    # df = pd.read_csv('5series_discharge/sweep_AR_h30_k30_3.csv', names=['index', 'H', 'D', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # for index in df['index'].unique():
    #     df_i = df[df['index'] == index]
    #     plt.plot(df_i['time'], df_i['outlet_T'], label='H = ' + str(df_i['H'].unique()[0]) + ', D = ' + str(df_i['D'].unique()[0]))
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Outlet Temperature (K)')
    # plt.tight_layout()
    df = pd.read_csv('5series_discharge/sweep_AR_h{}_k30_all.csv'.format(hours), names=['H', 'D', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    firstH = df['H'].unique()[0]
    firstD = df['D'].unique()[0]
    # get df for first H and D
    df_first = df[(df['H'] == firstH) & (df['D'] == firstD)]
    indexes = [element for element in np.arange(1,65,1) for i in range(len(df_first))]
    df['index'] = indexes[:len(df)]
    outlet_nonideals_all = []
    for H in df['H'].unique():
        outlet_nonideals = []
        Pes = []
        for D in df['D'].unique():
            df_i = df[(df['H'] == H) & (df['D'] == D)]
            if len(df_i)==0:
                outlet_nonideals.append(np.nan)
                outlet_nonideals_all.append(np.nan)
                Pes.append(None)
                continue
            time = df_i['time']/max(df_i['time'])*2
            Tout = (df_i['outlet_T']-2173)/500
            plt.figure(num=1)
            plt.plot(time, Tout, '-', label='H %0.2fm, D %0.2fm' % (df_i['H'].unique()[0], df_i['D'].unique()[0]))
            outlet_nonideal = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
            outlet_nonideals.append(outlet_nonideal)
            outlet_nonideals_all.append(outlet_nonideal)
            Pes.append(D)
        plt.figure(num=2)
        plt.plot(Pes, outlet_nonideals, '-', label='H %0.2fm' % (H))
    # for index in df['index'].unique():
    #     df_i = df[df['index'] == index]
    #     time = df_i['time']/max(df_i['time'])*2
    #     Tout = (df_i['outlet_T']-2173)/500
    #     plt.plot(time, Tout, '-', label='H %0.2fm, D %0.2fm' % (df_i['H'].unique()[0], df_i['D'].unique()[0]))
    #     outlet_nonideal = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
    #     outlet_nonideals.append(outlet_nonideal)
    # pad outlet_nonideals with zeros to make them the same length as indexes
    plt.figure(num=1)
    outlet_nonideals_all = np.pad(outlet_nonideals_all, (0,64-len(outlet_nonideals_all)), 'constant',constant_values=(np.nan,))
    outlet_nonideals_all = np.array(outlet_nonideals_all)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Outlet Temperature (K)')
    plt.tight_layout()
    plt.savefig('5series_discharge/sweep_AR_h{}_k30.png'.format(hours))
    series_blocks, block_D = np.mgrid[1:9:1, 0.1:0.9:0.1]
    plt.figure(num=3,clear=True)
    plt.pcolormesh(series_blocks, block_D, outlet_nonideals_all.reshape(8,8))
    cb = plt.colorbar()
    cb.set_label('T FOM')
    plt.xlabel('number of blocks')
    plt.ylabel('block diameter (m)')
    plt.tight_layout()
    plt.figure(num=2)
    plt.legend()
    plt.grid()


def sweep_Pe():
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    df = pd.read_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e_old.csv'%(10, 1.67e-4), names=['index', 'H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    for index in df['index'].unique():
        df_i = df[df['index'] == index]
        time = df_i['time']/max(df_i['time'])*2
        Tout = (df_i['outlet_T']-2173)/500
        H = df_i['H'].unique()[0]
        D = df_i['D'].unique()[0]
        U = df_i['U'].unique()[0]
        k = df_i['k'].unique()[0]
        hours = H/U*(rho_C/rho_Sn*0.99/100*Cp_C/Cp_Sn)
        plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k))
    
    plt.legend()
    plt.grid()
    plt.xlabel('norm time')
    plt.ylabel('norm outlet temperature')
    plt.savefig('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e.png'%(10, 1.67e-4))
    pass


def sweep_Pe_2():
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    df = pd.read_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e.csv'%(10, 1.67e-4), names=['index', 'H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    for index in df['index'].unique()[5:]:
        df_i = df[df['index'] == index]
        time = df_i['time']/max(df_i['time'])*2
        Tout = (df_i['outlet_T']-2173)/500
        H = df_i['H'].unique()[0]
        D = df_i['D'].unique()[0]
        U = df_i['U'].unique()[0]
        k = df_i['k'].unique()[0]
        hours = H/U*(rho_C/rho_Sn*0.99/100*Cp_C/Cp_Sn)
        plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k))
    plt.legend()
    plt.grid()
    plt.xlabel('norm time')
    plt.ylabel('norm outlet temperature')
    plt.savefig('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e_2.png'%(10, 1.67e-4))

def hours_from_H_U(H, U):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    hours = H/U*(rho_C/rho_Sn*99*Cp_C/Cp_Sn)/3600
    return hours    

def H_from_hours_U(hours, U):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    H = hours*3600/(rho_C/rho_Sn*99*Cp_C/Cp_Sn)*U
    return H

def U_from_H_hours(H, hours):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    U = H/(hours*3600)*(rho_C/rho_Sn*99*Cp_C/Cp_Sn)
    return U

def flowrate_from_hours_D_H(hours, D, H):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    flowrate = rho_C*np.pi*0.99*D**2/4*H*Cp_C/Cp_Sn/(hours*3600)
    return flowrate

def UD_from_H_hours(H, hours, D):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    U = H/(hours*3600)*(rho_C/rho_Sn*99*Cp_C/Cp_Sn)
    return U*D

def mdot_from_U_D(U, D):
    rho_C = 1700
    rho_Sn = 6164
    Cp_C = 2000
    Cp_Sn = 240
    mdot = U*rho_Sn*(D/10)**2/4*np.pi
    return mdot

def sweep_Pe_H_U():
    # df = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df2 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_30hrs.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df3 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highU.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df4 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highH.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    df_all = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_all_1.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # setk = 10
    fig10 = plt.figure(num=10, clear=True)
    ax10 = fig10.add_subplot(111)
    for setk in df_all['k'].unique():
        df = df_all[df_all['k']==setk]
        # df = df.append(df2).append(df3).append(df4)
        data = []
        cmap = mpl.cm.get_cmap('nipy_spectral')
        H_unique = np.array(sorted(df['H'].unique()))
        U_unique = np.array(sorted(df['U'].unique()))
        for H in H_unique:
            for U in U_unique:
                df_i = df[(df['H'] == H) & (df['U'] == U)]
                try:
                    time = df_i['time']/max(df_i['time'])*2.22
                except:
                    continue
                Tout = (df_i['outlet_T']-2170)/500
                # if U == 1e-3:
                #     continue
                D = df_i['D'].unique()[0]
                k = df_i['k'].unique()[0]
                hours = hours_from_H_U(H, U)
                Pe = U*D**2/H
                # print(Pe)
                FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
                data.append([H,U,FOM, hours, Pe])
                # plt.figure(num=4)
                Pe_norm = np.sqrt((Pe)/(0.04))
                # if abs((Pe_norm*10) - np.round(Pe_norm*10))< 0.05:
                # plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k), color=cmap(Pe_norm*0.9))
        # print(sorted(H_unique))
        # print(sorted(U_unique))
        fig = plt.figure(num=1,clear=True,figsize=(10,8))
        ax = fig.add_subplot(211)
        # ax = plt.axes(projection='3d')
        df = pd.DataFrame(data, columns=['H', 'U', 'FOM','hours','Pe'])
        U,H = np.meshgrid(sorted(U_unique), sorted(H_unique))
        # plt.contourf(H/D, U*D, df['FOM'].values.reshape(len(H_unique), len(U_unique)), cmap='viridis')
        colormap = mpl.cm.get_cmap('tab10')
        plt.figure(num=10)
        plt.contourf(U*D/setk, H/D, df['FOM'].values.reshape(len(H_unique), len(U_unique)), levels=np.linspace(0,1,100), cmap='viridis')
        # plt.scatter(U*D/setk, H/D, df['FOM'].values.reshape(len(H_unique), len(U_unique)), cmap='viridis')
        ax10.set_yscale('log')
        ax10.set_xscale('log')
        # plt.colorbar()
        plt.figure(num=1)
        plt.contourf(U*D/setk, H/D, df['FOM'].values.reshape(len(H_unique), len(U_unique)), levels=100, cmap='viridis')
        # plt.scatter(df['U']*0.2, df['H']/0.2, c=df['FOM'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('H/D')
        plt.xlabel('U*D')
        Hs = np.linspace(min(df['H'].unique()), max(df['H'].unique()), 100)
        print(Hs)
        print(min(df['H'].unique()))
        print(max(df['H'].unique()))
        plt.plot(UD_from_H_hours(Hs, 1, D)/setk, Hs/D, label='1 hour, D=0.2', c=colormap(0))
        plt.plot(UD_from_H_hours(Hs, 5, D)/setk, Hs/D, label='5 hour, D=0.2', c=colormap(0.1))
        plt.plot(UD_from_H_hours(Hs, 10, D)/setk, Hs/D, label='10 hour, D=0.2', c=colormap(0.2))
        plt.plot(UD_from_H_hours(Hs, 30, D)/setk, Hs/D, label='30 hour, D=0.2', c=colormap(0.3))
        # set H = 10, sweep hours and D
        H = 10
        Ds = np.linspace(0.01, 100, 1000)
        plt.plot(UD_from_H_hours(H, 1, Ds)/setk, H/Ds, '--', label='1 hour, H=10', c=colormap(0))
        plt.plot(UD_from_H_hours(H, 5, Ds)/setk, H/Ds, '--',label='5 hour, H=10', c=colormap(0.1))
        plt.plot(UD_from_H_hours(H, 10, Ds)/setk, H/Ds, '--',label='10 hour, H=10', c=colormap(0.2))
        plt.plot(UD_from_H_hours(H, 30, Ds)/setk, H/Ds, '--',label='30 hour, H=10', c=colormap(0.3))
        # plt.plot(UD_from_H_hours(Hs, 1, D/5.477), Hs/(D/5.477), label='1 hour, D=0.0365')
        plt.xlim(min(U_unique)*D/setk, max(U_unique)*D/setk)
        plt.ylim(min(H_unique)/D, max(H_unique)/D)
        # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 40, D), label='40 hour')
        # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 80, D), label='80 hour')
        # UD2H = np.logspace(np.log10(min(df['U']*0.2**2/df['H'])), np.log10(max(df['U']*0.2**2/df['H'])), 10)
        # for val in UD2H:
        #     plt.plot(Hs/0.2, Hs/0.2*val, label='%0.2e' % val)
        # plt.ylim([min(df['U'].unique())*0.2, max(df['U'].unique())*0.2])
        ax.set_aspect('equal', 'box')
        
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(label='T FOM', location = 'left')
        plt.legend(loc='upper center', bbox_to_anchor=(1.5, 1),ncol=1)
        # plt.legend()
        # plt.legend(title='H/U/D$^2$, D=0.2')
        plt.title('k = %0.0f' % setk)
        ax.set_anchor('W')
        plt.tight_layout()
        # plt.figure(num=2,clear=True)
        # print(max(df['Pe']), min(df['Pe']))
        # plt.plot(df['Pe'], df['FOM'],'.')
        # plt.xlabel('UD$^2$/H $\propto$ D$^2$/hours')
        # plt.ylabel('T FOM')
        # plt.tight_layout()
        # plt.figure(num=3,clear=True)
        # plt.plot(df['hours'], df['FOM'],'.')
        # plt.xlabel('hours')
        # plt.ylabel('T FOM')
        # plt.tight_layout()
        # plt.xlim([-10,100])
        print(np.log10(U_unique*D))
        print(np.log10(H_unique/D))
        print(df)
        # if setk==3:
            # f = interpolate.interp2d(np.log10(df['U']*D), np.log10(df['H']/D), df['FOM'], kind='linear')
        f = interpolate.RegularGridInterpolator((np.log10(H_unique/D), np.log10(U_unique*D)), df['FOM'].values.reshape(len(H_unique), len(U_unique)))
        # else:
            # f = interpolate.RegularGridInterpolator((np.log10(H_unique/D), np.log10(U_unique*D)), df['FOM'].values.reshape(len(H_unique), len(U_unique)))
        # fig = plt.figure(num=5,clear=True)
        ax = fig.add_subplot(223)
        hours_loop = [1, 5, 10, 30]
        # H_loop = np.array([2,4,6,8,10,12,14,16,18,20])
        H_loop = np.linspace(2,20,100)
        for hour in hours_loop:
            H_FOMS = []
            for H in H_loop:
                ynew = np.log10(UD_from_H_hours(H, hour, D))
                xnew = np.log10(H/D)
                try:
                    znew = f(np.array([[xnew, ynew]]))
                except ValueError:
                    znew = np.nan
                H_FOMS.append(znew)
            plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour)
        plt.title('T FOM for D = 0.2')
        plt.legend()
        plt.xlabel('H')
        plt.ylabel('FOM')
        plt.grid()
        plt.tight_layout()

        # fig = plt.figure(num=6,clear=True)
        ax = fig.add_subplot(224)
        hours_loop = [1, 5, 10, 30]
        D_loop = np.linspace(0.01, 1, 100)
        H = 10
        for hour in hours_loop:
            D_FOMS = []
            for D in D_loop:
                ynew = np.log10(UD_from_H_hours(H, hour, D))
                xnew = np.log10(H/D)
                try:
                    znew = f(np.array([[xnew, ynew]]))
                except ValueError:
                    znew = np.nan
                D_FOMS.append(znew)
            plt.plot(D_loop, D_FOMS, label='%0.0f hours' % hour)
        plt.title('T FOM for H = 10')
        plt.legend()
        plt.xlabel('D')
        plt.ylabel('FOM')
        plt.grid()
        plt.tight_layout()
        
        
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.savefig('5series_discharge/sweep_HD_UD_k%0.0f.png' % setk, dpi=300)
        # plt.show()
    plt.figure(num=10)
    plt.xlabel('UD/k')
    plt.ylabel('H/D')
    plt.colorbar()

def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (i, j) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)


def polyfit_HD_UD():
    df_all = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_all_1.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    k = 10
    df = df_all[df_all['k']==k]
    D = df['D'].unique()[0]
    data = []
    H_unique = np.array(sorted(df['H'].unique()))
    U_unique = np.array(sorted(df['U'].unique()))
    for H in H_unique:
        for U in U_unique:
            df_i = df[(df['H'] == H) & (df['U'] == U)]
            try:
                time = df_i['time']/max(df_i['time'])*2.22
            except:
                continue
            Tout = (df_i['outlet_T']-2170)/500
            D = df_i['D'].unique()[0]
            k = df_i['k'].unique()[0]
            hours = hours_from_H_U(H, U)
            Pe = U*D**2/H
            FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
            data.append([np.log10(H),np.log10(U),FOM, hours, Pe])
    df_data = pd.DataFrame(data, columns=['H', 'U', 'FOM', 'hours', 'Pe'])
    x = np.array(sorted(df_data['H'].unique()/D))
    y = np.array(sorted(df_data['U'].unique()*D/k))
    z = df_data['FOM'].values.reshape(len(x), len(y))
    kx = 4
    ky = 4
    soln, residuals, rank, s = polyfit2d(x, y, z, kx=kx, ky=ky, order=None)
    print(soln, residuals, rank, s)
    X,Y = np.meshgrid(x,y)
    fitted_surf = np.polynomial.polynomial.polyval2d(X, Y, soln.reshape((kx+1, ky+1)))
    print(fitted_surf)
    plt.figure(num=10)
    # plt.plot(x, y, 'o')
    # plt.contour(x, y, fitted_surf, 10)
    # plt.matshow(fitted_surf)
    plt.pcolormesh(X, Y, z)
    plt.colorbar()
    plt.figure(num=11)
    plt.pcolormesh(X, Y, fitted_surf)
    plt.colorbar()
    pass
    
def sweep_Pe_H_U_2():
    df = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_30hrs.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    data = []
    cmap = mpl.cm.get_cmap('nipy_spectral')
    for H in df['H'].unique():
        for U in df['U'].unique():
            df_i = df[(df['H'] == H) & (df['U'] == U)]
            try:
                time = df_i['time']/max(df_i['time'])*2.22
            except:
                continue
            Tout = (df_i['outlet_T']-2173)/500
            if U == 1e-3:
                continue
            D = df_i['D'].unique()[0]
            k = df_i['k'].unique()[0]
            hours = hours_from_H_U(H, U)
            Pe = U*D**2/H
            # print(Pe)
            FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
            data.append([H,U,FOM, hours, Pe])
            plt.figure(num=6)
            Pe_norm = np.sqrt((Pe-2.5e-5)/(0.04-2.4e-5))
            # if abs((Pe_norm*10) - np.round(Pe_norm*10))< 0.05:
            plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k), color=cmap(Pe_norm*0.9))
    plt.figure(num=1)
    df = pd.DataFrame(data, columns=['H', 'U', 'FOM','hours','Pe'])
    plt.scatter(df['H']/0.2, df['U']*0.2, c=df['FOM'])
    plt.colorbar()
    plt.figure(num=2)
    plt.plot(df['Pe'], df['FOM'],'.',c='r')
    plt.figure(num=3)
    plt.plot(df['hours'], df['FOM'],'.', c='r')


def sweep_HD_UDk_k():
    # df = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df2 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_30hrs.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df3 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highU.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df4 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highH.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    df_all = pd.read_csv('../data/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # setk = 10
    fig10 = plt.figure(num=10, clear=True, figsize=(25,2))
    # ax10 = fig10.add_subplot(111)
    
    for index, setk in enumerate(df_all['k'].unique()):
        # for index, setk in enumerate([1]):
        D = 0.2
        print(setk, D)
        df = df_all[df_all['k']==setk]
        # df = df.append(df2).append(df3).append(df4)
        data = []
        cmap = mpl.cm.get_cmap('nipy_spectral')
        H_D_unique = np.array(sorted(df['H/D'].unique()))
        U_D_k_unique = np.array(sorted(df['UD/k'].unique()))
        for H in H_D_unique:
            for U in U_D_k_unique:
                df_i = df[(df['H/D'] == H) & (df['UD/k'] == U)]
                try:
                    time = df_i['time']/max(df_i['time'])*2
                except:
                    data.append([H,U,np.nan, np.nan, np.nan])
                    continue
                Tout = (df_i['outlet_T']-2170)/500
                # if U == 1e-3:
                #     continue
                # D = df_i['D'].unique()[0]
                k = df_i['k'].unique()[0]
                hours = hours_from_H_U(H, U)
                Pe = U*D**2/H
                # print(Pe)
                FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
                data.append([H,U,FOM, hours, Pe])
                # plt.figure(num=4)
                Pe_norm = np.sqrt((Pe)/(0.04))
                # if abs((Pe_norm*10) - np.round(Pe_norm*10))< 0.05:
                # plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k), color=cmap(Pe_norm*0.9))
        # print(sorted(H_D_unique))
        # print(sorted(U_unique))
        fig = plt.figure(num=index+1,clear=True,figsize=(10,8))
        ax = fig.add_subplot(211)
        # ax = plt.axes(projection='3d')
        df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'FOM','hours','Pe'])
        U,H = np.meshgrid(sorted(U_D_k_unique), sorted(H_D_unique))
        # plt.contourf(H/D, U*D, df['FOM'].values.reshape(len(H_D_unique), len(U_unique)), cmap='viridis')
        colormap = mpl.cm.get_cmap('tab10')
        plt.figure(num=10)
        ax10 = fig10.add_subplot(1,5,index+1)
        plt.contourf(U, H, df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)), levels=np.linspace(0,1,101), cmap='viridis')
        # plt.scatter(U*D/setk, H/D, df['FOM'].values.reshape(len(H_D_unique), len(U_unique)), cmap='viridis')
        ax10.set_yscale('log')
        ax10.set_xscale('log')
        ax10.set_title('k = %0.0f' % setk, fontsize=20)
        plt.tight_layout()
        if setk == 10:
            plt.ylabel('$\\dfrac{L}{D}$', rotation=0)
            plt.colorbar(label = '$FOM_T$')
        else:
            plt.colorbar()
        # plt.colorbar()
        plt.figure(num=index+1)
        plt.contourf(U, H, df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)), levels=np.linspace(0,1,1001, endpoint=True), cmap='viridis')
        # plt.scatter(df['U']*0.2, df['H']/0.2, c=df['FOM'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.ylabel('$\\dfrac{L}{D}$', rotation=0)
        plt.xlabel('$\\dfrac{UD}{k}$')
        # Hs = np.linspace(min(df['H/D'].unique()), max(df['H/D'].unique()), 100)
        # Hs = np.linspace(min(df['H'].unique()), max(df['H'].unique()), 100)
        Hs = np.linspace(0.1, 100, 100)
        # plt.plot(UD_from_H_hours(Hs, 1, D)/setk, Hs/D, label='1 hour, D=0.2', c=colormap(0))
        # plt.plot(UD_from_H_hours(Hs, 5, D)/setk, Hs/D, label='5 hour, D=0.2', c=colormap(0.1))
        # plt.plot(UD_from_H_hours(Hs, 10, D)/setk, Hs/D, label='10 hour, D=0.2', c=colormap(0.2))
        # plt.plot(UD_from_H_hours(Hs, 30, D)/setk, Hs/D, label='30 hour, D=0.2', c=colormap(0.3))
        # set H = 10, sweep hours and D
        H = 10
        Ds = np.linspace(0.01, 100, 1000)
        # plt.plot(UD_from_H_hours(H, 1, Ds)/setk, H/Ds, '--', label='1 hour, L=10', c=colormap(0))
        # plt.plot(UD_from_H_hours(H, 5, Ds)/setk, H/Ds, '--',label='5 hour, L=10', c=colormap(0.1))
        # plt.plot(UD_from_H_hours(H, 10, Ds)/setk, H/Ds, '--',label='10 hour, L=10', c=colormap(0.2))
        # plt.plot(UD_from_H_hours(H, 30, Ds)/setk, H/Ds, '--',label='30 hour, L=10', c=colormap(0.3))
        # plt.plot(UD_from_H_hours(Hs, 1, D/5.477), Hs/(D/5.477), label='1 hour, D=0.0365')
        plt.xlim(min(U_D_k_unique), max(U_D_k_unique))
        plt.ylim(min(H_D_unique), max(H_D_unique))
        # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 40, D), label='40 hour')
        # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 80, D), label='80 hour')
        # UD2H = np.logspace(np.log10(min(df['U']*0.2**2/df['H'])), np.log10(max(df['U']*0.2**2/df['H'])), 10)
        # for val in UD2H:
        #     plt.plot(Hs/0.2, Hs/0.2*val, label='%0.2e' % val)
        # plt.ylim([min(df['U'].unique())*0.2, max(df['U'].unique())*0.2])
        # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4)
        ax.set_aspect('equal', 'box')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.legend()
        # plt.legend(title='H/U/D$^2$, D=0.2')
        plt.title('k = %0.0f' % setk)
        ax.set_anchor('W')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("left", size="5%", pad=0.05)
        # cbaxes = fig.add_axes([-0.03, 0.521, 0.03, 0.435]) 
        plt.colorbar(label='$FOM_T$')
        # cbaxes.yaxis.tick_left()
        # cbaxes.yaxis.set_label_position('left')
        plt.tight_layout()
        # plt.figure(num=2,clear=True)
        # print(max(df['Pe']), min(df['Pe']))
        # plt.plot(df['Pe'], df['FOM'],'.')
        # plt.xlabel('UD$^2$/H $\propto$ D$^2$/hours')
        # plt.ylabel('T FOM')
        # plt.tight_layout()
        # plt.figure(num=3,clear=True)
        # plt.plot(df['hours'], df['FOM'],'.')
        # plt.xlabel('hours')
        # plt.ylabel('T FOM')
        # plt.tight_layout()
        # plt.xlim([-10,100])
        print(np.log10(U_D_k_unique))
        print(np.log10(H_D_unique))
        print(df)
        # if setk==3:
            # f = interpolate.interp2d(np.log10(df['U']*D), np.log10(df['H']/D), df['FOM'], kind='linear')
        # f = interpolate.RegularGridInterpolator((np.log10(H_D_unique), np.log10(U_D_k_unique)), df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)))
        # # else:
        #     # f = interpolate.RegularGridInterpolator((np.log10(H_D_unique/D), np.log10(U_unique*D)), df['FOM'].values.reshape(len(H_D_unique), len(U_unique)))
        # # fig = plt.figure(num=5,clear=True)
        # ax = fig.add_subplot(223)
        # hours_loop = [1, 5, 10, 30]
        # # H_loop = np.array([2,4,6,8,10,12,14,16,18,20])
        # H_loop = np.linspace(2,20,100)
        # for hour in hours_loop:
        #     H_FOMS = []
        #     for H in H_loop:
        #         ynew = np.log10(UD_from_H_hours(H, hour, D)/setk)
        #         xnew = np.log10(H/D)
        #         try:
        #             znew = f(np.array([[xnew, ynew]]))
        #         except ValueError:
        #             znew = np.nan
        #         H_FOMS.append(znew)
        #     plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour)
        # plt.title('$FOM_T$ for D = 0.2')
        # plt.ylim([0,1])
        # plt.legend()
        # plt.xlabel('L')
        # plt.ylabel('FOM')
        # # plt.grid()
        # plt.tight_layout()

        # # fig = plt.figure(num=6,clear=True)
        # ax = fig.add_subplot(224)
        # hours_loop = [1, 5, 10, 30]
        # D_loop = np.linspace(0.01, 1, 100)
        # H = 10
        # for hour in hours_loop:
        #     D_FOMS = []
        #     for D in D_loop:
        #         ynew = np.log10(UD_from_H_hours(H, hour, D)/setk)
        #         xnew = np.log10(H/D)
        #         try:
        #             znew = f(np.array([[xnew, ynew]]))
        #         except ValueError:
        #             znew = np.nan
        #         D_FOMS.append(znew)
        #     plt.plot(D_loop, D_FOMS, label='%0.0f hours' % hour)
        # plt.title('$FOM_T$ for L = 10')
        # plt.legend()
        # plt.ylim([0,1])
        # plt.xlabel('D')
        # plt.ylabel('FOM')
        # # plt.grid()
        # plt.tight_layout()
        
        
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.savefig('../plots/sweep_HD_UDk_k%0.0f.png' % setk, dpi=300)
        # plt.show()
    plt.figure(num=10)
    plt.xlabel('$\\dfrac{UD}{k}$')
    plt.tight_layout()
    plt.savefig('../plots/sweep_HD_UDk_k_all.png', dpi=300)
    # plt.ylabel('H/D')
    # plt.colorbar()


def sweep_HD_UDk_k_hoursfrac():
    # df = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df2 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_30hrs.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df3 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highU.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # df4 = pd.read_csv('5series_discharge/sweep_Pe_HD_UD_k10_highH.csv', names=['H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    df_all = pd.read_csv('5series_discharge/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    # setk = 10
    
    # ax10 = fig10.add_subplot(111)
    
    for hours_frac in [0.5,1,2]:
        fig10 = plt.figure(num=10, clear=True, figsize=(5,10))
        fig11 = plt.figure(num=11, clear=True, figsize=(5,10))
        for index, setk in enumerate(df_all['k'].unique()):
            D = 0.2
            print(setk, D)
            df = df_all[df_all['k']==setk]
            # df = df.append(df2).append(df3).append(df4)
            data = []
            cmap = mpl.cm.get_cmap('nipy_spectral')
            H_D_unique = np.array(sorted(df['H/D'].unique()))
            U_D_k_unique = np.array(sorted(df['UD/k'].unique()))
            
            for H in H_D_unique:
                for U in U_D_k_unique:
                    df_i = df[(df['H/D'] == H) & (df['UD/k'] == U)]
                    hours = hours_from_H_U(H*D, U*setk/D)*hours_frac
                    try:
                        time = df_i['time']/(hours*3600)
                    except:
                        continue
                    Tout = (df_i['outlet_T']-2170)/500
                    # if U == 1e-3:
                    #     continue
                    # D = df_i['D'].unique()[0]
                    k = df_i['k'].unique()[0]
                    Pe = U*D**2/H
                    # print(Pe)
                    # integrate Tout over time from 0 to 1
                    Tout1 = Tout[time<=1]
                    time1 = time[time<=1]
                    # plt.plot(time1, Tout1)
                    # plt.show()
                    FOM1 = np.trapz(Tout1, time1)
                    Tout2 = Tout[time<=2]
                    if len(Tout2) < 2*len(Tout1)-1:
                        # time = np.pad(time, (0, 2*len(Tout1)-len(Tout2)), 'linear_ramp')
                        # print(len(Tout1), len(Tout2))
                        Tout2 = np.pad(Tout2, (0, 2*len(Tout1)-len(Tout2)), 'linear_ramp', end_values=(0,0))
                        time = np.linspace(0,2,2*len(Tout1))
                        # plt.plot(time, Tout2)
                        # plt.title('H/D = %0.2f, UD/k = %0.2e, k = %0.2f, frac=%0.2f' % (H, U, k, hours_frac))
                        # plt.show()
                    # print(len(Tout2), len(time))
                    # print(max(df_i['time']), hours*3600, max(time))
                    FOM2 = np.trapz(Tout2[time<=2], time[time<=2])
                    FOM2 = 1 - (np.abs(1 - FOM2))
                    data.append([H,U,FOM1, FOM2, hours, Pe])
                    # plt.figure(num=4)
                    Pe_norm = np.sqrt((Pe)/(0.04))
                    # if abs((Pe_norm*10) - np.round(Pe_norm*10))< 0.05:
                    # plt.plot(time, Tout, '-', label='H %0.0fm, D %0.1fm, U %0.2em/s, k %0.0f' % (H, D, U, k), color=cmap(Pe_norm*0.9))
            # print(sorted(H_D_unique))
            # print(sorted(U_unique))
            fig = plt.figure(num=1,clear=True,figsize=(10,8))
            ax = fig.add_subplot(211)
            # ax = plt.axes(projection='3d')
            df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'FOM','FOM2','hours','Pe'])
            U,H = np.meshgrid(sorted(U_D_k_unique), sorted(H_D_unique))
            # plt.contourf(H/D, U*D, df['FOM'].values.reshape(len(H_D_unique), len(U_unique)), cmap='viridis')
            colormap = mpl.cm.get_cmap('tab10')
            plt.figure(num=10)
            ax10 = fig10.add_subplot(5,1,index+1)
            plt.contourf(U, H, df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)), levels=np.linspace(0,1,101), cmap='viridis', extend='both')
            # plt.scatter(U*D/setk, H/D, df['FOM'].values.reshape(len(H_D_unique), len(U_unique)), cmap='viridis')
            ax10.set_yscale('log')
            ax10.set_xscale('log')
            ax10.set_title('k = %0.0f' % setk)
            plt.tight_layout()
            if setk == 10:
                plt.ylabel('H/D')
                plt.colorbar(label = 'T FOM')
            else:
                plt.colorbar()
            # plt.colorbar()
            plt.figure(num=11)
            ax11 = fig11.add_subplot(5,1,index+1)
            plt.contourf(U, H, df['FOM2'].values.reshape(len(H_D_unique), len(U_D_k_unique)), levels=np.linspace(0,1,101), cmap='viridis', extend='both')
            # plt.scatter(U*D/setk, H/D, df['FOM'].values.reshape(len(H_D_unique), len(U_unique)), cmap='viridis')
            ax11.set_yscale('log')
            ax11.set_xscale('log')
            ax11.set_title('k = %0.0f' % setk)
            plt.tight_layout()
            if setk == 10:
                plt.ylabel('H/D')
                plt.colorbar(label = 'T FOM')
            else:
                plt.colorbar()
            # plt.colorbar()
            plt.figure(num=1)
            plt.contourf(U, H, df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)), levels=np.linspace(0,1,1001, endpoint=True), cmap='viridis', extend='both')
            # plt.scatter(df['U']*0.2, df['H']/0.2, c=df['FOM'])
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.ylabel('H/D')
            plt.xlabel('U*D/k')
            # Hs = np.linspace(min(df['H/D'].unique()), max(df['H/D'].unique()), 100)
            # Hs = np.linspace(min(df['H'].unique()), max(df['H'].unique()), 100)
            Hs = np.linspace(0.1, 100, 100)
            plt.plot(UD_from_H_hours(Hs, 1, D)/setk, Hs/D, label='1 hour, D=0.2', c=colormap(0))
            plt.plot(UD_from_H_hours(Hs, 5, D)/setk, Hs/D, label='5 hour, D=0.2', c=colormap(0.1))
            plt.plot(UD_from_H_hours(Hs, 10, D)/setk, Hs/D, label='10 hour, D=0.2', c=colormap(0.2))
            plt.plot(UD_from_H_hours(Hs, 30, D)/setk, Hs/D, label='30 hour, D=0.2', c=colormap(0.3))
            # set H = 10, sweep hours and D
            H = 10
            Ds = np.linspace(0.01, 100, 1000)
            plt.plot(UD_from_H_hours(H, 1, Ds)/setk, H/Ds, '--', label='1 hour, H=10', c=colormap(0))
            plt.plot(UD_from_H_hours(H, 5, Ds)/setk, H/Ds, '--',label='5 hour, H=10', c=colormap(0.1))
            plt.plot(UD_from_H_hours(H, 10, Ds)/setk, H/Ds, '--',label='10 hour, H=10', c=colormap(0.2))
            plt.plot(UD_from_H_hours(H, 30, Ds)/setk, H/Ds, '--',label='30 hour, H=10', c=colormap(0.3))
            # plt.plot(UD_from_H_hours(Hs, 1, D/5.477), Hs/(D/5.477), label='1 hour, D=0.0365')
            plt.xlim(min(U_D_k_unique), max(U_D_k_unique))
            plt.ylim(min(H_D_unique), max(H_D_unique))
            # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 40, D), label='40 hour')
            # plt.plot(Hs/0.2, UD_from_H_hours(Hs, 80, D), label='80 hour')
            # UD2H = np.logspace(np.log10(min(df['U']*0.2**2/df['H'])), np.log10(max(df['U']*0.2**2/df['H'])), 10)
            # for val in UD2H:
            #     plt.plot(Hs/0.2, Hs/0.2*val, label='%0.2e' % val)
            # plt.ylim([min(df['U'].unique())*0.2, max(df['U'].unique())*0.2])
            ax.set_aspect('equal', 'box')
            
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.colorbar(label='T FOM', location = 'left')
            plt.legend(loc='upper center', bbox_to_anchor=(1.25, 1),ncol=1)
            # plt.legend()
            # plt.legend(title='H/U/D$^2$, D=0.2')
            plt.title('k = %0.0f' % setk)
            ax.set_anchor('W')
            plt.tight_layout()
            # plt.figure(num=2,clear=True)
            # print(max(df['Pe']), min(df['Pe']))
            # plt.plot(df['Pe'], df['FOM'],'.')
            # plt.xlabel('UD$^2$/H $\propto$ D$^2$/hours')
            # plt.ylabel('T FOM')
            # plt.tight_layout()
            # plt.figure(num=3,clear=True)
            # plt.plot(df['hours'], df['FOM'],'.')
            # plt.xlabel('hours')
            # plt.ylabel('T FOM')
            # plt.tight_layout()
            # plt.xlim([-10,100])
            print(np.log10(U_D_k_unique))
            print(np.log10(H_D_unique))
            print(df)
            # if setk==3:
                # f = interpolate.interp2d(np.log10(df['U']*D), np.log10(df['H']/D), df['FOM'], kind='linear')
            f = interpolate.RegularGridInterpolator((np.log10(H_D_unique), np.log10(U_D_k_unique)), df['FOM'].values.reshape(len(H_D_unique), len(U_D_k_unique)))
            # else:
                # f = interpolate.RegularGridInterpolator((np.log10(H_D_unique/D), np.log10(U_unique*D)), df['FOM'].values.reshape(len(H_D_unique), len(U_unique)))
            # fig = plt.figure(num=5,clear=True)
            ax = fig.add_subplot(223)
            hours_loop = [1, 5, 10, 30]
            # H_loop = np.array([2,4,6,8,10,12,14,16,18,20])
            H_loop = np.linspace(2,20,100)
            for hour in hours_loop:
                H_FOMS = []
                for H in H_loop:
                    ynew = np.log10(UD_from_H_hours(H, hour, D)/setk)
                    xnew = np.log10(H/D)
                    try:
                        znew = f(np.array([[xnew, ynew]]))
                    except ValueError:
                        znew = np.nan
                    H_FOMS.append(znew)
                plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour)
            plt.title('T FOM for D = 0.2')
            plt.ylim([0,1])
            plt.legend()
            plt.xlabel('H')
            plt.ylabel('FOM')
            plt.grid()
            plt.tight_layout()

            # fig = plt.figure(num=6,clear=True)
            ax = fig.add_subplot(224)
            hours_loop = [1, 5, 10, 30]
            D_loop = np.linspace(0.01, 1, 100)
            H = 10
            for hour in hours_loop:
                D_FOMS = []
                for D in D_loop:
                    ynew = np.log10(UD_from_H_hours(H, hour, D)/setk)
                    xnew = np.log10(H/D)
                    try:
                        znew = f(np.array([[xnew, ynew]]))
                    except ValueError:
                        znew = np.nan
                    D_FOMS.append(znew)
                plt.plot(D_loop, D_FOMS, label='%0.0f hours' % hour)
            plt.title('T FOM for H = 10')
            plt.legend()
            plt.ylim([0,1])
            plt.xlabel('D')
            plt.ylabel('FOM')
            plt.grid()
            plt.tight_layout()
            
            
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # plt.savefig('5series_discharge/sweep_HD_UDk_k%0.0f.png' % setk, dpi=300)
            # plt.show()
        plt.figure(num=10)
        plt.xlabel('UD/k')
        plt.tight_layout()
        plt.savefig('5series_discharge/sweep_HD_UDk_k_all_hf_%0.2f.png'%hours_frac, dpi=300)
        plt.figure(num=11)
        plt.xlabel('UD/k')
        plt.tight_layout()
        plt.savefig('5series_discharge/sweep_HD_UDk_k_all_hf_%0.2f_FOM2.png'%hours_frac, dpi=300)
        plt.show()
        # plt.ylabel('H/D')
        # plt.colorbar()


def interp_HD_UDk_k():
    df_all = pd.read_csv('5series_discharge/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    data = []
    k_unique = np.array(sorted(df_all['k'].unique()))
    for index, setk in enumerate(k_unique):
        D = 0.2
        print(setk, D)
        df = df_all[df_all['k']==setk]
        cmap = mpl.cm.get_cmap('nipy_spectral')
        H_D_unique = np.array(sorted(df['H/D'].unique()))
        U_D_k_unique = np.array(sorted(df['UD/k'].unique()))
        for H in H_D_unique:
            for U in U_D_k_unique:
                df_i = df[(df['H/D'] == H) & (df['UD/k'] == U)]
                try:
                    time = df_i['time']/max(df_i['time'])*2
                except:
                    continue
                Tout = (df_i['outlet_T']-2170)/500
                k = df_i['k'].unique()[0]
                hours = hours_from_H_U(H, U)
                Pe = U*D**2/H
                FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
                data.append([H,U,k,FOM, hours, Pe])

    df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'k', 'FOM','hours','Pe'])
    U,H = np.meshgrid(sorted(U_D_k_unique), sorted(H_D_unique))
    
    f = interpolate.RegularGridInterpolator((np.log10(k_unique), np.log10(H_D_unique), np.log10(U_D_k_unique)), df['FOM'].values.reshape(len(k_unique), len(H_D_unique), len(U_D_k_unique)))
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(111)
    hours_loop = [1, 5, 10, 30]
    H_loop = np.linspace(2,20,100)
    for hour in hours_loop:
        H_FOMS = []
        for H in H_loop:
            wnew = np.log10(setk)
            xnew = np.log10(H/D)
            ynew = np.log10(UD_from_H_hours(H, hour, D)/setk)
            try:
                znew = f(np.array([[wnew, xnew, ynew]]))
            except ValueError:
                znew = np.nan
            H_FOMS.append(znew)
        plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour) 
    plt.legend()
    plt.xlabel('H')
    plt.ylabel('FOM')
    plt.grid()
    plt.tight_layout()

    fig = plt.figure(num=2, clear=True)
    ax = fig.add_subplot(111)
    hours_loop = [1, 5, 10, 30]
    k_loop = np.linspace(1,100,100)
    H_D = 50
    UD_ks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for UD_k in UD_ks:
        H_FOMS = []
        for setk in k_loop:
            wnew = np.log10(setk)
            xnew = np.log10(H_D)
            ynew = np.log10(UD_k)
            try:
                znew = f(np.array([[wnew, xnew, ynew]]))
            except ValueError:
                znew = np.nan
            H_FOMS.append(znew)
        plt.plot(k_loop, H_FOMS, label='UD/k = %0.1e' % UD_k) 
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('FOM')
    plt.grid()
    plt.tight_layout()


def KRR_interp_HD_UDk_k():
    df_all = pd.read_csv('../data/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    data = []
    k_unique = np.array(sorted(df_all['k'].unique()))
    for index, setk in enumerate(k_unique):
        D = 0.2
        # print(setk, D)
        df = df_all[df_all['k']==setk]
        cmap = mpl.cm.get_cmap('nipy_spectral')
        H_D_unique = np.array(sorted(df['H/D'].unique()))
        U_D_k_unique = np.array(sorted(df['UD/k'].unique()))
        for H in H_D_unique:
            for U in U_D_k_unique:
                df_i = df[(df['H/D'] == H) & (df['UD/k'] == U)]
                try:
                    time = df_i['time']/max(df_i['time'])*2
                except:
                    continue
                Tout = (df_i['outlet_T']-2170)/500
                k = df_i['k'].unique()[0]
                hours = hours_from_H_U(H, U)
                Pe = U*D**2/H
                FOM = np.trapz(Tout[:int(len(Tout)/2)], time[:int(len(time)/2)])
                data.append([H,U,k,FOM, hours, Pe])

    df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'k', 'FOM','hours','Pe'])
    # U,H = np.meshgrid(sorted(U_D_k_unique), sorted(H_D_unique))
    X = df[['H/D', 'UD/k', 'k']]
    y = df['FOM']
    X = np.log10(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    krr = KernelRidge(kernel='rbf', alpha=1e-10)
    krr.fit(X_train, y_train)
    # print(X_test)
    y_pred = krr.predict(X_test)
    print('R2 score: %0.2f' % r2_score(y_test, y_pred))
    print('MAE score: %0.2f' % mean_absolute_error(y_test, y_pred))
    plt.figure(num=12)
    plt.scatter(X_test['UD/k'], X_test['H/D'], c=y_test, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.figure(num=13)
    plt.scatter(X_test['UD/k'], X_test['H/D'], c=y_pred, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    

    # f = interpolate.RegularGridInterpolator((np.log10(k_unique), np.log10(H_D_unique), np.log10(U_D_k_unique)), df['FOM'].values.reshape(len(k_unique), len(H_D_unique), len(U_D_k_unique)))

    fig = plt.figure(num=14, clear=True, figsize=(2.6,3))
    ax = fig.add_subplot(111)
    hours_loop = [1, 5, 10, 30]
    H_loop = np.linspace(2,20,100)
    setk=10
    for hour in hours_loop:
        H_FOMS = []
        for H in H_loop:
            wnew = H/D
            xnew = UD_from_H_hours(H, hour, D)/setk
            ynew = setk
            try:
                df_test = pd.DataFrame([[wnew, xnew, ynew]], columns=['H/D', 'UD/k', 'k'])
                X = np.log10(df_test)
                znew = krr.predict(X)
            except ValueError:
                znew = np.nan
            H_FOMS.append(znew)
        plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour) 
    plt.legend()
    plt.xlabel('L')
    plt.ylabel('FOM')
    plt.title('$FOM_T$ for D = 0.2')
    # plt.grid()
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig('../plots/sweep_HD_UDk_k_interp_KRR_H.pdf')

    fig = plt.figure(num=15, clear=True, figsize=(2.6,3))
    ax = fig.add_subplot(111)
    hours_loop = [1, 5, 10, 30]
    k_loop = np.linspace(1,100,100)
    H_set = 10
    # UD_ks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    Ds = np.linspace(0.01, 1, 100)
    for hour in hours_loop:
        H_FOMS = []
        for D in Ds:
            wnew = H_set/D
            xnew = UD_from_H_hours(H_set, hour, D)/setk
            ynew = setk
            try:
                df_test = pd.DataFrame([[wnew, xnew, ynew]], columns=['H/D', 'UD/k', 'k'])
                X = np.log10(df_test)
                znew = krr.predict(X)
            except ValueError:
                znew = np.nan
            H_FOMS.append(znew)
        plt.plot(Ds, H_FOMS, label='%0.0f hours' % hour) 
    plt.legend()
    plt.xlabel('D')
    plt.ylabel('FOM')
    plt.title('$FOM_T$ for L = 10')
    # plt.grid()
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig('../plots/sweep_HD_UDk_k_interp_KRR_D.pdf')

    return krr


def KRR_interp_HD_UDk_k_2():
    df_all = pd.read_csv('../data/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    data = []
    k_unique = np.array(sorted(df_all['k'].unique()))
    for index, setk in enumerate(k_unique):
        D = 0.2
        # print(setk, D)
        df = df_all[df_all['k']==setk]
        cmap = mpl.cm.get_cmap('nipy_spectral')
        H_D_unique = np.array(sorted(df['H/D'].unique()))
        U_D_k_unique = np.array(sorted(df['UD/k'].unique()))
        # hours_frac = 1
        for H in H_D_unique:
            for U in U_D_k_unique:
                for hours_frac in np.logspace(-2,1,20):
                    df_i = df[(df['H/D'] == H) & (df['UD/k'] == U)]
                    hours = hours_from_H_U(H*D, U*setk/D)*hours_frac
                    try:
                        time = df_i['time']/(hours*3600)
                    except:
                        continue
                    Tout = (df_i['outlet_T']-2170)/500
                    k = df_i['k'].unique()[0]
                    Pe = U*D**2/H
                    Tout1 = Tout[time<=1]
                    time1 = time[time<=1]
                    # plt.plot(time1, Tout1)
                    # plt.show()
                    FOM1 = np.trapz(Tout1, time1)
                    Tout2 = Tout[time<=2]
                    if len(Tout2) < 2*len(Tout1)-1:
                        # time = np.pad(time, (0, 2*len(Tout1)-len(Tout2)), 'linear_ramp')
                        # print(len(Tout1), len(Tout2))
                        Tout2 = np.pad(Tout2, (0, 2*len(Tout1)-len(Tout2)), 'linear_ramp', end_values=(0,0))
                        time = np.linspace(0,2,2*len(Tout1))
                        # plt.plot(time, Tout2)
                        # plt.title('H/D = %0.2f, UD/k = %0.2e, k = %0.2f, frac=%0.2f' % (H, U, k, hours_frac))
                        # plt.show()
                    # print(len(Tout2), len(time))
                    # print(max(df_i['time']), hours*3600, max(time))
                    FOM2 = np.trapz(Tout2[time<=2], time[time<=2])
                    data.append([np.log10(H),np.log10(U),np.log10(k),np.log10(hours_frac),FOM1, FOM2, hours, Pe])

    df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'k', 'hours_frac','FOM','FOM2','hours','Pe'])
    # U,H = np.meshgrid(sorted(U_D_k_unique), sorted(H_D_unique))
    X = df[['H/D', 'UD/k', 'k', 'hours_frac']]
    y = df['FOM']
    # X = np.log10(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    krr = KernelRidge(kernel='rbf', alpha=1e-10)
    krr.fit(X_train, y_train)
    # print(X_test)
    y_pred = krr.predict(X_test)
    print('R2 score: %0.2f' % r2_score(y_test, y_pred))
    print('MAE score: %0.2f' % mean_absolute_error(y_test, y_pred))
    plt.figure(num=2)
    plt.scatter(X_test['UD/k'], X_test['H/D'], c=y_test, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.figure(num=3)
    plt.scatter(X_test['UD/k'], X_test['H/D'], c=y_pred, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()

    y = df['FOM2']
    # X = np.log10(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    krr2 = KernelRidge(kernel='rbf', alpha=1e-10)
    krr2.fit(X_train, y_train)
    # print(X_test)
    y_pred = krr2.predict(X_test)
    print('R2 score: %0.2f' % r2_score(y_test, y_pred))
    print('MAE score: %0.2f' % mean_absolute_error(y_test, y_pred))
    

    # f = interpolate.RegularGridInterpolator((np.log10(k_unique), np.log10(H_D_unique), np.log10(U_D_k_unique)), df['FOM'].values.reshape(len(k_unique), len(H_D_unique), len(U_D_k_unique)))

    fig = plt.figure(num=4, clear=True)
    ax = fig.add_subplot(111)
    H_loop = np.linspace(2,20,10)
    D = 0.2
    desired_hours = 20
    U_loop = np.linspace(U_from_H_hours(1, desired_hours), U_from_H_hours(20, desired_hours), 10)
    k = 10
    data = []
    for H in H_loop:
        for U in U_loop:
            hours = hours_from_H_U(H, U)
            hours_frac = desired_hours/hours
            dfdata = [np.log10(H/D), np.log10(U*D/k), np.log10(k), np.log10(hours_frac)]
            df = pd.DataFrame([dfdata], columns=['H/D', 'UD/k', 'k', 'hours_frac'])
            FOM1 = krr.predict(df)
            if FOM1 < 0:
                print(U,H,U*D/k,H/D, hours_frac)
            # FOM1 = krr.predict([[np.log10(H/D), np.log10(U*D/k), np.log10(k), hours_frac]])
            # FOM2 = krr2.predict([[np.log10(H/D), np.log10(U*D/k), np.log10(k), hours_frac]])
            # FOM = 1 - (np.abs(1 - FOM1) + np.abs(1 - FOM2))
            data.append([H, U, FOM1])
    df = pd.DataFrame(data, columns=['H', 'U', 'FOM'])
    # plt.scatter(df['U'], df['H'], c=df['FOM'], cmap='viridis', vmin=0, vmax=1)
    U, H = np.meshgrid(sorted(df['U'].unique()), sorted(df['H'].unique()))
    plt.contourf(U,H, df['FOM'].values.reshape(len(H_loop), len(U_loop)), cmap='viridis', levels=np.linspace(0,1,101), extend='both')
    Hs = np.linspace(0,max(H_loop),100)
    plt.plot(U_from_H_hours(Hs, desired_hours), Hs, 'k--')
    plt.xlim(min(U_loop), max(U_loop))
    plt.ylim(min(H_loop), max(H_loop))
    plt.colorbar(label='T FOM 1')
    plt.xlabel('U')
    plt.ylabel('H')
    plt.tight_layout()
    plt.savefig('../plots/U_H_hours_FOM1.png')


    fig = plt.figure(num=5, clear=True)
    ax = fig.add_subplot(111)
    data = []
    for H in H_loop:
        for U in U_loop:
            hours = hours_from_H_U(H, U)
            desired_H = H_from_hours_U(desired_hours, U)
            hours_frac = desired_hours/hours
            dfdata = [np.log10(H/D), np.log10(U*D/k), np.log10(k), np.log10(hours_frac)]
            df = pd.DataFrame([dfdata], columns=['H/D', 'UD/k', 'k', 'hours_frac'])
            FOM = (2-krr2.predict(df))/1.44
            # FOM2 = H/desired_H
            # FOM = 1 - (np.abs(1 - FOM2))
            # FOM = H*D**2*np.pi/4
            # FOM = 1/(H/desired_H)
            data.append([H, U, FOM])
    df = pd.DataFrame(data, columns=['H', 'U', 'FOM'])
    # plt.scatter(df['U'], df['H'], c=df['FOM'], cmap='viridis', vmin=0, vmax=1)
    U, H = np.meshgrid(sorted(df['U'].unique()), sorted(df['H'].unique()))
    plt.contourf(U,H, df['FOM'].values.reshape(len(H_loop), len(U_loop)), cmap='viridis', levels=101, extend='both')
    Hs = np.linspace(0,max(H_loop),100)
    plt.plot(U_from_H_hours(Hs, desired_hours), Hs, 'k--')
    plt.xlim(min(U_loop), max(U_loop))
    plt.ylim(min(H_loop), max(H_loop))
    plt.colorbar(label='T FOM 2')
    plt.xlabel('U')
    plt.ylabel('H')
    plt.tight_layout()
    plt.savefig('../plots/U_H_hours_FOM2.png')

    fig = plt.figure(num=6, clear=True)
    ax = fig.add_subplot(111)
    data = []
    for H in H_loop:
        for U in U_loop:
            hours = hours_from_H_U(H, U)
            hours_frac = desired_hours/hours
            dfdata = [np.log10(H/D), np.log10(U*D/k), np.log10(k), np.log10(hours_frac)]
            df = pd.DataFrame([dfdata], columns=['H/D', 'UD/k', 'k', 'hours_frac'])
            FOM1 = krr.predict(df)
            FOM2 = krr2.predict(df)
            # FOM2 = 1/(H/desired_H)
            FOM = (FOM1 + (2-FOM2)/2.2)*1.51
            data.append([H, U, FOM])
    df = pd.DataFrame(data, columns=['H', 'U', 'FOM'])
    # plt.scatter(df['U'], df['H'], c=df['FOM'], cmap='viridis', vmin=0, vmax=1)
    U, H = np.meshgrid(sorted(df['U'].unique()), sorted(df['H'].unique()))
    plt.contourf(U,H, df['FOM'].values.reshape(len(H_loop), len(U_loop)), cmap='viridis', levels=101, extend='both')
    Hs = np.linspace(0,max(H_loop),100)
    plt.plot(U_from_H_hours(Hs, desired_hours), Hs, 'k--')
    plt.xlim(min(U_loop), max(U_loop))
    plt.ylim(min(H_loop), max(H_loop))
    plt.colorbar(label='T FOM')
    plt.xlabel('U')
    plt.ylabel('H')
    plt.tight_layout()
    plt.savefig('../plots/U_H_hours_FOM.png')

    # fig = plt.figure(num=5, clear=True)
    # ax = fig.add_subplot(111)
    # H_loop = np.linspace(1,20,10)
    # D = 0.2
    # U_loop = np.linspace(U_from_H_hours(1, 10), U_from_H_hours(20, 10), 10)
    # k = 10
    # desired_hours = 10
    # data = []
    # for H in H_loop:
    #     for U in U_loop:
    #         hours = hours_from_H_U(H, U)
    #         hours_frac = desired_hours/hours
    #         FOM2 = krr2.predict([[np.log10(H/D), np.log10(U*D/k), np.log10(k), hours_frac]])
    #         data.append([H, U, FOM2])
    # df = pd.DataFrame(data, columns=['H', 'U', 'FOM'])
    # # plt.scatter(df['U'], df['H'], c=df['FOM'], cmap='viridis', vmin=0, vmax=1)
    # U, H = np.meshgrid(sorted(df['U'].unique()), sorted(df['H'].unique()))
    # plt.contourf(U,H, df['FOM'].values.reshape(len(H_loop), len(U_loop)), cmap='viridis', levels=np.linspace(0,2,100))
    # Hs = np.linspace(0,max(H_loop),100)
    # plt.plot(U_from_H_hours(Hs, desired_hours), Hs, 'k--')
    # plt.xlim(min(U_loop), max(U_loop))
    # plt.ylim(min(H_loop), max(H_loop))
    # plt.colorbar()

    # fig = plt.figure(num=4, clear=True)
    # ax = fig.add_subplot(111)
    # hours_loop = [1, 5, 10, 30]
    # H_loop = np.linspace(2,20,100)
    # setk=30
    # for hour in hours_loop:
    #     H_FOMS = []
    #     for H in H_loop:
    #         wnew = np.log10(H/D)
    #         xnew = np.log10(UD_from_H_hours(H, hour, D)/setk)
    #         ynew = np.log10(setk)
    #         hours_frac = 1
    #         try:
    #             df_test = pd.DataFrame([[wnew, xnew, ynew, hours_frac]], columns=['H/D', 'UD/k', 'k','hours_frac'])
    #             X = df_test
    #             # X = np.log10(df_test)
    #             znew = krr.predict(X)
    #         except ValueError:
    #             znew = np.nan
    #         H_FOMS.append(znew)
    #     plt.plot(H_loop, H_FOMS, label='%0.0f hours' % hour) 
    # plt.legend()
    # plt.xlabel('H')
    # plt.ylabel('FOM')
    # plt.grid()
    # plt.ylim(0,1)
    # plt.tight_layout()
    # plt.savefig('5series_discharge/sweep_HD_UDk_k_interp_KRR_H.png', dpi=300)

    # fig = plt.figure(num=5, clear=True)
    # ax = fig.add_subplot(111)
    # hours_loop = [1, 5, 10, 30]
    # k_loop = np.linspace(1,100,100)
    # H_D = 50
    # UD_ks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # for UD_k in UD_ks:
    #     H_FOMS = []
    #     for setk in k_loop:
    #         wnew = np.log10(H_D)
    #         xnew = np.log10(UD_k)
    #         ynew = np.log10(setk)
    #         hours_frac = 1
    #         try:
    #             df_test = pd.DataFrame([[wnew, xnew, ynew, hours_frac]], columns=['H/D', 'UD/k', 'k','hours_frac'])
    #             X = df_test
    #             # X = np.log10(df_test)
    #             znew = krr.predict(X)
    #         except ValueError:
    #             znew = np.nan
    #         H_FOMS.append(znew)
    #     plt.plot(k_loop, H_FOMS, label='UD/k = %0.1e' % UD_k) 
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('FOM')
    # plt.grid()
    # plt.ylim(0,1)
    # plt.tight_layout()
    # plt.savefig('5series_discharge/sweep_HD_UDk_k_interp_KRR_k.png', dpi=300)

    return krr, krr2


def opt_H_D_from_hours_k(hours,k):
    plt.style.use(['science','grid'])
    fig = plt.figure(num=16, figsize=(4,3), clear=True)
    data = []
    H_sweep = np.linspace(1e-5,100,100)
    D_sweep = np.linspace(1e-5,1,100)
    # print(H_sweep)
    # print(D_sweep)
    # exit()
    for H in H_sweep:
        for D in D_sweep:
            U = U_from_H_hours(H, hours)
            data.append([H/D, U*D/k, k, H, D, U])
    df = pd.DataFrame(data, columns=['H/D', 'UD/k', 'k', 'H', 'D', 'U'])
    X = np.log10(df[['H/D', 'UD/k', 'k']])
    FOM = krr.predict(X)
    df['FOM'] = FOM
    df_all = pd.read_csv('../data/sweep_Pe_HD_UDk_k_all_new.csv', names=['UD/k', 'H/D', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
    max_H_D = df_all['H/D'].max()
    max_UD_k = df_all['UD/k'].max()
    # set FOM to nan if outside of range of data
    df.loc[df['H/D'] > max_H_D, 'FOM'] = np.nan
    df.loc[df['UD/k'] > max_UD_k, 'FOM'] = np.nan
    ax = fig.add_subplot(111)
    # plt.scatter(df['D'], df['H'], c=df['FOM'], cmap='viridis')
    # contourf of D, H, FOM in df
    D, H = np.meshgrid(sorted(df['D'].unique()), sorted(df['H'].unique()))
    # U, H = np.meshgrid(sorted(df['U'].unique()), sorted(df['H'].unique()))
    FOM = df['FOM'].values.reshape(len(df['D'].unique()), len(df['H'].unique()))
    ctf = plt.contourf(D, H, FOM, levels=np.linspace(0,1,100), cmap='viridis', extend='both')
    # add second y axis with H translated to U
    plt.xlabel('D (m)')
    plt.ylabel('L (m)')
    ax.tick_params(axis='x', which='major', pad=5, direction='out', top=False)
    # ax.tick_params(zorder=10)
    plt.xlim(0, max(D_sweep))
    plt.ylim(0, max(H_sweep))
    plt.title('$\\tau$=%0.0f hours, k=%0.0f W/m/K' % (hours, k))
    # cbaxes = fig.add_axes([1, 0.15, 0.03, 0.7])
    plt.grid(axis='both')
    plt.colorbar(label='$FOM_T$', ax=ax, pad=0.25)
    plt.contour(D, H, FOM, levels=[0.9], colors='k')
    idx = 10
    idx_D = 20
    plt.hlines(idx, 0, 1, 'b', linestyles='dashed')
    plt.vlines(idx_D/100, 0, 100, 'r', linestyles='dashed')
    plt.tight_layout()
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(['%0.3f' % U_from_H_hours(H, hours) for H in ax.get_yticks()])
    ax2.set_ylabel('U (m/s)')
    plt.grid(axis='both')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # reshape FOM
    # FOM = FOM.reshape(len(H_sweep), len(D_sweep))
    # df['FOM'] = FOM
    # Hs, Ds = np.meshgrid(H_sweep, D_sweep)
    # plt.contourf(Hs, Ds, FOM, cmap='viridis')
    plt.tight_layout()
    for c in ctf.collections:
        c.set_edgecolor("face")
    plt.savefig('../plots/opt_H_D_from_hours_k_%0.0f_%0.0f.pdf' % (hours, k))

    fig = plt.figure(num=17, figsize=(3,3), clear=True)
    
    df_10 = df[df['H']==H_sweep[idx]]
    plt.plot(df_10['D'], df_10['FOM'],'b')
    plt.hlines(0.9, -1, 2, 'k')
    plt.xlabel('D (m)')
    plt.ylabel('$FOM_T$')
    plt.ylim(0.5,1)
    plt.xlim(0,1)
    plt.title('FOM$_T$ vs. D for L = %0.0f m' % idx)
    plt.tight_layout()
    plt.savefig('../plots/opt_H_D_from_hours_k_%0.0f_%0.0f_2.pdf' % (hours, k))

    df_02 = df[df['D']==D_sweep[idx_D]]
    plt.figure(num=18, figsize=(3,3), clear=True)
    plt.plot(df_02['H'], df_02['FOM'],'r')
    plt.hlines(0.9, -1, 101, 'k')
    plt.xlabel('L (m)')
    plt.ylabel('$FOM_T$')
    plt.ylim(0,1)
    plt.xlim(0,50)
    plt.title('FOM$_T$ vs. L for D = %0.1f m' % (idx_D/100))
    plt.tight_layout()
    plt.savefig('../plots/opt_H_D_from_hours_k_%0.0f_%0.0f_3.pdf' % (hours, k))

    pass

# data = import_data()
# plot_data(data)
# analyze_data(data)
# sweep_AR(10)
# df = pd.read_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e.csv'%(10, 1.67e-4), names=['index', 'H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
# df2 = pd.read_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e_2.csv'%(10, 1.67e-4), names=['index', 'H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T'], skiprows=5)
# df2['index'] = df2['index'] + 5
# df = df.append(df2)
# df.to_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e_new.csv'%(10, 1.67e-4), index=False)
# df['k'] = 30
# df = df[['index', 'H', 'D', 'U', 'k', 'time', 'inlet_T','outlet_T','block_T','Sn_T','bottom_right_T']]
# df.to_csv('5series_discharge/sweep_Pe_pi1_%0.0f_pi2_%0.2e.csv'%(10, 1.67e-4), index=False)
# sweep_Pe_2()
# for H in np.arange(2,18,2):
#     U = U_from_H_hours(H, 30)
#     mdot = mdot_from_U_D(U, 0.2)
#     print(U,mdot)
# print(flowrate_from_hours_D_H(30, 0.2, 10))
# U = U_from_H_hours(10, 30)
# print(U)
# print()
# sweep_Pe_H_U()
# plt.show()
# sweep_Pe_H_U_2()
# polyfit_HD_UD()
sweep_HD_UDk_k()
# plt.show()
# sweep_HD_UDk_k_hoursfrac()
# plt.show()
# interp_HD_UDk_k()
krr = KRR_interp_HD_UDk_k()
# plt.show()
# krr2, krr3 = KRR_interp_HD_UDk_k_2()
# plt.show()
opt_H_D_from_hours_k(30, 10)
plt.show()
# opt_H_D_from_hours_k(10, 10)
# plt.show()
# opt_H_D_from_hours_k(5, 10)
# plt.show()
# opt_H_D_from_hours_k(4, 10)
# plt.show()
# opt_H_D_from_hours_k(1, 10)
# plt.show()