import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import fsolve

plt.style.use(['science','grid'])

cp_Sn = 240
cp_air = 1170
df_data = pd.read_csv('data/COMSOL_5block_10_k_discharge_sweep_log_maxflow.csv', names=[
                        'hours', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
# hours = 30
# max_ff = 2
all_hours = df_data['hours'].unique()
hours = all_hours[6]
# print(all_hours)
all_max_ff = df_data['max_ff'].unique()
max_ff = all_max_ff[3]
P_FOMs = []
P_heights_add = []
index = 1

df = df_data[df_data['hours'] == hours]
df = df[df['max_ff'] == max_ff]
df_1 = df_data[df_data['hours'] == hours]
df_1 = df_1[df_1['max_ff'] == 1]
df['time'] = df['time']/3600
delT = df['outlet_T'] - df['inlet_T']
print(np.array(df['mass_flow'])[0])
Pout = df['mass_flow']*delT*cp_Sn
Poutnp = np.array(df['mass_flow']*delT*cp_Sn)

fig = plt.figure(num=index, clear=True, figsize=(7.5, 2.5))
index += 1
# fig.suptitle('hours = %0.1f, max$_{ff}$ = %0.1f' % (hours, max_ff))
ax = fig.add_subplot(131)
plt.plot(df['time'], delT,'b', label='f = %0.1f' % max_ff)
plt.xlabel('time (hr)')
plt.ylabel('delta T ($^{\circ}$C)')
plt.legend(fontsize=8)
plt.title(r'$\tau$ = %0.1f hours' % (hours), fontsize=10)
# plt.grid()
plt.tight_layout()
ax = fig.add_subplot(132)
plt.plot(df['time'], df['mass_flow'],'b')
plt.ylim(min(df['mass_flow'])*0.9, max(df['mass_flow'])*1.1)
plt.xlabel('time (hr)')
plt.ylabel('mass flow (kg/s)')
plt.title(r'$\tau$ = %0.1f hours' % (hours), fontsize=10)
# plt.grid()
plt.tight_layout()
ax = fig.add_subplot(133)
plt.plot(df['time'], Pout,'b', label='Thermal')
# plt.plot(df['time'], Pout*0.4,'r', label='Electrical')
plt.xlabel('time (hr)')
plt.ylabel('Power (W)')
plt.title(r'$\tau$ = %0.1f hours' % (hours), fontsize=10)
# plt.grid()
plt.tight_layout()
# ax = fig.add_subplot(224)
# plt.plot(df['time'], P_heights,'r', label='f = %0.1f' % max_ff)
# plt.plot(df_1['time'], P_heights_1,'r--', label='f = 1')
# plt.xlabel('time (hr)')
# plt.ylabel('TPV area (norm)')
# plt.legend(fontsize=8)
# plt.title(r'$\tau$ = %0.1f hours' % (hours), fontsize=10)
# plt.grid()
plt.tight_layout()
plt.savefig('plots/COMSOL_5block_10_k_discharge_sweep_maxflow_turbine_%0.1f_%0.1f.png' %
            (hours, max_ff), dpi=300)
# plt.savefig('../plots/COMSOL_5block_10_k_discharge_sweep_maxflow_%0.1f_%0.1f.png' %
#             (hours, max_ff), dpi=300)
# plt.show()

def calc_eff(mdot_air, index):
    C_h = np.array(df['mass_flow']*cp_Sn)[index]
    cp_air = 1170
    C_c = mdot_air*cp_air
    C_min = min(C_h, C_c)
    # if C_h == C_min:
    #     print('hot')
    # else:
    #     print('cold')
    T_h_in = np.array(df['outlet_T'])[index]-273
    T_c_in = 25
    T_h_out = 1900
    Q = C_h*(T_h_in - T_h_out)
    Q_max = C_min*(T_h_in - T_c_in)
    eff = Q/Q_max
    UA = eff*C_min
    return eff

def calc_mdot_air_from_eff(eff, index):
    C_h = np.array(df['mass_flow']*cp_Sn)[index]
    cp_air = 1170
    # C_min = min(C_h, C_c)
    T_h_in = np.array(df['outlet_T'])[index]-273
    T_c_in = 25
    T_h_out = 1900
    Q = C_h*(T_h_in - T_h_out)
    mdot_air = Q/(eff*cp_air*(T_h_in - T_c_in))
    
    # check if satisfies min
    C_c = mdot_air*cp_air
    C_min = min(C_h, C_c)
    if C_min == C_h:
        return np.nan
    
    return mdot_air

def solve_mdot_air(mdot_air, T_c_in, index):
    C_h = np.array(df['mass_flow']*cp_Sn)[index]
    cp_air = 1170
    C_c = mdot_air*cp_air
    C_min = min(C_h, C_c)
    # eff = 0.626
    T_h_in = np.array(df['outlet_T'])[index]-273
    # T_c_in = 25
    T_h_out = 1900
    Q_max = C_min*(T_h_in - T_c_in)
    # Q = eff*Q_max
    Q = C_h*(T_h_in - T_h_out)
    T_c_out = Q/C_c + T_c_in
    return T_c_out - 1500

# def solve_Tout_air(mdot_air, index):
#     mdot_Sn = np.array(df['mass_flow'])[index]
#     C_h = mdot_Sn*cp_Sn
#     cp_air = 1170
#     T_h_in = np.array(df['outlet_T'])[index]-273
#     T_h_out = 1900
#     C_c = mdot_air*cp_air
#     T_c_in = 25
#     Q = C_h*(T_h_in - T_h_out)
#     T_c_out = Q/C_c + T_c_in
#     return T_c_out


turbine_mdot = 0.002
mdot_airs = []
effs = []
for i in tqdm(range(len(df))):
    if i == 0:
        mdot_air = fsolve(solve_mdot_air, 1e-3, args=(25, i))
    else:
        # if mdot_air > turbine_mdot:
        #     excess_air = mdot_air - turbine_mdot
        #     if excess_air > turbine_mdot:
        #         T_c_in = 1400
        #     else:
        #         T_c_in = (1500*excess_air + 25*(turbine_mdot - excess_air))/turbine_mdot
        # else:
        #     T_c_in = 25
        # print(T_c_in)
        mdot_air = fsolve(solve_mdot_air, mdot_air, args=(25, i))
    mdot_airs.append(mdot_air)
    eff = calc_eff(mdot_air, i)
    effs.append(eff)

mdot_airs = np.array(mdot_airs)
fig = plt.figure(figsize=(7.5, 2.5))
fig.add_subplot(131)
plt.plot(df['time'], 1500*np.ones(len(df['time'])),'b-')
plt.xlabel('time (hr)')
plt.ylabel('T$_{c,out}$ ($^{\circ}$C)')

fig.add_subplot(132)
plt.plot(df['time'], mdot_airs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('mdot air (kg/s)')

fig.add_subplot(133)
plt.plot(df['time'], effs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('effectiveness')
plt.tight_layout()

plt.savefig('plots/turbine_HX_maxflow_%0.1f_%0.1f.png' % (hours, max_ff), dpi=300)

mdot_airs = []
T_c_outs = []
effs = []
for i in tqdm(range(len(df))):
    mdot_air = np.array(df['mass_flow'])[0] * cp_Sn * 500 / (cp_air * 1475)
    mdot_airs.append(mdot_air)
    T_c_out = solve_mdot_air(mdot_air, 25, i)+1500
    T_c_outs.append(T_c_out)
    eff = calc_eff(mdot_air, i)
    effs.append(eff)

T_c_outs = np.array(T_c_outs)

fig = plt.figure(figsize=(7.5, 2.5))
fig.add_subplot(131)
plt.plot(df['time'], T_c_outs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('T$_{c,out}$ ($^{\circ}$C)')
plt.tight_layout()

fig.add_subplot(132)
plt.plot(df['time'], mdot_airs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('mdot air (kg/s)')
plt.tight_layout()

fig.add_subplot(133)
plt.plot(df['time'], effs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('effectiveness')
plt.tight_layout()

plt.savefig('plots/turbine_HX_maxflow_constmdot_%0.1f_%0.1f.png' % (hours, max_ff), dpi=300)

df_data = pd.read_csv('data/COMSOL_5block_10_k_discharge_sweep_log_maxflow.csv', names=[
                          'hours', 'max_ff', 'time', 'inlet_T', 'outlet_T', 'block_T', 'Sn_T', 'bottom_right_T', 'mass_flow'], skiprows=5)
all_hours = df_data['hours'].unique()
# print(all_hours)
hours = all_hours[6]
all_max_ff = df_data['max_ff'].unique()
# print(all_max_ff)
ff = all_max_ff[3]
df = df_data[df_data['hours'] == hours]
df = df[df['max_ff'] == ff]
# print(df.info())
df['time'] = df['time']/3600
delT = df['outlet_T'] - df['inlet_T']
Pout = df['mass_flow']*delT*cp_Sn
Poutnp = np.array(df['mass_flow']*delT*cp_Sn)

df_turbine = pd.read_csv('data/temp_eff.csv', header=None, names=['temp','eff'])
df_turbine['temp'] = df_turbine['temp'] - 273
df_turbine = df_turbine.append({'temp': 0, 'eff': 0}, ignore_index=True)
p = np.polyfit(df_turbine['temp'], df_turbine['eff'], 3)
plt.figure()
temps = np.linspace(0,1700,100)
plt.plot(df_turbine['temp'], df_turbine['eff'],'bo')
plt.plot(temps, np.polyval(p, temps),'r-')
plt.xlabel('turbine inlet temp ($^{\circ}$C)')
plt.tight_layout()
plt.figure(figsize=(2.5, 2.5))
plt.plot(df['time'], Poutnp, 'b', label='Thermal')
plt.plot(df['time'], Poutnp*np.polyval(p, T_c_outs), 'r', label='Electrical')
plt.legend(prop={'size': 7})
plt.xlabel('time (hr)')
plt.ylabel('Power (W)')
plt.title(r'$\tau$ = %0.1f hours' % (hours), fontsize=10)
plt.tight_layout()
plt.savefig('plots/turbine_HX_power_%0.1f_%0.1f.png' % (hours, max_ff), dpi=300)


## constant effectiveness

mdot_airs = []
T_c_outs = []
effs = []
for i in tqdm(range(len(df))):
    eff = 0.75
    mdot_air = calc_mdot_air_from_eff(eff, i)
    mdot_airs.append(mdot_air)
    T_c_out = solve_mdot_air(mdot_air, 25, i)+1500
    T_c_outs.append(T_c_out)
    eff = calc_eff(mdot_air, i)
    effs.append(eff)

T_c_outs = np.array(T_c_outs)

fig = plt.figure(figsize=(7.5, 2.5))
fig.add_subplot(131)
plt.plot(df['time'], T_c_outs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('T$_{c,out}$ ($^{\circ}$C)')
plt.tight_layout()

fig.add_subplot(132)
plt.plot(df['time'], mdot_airs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('mdot air (kg/s)')
plt.tight_layout()

fig.add_subplot(133)
plt.plot(df['time'], effs,'b-')
plt.xlabel('time (hr)')
plt.ylabel('effectiveness')
plt.tight_layout()

plt.savefig('plots/turbine_HX_maxflow_consteff_%0.1f_%0.1f.png' % (hours, max_ff), dpi=300)

plt.show()
