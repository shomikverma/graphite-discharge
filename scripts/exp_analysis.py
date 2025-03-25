import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use(['science','grid'])

df = pd.read_csv('data/1670 RPM.txt', header=None)
df.columns = ['time', 'T_out','T_in','TU1','T_block','TU3']

df['time'] = df['time'] - df['time'][0]

plt.plot(df['time'], df['T_out'], label='T$_{out}$')
plt.plot(df['time'], df['T_in'], label='T$_{in}$')
plt.plot(df['time'], df['T_block'], label='T$_{block}$')
plt.legend()
plt.tight_layout()

plt.figure()
df_2 = df[(df['time'] >= 10277) & (df['time'] <= 16983)]
df_2.reset_index(drop=True, inplace=True)
df_2['time'] = df_2['time'] - df_2['time'][0]
plt.plot(df_2['time'], df_2['T_out'], label='T$_{out}$')
plt.plot(df_2['time'], df_2['T_in'], label='T$_{in}$')
plt.plot(df_2['time'], df_2['T_block'], label='T$_{block}$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Temperature ($^\circ$C)')
plt.tight_layout()
plt.savefig('plots/1670RPM_temps.pdf')


plt.figure()
# pick times between 10277 and 16983
# df = df[(df['time'] >= 10277) & (df['time'] <= 16983)]
df = df[(df['time'] >= 11375) & (df['time'] <= 16983)]
df_model = pd.read_csv('data/model_data_28.csv', header=None, skiprows=5)
df_model.columns = ['time', 'T_out', 'T_in', 'T_block']
df.reset_index(drop=True, inplace=True)
df['time'] = df['time'] - df['time'][0]

plt.plot(df['time'], df['T_out'], 'b', label='T$_{out}$')
plt.plot(df['time'], df['T_block'], 'g', label='T$_{block}$')
plt.plot(df_model['time'], df_model['T_out']-10, 'orange', label='T$_{out,model}$', linestyle='--')
# plt.plot(df_model['time'], df_model['T_in'], label='T$_{in,model}$', linestyle='--')
plt.plot(df_model['time'], df_model['T_block']-10, 'r', label='T$_{block,model}$', linestyle='--')
plt.plot(df['time'], df['T_in'], 'k-', label='T$_{in}$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Temperature ($^\circ$C)')
plt.tight_layout()
plt.savefig('plots/1670RPM_temps_model.pdf')

df = df[['time','T_in']]
df.to_csv('data/1670RPM_temps_heatoff.csv', index=False)

df2 = pd.read_csv('data/model_data_10.csv', header=None, skiprows=5)
df2.columns = ['time', 'T_out', 'T_in', 'T_block']
df2['time'] = df2['time'] - df2['time'][0]
plt.figure()
plt.plot(df2['time'], df2['T_out'], label='T$_{out}$')
plt.plot(df2['time'][20:], df2['T_in'][20:], label='T$_{in}$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Temperature ($^\circ$C)')
plt.tight_layout()
plt.savefig('plots/1670RPM_temps_constTin.pdf')

df2['theta'] = (df2['T_out'] - 500) / (max(df2['T_out']) - 500)
df2['theta_in'] = (df2['T_in'] - 500) / (max(df2['T_in']) - 500)
m_block = 4.5*4.5*13*16.387*1.72/1000 # kg
cp_block = 1800 # J/kgK
mdot_flow = 0.062 # kg/s
cp_flow = 248 # J/kgK
tau = m_block*cp_block/(mdot_flow*cp_flow)
df2['t_star'] = df2['time']/tau

plt.figure()
plt.plot(df2['t_star'], df2['theta'],'b-', label='T$_{out}$')
plt.plot(df2['t_star'][20:], df2['theta_in'][20:],'k-', label='T$_{in}$')
plt.xlim([-0.1, 2.2])
plt.xlabel('$t^*$')
plt.ylabel('$\Theta^*$')
plt.legend()
plt.tight_layout()
plt.savefig('plots/1670RPM_theta_constTin.pdf')

plt.show()