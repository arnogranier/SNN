from .core import Variable, Model

v = Variable(name='v', ddt='-u+0.04*v**2+5*v+140+I', init_value=-65, 
			 reset_value='c', unit='mV')
u = Variable(name='u', ddt='a*(b*v-u)', init_value=-15, reset_value='u+d')
izhi_model = Model(v, u, spike_when='v>=30', max_spike_value=30, 
        a=0.02, b=0.2, c=-65, d=8, I=0) 

#HH
from numpy import exp
V = Variable(name='V', init_value=-60,
			 ddt='(1/Cm)*(-gk*n**4*(V-Vk)-gna*m**3*h*(V-Vna)-gl*(V-Vl)+Iapp)')
n = Variable(name='n', ddt='alpha_n*(1-n)-beta_n*n', init_value=1/3)
m = Variable(name='m', ddt='alpha_m*(1-m)-beta_m*m', init_value=0)
h = Variable(name='h', ddt='alpha_h*(1-h)-beta_h*h', init_value=2/3)
HH_model = Model(V, n, m, h, Cm=1, gk=36, gna=120, 
				 gl=0.3, Vk=-77, Vna=50, Vl=-54.4, 
				 alpha_n='0.01*(-V-55)/(exp((-V-55)/10) -1)', 
				 beta_n='0.125*exp((-V-65)/80)',
				 alpha_m='0.1*(-V-40)/(exp((-V-40)/10) -1)', 
				 beta_m='4*exp((-V-65)/18)',
				 alpha_h='0.07*exp((-V-65)/20)',
				 beta_h='1/(1+exp((-V-36)/10))', 
				 Iapp=0)

#FHN
v = Variable(name='v', ddt='-w+v-(1/3)*v**3+I', init_value=-1.25)
w = Variable(name='w', ddt='(1/tau)*(v+a-b*w)', init_value=-1)
FHN_model = Model(v, w, a=0.7, b=1.2, tau=12.5, I=0)

#Leaky integrate and fire
u = Variable(name='u', init_value=0,
	ddt = '(1/tau_m)*(-u+R*I)', reset_value='ur', unit='mV')
LEAKY_INTEGRATE_AND_FIRE_model = Model(u, spike_when='u>=1',
	max_spike_value=1, tau_m=10, R=1, ur=0, I=0)
