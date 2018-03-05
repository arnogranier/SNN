from .core import Variable, Model
from .tools import Function as F
#izhi

v = Variable(name='v', ddt='0.04*v**2+5*v+140-u+I', init_value=-65, 
			 reset_value='c', unit='mV')
u = Variable(name='u', ddt='a*(b*v-u)', init_value=-15, reset_value='u+d')
izhi_model = Model(v, u, spike_when='v>=30', max_spike_value=30, 
        a=0.02, b=0.2, c=-65, d=8, I=0) 
#F('t', lambda t :0 if t<200 or t>800 else 5)


#HH
from numpy import exp
V = Variable(name='V', init_value=-60, forbidden_values=-65,
			 ddt='(1/Cm)*(-gk*n**4*(V-Vk)-gna*m**3*h*(V-Vna)-gl*(V-Vl)+Iapp)')
n = Variable(name='n', ddt='alpha_n*(1-n)-beta_n*n', init_value=1/3)
m = Variable(name='m', ddt='alpha_m*(1-m)-beta_m*m', init_value=0.01)
h = Variable(name='h', ddt='alpha_h*(1-h)-beta_h*h', init_value=2/3)
HH_model = Model(V, n, m, h, Cm=1, gk=36, gna=120, 
				 gl=0.3, Vk=-77, Vna=50, Vl=-54.4, 
				 alpha_n=F('V', lambda V: 0.01*(-V-55)/(exp((-V-55)/10) -1)), 
				 beta_n=F('V', lambda V: 0.125*exp((-V-65)/80)),
				 alpha_m=F('V', lambda V: 0.1*(-V-40)/(exp((-V-40)/10) -1)), 
				 beta_m=F('V', lambda V: 4*exp((-V-65)/18)),
				 alpha_h=F('V', lambda V: 0.07*exp((-V-65)/20)),
				 beta_h=F('V', lambda V: 1/(1+exp((-V-36)/10))), 
				 Iapp=0)

#FHN
v = Variable(name='v', ddt='v-(1/3)*v**3-w+I', init_value=0)
w = Variable(name='w', ddt='(1/tau)*(v+alpha-beta*w)', init_value=0)
FHN_model = Model(v, w, alpha=0.7, beta=0.8, tau=13, I=0.5)