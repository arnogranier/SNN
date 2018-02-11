from brnn.single import *

v = Variable(name='v', ddt='0.04*v**2+5*v+140-u+I', init_value=-65, 
			 reset_value='c', unit='mV')
u = Variable(name='u', ddt='a*(b*v-u)', init_value=-15, reset_value='u+d')
izhi_model = Model(v, u, spike_when='v>=30', max_spike_value=30, 
        a=0.02, b=0.2, c=-65, d=8, I=lambda t : 0 if t<200 or t>800 else 5)
fig1 = model.plot(1000, 1, keep=['I', 'u', 'v'])
fig2 = model.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       interactive=True, rescale=True)
plt.show()
