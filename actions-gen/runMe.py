import mujoco as mj
import numpy as np
import os,time,ctypes
from lib_ZMPctrl import selectRobot,q2data,trnparam,DepthvsForce
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

simend = 5.0 #simulation time
simfreq = 100 # 100 fps

# vel=0.1
# spno=1 #1-SSP, 2-DSP
# step_len=0.5 # step length for MPC

#Walking pattern mode
spno=2 #1-SSP, 2-DSP

#Select Robot - # 1-Kondo, 2-G1, 3-Biped
# humn,model,data = selectRobot(num=1,vel=0.1,step_len=0.1,stairs_height=0.005,spno=spno) #1-Kondo
humn,model,data = selectRobot(num=2,vel=0.15,step_len=0.25,stairs_height=0.005,spno=spno) #2-G1

# model.opt.timestep=0.00001

# Terrain parameters
humn.plno = 0  # 0-default 1-virtual terrain
humn.zpln = 0  # height of terrain plane

# Terrain parameters
zeta1=1.5 # damping ratio=25
nocp=4 #number of contact points of foot
trn=trnparam(nocp,zeta1,humn.zpln) #hard terrain parameters for left foot terrain
trn.mjparam(model)

print('stiffness and damping =',trn.solref)
time.sleep(2)

# Actual parameters of terrain
# model.geom_solimp[0][0]=0.0
# model.geom_solimp[0][2]=0.01
#model.geom_solimp[1]=[0.0, 0.95, 0.0002, 0.5, 2]
for nocp1 in np.arange(nocp,0,-1/10): #[nocp]:#[0.3]:#for defT, [1.8]:for hardT #
    trn1=trnparam(nocp1,zeta1,humn.zpln) #hard terrain parameters for left foot terrain
    trn1.mjparam(model)

    #Change terrain solref of Humanoid xml model
    i=0
    while model.geom_bodyid[i] == 0:
        model.geom_solimp[i] = trn1.solimp[i]
        model.geom_solref[i]=trn1.solref[i]
        i=i+1
    # print(DepthvsForce(model,data,0))
    if abs(DepthvsForce(model,data,0))<(model.geom_solimp[0][2]/2):
        print('nocp=',nocp1)
        print('stiffness=', model.geom_solref[0][0], 'damping=', model.geom_solref[0][1], ', ...wait for 2 sec')
        print('Des vs Act Deformation is:',(model.geom_solimp[0][2]/2),DepthvsForce(model,data,1))
        break

time.sleep(2)

st_time=time.time()

#LIPM MPC to Kondo traj
humn.mpc2humn(1/1000,simend,trn,50,humn.step_time,humn.step_len,1)

print('!!!!!!! Walking pattern is generated/saved')

# Generate Gait / Cartesian traj to joint space traj
ttraj=np.arange(0,simend,1/1000)
qtraj=humn.cart2joint(model,data,ttraj,0,0)
humn.pltqTraj(ttraj,qtraj,linestyl='-',overlap=0)
# plt.show()
plt.pause(0.1)
# humn.k_ub=0.1 #Reset k_ub

# Or open qtraj
# with open('qtraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     ttraj,qtraj = pickle.load(f)

print('Time taken in traj=',time.time()-st_time,'sec for sim time of ',simend,'sec, ...wait for 1 sec')
time.sleep(0.1)

# Generate spline from qtraj
humn.qspl=[]
for i in np.arange(0,model.nv):
    humn.qspl.append(CubicSpline(ttraj,qtraj[:,i])) #,bc_type='clamped'
#     #dqi.append(CubicSpline.derivative(qi[i]))

# Initialize
data=q2data(data,humn.q0)


# Simulate
humn.sim(model,data,trn,simfreq,simend,saveVid=True)

