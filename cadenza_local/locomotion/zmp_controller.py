# This file contains functions used for footstep planning of Kondo khr3hv using ZMP preview control of Kajita et al.
# Author : Sunil Gora, Shakti S. Gupta and Ashish Dutta
import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle
from scipy.optimize import minimize,LinearConstraint,differential_evolution
from copy import deepcopy
import time
# import keyboard
from scipy.linalg import solve_discrete_are
from moviepy import ImageSequenceClip

#Plot set
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'cmr10'  # Computer Modern Roman
# To ensure correct rendering of minus signs in mathematical expressions
plt.rcParams["axes.formatter.use_mathtext"] = True        
plt.rcParams['pdf.fonttype'] = 42  # avoid type3 font
# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# Set the default linewidth to 2
plt.rcParams['lines.linewidth'] = 3
fsize=16
plt.rc('font', size=fsize)  # controls default text sizes
plt.rc('axes', titlesize=fsize)  # fontsize of the axes title
plt.rc('axes', labelsize=fsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize)  # fontsize of the tick labels
plt.rc('legend', fontsize=fsize)  # legend fontsize
plt.rc('figure', titlesize=fsize+2)  # fontsize of the figure title
plt.rcParams["axes.spines.right"] = "False"
plt.rcParams["axes.spines.top"] = "False"
plt.rcParams['axes.autolimit_mode'] = 'round_numbers' # to avoid offset in axis
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def selectRobot(num,vel,step_len,stairs_height,spno):
    dirname = os.path.dirname(__file__) #os.getcwd() 
    #Generate XML for scene
    xml_path='/scene_basic.xml' #load basic xml file of scene
    xml_path = os.path.join(dirname + xml_path)
    xml_tree = ET.parse(xml_path) 
    #Modify xml_tree for 3D terrain generation
    #Stairs
    trnnum=20
    # stairs_height=0.005
    trnheight=[i * stairs_height for i in range(trnnum)] #height of each step
    xml_str = scenegen(xml_tree,trnnum=trnnum,trnlength=step_len,trnwidth=0.50,trnheight=trnheight)
    # print(xml_str)
    # print(asd)
    xml_tree = ET.ElementTree(ET.fromstring(xml_str))

    # make cases for different robots
    if num==1: #Kondo khr3hv
        # xml_path= 'kondo/scene_Stairs.xml' #Scene
        # xml_path = os.path.join(dirname + "/" + xml_path)
        # xml_tree = ET.parse(xml_path) 

        robotpath='kondo/kondo_khr3hv.xml' #Robot
        robotpath = os.path.join(dirname + "/" + robotpath)
        #robotpath='example/model/humanoid/humanoid.xml' #Robot <!-- Include sites at hip and both foot -->
        xml_str=addrobot2scene(xml_tree,robotpath)

        # MuJoCo data structures
        model = mj.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mj.MjData(model)  # MuJoCo data
        # cam = mj.MjvCamera()                        # Abstract camera
        # opt = mj.MjvOption()                        # visualization options
        
        model.opt.timestep=0.0001
        
        ub_jnts=np.arange(18,model.nv)
        left_legjnts=np.arange(6,12)
        right_legjnts=np.arange(12,18)
        foot_size = np.array([0.050, 0.040]) #50mm,40mm np.array([length, width])
        
        #init_controller(model, data)
        Kp=np.zeros(model.nu)
        Kv=np.zeros(model.nu)
        Ki=np.zeros(model.nu)
        Kp[0:12]=10 #7
        Kv[0:12]=.1  #0.5 #0.003
        # Ki[0:12]=0.1
        Kp[12:]=1
        Kv[12:]=0.01 #0.05
        # Ki[12:]=0.01
        # Kp=10*Kp
        # Kv=10*Kv
        #FW
        # Kp[-1:]=10
        # Kv[-1:]=0.01

        # data to humanoid parameters
        humn=myRobot(ub_jnts,left_legjnts,right_legjnts,foot_size,vel)

        humn.mj2humn(model,data)

        # Initial joint angles and velocity
        q0=data2q(data)
        q0[[2,9,10,15,16]]=[0.95*q0[2],0.5,-0.5,0.5,-0.5]

        # Initial COM position
        humn.r_com[0]=-0.0#1#-0.01
        humn.r_com[1]=(0+humn.o_left[1])/3
        humn.r_com[2]=0.9*humn.r_com[2]

        #Step length and height
        # step_len=0.01 # Max Steplength
        # step_time= step_len/(2*vel) #0.2
        zSw=0.03 #swing foot lift
        cam_dist=0.75 #camera distance

    elif num==2: #Unitree G1
        # xml_path= 'sceneG1.xml' #Scene
        # #xml_path= 'scene_3DT.xml' #Scene
        # xml_path = os.path.join(dirname + "/" + xml_path)
        # xml_tree = ET.parse(xml_path) 
        robotpath='/Unitree/g1.xml' #Robot <!-- Include sites at hip and both foot -->
        robotpath = os.path.join(dirname + robotpath)
        xml_str=addrobot2scene(xml_tree,robotpath)

        # MuJoCo data structures
        model = mj.MjModel.from_xml_string(xml_str)  # MuJoCo model
        data = mj.MjData(model)  # MuJoCo data
        # cam = mj.MjvCamera()                        # Abstract camera
        # opt = mj.MjvOption()                        # visualization options

        # model.opt.timestep=0.0001

        ub_jnts=np.arange(18,model.nv)
        left_legjnts=np.arange(6,12)
        right_legjnts=np.arange(12,18)
        foot_size = np.array([0.050, 0.040]) #50mm,40mm np.array([length, width])

        #init_controller(model, data)
        Kp=np.zeros(model.nu)
        Kv=np.zeros(model.nu)
        Ki=np.zeros(model.nu)
        Kp[0:12]=5000 #7
        Kv[0:12]=50  #0.003
        Kp[3]=10000
        Kv[3]=100
        Kp[9]=10000
        Kv[9]=100
        Kp[12:]=50
        Kv[12:]=1

        # data to humanoid parameters
        humn=myRobot(ub_jnts,left_legjnts,right_legjnts,foot_size,vel)

        humn.mj2humn(model,data)

        # Initial joint angles and velocity
        q0=data2q(data)
        q0[[2,9,10,15,16]]=[0.9*q0[2],0.5,-0.5,0.5,-0.5]

        # Initial COM position
        humn.r_com[0]=-0.0#1#-0.01
        humn.r_com[1]=(0+humn.o_left[1])/3
        humn.r_com[2]=0.9*humn.r_com[2]

        humn.o_left[1]=0.5*humn.o_left[1]
        humn.o_right[1]=0.5*humn.o_right[1]


        #Step length, time and height
        # step_len=0.1 # Max Steplength
        # step_time= step_len/(2*vel) #0.5
        zSw=0.05 #swing foot lift
        cam_dist=2 #camera distance

    
    # Set pos and orientation
    # print('o_left=',humn.o_left)
    humn.o_left[0]=0.0
    humn.o_left[2]=0#-model.geom_solimp[0][2]/2
    # print('o_right=',humn.o_right)
    humn.o_right[0]=0.0
    humn.o_right[2]=-0.0

    data.qvel[0]=vel#0.25#5#0.1 #0.1
    data.qvel[1]=0.0#5#0.1 #0.1

    humn.spno = spno  # SSP=1, DSP=2

    if humn.spno==2:
        data.qvel[0]=0#0.55#0.1 #0.1
        humn.r_com[1]=0.0#(0+humn.o_left[1])/3
        humn.r_com[2]=0.8/0.9*humn.r_com[2]


    #print(q0)
    # Find initial IK solution
    # q=numik(model,data,q0,1,humn.r_com,humn.o_left,humn.o_right,humn.ub_jnts,0)
    ti=0
    while ti<100:
        delt=1/100
        q,gradH0=numik(model,data,q0,delt,humn.r_com,humn.o_left,humn.o_right,humn.ub_jnts,np.zeros([model.nv]),False,0,0)
        q0=q.copy()
        ti +=delt

    data=q2data(data,q)
    humn.q0=q.copy()
    # print("Initial joint coord q0=",q0)
    #glfw.terminate()

    # Humanoid parameters
    humn.mj2humn(model,data)
    print('mass=',humn.m,' rCOM=',humn.r_com,' drCOM=',humn.dr_com)
    print('o_left=',humn.o_left, 'o_right=',humn.o_right)
    # Check leg transition required
    humn.xlimft = 0.5 #10*step_len # Max Steplength
    humn.ylimft = abs(humn.o_left[1]-humn.o_right[1])   #max(0.1 * l, 2 * abs(qcm[1] - qcp[1]))
    # humn.spno = 1  # SSP=1, DSP=2
    humn.Stlr=np.array([1, 0])
    humn.zSw=zSw #swing foot lift
    humn.step_time= step_len/(2*vel) #step_time for MPC
    humn.step_len=step_len #steplength for MPC
    humn.sspbydsp=2
    humn.Tsip = 0  # Cycle Time for footstep control
    humn.cam_dist=cam_dist


    # initialize the controller
    humn.init_controller(Kp, Kv,Ki)
    if num==2: #G1 is position controlled
        humn.posCTRL=True #Control mode is Position or Torque

    return humn,model,data





# Find Euler angles from COM position and Contact position of inverted pendulum model
def findeulr(qcm,qcp,l):
    return np.array([np.arctan2(qcp[1] - qcm[1], qcm[2] - qcp[2]), np.arctan2(qcm[0] - qcp[0], np.linalg.norm(qcm[1:] - qcp[1:]) ), 0.0])

### OpenAI codes for Euler-quat conversion
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


#MuJoCo model to Humanoid Robot data
class myRobot:
    def __init__(self, ub_jnts,left_legjnts,right_legjnts,foot_size,vel):
        #Design
        self.ub_jnts=ub_jnts
        self.left_legjnts=left_legjnts
        self.right_legjnts=right_legjnts
        self.foot_size=foot_size
        self.cam_dist=1
        #Control
        self.vel=vel
        self.Tsip = 0  # Cycle Time for footstep control
        self.WD = 0 # Work Done
        self.posCTRL=False #Control mode is Position or Torque
        self.KINctrl=False #State estimation and traj correction using FK
        self.ZMPctrl=0#-1/(10**5)#.0001#01
        self.AMctrl=0.0
        self.k_ub=0.1000 #task priority weight for ub
        self.k_L=self.AMctrl*1/(1+self.k_ub*0*np.linalg.norm(np.zeros([6])))
        self.FWctrl=0
        self.ftctrl=0
        #Identification
        self.xn = np.array([])
        self.dxn = np.array([])
        self.fn = np.array([])
        self.AM_CMspl=[]
        for i in range(3):
            self.AM_CMspl.append(CubicSpline([0,1], [0,0], bc_type='clamped')) #torque on COM due to change of AM

    def mj2humn(self,model,data):
        # Forward kinematics for position and velocity terms
        # Forward Position kinematics
        mj.mj_fwdPosition(model, data)
        # mj.mj_kinematics(model, data)
        mj.mj_comVel(model, data)
        mj.mj_subtreeVel(model, data)
        self.ti=data.time
        self.m =mj.mj_getTotalmass(model) #mass
        self.r_com=data.subtree_com[0].copy() #com position
        self.dr_com=data.subtree_linvel[0].copy()  #com velocity
        self.dq_com=data.cvel[1].copy() # com rot:lin velocity
        self.o_left=data.site('left_foot_site').xpos.copy()  # current Left foot position
        self.o_right=data.site('right_foot_site').xpos.copy()  # current Right foot position
        self.data2q(data) #write self.q and self.dq
        self.q_err = np.zeros([model.nv])
        self.Eqdt=0*data.qvel #Integral error
        self.E_init=1/2*self.m*np.linalg.norm(self.dr_com)**2 + self.m*9.81*self.r_com[2] #Initial energy
        self.gradH = np.ones([model.nv]) #Gradient for joint limits check

        #Traj save
        #if data.time==0:
        #print(data.time)
        # self.tCMtraj = np.array([data.time])
        # self.oCMtraj = np.array([self.r_com])
        # self.tLtraj = np.array([data.time])
        # self.oLtraj = np.array([self.o_left])
        # self.tRtraj = np.array([data.time])
        # self.oRtraj = np.array([self.o_right])

        #Save desired and actual traj data
        self.qTraj_des=np.array([self.q.copy()])
        self.dqtTraj_des=self.dq.copy()
        self.qTraj_act=self.q.copy()
        self.dqTraj_act=self.dq.copy()
        self.tTraj=data.time
        self.CAMTraj=np.zeros(3)
        self.torqueTraj=np.zeros(model.nu)
        self.fLTraj=np.zeros(6)
        self.fRTraj=np.zeros(6)

    def updateTrajData(self,model,data):
        # Save desired and actual traj data
        self.qTraj_des=np.vstack((self.qTraj_des,self.q.copy()))
        self.dqtTraj_des=np.vstack((self.dqtTraj_des,self.dq.copy()))
        self.qTraj_act=np.vstack((self.qTraj_act,self.q.copy()))    
        self.dqTraj_act=np.vstack((self.dqTraj_act,self.dq.copy()))
        self.tTraj=np.vstack((self.tTraj,data.time))
        self.torqueTraj=np.vstack((self.torqueTraj,data.actuator_force.copy()))
        #Contact force on left foot
        self.fLTraj=np.vstack((self.fLTraj,data.sensordata[0:6].copy()))
        #Contact force on right foot
        self.fRTraj=np.vstack((self.fRTraj,data.sensordata[6:12].copy()))
        #Work done


    # Copy data.qpos (with quaternion) to q (with euler angles)
    def data2q(self,data):
        self.q = 0 * data.qvel.copy()
        self.dq = 0 * data.qvel.copy()
        qqt = data.qpos[3:7].copy()
        qeulr = quat2euler(qqt)
        for i in np.arange(0, 3):
            self.q[i] = data.qpos[i].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(3, 6):
            self.q[i] = qeulr[i - 3].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(6, len(data.qvel)):
            self.q[i] = data.qpos[i + 1].copy()
            self.dq[i] = data.qvel[i].copy()


    def lipm2humn(self,dt,Tf,sspbydsp,vis):
        i = 0
        ti = dt#sipdata[i][0]
        t0=0
        # dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=self.spno
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
            rct = self.o_left
        else:
            self.oCPtraj = np.array([self.o_right])
            rct = self.o_right

        # LIPM motion
        r1=rct.copy()
        rcm0=self.r_com.copy()
        drcm0=self.dr_com.copy()
        tcm = np.empty((0))
        ocm = np.empty((0, 3))
        oct = np.empty((0, 3))
        ftplac=[]
        for ti in np.arange(dt,2*Tf,dt):
            if spno==1: #SSP
                Ts = np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                # x(t)=1/2*(x(0)+Ts xdot(0)) * np.exp(t/Ts) + 1/2*(x(0)-Ts xdot(0)) * np.exp(-t/Ts)
                rcm=rct+1/2*(rcm0-rct + Ts * drcm0) * np.exp((ti-t0)/Ts) + 1/2*(rcm0-rct - Ts * drcm0) * np.exp(-(ti-t0)/Ts)
                drcm=1/2*(rcm0-rct + Ts * drcm0) * (1/Ts) * np.exp((ti-t0)/Ts) - 1/2*(rcm0-rct - Ts * drcm0) *(1/Ts)* np.exp(-(ti-t0)/Ts)
            else: #DSP
                # x(t) = x(0)*cos(t/Td) + xdot(0)* Td * sin(t/Td)
                Td=np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                rcm = rct + (rcm0-rct) * np.cos((ti-t0)/Td) + drcm0 * Td * np.sin((ti-t0) / Td)
                drcm= -(rcm0-rct) *(1/Td)* np.sin((ti-t0)/Td) + drcm0 * 1 * np.cos((ti-t0) / Td)
            rcm[2]=rcm0[2]
            drcm[2]=0
            tcm = np.append(tcm, np.array([ti]), axis=0)
            ocm = np.append(ocm, np.array([rcm]), axis=0)
            oct = np.append(oct, np.array([rct]), axis=0)
            if spno==1:
                rcp=drcm*Ts
                r2=rcm+(1/sspbydsp)*np.linalg.norm(rcm-rct)*drcm/np.linalg.norm(drcm)
                r2[2]=rcm[2]/sspbydsp
                r3=r1+2*(r2-r1)
                r3[2]=r1[2]

            # print(rcm,r1,r2,r3)
            if (spno==1)*(rcm[0]>rct[0])*( (abs(r3[0]-r1[0]) > self.xlimft) + (abs(r3[1]-r1[1]) > self.ylimft) ) or (spno==2)*(rcm[0]>rct[0])*(abs(rcm[0]-rct[0])>abs(rcm0[0]-rct[0]))*(abs(rcm[1]-rct[1])>abs(rcm0[1]-rct[1])):
                print(ti,oLeft,oRight)
                # Foot placement data
                ftplac.append([ti, r1, r2, r3])

                self.updateGait(tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac)  
                if spno==2:
                    Stlr = np.array([1, 1]) - Stlr
                t0 = ti
                tcyc.append(t0)
                ncyc = ncyc + 1
                spno = 3 - spno
                if spno==2:
                    rct=r2.copy()
                else:
                    rct = r3.copy()
                    r1 = r3.copy()
                rcm0 = rcm.copy()
                drcm0 = drcm.copy() #(ocm[-1,:] - ocm[-2]) / dt

                # plt.figure(18)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm, (np.array([sublist[0] for sublist in ocm])))
                # print(oct[-1])
                # plt.plot(tcm[-1],oct[-1][0],'*r')
                # plt.plot(tcm, (np.array([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[-1],oct[-1][1],'*b')
                # plt.figure(19)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[0:-1],(np.diff([sublist[0] for sublist in ocm])/dt))
                # plt.plot(tcm[0:-1],(np.diff([sublist[1] for sublist in ocm])/dt))


                oLeft = self.oLtraj[-1]
                oRight = self.oRtraj[-1]
                tcm = np.empty((0))
                ocm = np.empty((0, 3))
                oct = np.empty((0, 3))
                if ti >= Tf:
                    break
            if vis==1 and (ti%0.01)<dt:
                # print(ti) # ,drcm,rcm,r2)
                plt.figure(10)
                # plt.plot(ctime+data.time,qcp[2],'.r')
                plt.plot(ti,drcm[0],'*r',ti,drcm[1],'*g',ti,drcm[2],'*b')
                plt.xlabel('Time')
                plt.ylabel('dq/dt')
                plt.pause(0.001)
        # plt.show()
        # print(self.tCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz

    def mpc2humn(self,dt,Tf,trn,sspbydsp,step_time,step_len,vis):
        i = 0
        ti = dt#sipdata[i][0]
        t0=0
        # dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=2 #self.spno #start with DSP in rest
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
            rct = self.o_left
        else:
            self.oCPtraj = np.array([self.o_right])
            rct = self.o_right

        # LIPM motion
        r1=rct.copy()
        rcm0=self.r_com.copy()
        rct0=rct.copy()
        drcm0=self.dr_com.copy()
        tcm = np.empty((0))
        ocm = np.empty((0, 3))
        oct = np.empty((0, 3))
        ftplac=[]

        #MPC control
        #Kajita's 2003 paper on LIPM preview control using Katayama et al. 1985 LQI control
        # LIPM parameters
        h=rcm0[2]-rct[2] #height of COM from ZMP
        g=9.81 #gravity

        Ts = np.sqrt(h / g) #Tc = np.sqrt(h/g)
        # step_time=np.around(Ts,1) #cycle time
        # dt = 0.005 #5ms
        N = 1000  # Preview horizon

        # State-space model for LIPM (discretized)
        A = np.array([[1, dt, dt**2/2],
                    [0, 1, dt],
                    [0, 0, 1]])
        B = np.array([[dt**3/6],
                    [dt**2/2],
                    [dt]])
        C = np.array([[1, 0, -h/g]])

        # Cost weights
        Qe = 1.0   # ZMP error weight
        Qx = np.zeros((3, 3))  # State weight        
        R = 1e-6   # Control input weight

        # Augmented system for preview control
        nx = A.shape[0]
        G = np.vstack((C, np.zeros((1, nx))))
        A_aug = np.vstack([
            np.hstack([np.eye(1), C @ A]),
            np.hstack([np.zeros((nx, 1)), A])
        ])
        B_aug = np.vstack([C @ B, B])

        # Riccati equation for augmented system
        Q = np.zeros((nx+1, nx+1))
        Q[0, 0] = Qe
        Q[1:, 1:] = Qx
        P = solve_discrete_are(A_aug, B_aug, Q, R)
        #print("Algebric Riccati Equation solution P:\n", P)

        # Compute feedback and preview gains
        K = np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ P @ A_aug)
        Gi = K[0, 0]
        Gx = K[0, 1:]

        # Compute preview gains
        Gd = np.zeros(N)
        AcBK = A_aug - B_aug @ K
        X = -AcBK.T @ P @ np.array([[1], [0], [0], [0]])
        for i in range(N):
            Gd[i] = (np.linalg.inv(B_aug.T @ P @ B_aug + R) @ (B_aug.T @ X)).item()
            X = AcBK.T @ X

        # Plot preview gains
        # plt.figure(figsize=(8,3))
        # plt.plot(np.arange(1, N+1)*dt, Gd, marker='o')
        # plt.xlabel('Preview Time (s)')
        # plt.ylabel('Preview Gain $G_d$')
        # plt.title('Preview Gains vs. Preview Time (Kajita 2003)')
        # plt.grid()
        # plt.tight_layout()
        # plt.show()

        # --- Generate a sample ZMP (footstep) reference trajectory ---
        # step_time = 1 #cycle time
        num_steps = np.ceil((Tf+N*dt)/step_time).astype(int) #10
        zmp_ref_x = []
        zmp_ref_y = []
        zmp_ref_z = []
        com_z = [] #rcm0[2]
        zmp_crt=np.zeros(3)
        i=0
        #Current ZMP position
        zmp_crt[0]= rct0[0]+(i)*step_len #x_val
        zmp_crt[1]=0 #y_val
        zmp_crt[2]=0
        #Next ZMP position
        zmp_nxt=np.zeros(3)
        for i in range(num_steps):
            if (i+1) % 2 == 0:
                zmp_nxt[0]= rct0[0]+(i)*step_len #x_val
                zmp_nxt[1]= self.o_right[1] #y_step
                zmp_nxt[2]=0
            else:
                zmp_nxt[0]= rct0[0]+(i)*step_len #x_val
                zmp_nxt[1]=self.o_left[1] #-y_step
                zmp_nxt[2]=0

            # Check step height
            trn.cntplane(zmp_nxt, 1)
            zmp_nxt[2]=trn.cntpos[2]
            zmp_ref_x += [zmp_crt[0]]*int(step_time/dt)
            zmp_ref_y += [zmp_crt[1]]*int(step_time/dt)
            zmp_ref_z += [zmp_crt[2]]*int(step_time/dt)
            # print(zmp_ref_x)
            if i==0:
                com_z += [rcm0[2]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))] 
            else:
                # print("com_z last:", com_z[-1])
                com_z += [com_z[-1]+(zmp_nxt[2]-zmp_crt[2])*i/int(step_time/dt) for i in range(int(step_time/dt))]
            zmp_crt=zmp_nxt.copy()

        zmp_ref_x = np.array(zmp_ref_x)
        zmp_ref_y = np.array(zmp_ref_y)
        zmp_ref_z = np.array(zmp_ref_z)
        print("ZMP ref len:", len(zmp_ref_x), "Preview steps:", N)

        # --- Preview control simulation ---
        x = np.array([[rcm0[0]], [drcm0[0]], [0.0]])  # [CoM pos, vel, acc]
        com_x = []
        zmp_x = []
        e_sum = 0.0

        for k in range(len(zmp_ref_x)-N):
            # Output (current ZMP)
            p = (C @ x)[0, 0]
            # Error integration
            e = p - zmp_ref_x[k]
            e_sum += e

            # Preview control law (Kajita 2003)
            preview_sum = 0.0
            for j in range(N):
                if (k + j + 1) < len(zmp_ref_x):
                    preview_sum += float(Gd[j]) * zmp_ref_x[k + j + 1]
                else:
                    preview_sum += float(Gd[j]) * zmp_ref_x[-1]
            u = -Gi * e_sum - Gx @ x.flatten() - preview_sum

            # State update
            x = A @ x + B * u

            com_x.append(float(x[0, 0]))
            zmp_x.append(p)


        # --- Generate a sample ZMP (footstep) reference trajectory for y ---
        # For example, alternate left/right footsteps
        # y_step = 0.1
        # zmp_ref_y = []

        # --- Preview control simulation for y direction ---
        y = np.array([[rcm0[1]], [drcm0[1]], [0.0]])  # [CoM pos, vel, acc] in y
        com_y = []
        zmp_y = []
        e_sum_y = 0.0

        for k in range(len(zmp_ref_y)-N):
            # Output (current ZMP)
            p_y = (C @ y)[0, 0]
            # Error integration
            e_y = p_y - zmp_ref_y[k]
            e_sum_y += e_y

            # Preview control law (Kajita 2003) for y
            preview_sum_y = 0.0
            for j in range(N):
                if (k + j + 1) < len(zmp_ref_y):
                    preview_sum_y += float(Gd[j]) * zmp_ref_y[k + j + 1]
                else:
                    preview_sum_y += float(Gd[j]) * zmp_ref_y[-1]
            u_y = -Gi * e_sum_y - Gx @ y.flatten() - preview_sum_y

            # State update
            y = A @ y + B * u_y

            com_y.append(float(y[0, 0]))
            zmp_y.append(p_y)        
        if vis==1:
            plt.figure(figsize=(10,6))
            time_axis = np.arange(0, len(com_x)*dt, dt)
            plt.subplot(3,1,1)
            plt.plot(time_axis, com_x, label='CoM x')
            plt.plot(time_axis, zmp_x, label='ZMP x')
            plt.plot(time_axis, zmp_ref_x[:len(zmp_x)], '--', label='ZMP ref x')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('LIPM Preview Control in X Direction')
            plt.legend()
            plt.grid()

            plt.subplot(3,1,2)
            plt.plot(time_axis, com_y, label='CoM y')
            plt.plot(time_axis, zmp_y, label='ZMP y')
            plt.plot(time_axis, zmp_ref_y[:len(zmp_y)], '--', label='ZMP ref y')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('LIPM Preview Control in Y Direction')
            plt.legend()
            plt.grid()

            plt.subplot(3,1,3)
            plt.plot(time_axis, com_z[:len(time_axis)], label='CoM z')
            plt.xlabel('Time (s)')
            plt.ylabel('Z Position (m)')
            plt.xlim(0, max(time_axis))
            plt.title('CoM Height Trajectory')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.pause(2)
            # print(rcm0,com_x[0],com_y[0])
            # plt.show()

        # Generate Gait starting with DSP
        #spno=2
        i=0
        for ti in np.arange(dt,2*Tf,dt):
            if i>=len(com_x):
                print(ti,Tf,"End of ref traj")
                break
            rcm=np.array([com_x[i],com_y[i],com_z[i]])
            rct=np.array([zmp_ref_x[i],zmp_ref_y[i],rct[2]])
            tcm = np.append(tcm, np.array([ti]), axis=0)
            ocm = np.append(ocm, np.array([rcm]), axis=0)
            oct = np.append(oct, np.array([rct]), axis=0)
            #     rcp=drcm*Ts
            #     r2=rcm+(1/sspbydsp)*np.linalg.norm(rcm-rct)*drcm/np.linalg.norm(drcm)
            #     r2[2]=rcm[2]/sspbydsp
            #     r3=r1+2*(r2-r1)
            #     r3[2]=r1[2]

            # print(rcm,r1,r2,r3)
            if (i+1)%int(step_time/dt)==0:# or ti%step_time<dt: #time to change foot
                print(ncyc,ti)
                # Foot placement data
                if spno==2: #DSP no change in foot placement
                    r1=self.o_left.copy()
                    r3=self.o_right.copy()
                else:
                    r3=np.array([zmp_ref_x[i+1],zmp_ref_y[i+1],rct[2]])
                    trn.cntplane(r3, 1)
                    r3[2]=trn.cntpos[2]
                    print(r1,r3)

                ftplac.append([ti, r1, rct, r3])
                #print(ti,ti%step_time,ftplac[-1])

                self.updateGait(tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac)
                if spno==1:
                    Stlr = np.array([1, 1]) - Stlr 
                    # rct0 = rct.copy()      
                    rct[2]=r3[2]
                    r1 = r3.copy()          
                else: #No DSP in MPC after first DSP
                    spno=1
                t0 = ti
                tcyc.append(t0)
                ncyc = ncyc + 1
                #spno = 3 - spno
                # if spno==2:
                #     rct=r2.copy()
                # else:
                #     rct = r3.copy()
                #     r1 = r3.copy()
                rcm0 = rcm.copy()
                # drcm0 = drcm.copy() #(ocm[-1,:] - ocm[-2]) / dt

                # plt.figure(18)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm, (np.array([sublist[0] for sublist in ocm])))
                # print(oct[-1])
                # plt.plot(tcm[-1],oct[-1][0],'*r')
                # plt.plot(tcm, (np.array([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[-1],oct[-1][1],'*b')
                # plt.figure(19)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[0:-1],(np.diff([sublist[0] for sublist in ocm])/dt))
                # plt.plot(tcm[0:-1],(np.diff([sublist[1] for sublist in ocm])/dt))


                oLeft = self.oLtraj[-1]
                oRight = self.oRtraj[-1]
                tcm = np.empty((0))
                ocm = np.empty((0, 3))
                oct = np.empty((0, 3))
                if ti >= Tf:
                    break
            i=i+1
            # if vis==1 and (ti%0.01)<dt:
            #     print(ti) # ,drcm,rcm,r2)
            #     plt.figure(10)
            #     # plt.plot(ctime+data.time,qcp[2],'.r')
            #     # plt.plot(ti,drcm[0],'*r',ti,drcm[1],'*g',ti,drcm[2],'*b')
            #     plt.xlabel('Time')
            #     plt.ylabel('dq/dt')
            #     plt.pause(0.001)
        # plt.show()
        # print(self.tCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz

    # def LIPMmpc(self,):

    # Update gait from simplified model of one cycle
    def updateGait(self,tcm, ocm, oct, oLeft, oRight, ncyc, Stlr, spno, ftplac):
        self.tCMtraj = np.append(self.tCMtraj, tcm, axis=0)
        self.oCMtraj = np.append(self.oCMtraj, ocm, axis=0)
        self.oCPtraj = np.append(self.oCPtraj, oct, axis=0)
        if spno == 1:  # SSP
            if Stlr[0] == 1:  # left foot is stance
                if abs(oRight[2] - ftplac[ncyc][3][2])>self.zSw/2: #Increase Step height on stairs if needed
                    self.zSw=2*abs(oRight[2] - ftplac[ncyc][3][2])

                self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                self.oLtraj = np.append(self.oLtraj, oct, axis=0)
                self.tRtraj = np.append(self.tRtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oRtraj = np.append(
                    np.append(self.oRtraj, np.array([0.5 * (oRight + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                    np.array([ftplac[ncyc][3]]), axis=0)
            else:
                if abs(oLeft[2] - ftplac[ncyc][3][2])>self.zSw/2: #increase step height if needed
                    self.zSw=2*abs(oLeft[2] - ftplac[ncyc][3][2])
                self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                self.oRtraj = np.append(self.oRtraj, oct, axis=0)
                self.tLtraj = np.append(self.tLtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                self.oLtraj = np.append(
                    np.append(self.oLtraj, np.array([0.5 * (oLeft + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                    np.array([ftplac[ncyc][3]]), axis=0)
        else:  # DSP
            Stlr = np.array([1, 1]) - Stlr
            self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
            self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
            if Stlr[0] == 1:  # left foot is stance
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]),
                                    bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]),
                                    bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]),
                                    bc_type='clamped')
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]), bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]), bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]), bc_type='clamped')
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
            else:
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]), bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]), bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                    [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]), bc_type='clamped')
                self.oRtraj = np.append(self.oRtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)
                csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]),
                                    bc_type='clamped')
                csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]),
                                    bc_type='clamped')
                csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                    np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]),
                                    bc_type='clamped')
                self.oLtraj = np.append(self.oLtraj, np.transpose(
                    np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                np.array([csplz(tcm)]), axis=0)), axis=0)

    
    # Joint angles from cartesian trajectories of Kondo humanoid robot
    def cart2joint(self,model, data0, ttraj, WLN, zeroAM):

        data = deepcopy(data0)
        qtraj = []
        AM_CM = []
        delt = ttraj[1] - ttraj[0]
        Lw = np.zeros(3)
        # self.tau = []
        rcom=data.subtree_com[0]
        gradH0=np.zeros([model.nv])            
        for ti in ttraj:
            mj.mj_fwdPosition(model, data);
            # Current joint angles
            q0 = data2q(data)
            drcom=(data.subtree_com[0]-rcom)/delt
            rcom = data.subtree_com[0].copy()
            # Desired traj
            ocm = np.array([self.oCMx(ti), self.oCMy(ti), self.oCMz(ti)])
            oLeft = np.array([self.oLx(ti), self.oLy(ti), self.oLz(ti)])
            oRight = np.array([self.oRx(ti), self.oRy(ti), self.oRz(ti)])
            qdes,gradH0 = numik(model, data, q0, delt, ocm, oLeft, oRight, self.ub_jnts, gradH0, WLN, self.k_ub, zeroAM)  # 0 - Upperbody locked and Nonzero ang. momentum about COM, 1 - ZAM abt COM using Arms swing
            data = q2data(data, qdes)
            qtraj.append(qdes)
            if ti%0.01<delt: print(ti)

            # Angular momentum matrix
            Iwb = np.zeros([3, model.nv])
            mj.mj_angmomMat(model, data, Iwb, 0)
            # Angular momentum
            AM_CM.append((Iwb @ (qdes-q0)/delt)) #tau is change in ang momentum
            # Lw = Iwb @ (qdes - q0) / delt
            # plt.figure(16)
            # plt.plot(drcom[0],AM_CM[-1][1],'*r')
            # plt.pause(0.0001)
        qi = []
        # dqi=[]
        self.CAMTraj=np.array(AM_CM)
        self.qTraj_des = np.append(self.qTraj_des, [qdes], axis=0)            
        self.AM_CMspl[0] = CubicSpline(ttraj, self.CAMTraj[:, 0], bc_type='clamped')
        self.AM_CMspl[1] = CubicSpline(ttraj, self.CAMTraj[:, 1], bc_type='clamped')
        self.AM_CMspl[2] = CubicSpline(ttraj, self.CAMTraj[:, 2], bc_type='clamped')

        self.qTraj_des = np.array(qtraj)
        # if len(ttraj) > 2:
        #     fig1 = plt.figure(1)
        #     fig2 = plt.figure(2)
        #     for i in np.arange(6, model.nv):
        #         plt.figure(1)
        #         plt.plot(ttraj, qtraj[:, i],label=f'th_{i}')
        #         plt.figure(2)
        #         plt.plot(ttraj, np.append(0, 1 / delt * np.diff(qtraj[:, i])),label=f'dth_{i}')
        #     plt.figure(1)
        #     plt.legend()
        #     fig1.savefig('qtraj.jpeg')
        #     plt.close(fig1)
        #     plt.figure(2)
        #     plt.legend()
        #     fig2.savefig('dqtraj.jpeg')
        #     plt.close(fig2)
        #     # Saving the data:
        #     with open('qtraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #         pickle.dump([ttraj, qtraj], f)

        #Plot position vs velocity in subplot
        # numrow=4
        # numcol=3
        # #Find joint velocities by numerical differentiation
        # dqtraj = np.zeros(qtraj.shape)
        # for i in range(model.nv):
        #     dqtraj[:, i] = np.append(data0.qvel[i], 1 / delt * np.diff(qtraj[:, i]))

        # fig2, ax2 = plt.subplots(nrows=numrow, ncols=numcol)
        # for i in range(12):
        #     ax2[i // numcol, i % numcol].plot(qtraj[:,i],dqtraj[:,i]) #, col[i % 6])
        #     ax2[i // numcol, i % numcol].set_title(f'Joint {i}')
        #     ax2[i // numcol, i % numcol].set_xlabel('Position (rad    )')
        #     ax2[i // numcol, i % numcol].set_ylabel('Velocity (rad/s)')
        #     ax2[i // numcol, i % numcol].grid(visible=None, which='major', axis='both')
        # # fig2.tight_layout()

        # plt.pause(1)
        # plt.show()

        return self.qTraj_des



    def mjfc(self,model,data):
        # Normal Contact Force
        self.fcl = np.zeros([3])
        self.fcl[0]=data.xfrc_applied[model.site_bodyid[data.site("left_foot_site").id]][2]
        self.fcr = np.zeros([3])
        self.fcr[0] = data.xfrc_applied[model.site_bodyid[data.site("right_foot_site").id]][2]
        self.fcn = self.fcl[0] + self.fcr[0]
        self.rfl = np.zeros([3])
        self.rfr = np.zeros([3])
        self.rf = np.zeros([3])
        for i in np.arange(0, data.ncon):
            # conid = min(data.contact[i].geom1,data.contact[i].geom2)
            fci = np.zeros([6])
            try:
                mj.mj_contactForce(model, data, i, fci)
                # print(fci[0])
                self.fcn = self.fcn + abs(fci[0])
                self.rf = self.rf + np.array(data.contact[i].pos) * abs(fci[0])  # pos*normal force
                #fc0 = np.matmul(np.array(data.contact[i].frame).reshape((3, 3)), -fci[0:3])  # force vector in world frame

                if model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("left_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("left_foot_site").id]:  # Left foot body
                    self.fcl = self.fcl + fci[0:3]
                    self.rfl = self.rfl + np.array(data.contact[i].pos) * abs(fci[0])
                elif model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("right_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("right_foot_site").id]:  # Right foot body
                    self.fcr = self.fcr + fci[0:3]
            except:
                print('no contact')

    #def mjaref(self,cnt):


    def init_controller(self,Kp,Kv,Ki):
        self.Kp=Kp
        self.Kv=Kv
        self.Ki=Ki

    def controller(self,model,data):
        self.tau = 0 * data.ctrl
        # Desired traj
        # thdes=np.zeros([model.nu])
        # dthdes = np.zeros([model.nu])
        self.qdes = np.zeros([model.nv])
        self.dqdes = np.zeros([model.nv])
        self.ddqdes = np.zeros([model.nv])

        # Current joint angles
        q = data2q(data)

        # current COM position
        self.rcom = data.subtree_com[0].copy()
        # current COP position
        self.ocp_des = np.array( [self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])  # Desired COP Pos

        #Desired cartesian trajt
        self.ocm_des = np.array( [self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])  # Desired COM Pos
        self.oL_des = np.array([self.oLx(data.time), self.oLy(data.time), self.oLz(data.time)])  # Desired Left Pos
        self.oR_des = np.array([self.oRx(data.time), self.oRy(data.time), self.oRz(data.time)])  # Desired Right Pos

        # Normal contact force
        self.mjfc(model, data)

        # COP pos
        if abs(self.fcn) > 0:
            self.rcop = self.rf / abs(self.fcn)
        else:  # No contact
            self.rcop = self.ocp_des  # np.array([np.nan, np.nan, np.nan])

        if self.ocp_des[2] > self.rcom[2]:
            if abs(self.fcn) == 0:
                self.rcop[2] = 0
            self.ocp_des[0] = self.ocm_des[0] + (self.ocm_des[0] - self.ocp_des[0]) * abs(
                self.rcop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] + (self.ocm_des[1] - self.ocp_des[1]) * abs(
                self.rcop[2] - self.ocm_des[2]) / (self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.rcop[2]
        if abs(self.fcn) == 0:
            self.rcop = self.ocp_des  # np.array([np.nan, np.nan, np.nan])
        # Angular momentum
        self.Iwb = np.zeros([3, model.nv])
        Lw = self.Iwb @ (0 * data.qvel)
        mj.mj_angmomMat(model, data, self.Iwb, 0)
            
        # Follow the pre-defined joint trajectory

        for i in np.arange(0, model.nv):
            self.qdes[i] = self.qspl[i](1 * data.time)  # + self.q_err[i]
            self.dqdes[i] = self.qspl[i](1 * data.time, 1)
            self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
            #     # qdes[i+6]=thdes[i].copy()
            # data.qacc[i]=ddqdes[i].copy()
        #Desired joint velocity
        self.dqdes = 1*(self.qdes - q) / model.opt.timestep
        # PID Controller for torque control mode
        self.tau_PID = 0*data.ctrl
        self.tau_PID = self.PIDcontrol(model, data)
        if self.posCTRL==True:
            self.tau=self.qdes[6:model.nv].copy() #Position control mode
        else:
            # Gear Ratio for tau_PID to data.ctrl
            for i in range(model.nu):
                self.tau[i] = self.tau_PID[i] / model.actuator_gear[i][0]
        # print(max(abs(self.tau)),max(abs(self.dqdes)))
        # print(min(abs(self.tau)))
        # return self.tau_PD


        # # PD control
        # self.tau += self.PDcontrol(model, data)
        #
        # self.q_err = np.zeros([model.nv])
        #
        # # Ankle Torque Control
        # # tau_ankle = 0
        #
        # # COM control # Choi et al. without ZMP control
        # self.tau += self.COMcontrol(model,data)
        #
        # # ZMP control # Choi et al.
        # self.tau += self.ZMPcontrol(model,data)
        #
        # # Angular momentum control
        # self.tau += self.AMcontrol(model,data)
        #
        # # FW control # FW inverted pendulum to control COM
        # self.tau +=self.FWcontrol(model,data)

        # Controller data.ctrl
        #data.ctrl = tau_PD + COMctrl * tau_COM + AMctrl * tau_AM + FWctrl * tau_FW

        # Inverse dynamics for model-based control
        # tauid = tauinvd(model, data, self.ddqdes)
        # print(max(data.actuator_force))
        # self.tau +=tauid

        #Torque limits
        # np.clip(self.tau, -1.37, 1.37, out=self.tau)

        return self.tau

    def PIDcontrol(self, model, data):
        # Current joint angles
        q = data2q(data)
        # Desired joint angles

        # for i in np.arange(0, model.nv):
        #     self.qdes[i] = self.qspl[i](1 * data.time) #+ self.q_err[i]
        #     self.dqdes[i] = self.qspl[i](1 * data.time, 1)
        #     self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
        #     # qdes[i+6]=thdes[i].copy()
        #     # data.qacc[i]=ddqdes[i].copy()
        # PID Controller
        self.tau_PID = 0 * data.ctrl

        for i in range(model.nu):
            self.tau_PID[i] = (self.Kp[i]) * (self.qdes[i + 6] - q[i + 6]) + self.Kv[i] * (
                    self.dqdes[i + 6] - data.qvel[i + 6]) + self.Ki[i] * (self.Eqdt[i+6])
            #Integral error
            self.Eqdt[i+6]=self.Eqdt[i+6]+(self.qdes[i + 6] - q[i + 6])*model.opt.timestep

        return self.tau_PID


    def COMcontrol(self,model, data):
        self.ocm_des = np.array(
            [self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])  # Desired COM Pos
        self.rcom = data.subtree_com[0].copy()  # current COM position
        # drcom = data.subtree_linvel[0].copy()
        if self.COMctrl == 0:
            self.tau_COM = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, rcom, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            Jct = np.zeros((3, model.nv))  # Stance foot center position jacobian
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0]>self.o_left[0]:
                    mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm - Jct
            delx2 = 1 * self.COMctrl * (self.ocm_des - self.rcom) / model.opt.timestep
            Jt2 = np.matmul(J2, InJ1)
            # dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            dq_err=1*np.matmul(np.linalg.pinv(Jcm - Jct), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_COM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_COM[i] = self.Kp[i] * (dq_err[i + 6] * model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_COM

    def ZMPcontrol(self,model, data):
        self.ocp_des = np.array(
            [self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])  # Desired COP Pos
        # Normal contact force
        self.mjfc(model, data)
        # COP pos
        if abs(self.fcn) > 0:
            self.rcop = self.rf / abs(self.fcn)
        else: #No contact
            self.rcop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        if self.ocp_des[2]>self.rcom[2]:
            if abs(self.fcn) == 0:
                self.rcop[2]=0
            self.ocp_des[0]=self.ocm_des[0] + (self.ocm_des[0]-self.ocp_des[0])*abs(self.rcop[2]-self.ocm_des[2])/(self.ocp_des[2]-self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] +  (self.ocm_des[1] - self.ocp_des[1])*abs(self.rcop[2] - self.ocm_des[2])/(self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.rcop[2]
        if abs(self.fcn) == 0:
            self.rcop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        # self.rcom = data.subtree_com[0].copy()  # current COM position
        # drcom = data.subtree_linvel[0].copy()
        if self.ZMPctrl == 0:
            self.tau_ZMP = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, rcom, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            Jct = np.zeros((3, model.nv))  # Stance foot center position jacobian
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0] > self.o_left[0]:
                    mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm - Jct
            delx2 =  1 * self.ZMPctrl * (self.ocp_des - self.rcop) / model.opt.timestep
            # print(delx2,self.ocp_des,self.rcop)
            Jt2 = np.matmul(J2, InJ1)
            # dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            dq_err=1*np.matmul(np.linalg.pinv(Jcm - Jct), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_ZMP = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_ZMP[i] = self.Kp[i] * (dq_err[i + 6]* model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_ZMP


    def sim(self,model,data,trn,simfreq,simend,saveVid=False):

        # Humanoid parameters
        self.mj2humn(model, data)

        # Parameters of SIP
        # sip = humn2SIP(self,trn, model, data)
        # sip.trn.cntplane(sip.qcp,sip.spno)

        # Mocap body we will control with our mouse. for COP
        mocap_id_COP = model.body("COP").mocapid[0]
        mocap_id_COM_des = model.body("COM_des").mocapid[0]

        ActData = []
        DesData = []
        
        if saveVid == True:
            # Create a renderer
            renderer = mj.Renderer(model, width=1280, height=720)

            frames = []
            #duration = 5  # seconds
            framerate = int(simfreq/4) #60 #fps # saved Vid is 0.25 x real-time

        with mj.viewer.launch_passive(
                model=model, data=data, show_left_ui=True, show_right_ui=False
        ) as viewer:

            # Initialize the camera view to that of the free camera.
            mj.mjv_defaultFreeCamera(model, viewer.cam)

            # Visualization.
            # viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE #Site frame
            # viewer.opt.flags[2] = 1  # Joints
            # viewer.opt.flags[4] = 1  # Actuators
            # viewer.opt.flags[14] = 1 #Contact Points
            viewer.opt.flags[16] = 1 #Contact Forces
            viewer.opt.flags[18] = 1 #Transparent
            # viewer.opt.flags[20] = 1 #COM

            if saveVid == True:
                # Make new camera, set distance.
                camera = mj.MjvCamera()
                mj.mjv_defaultFreeCamera(model, camera)
                camera.distance = self.cam_dist

                # Enable contact force visualisation.
                scene_option = mj.MjvOption()
                scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            # Update scene and render
            # viewer.sync()

            # print("Press any key to proceed.")
            # key = keyboard.read_key()
            print("Simulation starting.....")
            time.sleep(2)

            while viewer.is_running() and data.time < simend:
                time_prev = data.time

                clock_start = time.time()
                while (data.time - time_prev < 1.0 / simfreq) and data.time < simend:
                    # Current joint angles
                    #q = data2q(data)
                    #dq = data.qvel.copy()

                    # Spring damper contact model
                    # data.xfrc_applied=self.sdmodel(model,data).copy()

                    #Forward dyn
                    data.ctrl=self.controller(model,data).copy()
                    # data.ctrl=0*data.ctrl #Zero Torque
                    mj.mj_step(model, data)  # Forward dynamics

                    # print('solver_fwdinv[2] flag',data.solver_fwdinv)

                    # Inv dyn
                    # mj.mj_inverse(model, data)  # Inverse  dynamics
                    # print(data.qfrc_inverse[self.left_legjnts],data.qfrc_inverse[self.right_legjnts])
                    # for i in np.arange(0, model.nv):
                    #     self.qdes[i] = self.qspl[i](1 * data.time)
                    #     self.dqdes[i] = self.qspl[i](1 * data.time, 1)
                    #     self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
                    #     # qdes[i+6]=thdes[i].copy()
                    #     # data.qacc[i]=ddqdes[i].copy()
                    # data = q2data(data, self.qdes)  # Joint traj for inv dynamics
                    # data.qvel = self.dqdes
                    # data.qacc = self.ddqdes
                    # data.time = data.time + 1 / simfreq

                    # Work Done

                    for i in range(model.nu):
                        #self.WD += abs(data.ctrl[i] * model.actuator_gear[i][0] * data.qvel[i+6] * model.opt.timestep)
                        self.WD += abs(data.actuator_force[i] * data.qvel[i+6] * model.opt.timestep)


                if (data.time >= simend):
                    break

                # MuJoCo to Humanoid Parameters
                self.mj2humn(model, data)
                # print(data.ctrl[self.left_legjnts - 6], data.ctrl[self.right_legjnts - 6])
                # plt.figure(10)
                # plt.plot(data.time, self.dr_com[0], '.r')
                # plt.plot(data.time, self.dr_com[1], '.g')
                # plt.plot(data.time, self.dr_com[2], '.b')
                # plt.pause(0.001)

                # Normal contact force
                # print('self.fmj:')
                self.mjfc(model, data)
                # print('ncon,efc_force',data.ncon,data.efc_force)
                # fmj,fsd,fdef,dddef=mjforce(model,data)
                # print('mj_force',fmj)
                # time.sleep(10)
                # print('nefc,efc_pos,efc_force',data.nefc,data.efc_pos,data.efc_force)
                # print('efc_aref,force',data.efc_aref,data.efc_force)
                #COP COM mocap
                if abs(self.fcn) > 0:
                    rcop = self.rf / abs(self.fcn)
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = rcop
                else:
                    rcop = np.array([np.nan, np.nan, np.nan])
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = np.array([1000, 1000, 1000])

                sipCnt=np.array([self.oCPx(data.time),self.oCPy(data.time),self.oCPz(data.time)])
                sipCOM=np.array([self.oCMx(data.time),self.oCMy(data.time),self.oCMz(data.time)])
                data.mocap_pos[mocap_id_COM_des, 0:3] = sipCOM
                #Visualize SIP Model
                # iterator for decorative geometry objects
                idx_geom = 0
                for i in range(100):
                    # mj Geometry from vyankatesh's code
                    sipPt=sipCOM+i/100*(sipCnt-sipCOM)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx_geom],
                                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                        size=[0.005, 0, 0],
                                        pos=sipPt,
                                        mat=np.eye(3).flatten(),
                                        rgba=np.array([1, 0, 0, 0.3]))
                    idx_geom += 1
                    viewer.user_scn.ngeom = idx_geom
                    # Reset if the number of geometries hit the limit
                    if idx_geom > (viewer.user_scn.maxgeom - 50):
                        # Reset
                        idx_geom = 1

                # Reproduce MuJoCo Forces
                # fmj,fsd =mjforce(model,data)

                # Save data for plots [t,q,dq,rcom,drcom,oL,oR,rcop,fcl,fcr,tau,I*dq,WD]
                self.updateTrajData(model, data)

                # Update scene and render
                viewer.sync()
                if saveVid == True:
                    # Set the lookat point to the humanoid's center of mass.
                    camera.lookat = self.r_com
                    renderer.update_scene(data, camera, scene_option)
                    # initialize the geom, here is a ball, if you want the label only, just change it to the mujoco.mjtGeom.mjGEOM_LABEL
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.0001, 0.0001, 0.0001]),  # label_size
                        pos=self.r_com + 0.5*self.r_com[2]*np.array([2.0, 0.0, 1.0]),  # label position
                        mat=np.eye(3).flatten(),  # label orientation
                        rgba=np.array([1, 0, 0, 1])  # red for the sphere
                    )
                    # add label
                    geom.label = "0.25 x real-time"
                    # add geom into scene
                    renderer.scene.ngeom += 1
                    pixels = renderer.render()
                    frames.append(pixels)

                time_until_next_step = 1 / simfreq - (time.time() - clock_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                print(data.time)
                # time.sleep(500)

        if saveVid==True:
            # Convert frames to a MoviePy clip and save
            clip = ImageSequenceClip(frames, fps=framerate)
            # saving video clip as gif or mp4
            clip.write_gif("simulation_video.gif")
            # clip.write_videofile("simulation_video.mp4")

        #End of simulation
        return DesData, ActData
    
    def plotData(self,model,data):
        #Plot des and act joint traj
        self.pltqTraj(self.tTraj,self.qTraj_des)
        # self.pltfc()
        # self.pltcom()
        # self.pltwd()
        plt.show()
    
    def pltqTraj(self,ttraj,qtraj,linestyl='-',overlap=0):
        #Plot joint angles in subplots
        print(np.shape(ttraj),np.shape(qtraj))
        if overlap==0:
            fig1, self.qpltAaxis = plt.subplots(nrows=5, sharex=True, num=100)  # joint angles
        ax1=self.qpltAaxis
        col = ['r', 'g', 'b', 'c', 'm', 'k']
        for i in self.left_legjnts:
            ax1[0].plot(ttraj, qtraj[:, i] * 180 / np.pi, col[i - min(self.left_legjnts)],linestyle=linestyl,
                        label='$\u03B8_{' + str(i - min(self.left_legjnts) + 1) + '}$')
        for i in self.right_legjnts:
            ax1[1].plot(ttraj, qtraj[:, i] * 180 / np.pi, col[i - min(self.right_legjnts)],linestyle=linestyl,
                        label='$\u03B8_{' + str(i - min(self.right_legjnts) + 1 + len(self.left_legjnts)) + '}$')

        for i in self.ub_jnts[0:2]:
            ax1[2].plot(ttraj, qtraj[:, i] * 180 / np.pi, col[(i - min(self.ub_jnts))%6],linestyle=linestyl, label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')
        for i in self.ub_jnts[2:6]:
            ax1[3].plot(ttraj, qtraj[:, i] * 180 / np.pi, col[i - min(self.ub_jnts) - 2],linestyle=linestyl, label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')
        for i in self.ub_jnts[6:10]:
            ax1[4].plot(ttraj, qtraj[:, i] * 180 / np.pi, col[i - min(self.ub_jnts) - 6],linestyle=linestyl, label='$\u03B8_{' + str(
                i - min(self.ub_jnts) + 1 + len(self.left_legjnts) + len(self.right_legjnts)) + '}$')

        ax1[len(ax1)-1].set_xlabel('Time (s)')
        for i in range(len(ax1)):
            ax1[i].set_ylabel('Angle (deg)')
            ax1[i].grid(visible=None, which='major', axis='both')
            ax1[i].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=max(1,len(ax1[i].lines)))


# Copy data.qpos (with quaternion) to q (with euler angles)
def data2q(data):
    q=0*data.qvel.copy()
    qqt = data.qpos[3:7].copy()
    qeulr=quat2euler(qqt)
    for i in np.arange(0,3):
        q[i]=data.qpos[i].copy()
    for i in np.arange(3,6):
        q[i]=qeulr[i-3].copy()
    for i in np.arange(6,len(data.qvel)):
        q[i]=data.qpos[i+1].copy()
    return q

# Copy q (with euler angles) to data.qpos (with quaternion)
def q2data(data,q0):
    import copy
    qqt=euler2quat(q0[3:6])
    for i in np.arange(0,3):
        data.qpos[i]=copy.copy(q0[i])
    for i in np.arange(3,7):
        data.qpos[i]=copy.copy(qqt[i-3])
    for i in np.arange(7,len(data.qpos)):
        data.qpos[i]=copy.copy(q0[i-1])
    return data


def COT(m,WD,dist):
    COT = WD / (m*9.81*dist)
    return COT

# Numerical inverse kinematics of the robot
def numik(model,data,q0, delt, ocm,oleft,oright,ubjnts,gradH0,WLN,k_ub,zeroAM):
    data=q2data(data,q0)
    mj.mj_fwdPosition(model, data)
    ocmi=data.subtree_com[0].copy() #current COM position

    olefti=data.site('left_foot_site').xpos.copy() #current Left foot position
    orighti=data.site('right_foot_site').xpos.copy() #current right foot position

    quat_hip=data.qpos[3:7].copy()
    quat_left=np.zeros([4])
    quat_right=np.zeros([4])

    quat_conj=np.zeros([4])
    err_quat=np.zeros([4])
    err_ori_hip = np.zeros([3])
    err_ori_left=np.zeros([3])
    err_ori_right=np.zeros([3])
    # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
    mujoco.mju_negQuat(quat_conj, quat_hip)
    mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
    mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

    if model.nu-len(ubjnts)>=6:
        mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_left)
        mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

        # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
        mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_right)
        mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

    # Error to minimize for numerical inverse kinematics
    delE=np.linalg.norm(oleft-olefti)+np.linalg.norm(err_ori_left)+np.linalg.norm(oright-orighti)+np.linalg.norm(err_ori_right)+np.linalg.norm(ocm-ocmi)
    k=1 # Increase k to increase accuracy of q
    while delE>1e-8:
        Jcm = np.zeros((3, model.nv)) # COM position jacobian
        mj.mj_jacSubtreeCom(model, data, Jcm,0)
        Jwb = np.zeros((3, model.nv))  # Base orientation jacobian
        Jwb[0:3,3:6]=np.eye(3)
        Jvleft = np.zeros((3, model.nv)) # Left foot center jacobian
        Jwleft = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvleft, Jwleft, model.site('left_foot_site').id)
        #mj.mju_mat2Quat(quat_left, data.site(model.site('left_foot_site').id).xmat)
        #mj.mju_negQuat(quat_left, quat_left)
        #mj.mju_quat2Vel(err_ori_left, quat_left, 1.0)
        Jvright = np.zeros((3, model.nv)) # right foot center jacobian
        Jwright = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvright, Jwright, model.site('right_foot_site').id)
        #mj.mju_mat2Quat(quat_right, data.site(model.site('right_foot_site').id).xmat)
        #mj.mju_negQuat(quat_right, quat_right)
        #mj.mju_quat2Vel(err_ori_right, quat_right, 1.0)
        # lock upperbody
        if ubjnts.size:
            #ubjnts=np.arange(18,model.nv) #kondo khr3hv
            #ubjnts=np.append([6,7,8],np.arange(21,model.nv)) #MuJoCo humanoid model
            Jub = np.zeros((len(ubjnts), model.nv))  # Base orientation jacobian
            Jub[:,ubjnts]=np.eye(len(ubjnts))
        # Ang momentum
        Iwb=np.zeros([3,model.nv])
        mj.mj_angmomMat(model, data, Iwb, 0)

        Avec=np.zeros([18+len(ubjnts)+3+3,model.nv])
        bvec=np.zeros([18+len(ubjnts)+3+3])
        #COM traj
        Avec[0:3,0:model.nv]=Jcm
        bvec[0:3] = ocm-ocmi
        #Hip orient
        Avec[3:6,0:model.nv]=Jwb
        bvec[3:6]=err_ori_hip #np.zeros([3])
        #Left ankle lin vel
        Avec[6:9,0:model.nv]=Jvleft
        bvec[6:9] = oleft-olefti
        #Left ankle ang vel
        Avec[9:12,0:model.nv]=Jwleft
        bvec[9:12] = err_ori_left #np.zeros([3])
        #Right ankle lin vel
        Avec[12:15,0:model.nv]=Jvright
        bvec[12:15] = oright-orighti
        #Right ankle ang vel
        Avec[15:18,0:model.nv]=Jwright
        bvec[15:18] = err_ori_right #np.zeros([3])
        #Upper body joints
        invWroot=np.eye(model.nv) #WLN
        if ubjnts.size:
            qref=0*np.ones([model.nv])            
            #Lock upper body joints
            Avec[18:18+len(ubjnts),0:model.nv]=Jub
            bvec[18:18+len(ubjnts)] = ( qref[ubjnts] - q0[ubjnts] )/1 #np.zeros([len(ubjnts)])#
            #Zero angular momentum using upper body joints
            Avec[18+len(ubjnts):18+len(ubjnts)+3,0:model.nv]=Iwb
            bvec[18+len(ubjnts):18+len(ubjnts)+3] = np.zeros([3])
            # #Sym. motion of upper body joints
            # Avec[18+len(ubjnts)+3:18+len(ubjnts)+6,ubjnts]=Iwb[:,ubjnts]
            # bvec[18+len(ubjnts)+3:18+len(ubjnts)+6] = np.zeros([3])
        else:
            Avec[18:18+3,0:model.nv]=Iwb
            bvec[18:18+3] = np.zeros([3])


        #J=np.append(np.append(np.append(np.append(Jcm,Jwb,axis=0), np.append(Jvleft,Jwleft,axis=0),axis=0), np.append(Jvright,Jwright,axis=0), axis=0),Jub,axis=0)
        #delx=np.append(np.append(np.append( np.append(ocm-ocmi,np.zeros([3]),axis=0), np.append(oleft-olefti,np.zeros([3]),axis=0),axis=0), np.append(oright-orighti,np.zeros([3]),axis=0), axis=0),np.zeros([model.nv-18]),axis=0)
        if (model.nv-len(ubjnts) )<18: #Planer biped
            eqnJ1=np.append(np.array([0,2]),np.array([6,8,10, 12,14,16])) # remove foot orientation
        else: #Spatial biped
            eqnJ1 = np.append(np.array([0, 1, 2]), np.arange(6, 18))

        J1 = Avec[eqnJ1, :].copy()
        delx1 = bvec[eqnJ1].copy()/delt
        dqN1 = np.matmul(np.linalg.pinv(J1), delx1)
        InJ1=np.eye(model.nv)-np.matmul(np.linalg.pinv(J1),J1)

        eqnJ2=np.append(np.array([3,4,5]),np.arange(18,18+len(ubjnts)))

        if WLN==False:
            invWroot=np.eye(model.nv)
        J2 = Avec[eqnJ2, :] @ invWroot #.copy()
        delx2 = bvec[eqnJ2].copy()/delt
        Jt2=np.matmul(J2,InJ1)
        dqN2=dqN1+ invWroot @ np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2,dqN1))
        InJ2=InJ1-np.matmul(np.linalg.pinv(Jt2),Jt2)
        dq=dqN2.copy() #+np.matmul(InJ2,qref-qi)
        # print(asd)


        if delt<1:
            # Integrate joint velocities to obtain joint positions.
            # q0=q0+dq*delt
            data=q2data(data,q0)
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, delt)
            # np.clip(q, *model.jnt_range.T, out=q)

            # q0=data2q(data)
            q0 = 0 * data.qvel.copy()
            qqt = q[3:7].copy()
            qeulr = quat2euler(qqt)
            for i in np.arange(0, 3):
                q0[i] = q[i].copy()
            for i in np.arange(3, 6):
                q0[i] = qeulr[i - 3].copy()
            for i in np.arange(6, len(data.qvel)):
                q0[i] = q[i + 1].copy()

            return q0,gradH0

        while delE<=(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
            # Integrate joint velocities to obtain joint positions.
            # qi=q0+dq*delt/k
            data=q2data(data,q0)
            q = data.qpos.copy()  # Note the copy here is important.
            mujoco.mj_integratePos(model, q, dq, delt/k)
            # np.clip(q, *model.jnt_range.T, out=q)

            # q0=data2q(data)
            qi = 0 * data.qvel.copy()
            qqt = q[3:7].copy()
            qeulr = quat2euler(qqt)
            for i in np.arange(0, 3):
                qi[i] = q[i].copy()
            for i in np.arange(3, 6):
                qi[i] = qeulr[i - 3].copy()
            for i in np.arange(6, len(data.qvel)):
                qi[i] = q[i + 1].copy()

            data=q2data(data,qi)
            mj.mj_fwdPosition(model, data)
            ocmi = data.subtree_com[0]
            olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
            orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
            # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
            # mujoco.mju_negQuat(quat_conj, quat_hip)
            # mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
            # mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

            if model.nu - len(ubjnts) >= 60:
                mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_left)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

                # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
                mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_right)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

            # print(np.linalg.norm(np.matmul(J,dq)-delx))
            k=2*k
            if k>2:
                print('Error (x_des-x_cur) is diverging')


        if delE>(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
            delE = np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)
            q0 = qi.copy()
            k=1
        """
        else:
            print('Error (x_des-x_cur) is diverging')
            k=2*k
            data = q2data(data, q0)
            mj.mj_fwdPosition(model, data)
            ocmi = data.subtree_com[0]
            olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
            orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
        """

    return q0, gradH0

# inv dynamics in MuJoCo
def tauinvd(model,data,ddqdes):
    data.qacc=ddqdes.copy()
    #mj.mj_fwdActuation(model,data)
    mj.mj_inverse(model,data)
    #mj.mj_fwdActuation(model,data)
    tau=data.qfrc_inverse[6:].copy()
    return tau

# Find terrain parameters (solref) given solimp=[d0,dwidth,width,midpt,power] and zeta
class trnparam:
    def __init__(self,nocp,zeta,zpln):
        i=0
        self.zeta=zeta
        self.zpln=zpln
        self.nocp=nocp
        # self.A=[]
        self.r=[]
        self.rdot=[]
        self.aref=[]
        self.f=[]
        self.efc_f=[]
        self.q=[]
        self.dq=[]
        self.ddq=[]

    def mjparam(self, model):
        self.solimp=[]
        self.solref=[]
        self.pos=[]
        self.size=[]
        self.xmean=[]
        i=0
        while model.geom_bodyid[i]==0:
            self.pos.append(model.geom_pos[i].copy())
            self.size.append(model.geom_size[i].copy())
            solimp=model.geom_solimp[i].copy()
            solref=model.geom_solref[i].copy()
            self.solimp.append(solimp)
            d0=solimp[0]
            dwidth=solimp[1]
            width=solimp[2]
            midpt=solimp[3]
            power=solimp[4]
            #trn.solref = (-Stiffness, -damping)
            #self.solimp = [d0, dwidth, width, midpt, power]
            dmean=(d0+dwidth)/2
            deln=width*self.nocp
            xmean = deln / 2
            if solref[0]<0:
                # kn=9.81/deln
                dampratio=self.zeta
                stiffness=(9.81*(1-dmean)*dwidth*dwidth)/(xmean*dmean*dmean) #/self.zeta**2
                timeconst=1/(dampratio*np.sqrt(stiffness))
                #dampratio = 1 / (timeconst * np.sqrt(stiffness))
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                # wn=np.sqrt(stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                # damping=2*wn*self.zeta
            else:
                timeconst=solref[0] #np.sqrt(1/(9.81*(1-dwidth)/width)) # 1*solref[0]
                dampratio = solref[1]
                # kn=9.81/deln
                stiffness=1/((timeconst**2)*(dampratio**2))#9.81*(1-dwidth)/deln
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                #wn=np.sqrt(nocp*stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                #damping=2*zeta*wn/nocp #
            damping=2/timeconst #2/(dwidth*timeconst)
            i=i+1

            #stiffness=stiffness/nocp
            #damping=damping/nocp

            self.solref.append([-stiffness,-damping])
            self.xmean.append(xmean)

    def cntplane(self,cntpt,spno):
        i=0
        for pos in self.pos:
            size=self.size[i].copy()
            if cntpt[0]>(pos[0]-size[0]) and cntpt[0]<(pos[0]+size[0]):
                if cntpt[1] > (pos[1] - size[1]) and cntpt[1] < (pos[1] + size[1]):
                    self.cntgeomid=i
                    self.cntpos=self.pos[i].copy()
                    self.cntsize=self.size[i].copy()
                    if spno==1:
                        self.cntsolref = self.solref[i].copy()
                        self.cntsolimp = self.solimp[i].copy()
                        self.cntnocp = self.nocp
                        self.cntpos[2] +=  self.cntsize[2]
                    else:
                        self.cntsolref = [0.02, 1] #self.solref[i].copy()
                        self.cntsolimp = [0.9,0.95,0.001,0.5,2] #self.solimp[i].copy()
                        self.cntnocp = 2*self.nocp
                        self.cntpos[2] = cntpt[2] + 0.00036 #self.cntsolimp[2]/2.5
                        # self.cntnocp=2*self.cntnocp
                    # else:
                    #     self.qcp[2]=self.cntpos[2] - self.cntsolimp[2]
                    break
            i=i+1



# Add robot xml to scene xml
def addrobot2scene(xml_tree,robotpath):
    #xml_path = r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\mujoco\kondo\scene_defT.xml" #xml file (assumes this is in the same folder as this file)

    # get the full path
    #dirname = os.getcwd() #os.path.dirname(__file__)
    #abspath = os.path.join(dirname + "/" + xml_path)
    #xml_path = abspath

    # xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    # Change mass,pos, orientation and length of pendulum
    bodyeul=np.zeros([1,3])
    #model.geom_size[2,1]=l/2 # length of cylindrical rod
    for tag1 in root.findall("include"):
        tag1.attrib['file']=robotpath #' '.join(map(str, np.array([0,0,zpln]))) #change contact plane pos

    # xmltree.write('robotwithscene.xml')
    xml_str = ET.tostring(root)
    # ET.dump(root)
    # xml_path = 'robotwithscene.xml'
    return xml_str

def scenegen(xml_tree,trnnum,trnlength,trnwidth,trnheight):
    #xml_path = r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\mujoco\kondo\scene_defT.xml" #xml file (assumes this is in the same folder as this file)

    # get the full path
    #dirname = os.getcwd() #os.path.dirname(__file__)
    #abspath = os.path.join(dirname + "/" + xml_path)
    #xml_path = abspath

    # xmltree = ET.parse(xml_path)
    root = xml_tree.getroot()

    if trnnum==1: #Add one plane geom
        for tag1 in root.findall("worldbody"):
            geom = ET.SubElement(tag1, "geom")
            geom.set("name", "terrain1")
            geom.set("type", "plane")
            geom.set("size", f"{trnlength/2} {trnwidth} {0.001+trnheight}")
            geom.set("pos", f"0 0 0.0+{trnheight}")
            geom.set("rgba", "0.2 0.9 0.2 1")
    else: #Add boxes as terrain
        for tag1 in root.findall("worldbody"):
            for i in range(trnnum):
                if i==0:
                    geom = ET.SubElement(tag1, "geom")
                    geom.set("name", f"terrain{i+1}")
                    geom.set("type", "box")
                    geom.set("size", f"{0.5+trnlength/4} {trnwidth} {0.05+trnheight[i]/2}")
                    geom.set("pos", f"{-1+0.5+trnlength/4} 0 {-0.05+trnheight[i]/2}")
                    geom.set("rgba", f"{0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} 1")
                else:
                    geom = ET.SubElement(tag1, "geom")
                    geom.set("name", f"terrain{i+1}")
                    geom.set("type", "box")
                    geom.set("size", f"{trnlength/2} {trnwidth} {0.05+trnheight[i]/2}")
                    geom.set("pos", f"{i*trnlength} 0 {-0.05+trnheight[i]/2}")
                    geom.set("rgba", f"{0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} {0.25+0.5*i/trnnum*(trnheight[i]>0)} 1")



                
    # xmltree.write('robotwithscene.xml')
    xml_str = ET.tostring(root)
    # ET.dump(root)
    # xml_path = 'robotwithscene.xml'
    return xml_str

class mydataparam:
    def __init__(self, d0, dwidth, width, midpt, power,nocp,zeta,zpln):
        self.solimp = [d0, dwidth, width, midpt, power]
        deln=width*nocp
        kn=9.81/deln
        stiffness=9.81*(1-dwidth)/deln
        wn=np.sqrt(nocp*stiffness)
        #timeconst = 1 / (zeta * wn) #0.02 default
        damping=2*zeta*wn/nocp #
        #damping=2/(dwidth*0.02)#timeconst
        self.solref=[-stiffness,-damping]
        self.zpln=zpln

def DepthvsForce(model,data,plotdata):
  # setting font sizeto 30
  # plt.rcParams['text.usetex'] = True
  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams.update({'font.size': 24})
  #Plot
  #   fig=plt.figure()#figsize=(8, 6))
  # Find the height at which the vertical force becomes less than the weight, i.e. contact is initiated
  weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
  mujoco.mj_inverse(model, data)
  if data.ncon: #Contact already exists
    dz=0.000001
  else: #No contact exists
    dz=-0.000001
  while True:
    data.qpos[2] += dz
    mujoco.mj_inverse(model, data)
    # print(data.qpos[2],data.qfrc_inverse[2],weight)
    if (dz>0)*(data.ncon==0) or (dz<0)*(data.ncon>0):
      z_0=data.qpos[2]
      break
  #Plot height vs Vertical Force
  height_arr = np.linspace(z_0-0.025, z_0, 101)
  vertical_forces = []
  contact_forces = []
  for z in height_arr:
    data.qpos[2] = z
    mujoco.mj_inverse(model, data)
    #if z%0.0005==0: print(z,data.efc_KBIP, data.efc_diagApprox)
    vertical_forces.append(data.qfrc_inverse[2])

    # contact force
    fc = np.zeros([6])
    for i in np.arange(0, data.ncon):
        #conid = data.contact[i].geom1
        fci = np.zeros([6])
        try:
            mj.mj_contactForce(model, data, i, fci)
            fc = fc + fci
        except:
            print('no contact')
    contact_forces.append(fc)
    # print('fc =', fc[0])
    # Reproduce MuJoCo Forces
    # fmj,fsd,defmj,margmj =mjforce(model,data)
    # print(data.qfrc_inverse[2],fmj,data.ncon)
    # if data.ncon>0:
    #     plt.plot(abs(z-z_0)*1, fc[0], 'g.', markersize=2)
    #     # plt.plot(abs(z-z_0)*1, fmj[-1], 'g.', markersize=2)
    #print(asd)

  height_offsets=height_arr-z_0
  vertical_forces=np.array(vertical_forces)
  contact_forces=np.array(contact_forces)[:,0]

  # Find the height-offset at which the vertical force is smallest.
  idx = np.argmin(np.abs(vertical_forces))
  best_offset = height_offsets[idx]
  # Plot the relationship.
  if plotdata==1:
      print('weight=', weight)
      fig, ax = plt.subplots()
      fig.subplots_adjust(right=0.75)

    #   twinax = ax.twinx()

      p1, =ax.plot(abs(height_offsets) * 1, contact_forces, 'r-', linewidth=3, label='force')
      #plt.plot(abs(height_offsets) * 1, vertical_forces, 'r-', linewidth=3)
      # Red vertical line at offset corresponding to smallest vertical force.
      ax.axvline(x=abs(best_offset) * 1, color='black', linestyle='-', linewidth=1)
      # Green horizontal line at the humanoid's weight.
      weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
      ax.axhline(y=weight, color='black', linestyle='-', linewidth=1)
      ax.set(xlabel='Deformation (m)')
      ax.set(ylabel='Contact force (N)')
      ax.set_xlim([0, max(abs(height_offsets) * 1)])
    #   ax.yaxis.label.set_color(p1.get_color())
      ax.grid(visible=None, which='major', axis='both')
    #   plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
    #   plt.ylabel('Vertical force on base (N)')
    #   plt.twinx()
      stiff_y=np.gradient(vertical_forces,height_offsets*1)
    #   p2, =twinax.plot(abs(height_offsets) * 1, stiff_y, 'b--', linewidth=2, label='Normal stiffness')
    #   twinax.set(ylabel='Stiffness (N/m')
    #   twinax.yaxis.label.set_color(p2.get_color())
    #   twinax.grid(visible=None, which='major', axis='both')
    #   plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
    #   plt.legend(handles=[p1, p2],loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    #   plt.minorticks_on()
    # plt.title(f'Min. vertical force at deformation {str(best_offset * 1000)[1:5]} mm.')
    #   plt.show(block=False)
      plt.pause(1)
    #   plt.show()

  return best_offset


