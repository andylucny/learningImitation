# developed by Andrej Lucny from Comenius University in Bratislava, www.agentspace.org/andy

import sys
import os
if sys.version_info[0] != 3:
    print("only Python 3 is supported by pyicubsim")
    os._exit(0)
    
import platform
if platform.architecture()[0][:2] != '64':
    print("only 64bit python is supported by pyicubsim")
    os._exit(0)
    
# download iCubSim if not installed yet
import io
import requests
import zipfile
import shutil

# for Python >= 3.8 which ignores %PATH%
if os.name == 'nt':

    def download_iCubSim():
        url = "https://www.agentspace.org/download/iCubSim.zip"
        print("downloading iCubSim")
        response = requests.get(url)
        #open("iCubSim.zip", "wb").write(response.content)
        if response.ok:
            file_like_object = io.BytesIO(response.content)
            zipfile_obj = zipfile.ZipFile(file_like_object)    
            zipfile_obj.extractall(".")
            if os.path.isdir('iCubSim'):
                _pwd = os.getcwd()
                os.chdir('iCubSim')
                os.system('init-bindings.bat')
                os.chdir(_pwd)
                print()
                print("Do you want to avoid objects (table, ball)? (y/n): ")
                ans = input()
                if ans == 'y' or ans == 'Y':
                    print('no objects are used')
                    shutil.copy('iCubSim/bin/iCub_parts_activation.ini','iCubSim/bin/iCub_parts_activation_Objects.ini')
                    shutil.copy('iCubSim/bin/iCub_parts_activation_noObjects.ini','iCubSim/bin/iCub_parts_activation.ini')
                    shutil.copy('iCubSim/run-iCubSim.bat','iCubSim/run-iCubSim_Objects.bat')
                    shutil.copy('iCubSim/run-iCubSim_noObjects.bat','iCubSim/run-iCubSim.bat')
                

    if sys.version >= '3.8':
        os.add_dll_directory(os.path.abspath(os.path.curdir)+'/iCubSim/bin') 
    else:
        os.environ['PATH'] += "iCumSim\\bin;" 
    
    if not os.path.isdir('iCubSim'):
        download_iCubSim()

    if not os.path.exists('yarp.py') or not os.path.exists('_yarp.pyd'):
        print("yarp protocol not available for pyicubsim")
        os._exit(0)

import yarp
import numpy as np
import cv2
import socket
import re
import time

class NoYarp:
    # parse respose from naming service
    @staticmethod
    def get_addr(s):
        m = re.match("registration name [^ ]+ ip ([^ ]+) port ([0-9]+) type tcp",s)
        return (m.group(1),int(m.group(2))) if m else None
    # get a single line of text from a socket
    @staticmethod
    def getline(sock):
        result = ""
        while result.find('\n')==-1:
            result = result + sock.recv(1024).decode()
        result = re.sub('[\r\n].*','',result)
        return result
    # send a message and expect a reply
    @staticmethod
    def command(addr,message):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(addr)
        sock.send('CONNECT extern\n'.encode())
        NoYarp.getline(sock) 
        if isinstance(message, tuple):
            result=()
            for command in message:
                #print('SENT: ',command);
                sock.send(('d\n%s\n' % command).encode())
                result += (NoYarp.getline(sock),)
                #print('RECEIVED: ',result[-1]);
        else:
            sock.send(('d\n%s\n' % message).encode())
            result = NoYarp.getline(sock)
        sock.close()
        return result
    # call YARP naming service
    @staticmethod
    def query(host, port_name):
        return NoYarp.get_addr(NoYarp.command((host,10000),"query %s"%port_name))

class Yarp:
    initialized = False
    @staticmethod
    def initialize():
        if not Yarp.initialized:
            yarp.Network.init()
            time.sleep(0.1)
            Yarp.initialized = True
            print('Yarp initialized')
    
class iCubLimb:
    def __init__(self,app_name,port_name):
        Yarp.initialize()
        # prepare a property object
        self.props = yarp.Property()
        self.props.put('device','remote_controlboard')
        self.props.put('local',app_name+port_name)
        self.props.put('remote',port_name)
        # create remote driver
        self.armDriver = yarp.PolyDriver(self.props)
        # query motor control interfaces
        self.iPos = self.armDriver.viewIPositionControl()
        #self.iVel = self.armDriver.viewIVelocityControl()
        self.iEnc = self.armDriver.viewIEncoders()
        # retrieve number of joints
        self.jnts = self.iPos.getAxes()
        time.sleep(0.1)
        print('Controlling', self.jnts, 'joints of', port_name)
        
    def get(self):
        # read encoders
        encs = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encs.data())
        values = ()
        for i in range(self.jnts):
            values += (encs.get(i),)
        return values
        
    def set(self,values=(), \
        joint0=None,joint1=None,joint2=None,joint3=None,joint4=None,joint5=None,joint6=None,joint7=None, \
        joint8=None,joint9=None,joint10=None,joint11=None,joint12=None,joint13=None,joint14=None,joint15=None):
        # read encoders
        encs = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(encs.data())
        # adjust joint positions
        for i in range(min(self.jnts,len(values))):
            if values[i] != None:
                encs.set(i,values[i])
        for i in range(16):
            value = eval('joint'+str(i))
            if value != None:
                #print('joint',i,'=',value)
                encs.set(i,value)
        # write to motors
        self.iPos.positionMove(encs.data())
        
    def size(self):
        # return number of joints
        return self.jnts

class iCubCamera:
    def __init__(self,app_name,port_name):
        Yarp.initialize()
        # open recipient port
        self.port = yarp.Port()
        self.port.open(app_name+port_name)
        yarp.delay(0.25)
        # connect the port to camera
        yarp.Network.connect(port_name,app_name+port_name)
        yarp.delay(0.25)
        # prepare data buffer for reception
        self.width = 320
        self.height = 240
        self.yarp_img = yarp.ImageRgb()
        self.yarp_img.resize(self.width,self.height)
        self.array_img = bytearray(self.width*self.height*3)
        self.yarp_img.setExternal(self.array_img,self.width,self.height)
        # prepare blank image to be returned when an error appears
        self.blank = np.zeros(self.shape())

    def grab(self):
        # receive one image
        self.port.read(self.yarp_img)
        # check if the image is correct
        if self.yarp_img.height() == self.height and self.yarp_img.width() == self.width:
            # turn the image to openCV format
            img = np.frombuffer(self.array_img, dtype=np.uint8)
            img = img.reshape(self.height,self.width,3)
            # correct its color model
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # return the OpenCV image
            return img
        else:
            return blank

    def shape(self): # can be called before any image is received
        # return shape of image provided by this camera 
        return (self.height,self.width,3)

class iCubEmotion:
    def __init__(self):
        host = 'localhost' #'192.168.56.1'
        port_name = '/emotion/in'
        self.query = NoYarp.query(host,port_name)
        self.neutral = 'neu'
        self.happy = 'hap'
        self.sad = 'sad'
        self.surprised = 'sur'
        self.angry = 'ang'
        self.evil = 'evi'
        self.shy = 'shy'
        self.cunning = 'cun'
        self.set(self.neutral)

    def set(self, emotion='neu'):
        commands = ('set all '+emotion,)
        NoYarp.command(self.query,commands)

class iCubBall:
    def __init__(self):
        host = 'localhost' #'192.168.56.1'
        port_name = '/icubSim/world'
        self.query = NoYarp.query(host, port_name)
        self.get()
        
    def get(self):
        command = 'world get ball'
        response = NoYarp.command(self.query,(command,))
        values = response[0].split()
        self.x = -(float(values[2])*1000-50)
        self.y = -float(values[0])*1000
        self.z = float(values[1])*1000-600
        return self.x, self.y, self.z

    def set(self, x=None, y=None, z=None):
        if x is not None and type(x) is tuple:
            x, y, z = x
        if x is None and y is None and z is None:
            x = 0.15*1000
            y = 0.5539755*1000
            z = 0.35*1000
        if x is None or y is None or z is None:
            self.get()
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        command = 'world set ball '+str(-self.y/1000)+' '+str((self.z+600)/1000)+' '+str((-self.x+50)/1000)
        NoYarp.command(self.query,(command,))
    
    def setDefault(self):
        self.set()

# start simulator if it is not started yet
def isRunning_iCubSim():
    try:
        NoYarp.command(('localhost',10000),"query /icubSim/world")
        return True
    except ConnectionRefusedError:
        return False

if os.name == 'nt':

    # start simulator
    def start_iCubSim():
        if isRunning_iCubSim():
            print('iCubSim already started')
        else:
            print('starting iCubSim')
            _pwd = os.getcwd()
            os.chdir('iCubSim')
            os.system('run-iCubSim.bat')
            os.chdir(_pwd)
            print('iCubSim started')
            
    start_iCubSim()

    # stop simulator
    def stop_iCubSim():
        if isRunning_iCubSim():
            os.system('taskkill /im BallControl.exe')
            time.sleep(1)
            os.system('taskkill /im emotionInterface.exe')
            time.sleep(1)
            os.system('taskkill /im simFaceExpressions.exe')
            time.sleep(1)
            os.system('taskkill /im yarpmotorgui.exe')
            time.sleep(1)
            os.system('taskkill /F /im iCub_SIM.exe') 
            time.sleep(1)
            os.system('taskkill /im yarpserver.exe')
            print('iCubSim stopped')
            
print('iCubSim ready')

class Kinematics:
    # iCub torso and arm have up to 11 thetas (12 points)
    waist = 0
    shoulder = 3
    elbow = 6
    wrist = 8 
    palm = 10
    
    # standard position
    poseA = (0.0,0.0,0.0) + (-80.0,80.0,0.0,50.0,0.0,0.0,0.0,59.0,20.0,20.0,20.0,10.0,10.0,10.0,10.0,10.0)
    
    # direct kinematics is written by Francesco Nori, Genova Jan 2008
    @staticmethod
    def directRightArm(thetas):

        def DH(a, d, alpha, theta):
            return np.array([
                [ np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), np.cos(theta)*a ],
                [ np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), np.sin(theta)*a ],
                [             0,                np.sin(alpha),                np.cos(alpha),               d ],
                [             0,                            0,                            0,               1 ]
            ])
        
        def rad(value):
            return value * np.pi/180.0

        Gs = [
            DH(      32,       0,     np.pi/2,     rad(thetas[2])),
            DH(       0,    -5.5,     np.pi/2,     rad(thetas[1])-np.pi/2),
            DH(-23.3647,  -143.3,     np.pi/2,     rad(thetas[0]) - 15*np.pi/180 - np.pi/2),
            DH(       0, -107.74,     np.pi/2,     rad(thetas[3])-np.pi/2),
            DH(       0,       0,    -np.pi/2,     rad(thetas[4])-np.pi/2),
            DH(     -15, -152.28,    -np.pi/2,     rad(thetas[5])-np.pi/2-15*np.pi/180),
            DH(      15,       0,     np.pi/2,     rad(thetas[6])), 
            DH(       0,  -137.3,     np.pi/2,     rad(thetas[7])-np.pi/2),
            DH(       0,       0,     np.pi/2,     rad(thetas[8])+np.pi/2),
            DH(    62.5,      16,           0,     rad(thetas[9])+np.pi)
        ]
        T_Ro0 = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
        XE = np.transpose(np.array([[0,0,0,1]]))
        positions = [(0,0,0)]
        T_0n = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        for G in Gs:
            T_0n = T_0n.dot(G)
            pos = T_Ro0.dot(T_0n).dot(XE)
            positions.append((pos[0][0],pos[1][0],pos[2][0]))

        return positions

    @staticmethod
    def directLeftArm(thetas):
        ts = (-thetas[0],-thetas[1]) + thetas[2:]
        ps = Kinematics.RightArm(thetas)
        positions = []
        for p in ps:
            positions.append((ps[0],-ps[1],ps[2]))
        return positions

    # inverse kinematics (FABRIK algorithm modified for D-H convention, need to be fixed)
    @staticmethod
    def inverseRightArm(goal,first,last): # first and last are from 1 to 10, e.g. Kinematics.shoulder, Kinematics.wrist
        rangesTorso = [
            [-50,50],
            [-30,30],
            [-9.6,69.6]
        ]
        rangesArm = [ # constraints for both left and right arm
            [-94.5,9.45],
            [0,160.8],
            [-36.27,79.56],
            [15.385,105.8],
            [-90,90],
            [-90,0],
            [-19.8,39.6],
            [0,60],
            [9.6,89.6],
            [0,90],
            [0,80],
            [0,90],
            [0,80],
            [0,90],
            [0,80],
            [0,270]
        ]
        ranges = np.array(rangesTorso+rangesArm)
        thetas = np.array(Kinematics.poseA)[:11] # pose A
        goal = np.array(goal)
        dts = 1.0
        i = last-1
        si = 1
        achieved = False
        no_progress = 0
        iters = 0
        positions = Kinematics.directRightArm(thetas)
        p = np.array(positions[last])
        d = np.linalg.norm(p-goal)
        while True:
            #print(i,d,thetas[i],dts,no_progress)
            if d < 1.0:
                achieved = True
                #print('achieved')
                break

            iters += 1

            thetas_plus = np.copy(thetas)
            thetas_plus[i] += dts
            positions_plus = Kinematics.directRightArm(thetas_plus)
            p_plus = np.array(positions_plus[last])
            d_plus = np.linalg.norm(p_plus-goal)

            thetas_minus = np.copy(thetas)
            thetas_minus[i] -= dts
            positions_minus = Kinematics.directRightArm(thetas_minus)
            p_minus = np.array(positions_minus[last])
            d_minus = np.linalg.norm(p_minus-goal)

            if d_plus < d and d_plus < d_minus and thetas_plus[i] <= ranges[i,1]:
                thetas = thetas_plus
                positions = positions_plus
                if d - d_plus > 0.001:
                    no_progress = 0
                p = p_plus
                d = d_plus
            elif d_minus < d and d_minus < d_plus and thetas_minus[i] >= ranges[i,0]:
                thetas = thetas_minus
                positions = positions_minus
                if d - d_minus > 0.001:
                    no_progress = 0
                p = p_minus
                d = d_minus
            else:
                no_progress += 1
            
            if i == first or i == last-1:
                si = -si
                if no_progress >= (last-first):
                    no_progress = 0
                    dts *= 0.5
            i += si
        
            if dts < 0.0001:
                #print('no progress')
                if d < 3.0:
                    achieved = True
                break
            
        return tuple(thetas), achieved, d, iters

    @staticmethod
    def inverseLeftArm(goal,first,last): # first and last are from 1 to 10, e.g. Kinematics.shoulder, Kinematics.wrist
        g = (goal[0],-goal[1],goal[2])
        thetas, achieved, distance, iterations = Kinematics.inverseRightArm(g,first,last)
        return (-thetas[0],-thetas[1])+thetas[2:], achieved, distance, iterations

appName = '/app/client'

def setApplicationName(name):
    appName = name
    
def iCubApplicationName(name):
    appName = name
        
class iCubRightArm(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/right_arm')
    def isRight(self):
        return True
    def reset(self):
        self.set(Kinematics.poseA[3:])
       
class iCubLeftArm(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/left_arm')
    def isRight(self):
        return False
    def reset(self):
        self.set(Kinematics.poseA[3:])

class iCubTorso(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/torso')
    def reset(self):
        self.set(Kinematics.poseA[:3])

class iCubHead(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/head')
    def reset(self):
        self.set((0,0,0,0,0,0))

class iCubRightLeg(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/right_leg')
    def reset(self):
        self.set((0,0,0,0,0,0))
        
class iCubLeftLeg(iCubLimb):
    def __init__(self):
        super().__init__(appName,'/icubSim/left_leg')
    def reset(self):
        self.set((0,0,0,0,0,0))

class iCubRightEye(iCubCamera):
    def __init__(self):
        super().__init__(appName,'/icubSim/cam/right')

class iCubLeftEye(iCubCamera):
    def __init__(self):
        super().__init__(appName,'/icubSim/cam/left')
        
class iCubGlobalCamera(iCubCamera):
    def __init__(self):
        super().__init__(appName,'/icubSim/cam')

def coord(torso, arm):
    thetas = torso.get()+arm.get()
    points = Kinematics.directRightArm(thetas) if arm.isRight() else Kinematics.directLeftArm(thetas)
    return points[Kinematics.palm]

def hit(torso, arm, goal, moveArmOnly=False):
    first = Kinematics.shoulder if moveArmOnly else Kinematics.waist
    thetas, achieved, distance, iterations = \
        Kinematics.inverseRightArm(goal,first,Kinematics.palm) if arm.isRight() else \
        Kinematics.inverseLeftArm(goal,first,Kinematics.palm)
    if achieved:
        torso.set(thetas[:3])
        arm.set(thetas[3:])
    else:
        print(distance)
    return achieved

def reset(limb):
    if type(limb) == list:
        for l in limb:
            l.reset()
    else:
        limb.reset()
