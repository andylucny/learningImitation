import numpy as np
import matplotlib.pyplot as plt

# load data
fname = 'rightarm_wrist.txt'
#fname = 'rightarm_elbow.txt'
with open(fname,'r') as f:
    lines = f.readlines()
    
data = [line[:-1].split(',')[:-1] for line in lines]
data = np.array(data,np.float32)

def data_for_cylinder_along_z(center_x,center_y,radius_x,radius_y,height_z0,height_z1):
    z = np.linspace(height_z0, height_z1, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius_x*np.cos(theta_grid) + center_x
    y_grid = radius_y*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1.0,1.0,1.0))

#head #308 - top of head, #230 - bottom of head
Xc,Yc,Zc = data_for_cylinder_along_z(0,0,20,40,240,310)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, cmap='Greys')

# torso
Xc,Yc,Zc = data_for_cylinder_along_z(0,0,10,120,0,230)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, cmap='Greys')

# arms
ad = 150
Xc,Yc,Zc = data_for_cylinder_along_z(0,ad,10,20,100,270)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, cmap='Greys')
Xc,Yc,Zc = data_for_cylinder_along_z(0,-ad,10,20,100,270)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, cmap='Greys')

# points reachable by the right arm
Xc = list(data[:,0])
Yc = list(data[:,1])
Zc = list(data[:,2])
ax.scatter(Xc,Yc,Zc,alpha=0.1)

# points reachable by the left arm
Xc = list(data[:,0])
Yc = list(-data[:,1])
Zc = list(data[:,2])
ax.scatter(Xc,Yc,Zc,alpha=0.1)

# legend
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")    

#ax.view_init(elev=45, azim=45)
#ax.view_init(elev=90, azim=90)
ax.view_init(elev=0, azim=0)
plt.savefig('reachable_by_wrist.png')
plt.show()

