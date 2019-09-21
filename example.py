import matplotlib.pyplot as plt
from morphing import *
from automatic_registration import *


#=======================================================================================================================
# Synthetic case
#=======================================================================================================================

# Choose synthetic case 1 or 2
case = 'syn2'   # 'syn1' or 'syn2'

# Choose where to save plot
folder_plot = "./result_syn2/"

#==================================================================
# Choose registration parameter

# Choose regulation coefficients
c1 = 0.1
c2 = 1
c3 = 1

# Choose number of morphing grids
I = 4

#==================================================================
# Define synthetic fields

# Create coordinates
y = np.arange(0,65)
x = np.arange(0,65)
x1 = x[:,np.newaxis]
ny = len(y)
nx = len(x)

s_u = 1/(2**8+1)
s_v = 1/(2**8+1)

u = 50*np.exp(-(((y - 40) /ny ) ** 2 + ((x1 - 25)/nx) **2 ) / s_u ) + 50*np.exp(-(((y - 30) / ny) ** 2 + ((x1 - 50)/nx) **2 )/ s_v)

if case == 'syn1':
    v = 50*np.exp(-(((y - 50) / ny) ** 2 + ((x1 - 30)/nx) **2 ) / s_v ) + 50*np.exp(-(((y - 20) / ny) ** 2 + ((x1 - 40)/nx) **2 )/ s_v)
elif case == 'syn2':
    v = 50*np.exp(-np.maximum(((y - 50) / ny) ** 2, ((x1 - 30)/nx) **2 ) / s_v ) + 50*np.exp(-np.maximum(((y - 20) / ny) ** 2, ((x1 - 40)/nx) **2 )/ s_v)


#==================================================================
# Automatic registration
mxT, myT = registration(u,v,x,y,I,c1=0.1,c2=1,c3=1,eps1=10**-5,eps2=10**-5)


# Morphing
u_warp = mapped(u, y, x, myT, mxT, I)
u_morph = morphing(u, v, y, x, myT, mxT, I, 1)

#==================================================================
# Statistics

# Mean Absolute Error
MAE_before = np.mean(np.abs(u - v))
MAE_morph = np.mean(np.abs(u_morph - v))
MAE_warp = np.mean(np.abs(u_warp - v))
print("MAE before: {:2f}, warped: {:2f} and morphed: {:2f}".format(MAE_before,MAE_warp,MAE_morph))

# Root Mean Square Error
RMSE_before = np.sqrt(np.mean((u - v) ** 2))
RMSE_morph = np.sqrt(np.mean((u_morph - v) ** 2))
RMSE_warp = np.sqrt(np.mean((u_warp - v) ** 2))
print("RMSE before: {:2f}, warped: {:2f} and morphed: {:2f}".format(RMSE_before,RMSE_warp,RMSE_morph))

#==================================================================
# Plot results

# Plot original fields
fig = plt.figure()
ax = plt.imshow(u, origin='lower',vmin=0,vmax=50)
plt.xlabel('x')
plt.ylabel('y', rotation=0, labelpad=15)
cbar = plt.colorbar(ax)
cbar.set_label(label='mm/h')
plt.title('Warped signal')
plt.tight_layout()
plt.savefig(folder_plot + 'u.png')
plt.close()

fig = plt.figure()
ax = plt.imshow(v, origin='lower',vmin=0,vmax=50)
plt.xlabel('x')
plt.ylabel('y', rotation=0, labelpad=15)
cbar = plt.colorbar(ax)
cbar.set_label(label='mm/h')
plt.title('Warped signal')
plt.tight_layout()
plt.savefig(folder_plot + 'v.png')
plt.close()


# Plot distorted grid
xx_l,yy_l = distort_grid(y,x,myT,mxT,I,1.0)
xx, yy = np.meshgrid(x,y,indexing='ij')
plt.figure()
plt.plot(xx_l,yy_l,'b')
plt.plot(xx_l.T,yy_l.T,'b')
plt.plot(xx,yy,'r',linewidth=0.9,linestyle='--')
plt.plot(xx.T,yy.T,'r',linewidth=0.9,linestyle='--')
plt.xlabel('x')
plt.ylabel('y', rotation=0, labelpad=15)
plt.title('Distorted grid')
plt.tight_layout()
plt.savefig(folder_plot + 'grid.png')
plt.close()

# Plot warped field
fig = plt.figure()
ax = plt.imshow(u_warp, origin='lower',vmin=0,vmax=50)
plt.xlabel('x')
plt.ylabel('y', rotation=0, labelpad=15)
cbar = plt.colorbar(ax)
cbar.set_label(label='mm/h')
plt.title('Warped signal')
plt.tight_layout()
plt.savefig(folder_plot + 'u_warp.png')
plt.close()

# Plot morphed field
fig = plt.figure()
ax = plt.imshow(u_morph, origin='lower',vmin=0,vmax=50)
plt.xlabel('x')
plt.ylabel('y', rotation=0, labelpad=15)
cbar = plt.colorbar(ax)
cbar.set_label(label='mm/h')
plt.title('Morphed signal')
plt.tight_layout()
plt.savefig(folder_plot + 'u_morph.png')
plt.close()


