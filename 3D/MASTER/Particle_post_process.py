import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import mpl_toolkits.mplot3d.axes3d as p3
try:
    from tqdm import tqdm
except ImportError:
    pass
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def plotScatter(fig_particle,ax_particle,i,X,rgbaTuple,pointSize,L,pointcolor1,pointcolor2,Np):
    ax_particle.scatter(X[i + 1][0, 0:int(Np/2)], X[i + 1][1, 0:int(Np/2)], X[i + 1][2, 0:int(Np/2)], s=pointSize, c=pointcolor1)
    ax_particle.scatter(X[i + 1][0, int(Np/2):], X[i + 1][1, int(Np/2):], X[i + 1][2, int(Np/2):], s=pointSize, c=pointcolor2)

    plt.xlim([0, L])
    plt.ylim([0, L])
    ax_particle.set_zlim(0, L)
    ax_particle.set_xlabel('x-axis')
    ax_particle.set_ylabel('y-axis')
    ax_particle.set_zlabel('z-axis')

    ax_particle.w_xaxis.set_pane_color(rgbaTuple)
    ax_particle.w_yaxis.set_pane_color(rgbaTuple)
    ax_particle.w_zaxis.set_pane_color(rgbaTuple)

    ax_particle.xaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax_particle.yaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.yaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax_particle.zaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.zaxis.set_major_locator(tck.MultipleLocator(base=np.pi))

    plt.savefig('particle_plots/particlePlot_t_' + str(i))
    #plt.pause(0.05)
    ax_particle.clear()

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[1]):
        scatters[i]._offsets3d = (data[iteration][0:1,i], data[iteration][1:2,i], data[iteration][2:,i])
    return scatters

if __name__ == '__main__':
    '''
    data = np.load('particleCoord_t10.npy')
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    print('begin scatters')
    scatters = [ax.scatter(data[0][0:1,i], data[0][1:2,i], data[0][2:,i]) for i in range(np.shape(data)[-1])]
    print('finished scatters')
    # Number of iterations
    iterations = np.shape(data)[0]

    # Setting the axes properties
    ax.set_xlim3d([0, 2*np.pi])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 2*np.pi])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 2*np.pi])
    ax.set_zlabel('Z')

    ax.set_title('3D Animated Scatter Example')

    # Provide starting angle for the view.
    ax.view_init(25, 10)
    print('start ani func')
    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                  interval=4, blit=False, repeat=True)
    print('start writer')
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    print('start save')
    ani.save('3d-scatted-animated.mp4', writer=writer)
    '''

    fig_particle = plt.figure()
    ax_particle = fig_particle.add_subplot(111, projection='3d')
    pointSize = 0.0001
    pointcolor1 = 'r'
    pointcolor2 = 'm'
    rgbaTuple = (167 / 255, 201 / 255, 235 / 255)
    L = 2*np.pi
    ims = []

    dataset = np.load('particleCoord_t10.npy')
    dataset_shape = np.shape(dataset)
    timesteps = dataset_shape[0]
    Np = dataset_shape[2]

    for i in range(0,timesteps,20):
        plotScatter(fig_particle,ax_particle,i,dataset,rgbaTuple,pointSize,L,pointcolor1,pointcolor2,Np)
        print(i)
