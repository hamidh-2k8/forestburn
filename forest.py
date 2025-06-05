import time
import random
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.style as mplstyle

class tree_status:
    NOT_PRESENT = 0
    DEAD = 1
    ON_FIRE = 2
    ALIVE = 3

side_len = 50
scalar = 5
fastmode = True
profiler_enable = False

tree_prob_arr = np.arange(0.4, 0.8, 0.01)
fire_spread_arr = np.arange(0.4, 0.8, 0.01)

forest_cmap = mcolors.ListedColormap([
    [0.23046875, 0.12109375, 0.0078125, 1.0],  # ground
    [0.26953125, 0.26953125, 0.26953125, 1.0], # dead
    [1.0, 0.64453125, 0.26953125, 1.0],        # on fire
    [0.0703125, 0.56640625, 0.00390625, 1.0],  # alive
])

profiler_arr = [1, 1, 1] #init, calc, display

assert scalar > 1, "Dimensionality Scalar musc be greater than 1"
assert len(tree_prob_arr) > 1, "Range of Tree Densities must be greater than 1"
assert len(fire_spread_arr) > 1, "Range of Fire Spread values must be greater than 1"

plt.ion()
mplstyle.use('fast')

sim_figure, sim_axes = plt.subplots(1, 2, figsize=(12, 4))

def burn(slice_arr: np.ndarray, fire_tick_prob):
    
    burn_arr: np.ndarray = np.where(np.random.uniform(0, 1, slice_arr.shape) < fire_tick_prob, 1, 0)

    return slice_arr - np.logical_and(burn_arr, np.where(slice_arr == tree_status.ALIVE, 1, 0)) 

def runSim(tree_spawn_prob, fire_tick_prob):
    
    init_start = time.time()
    
    burn_pcent_arr = []
    firemap = np.random.uniform(0, 1, (side_len, side_len))
    
    firemap = firemap < tree_spawn_prob
    firemap = firemap.astype(np.uint8) * tree_status.ALIVE
    
    firemap_small = np.zeros((int(side_len/scalar), int(side_len/scalar)))

    for i in range(int(side_len/scalar)):
        for j in range(int(side_len/scalar)):
            if tree_status.ALIVE in firemap[(i*scalar):(i*scalar + int(side_len/scalar)), (j*scalar):(j*scalar + int(side_len/scalar))]:
                firemap_small[i, j] = tree_status.ALIVE
                
    count_lg = len(np.where(firemap == tree_status.ALIVE)[0])
    count_sm = len(np.where(firemap_small == tree_status.ALIVE)[0])

    dim = (np.log10(count_lg) - np.log10(count_sm)) / np.log10(scalar)

    lightning_strike_index = tuple(random.choice(np.argwhere(firemap == tree_status.ALIVE))) or 0

    if not lightning_strike_index == 0:
        firemap[lightning_strike_index] = tree_status.ON_FIRE

    sim_plot = sim_axes[0].imshow(firemap, cmap=forest_cmap, vmin=tree_status.NOT_PRESENT, vmax=tree_status.ALIVE)
    sim_axes[0].set_title(f'Fire Map')
    sim_axes[0].tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False
        )
    
    progress,  = sim_axes[1].plot(range(len(burn_pcent_arr)), burn_pcent_arr)
    sim_axes[1].set_ylabel("% Burned")
    sim_axes[1].set_xlabel("Iterations")

    sim_figure.suptitle(f"Running Simulation with {str(int(tree_spawn_prob * 100))}% Tree Density, {str(int(fire_tick_prob*100))}% Fire Spread, and Grid Size {str(side_len)}x{str(side_len)}")

    init_end = time.time()
    profiler_arr[0] = init_end - init_start

    while len(np.where(firemap == tree_status.ON_FIRE)[0]) > 0:
        
        calc_start = time.time()

        global next_firemap
        next_firemap = firemap.copy()
        
        for x in range(len(firemap)):
            
            for y in range(len(firemap[x])):
                
                if firemap[x][y] == tree_status.ON_FIRE:
                    
                    row_start = max(0, x-1)
                    row_end = min(side_len, x+2)
                    col_start = max(0, y-1)
                    col_end = min(side_len, y+2)
                
                    slice_arr = next_firemap[row_start : row_end, col_start : col_end]
                    
                    next_firemap[row_start : row_end, col_start : col_end] = burn(slice_arr, fire_tick_prob)
                    
                    next_firemap[x][y] = tree_status.DEAD
            
        firemap = next_firemap        
        num_burned = len(np.where(firemap == tree_status.DEAD)[0])
        pcent_burned = num_burned / count_lg * 100
        
        burn_pcent_arr.append(pcent_burned)
        
        calc_end = time.time()
        profiler_arr[1] = calc_end - calc_start
        
        disp_start = time.time()
        
        if not fastmode:
                
            sim_plot.set_data(firemap)
            sim_axes[0].set_title(f'Fire Map (Burned: {num_burned} ({int(pcent_burned)}%)')
            progress.set_xdata(range(len(burn_pcent_arr)))
            progress.set_ydata(burn_pcent_arr)
            sim_axes[1].relim()
            sim_axes[1].autoscale_view()
            sim_figure.canvas.blit()
            sim_figure.canvas.flush_events()
        
        disp_end = time.time()
        profiler_arr[2] = disp_end - disp_start
    
    sim_axes[0].cla()
    sim_axes[1].cla()
    
    
    looptime = np.sum(profiler_arr)
    print("\x1b[2J\x1b[H", end="")
    print("--------------------------------------------------------------------------------------------------------------------")
    print(f"Running Forest Fire Simulation {'(FASTMODE)' if fastmode else '(SLOWMODE)'}")
    print(f"Current Simulation: {str(int(tree_spawn_prob * 100))}% Spawn | {str(int(fire_tick_prob*100))}% Burn")
    print(f"Result: Dimensionality: {dim} | Burned: {num_burned} ({int(pcent_burned)}%) | Iterations: {len(burn_pcent_arr)}")
    print(f"Time Taken: {np.round(looptime * 1000, 2)}ms ({int(len(burn_pcent_arr) / (profiler_arr[0] + profiler_arr[1]))} iterations/sec)")
    
    
    return firemap, dim, num_burned, pcent_burned, len(burn_pcent_arr), looptime

hide_res = len(tree_prob_arr) * len(fire_spread_arr) < 2 or len(tree_prob_arr) * len(fire_spread_arr) > 100

rows, cols = len(fire_spread_arr), len(tree_prob_arr)

if not hide_res:
    res_fig, res_axes = plt.subplots(rows, cols, figsize = (2*cols, 2*rows))
    res_fig.suptitle("Final Simulation Fire Maps")
    
ips_arr = []

summary_fig = plt.figure(figsize=(12, 4))
summary_fig.suptitle("Data Summary")

summary_2d = summary_fig.add_subplot(1, 2, 1)
summary_2d.set_title("Results Heatmap")

summary_3d = summary_fig.add_subplot(1, 2, 2, projection="3d")
summary_3d.set_title("Results Scatterplot")

if profiler_enable:
    profiler_fig = plt.figure(figsize=(12, 4))
    profiler_fig.suptitle("Performance")

    time_graph = profiler_fig.add_subplot(1, 2, 1)
    time_graph.set_title("Sim Loop Time")
    time_graph.set_xlabel("Simulation Index")
    time_graph.set_ylabel("Avg. Iterations/Sec")

    time_pie = profiler_fig.add_subplot(1, 2, 2)
    time_plot, = time_graph.plot(range(len(ips_arr)), ips_arr)

summary_arr = np.zeros((len(fire_spread_arr), len(tree_prob_arr)))
X, Y = np.meshgrid(np.arange(0, len(fire_spread_arr), 1), np.arange(0, len(tree_prob_arr), 1))


for i in range(len(tree_prob_arr)):
    for j in range(len(fire_spread_arr)):
        
        res_data, dim, num_burned, pcent_burned, iterations, elapsedtime = runSim(tree_prob_arr[i], fire_spread_arr[j])
        
        print(f"{np.ceil((i*len(tree_prob_arr) + j + 1) / (len(tree_prob_arr) * len(fire_spread_arr)) * 100)}% COMPLETE")
        print("--------------------------------------------------------------------------------------------------------------------")
        
        summary_arr[j][i] = pcent_burned
        ips_arr.append(iterations / (profiler_arr[0] + profiler_arr[1]))
        
        if profiler_enable:
            time_plot.set_xdata(range(len(ips_arr)))
            time_plot.set_ydata(ips_arr)
            time_graph.relim()
            time_graph.autoscale_view()
            
            time_pie.cla()
            time_pie.pie(profiler_arr, labels=["Init", "Calc", "Disp"], autopct='%1.1f%%')
            time_pie.set_title("Profiler")
        
        if not hide_res:    
            res_axes[j][i].imshow(res_data, cmap=forest_cmap, vmin=tree_status.NOT_PRESENT, vmax=tree_status.ALIVE)
            res_axes[j][i].annotate(f"T: {np.round(tree_prob_arr[i], 2)}\nF: {np.round(fire_spread_arr[j], 2)}\nD: {np.round(dim, 2)}\nB: {int(pcent_burned)}%\nI: {iterations}",
                xy=(0.75,0.5), xycoords='axes fraction',
                textcoords='offset points',
                size=8,
                bbox=dict(boxstyle="round", fc=(1.0, 0.4, 1.0), ec="none"))

            res_axes[j][i].tick_params(
                axis='both', which='both',
                bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False
            )
            
        summary_2d.imshow(summary_arr)
summary_3d.scatter(X, Y, summary_arr)
        
plt.close(sim_figure)
if profiler_enable: plt.close(profiler_fig)

plt.show()

np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True, linewidth=sys.maxsize)

desmosdata = ""

desmosdata += f"X={repr(np.tile(range(len(tree_prob_arr)), len(fire_spread_arr))).replace('array(', '').replace(')', '').strip(os.linesep)}" + os.linesep
desmosdata += f"Y={repr(np.repeat(range(len(fire_spread_arr)), len(tree_prob_arr))).replace('array(', '').replace(')', '').strip(os.linesep)}" + os.linesep
desmosdata += f"Z={repr(summary_arr.flatten() / 10).replace('array(', '').replace(')', '').strip(os.linesep)}" + os.linesep
desmosdata += "(X, Y, Z)"

try: desmosfile = open('desmos.dat', "x")
except FileExistsError: desmosfile = open('desmos.dat', "w")
    
desmosfile.write(desmosdata)
desmosfile.close()


input("Press [ENTER] to exit.")