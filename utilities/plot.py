import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from matplotlib import cm
import matplotlib.ticker as mtick
import scipy.interpolate as spinterp

def create_graphs_grid(x_gt, u_gt, u_pred, titles, figs_dir, heat = False):
    _, axs = plt.subplots(2,2, figsize=(10,8))

    for i, ax in enumerate(axs.flat):
        ax.plot(x_gt,u_gt[i],label="GT", color='blue')
        ax.plot(x_gt,u_pred[i], '--', label="Predicted", color='red')
        ax.set_title(titles[i])
        if heat:
            ax.set_ylim([250,1000])
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir,"t_grid.png"))

def create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir, gif_name="evolution.gif", duration=50, heat = False):
    os.makedirs("temp", exist_ok=True)

    ymin, ymax = np.min(u_gt), np.max(u_gt)
    
    frames = []
    try:
        for i in range(len(t_gt)):
            fig, ax = plt.subplots()
            if (np.isnan(u_gt[i,:]).sum()+np.isinf(u_gt[i,:]).sum())==0 :
                ax.plot(x_gt,u_gt[i,:],color='blue', label="GT")
            if (np.isnan(u_pred[i,:]).sum()+np.isinf(u_pred[i,:]).sum())==0 :
                ax.plot(x_gt,u_pred[i,:],'--', color='red', label="Predicted")
            ax.set_title(f"t={t_gt[i].item():.3f}")
            if heat:
                ax.set_ylim(250,1000)
            else:
                ax.set_ylim(ymin,ymax)
            ax.grid(False)
            ax.legend()

            fname = os.path.join("temp", f"_frame_{i:03d}.png")
            fig.savefig(fname)
            plt.close(fig)
            frames.append(Image.open(fname))
        
        frames[0].save(os.path.join(figs_dir,gif_name),save_all=True,append_images=frames[1:],duration=duration,loop=0)
    finally:
        for frame in frames:
            frame.close()
        for i in range(len(t_gt)):
            os.remove(os.path.join("temp", f"_frame_{i:03d}.png"))

def plot_prediction_vs_groundtruth(ts,u_pred,u_gt,extent=[0, 1, 0, 1],save_path=None,cmap="seismic", equation = None, vmin = None, vmax = None ):
    
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    for j, i in enumerate(idxs):

        if vmin == None:
            vmax = np.abs(u_gt[i]).max()
            vmin = -vmax

        if equation == 'heat':
            vmax = 1.0
            vmin = 0.0

        im1 = axs[0, j].imshow(u_pred[i].T, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, interpolation = 'bicubic')
        axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].tick_params(labelbottom=False, labelleft=False)

        im2 = axs[1, j].imshow(u_gt[i].T, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, interpolation = 'bicubic')
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].tick_params(labelbottom=False, labelleft=False)

    axs[0, 0].set_ylabel("Prediction", fontsize=12)
    axs[1, 0].set_ylabel("Ground-Truth", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # left, bottom, width, height
    fig.colorbar(im1, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)


    if save_path:
        plt.savefig(save_path, dpi=300)

def plot_error(ts, error, extent=[0, 1, 0, 1], save_path=None, cmap="seismic"):
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 3, 6))

    if num_frames == 1:
        axs = [axs]

    vmax = np.max(np.abs(error))
    vmin = -vmax

    for j, i in enumerate(idxs):
        im = axs[j].imshow(error[i].T, cmap=cmap,
                           extent=extent, vmin=vmin, vmax=vmax, interpolation = 'bicubic')
        axs[j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].tick_params(labelbottom=False, labelleft=False)

    axs[0].set_ylabel("Error", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    fig.colorbar(im, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_instant(x,y,u, projection="rectilinear", cmap=cm.seismic, title="",angles=(60,240,None),
                        y_name="y",x_name="x", axes=None, refine=False):
    
    if axes is None:
        fig,axes=plt.subplots(subplot_kw={"projection":projection})
        axes.set_title(title)
    else:
        fig=axes.get_figure()
        assert axes.name==projection
    fig.set_dpi(300)
    fig.set_size_inches(5.5,4)
    axes.set_xlabel(x_name)
    axes.set_ylabel(y_name)
    # max_val=np.max(np.abs(u))
    max_val=np.quantile(np.abs(u),0.99)
    vmin,vmax=-max_val, +max_val
    if projection=="rectilinear":
        u_plot = axes.pcolormesh(x,y,u.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    elif projection=="3d":
        if refine:
            interpolator = spinterp.RegularGridInterpolator((x,y),u, method="cubic")
            x_refined = np.linspace(x[0],x[-1],4*len(x))
            y_refined = np.linspace(y[0],y[-1],4*len(y))
            x_mesh,y_mesh = np.meshgrid(x_refined,y_refined,indexing="ij")
            xi = np.stack((x_mesh, y_mesh), axis=-1)
            u_refined = interpolator(xi)
            u = u_refined
        else:
            x_mesh,y_mesh = np.meshgrid(x,y,indexing="ij")
        u_plot = axes.plot_surface(x_mesh,y_mesh,u, cmap=cmap, vmin=vmin, vmax=vmax, antialiased=True, rstride=1, cstride=1)
        axes.view_init(elev=angles[0], azim=angles[1], roll=angles[2])
        axes.zaxis.set_major_formatter(mtick.NullFormatter())
    cbar=fig.colorbar(u_plot, shrink=1.0, ax=axes)

    return fig

def plot_1d_heatmap(t_plot, x_plot, u_pred, figs_dir, cmap="coolwarm"):
    '''Plots the solution of a 1D problem as a (t,x) heatmap'''
    fig = plot_instant(t_plot, x_plot, u_pred, cmap=cmap, x_name="t", y_name="x")
    fig.savefig(os.path.join(figs_dir, "u_heatmap.png"), dpi=300, bbox_inches="tight")

def figures_2d_subplots(t,x,y,u_pred,u_gt, folder, snapshot_times=(), order="txy", model_name="PINN", projection="rectilinear", x_name="x", y_name="y"):
    if not os.path.exists(folder): os.makedirs(folder)

    cmap=cm.seismic
    indices_t = []
    times = []
    for i, snapshot_time in enumerate(snapshot_times):
        ind_t = len(np.nonzero(snapshot_time>=t)[0])-1
        indices_t.append(ind_t)
        times.append(t[ind_t])
    
    if order=="txy":
        assert np.all(u_gt.shape[:-1]==np.concatenate((t.shape,x.shape,y.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[indices_t,:,:,i], u_pred[indices_t,:,:,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    elif order=="xyt":
        assert np.all(u_gt.shape[:-1]==np.concatenate((x.shape,y.shape, t.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[:,:,indices_t,i], u_pred[:,:,indices_t,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    if len(u_dimensions)==1: u_dimension_names = ["U"]
    elif len(u_dimensions)<=3: u_dimension_names= ["U","V","W"]
    else: u_dimension_names = [f"U{i}" for i in range(len(u_dimensions))]

    for (i,(gt,pred)) in enumerate(u_dimensions):
        u_name = u_dimension_names[i]

        fig, axes = plt.subplots(len(snapshot_times),2, subplot_kw={"projection":projection})
        fig.set_dpi(1200)
        fig.subplots_adjust(hspace=0.4,wspace=0.15)
        fig.set_size_inches(1+2*len(u_dimensions),5*len(snapshot_times))
        for (i, time) in enumerate(times):
            plot_instant(x,y,gt[i], projection=projection, cmap=cmap, title=f"Ground Truth {u_name}", axes=axes[i,0], x_name=x_name, y_name=y_name)
            plot_instant(x,y,pred[i], projection=projection, cmap=cmap, title=f"{model_name} {u_name}", axes=axes[i,1], x_name=x_name, y_name=y_name)
            axes[i,0].set_ylabel(y_name)
            axes[i,1].set_ylabel("")
            axes[i,1].yaxis.set_major_formatter(mtick.NullFormatter())
            for j in (0,1):
                axes[i,j].set_xlabel("")
                axes[i,j].xaxis.set_major_formatter(mtick.NullFormatter())  

            axes[i,0].set_title(f"Ground Truth {u_name}: t={time:.2f}")
            axes[i,1].set_title(f"{model_name} {u_name}: t={time:.2f}")
        for j in (0,1):
            axes[-1,j].set_xlabel(x_name)
        fig.savefig(os.path.join(folder,f"{model_name}_{u_name}.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(len(snapshot_times),1, subplot_kw={"projection":projection})
        fig.set_dpi(1200)
        fig.subplots_adjust(hspace=0.3,wspace=0.15)
        fig.set_size_inches(3,5*len(snapshot_times))
        for (i, time) in enumerate(times):
            error = pred[i]-gt[i]
            plot_instant(x,y,error, projection=projection, cmap=cmap, title=f"{model_name} Error {u_name}", axes=axes[i], x_name=x_name, y_name=y_name)
           
            axes[i].set_ylabel("y")
            axes[i].set_xlabel("")
            axes[i].xaxis.set_major_formatter(mtick.NullFormatter())  

            axes[i].set_title(f"Error {u_name}: t={time:.2f}")
        axes[-1].set_xlabel("x")
        fig.savefig(os.path.join(folder,f"{model_name}_{u_name}_error.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig)

def generate_figures_1d(t,x,u_pred,u_gt,folder,order="tx", model_name="PINN", x_name="x"):
    '''
    Generates figures for representing the solution as a heatmap across the full time series
    '''
    if not os.path.exists(folder): os.makedirs(folder)

    cmap=cm.seismic
    if order=="tx":
        assert np.all(u_gt.shape[:-1]==np.concatenate((t.shape,x.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[...,i].T, u_pred[...,i].T) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    elif order=="xt":
        assert np.all(u_gt.shape[:-1]==np.concatenate((x.shape, t.shape)))
        assert u_gt.shape==u_pred.shape
        u_dimensions = [(u_gt[...,i], u_pred[...,i]) for i in range(u_gt.shape[-1])] ##If the output is 2d, divides it in components U1 and U2
    if len(u_dimensions)==1: u_dimension_names = ["U"]
    elif len(u_dimensions)<=3: u_dimension_names= ["U","V","W"]
    else: u_dimension_names = [f"U{i}" for i in range(len(u_dimensions))]
    
    for (i,(gt,pred)) in enumerate(u_dimensions):
        u_name = u_dimension_names[i]
        fig_gt = plot_instant(x,t,gt, projection="rectilinear", cmap=cmap, title=f"Ground Truth {u_name}", y_name="t", x_name=x_name)
        fig_gt.savefig(os.path.join(folder,f"GT_{u_name}.png"), transparent=True, format="png", bbox_inches="tight")
        fig_pred = plot_instant(x,t,pred, projection="rectilinear", cmap=cmap, title=f"{model_name} {u_name}", y_name="t", x_name=x_name)
        fig_pred.savefig(os.path.join(folder,f"{model_name}_{u_name}_pred.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig_gt)
        plt.close(fig_pred)
        ##Plot 2D plot of error
        error = pred-gt
        fig_error = plot_instant(x,t,error, projection="rectilinear", cmap=cmap, title=f"{model_name} Error", y_name="t", x_name=x_name)
        fig_error.savefig(os.path.join(folder,f"{model_name}_{u_name}_error.png"), transparent=True, format="png", bbox_inches="tight")
        plt.close(fig_error)

def plot_extrap_1d(extrapolated_u, ground_truth_u, x, time, training_limit=1.0):
    '''PLots the extrapolated error for the model across the time series, clearly separating the training and extrapolation regions'''
    with plt.rc_context({'font.size': 14}):
        fig, axes = plt.subplots(1,3, figsize=(14,6))
        T, X = np.meshgrid(time, x, indexing="ij")
        axes[0].set_title("Ground Truth")
        axes[1].set_title("Extrapolated")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("x")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("x")
        axes[0].pcolormesh(T, X, ground_truth_u, cmap="coolwarm")
        axes[0].axvline(training_limit, color='black', linestyle='--')
        axes[1].pcolormesh(T, X, extrapolated_u, cmap="coolwarm")
        axes[1].axvline(training_limit, color='black', linestyle='--')
        rRMSE = np.sqrt(np.mean((extrapolated_u - ground_truth_u)**2, axis=1) / np.mean(ground_truth_u**2, axis=1))
        ax_err = axes[2]
        ax_err.grid()
        ax_err.set_xlabel("t")
        ax_err.set_ylabel("rRMSE")
        ax_err.plot(time, rRMSE,)
        ax_err.axvline(training_limit, color='black', linestyle='--')
        ax_err.set_xlim(time[0], time[-1])
        ax_err.set_ylim(time[0], time[-1])
        ax_err.fill_between(time[time>=training_limit], 0, 10, color='gray', alpha=0.2)
        ax_err.text(0.6, 1.1, "Training\nregion", ha='center')
        ax_err.text(1.4, 1.1, "Extrapolation\nregion", ha='center')

    plt.tight_layout()
    return fig

def plot_loss_track(loss_dict, save_path=None):
    fig, axes = plt.subplots()
    axes.grid(True)
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_yscale("log")
    for (name, loss_track) in loss_dict.items():
        axes.plot(loss_track, label=name)
    axes.set_xlim(0, len(loss_track))
    axes.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)

def plot_parameter(param_track, true_values=None, save_path=None):
    param_track = np.array(param_track)
    n_epochs, n_params = param_track.shape
    fig, axes = plt.subplots(n_params, sharex=True)
    axes[0].set_xlim(0,n_epochs)
    for i in range(n_params):
        axes[i].set_ylabel(f"λ_{i+1}")
        axes[i].plot(param_track[:,i])
        if true_values is not None:
            axes[i].hlines(true_values[i], 0, n_epochs, linestyles="dashed")
        axes[i].grid(True)
    axes[-1].set_xlabel("Epoch")
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
