"""
Wilson-Cowan equations on an NxN torus (periodic boundary conditions).
Translated from XPP ring model; convolutions done via FFT2.

Kernel convention matches XPP exactly:
  K(x) = exp(-(x/sigma)^2),  normalised to sum=1
  Fourier transform: exp(-sigma^2 * k^2 / 4)

Spatial input pattern (constant in time):
  ste[i,j] = qe * cos(2*pi*m*i/N)   (varies along x-axis)
  E input:  + ste
  I input:  + rie * ste

Usage
-----
Run directly:
    python wilson_cowan_2d.py

Or import and call simulate_wc2d() / make_video() / plot_timeseries().
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Sigmoid  f(u) = 1/(1+exp(-4u)),  gain at -theta = 4*exp(4*te)/(1+exp(4*te))^2
# ---------------------------------------------------------------------------
def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-4.0 * u))


def f_thresh(u, theta):
    """Thresholded sigmoid: f(u - theta) - f(-theta).  Zero at u=0."""
    return sigmoid(u - theta) - sigmoid(-theta)


# ---------------------------------------------------------------------------
# 2-D Gaussian kernel matching XPP convention: exp(-(r/sigma)^2)
# Toroidal (minimum-image) distances; normalised to sum = 1.
# Fourier transform at wavenumber k: exp(-sigma^2 * k^2 / 4)
# ---------------------------------------------------------------------------
def make_gaussian_kernel_2d(N, sigma):
    idx = np.arange(N)
    # minimum-image: wrap indices to [-N/2, N/2)
    dx = idx - N * np.round(idx / N).astype(int)
    X, Y = np.meshgrid(dx, dx, indexing='ij')
    r2 = X**2 + Y**2
    K = np.exp(-r2 / sigma**2)          # exp(-(r/sigma)^2)  <-- XPP convention
    K /= K.sum()
    return K


# ---------------------------------------------------------------------------
# Spatial input pattern (constant in time)
# ---------------------------------------------------------------------------

def make_spatial_input(N, qex, qey, rie, mx, my):
    """
    ste[i,j] = qex*cos(2*pi*mx*i/N) + qey*cos(2*pi*my*j/N)
    qex, qey : amplitudes in x and y directions
    mx, my   : mode numbers in x and y directions
    Returns ste_E (NxN) and ste_I (NxN).
    """
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    ste_E = (qex * np.cos(2.0 * np.pi * mx * i_idx / N)[:, np.newaxis] +
             qey * np.cos(2.0 * np.pi * my * j_idx / N)[np.newaxis, :])
    ste_I = rie * ste_E
    return ste_E, ste_I



# ---------------------------------------------------------------------------
# Filtered white noise stimulus
# ---------------------------------------------------------------------------
def make_filtered_noise(N, qe, rie, r0, g, mx, A, seed=None):
    """
    Generate a spatially filtered white-noise stimulus on an NxN grid.
 
    Steps:
      1. Draw NxN Gaussian white noise (variance 1).
      2. FFT2 to Fourier space.
      3. Apply circular low-pass logistic filter: 1/(1+exp(-g*(r0-r)))
         where r is the radial distance from the DC bin in pixels.
      4. Boost the annulus at radius nx by multiplying those coefficients by A
         (soft annulus: Gaussian ring of width ~1 pixel).
      5. Renormalise so the filtered transform has the same total energy
         as the original.
      6. IFFT2 back to real space; take real part.
 
    Parameters
    ----------
    N    : grid size
    qe   : overall amplitude scaling for E input
    rie  : I input amplitude relative to E  (ste_I = rie * ste_E)
    r0   : low-pass cutoff radius (pixels)
    g    : logistic gain (larger = sharper cutoff)
    mx    : radius of the boosted annulus (pixels); set A=1 to skip boost
    A    : boost factor for the annulus at radius p
    seed : random seed (None = unpredictable)
 
    Returns
    -------
    ste_E, ste_I : NxN arrays
    """
    rng = np.random.default_rng(seed)
 
    # Step 1: white noise
    noise = rng.standard_normal((N, N))
 
    # Step 2: FFT2
    F = np.fft.fft2(noise)
    energy_orig = np.sum(np.abs(F)**2)
 
    # Radial distance map in Fourier space (minimum-image / toroidal)
    idx = np.arange(N)
    dx  = idx - N * np.round(idx / N).astype(int)
    RX, RY = np.meshgrid(dx, dx, indexing='ij')
    R = np.sqrt(RX**2 + RY**2)          # radius in pixels for each Fourier bin
 
    # Step 3: low-pass logistic filter (passes r < r0)
    lp_filter = 1.0 / (1.0 + np.exp(-g * (r0 - R)))
 
    # Step 4: boost annulus at radius nx (soft Gaussian ring, sigma=1 pixel)
    boost = 1.0 + (A - 1.0) * np.exp(-((R - mx)**2))
 
    F_filtered = F * lp_filter * boost
 
    # Step 5: renormalise energy
    energy_filt = np.sum(np.abs(F_filtered)**2)
    if energy_filt > 0:
        F_filtered *= np.sqrt(energy_orig / energy_filt)
 
    # Step 6: inverse FFT, real part
    stim  = np.real(np.fft.ifft2(F_filtered))
 
    ste_E = qe * stim
    ste_I = rie * ste_E
    return ste_E, ste_I
 

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def simulate_wc2d(
    N=60,
    # Connectivity weights increase aee to get spontaneous pattern formation
    aee=5.4, aie=10.0, aei=10.0, aii=8.0,
    # Kernel widths (same units as XPP)
    sige=4.0, sigi=8.0,
    # Thresholds
    te=0.5, ti=0.3,
    # Time constants
    taue=3.0, taui=6.6,
    # Spatial input qex is also for noise and mx is as well
    # rie is ratio of stimulus to I cells vs E cells
    qex=0, qey=0,rie=0.0, mx=6, my=0, A=50, g=0.1, r0=12, stim_seed=0,
    # Noise amplitudes (Euler-Maruyama); set to 0 to disable
    # nz_i is initial conditions randomness
    noise_e=0.0, noise_i=0.0, nz_i=0.0,
    # Integration
    dt=0.1, total=1000.0, trans=500.0,
    nout=10,
    # use_noise_stim=True for filtered noise stimulus
    seed=42, use_noise_stim=False,
):
    """
    Simulate 2-D Wilson-Cowan on an NxN torus.

    Returns
    -------
    t_out : 1-D array  (n_saved,)
    UE    : 3-D array  (n_saved, N, N)
    UI    : 3-D array  (n_saved, N, N)
    """
    rng = np.random.default_rng(seed)

    # --- kernels ---
    Ke = make_gaussian_kernel_2d(N, sige)
    Ki = make_gaussian_kernel_2d(N, sigi)
    Ke_fft = np.fft.fft2(Ke)
    Ki_fft = np.fft.fft2(Ki)

    # --- spatial input (stripes constant in time) ---
   
    if use_noise_stim:
        ste_E, ste_I = make_filtered_noise(N, qex, rie, r0, g, mx, A, seed=stim_seed)
    else:    
        ste_E, ste_I = make_spatial_input(N, qex, qey, rie, mx,my)
    # --- initial conditions ---
    UE = rng.random((N, N)) * nz_i
    UI = np.zeros((N, N))

    # --- time loop ---
    n_steps    = int(round(total / dt))
    trans_step = int(round(trans / dt))
    sqrt_dt    = np.sqrt(dt)

    saved_UE, saved_UI, saved_t = [], [], []

    for step in range(n_steps):
        t = step * dt

        FE = f_thresh(UE, te)
        FI = f_thresh(UI, ti)

        conv_e = np.real(np.fft.ifft2(Ke_fft * np.fft.fft2(FE)))
        conv_i = np.real(np.fft.ifft2(Ki_fft * np.fft.fft2(FI)))

        dUE = (-UE + aee * conv_e - aie * conv_i + ste_E) / taue
        dUI = (-UI + aei * conv_e - aii * conv_i + ste_I) / taui

        UE = UE + dt * dUE + noise_e * sqrt_dt * rng.standard_normal((N, N))
        UI = UI + dt * dUI + noise_i * sqrt_dt * rng.standard_normal((N, N))

        if step >= trans_step and (step - trans_step) % nout == 0:
            saved_UE.append(UE.copy())
            saved_UI.append(UI.copy())
            saved_t.append(t)

    t_out  = np.array(saved_t)
    UE_out = np.stack(saved_UE, axis=0)
    UI_out = np.stack(saved_UI, axis=0)
    return t_out, UE_out, UI_out, ste_E


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def make_video(t, UE, UI, outfile="wc2d.mp4", fps=20, cmap="RdBu_r", every=1):
    """
    Save an mp4 animation of E and I fields side by side.

    Parameters
    ----------
    every : use every nth saved frame (thin for large runs)
    """
    import matplotlib.animation as animation

    frames   = list(range(0, len(t), every))
    vmin_e, vmax_e = UE.min(), UE.max()
    vmin_i, vmax_i = UI.min(), UI.max()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Force exact even pixel dimensions at dpi=150: 1500x600
    fig.set_size_inches(1500/150, 600/150)

    im0 = axes[0].imshow(UE[0], origin='lower', cmap=cmap,
                         vmin=vmin_e, vmax=vmax_e, animated=True)
    axes[0].set_title("E population")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(UI[0], origin='lower', cmap=cmap,
                         vmin=vmin_i, vmax=vmax_i, animated=True)
    axes[1].set_title("I population")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    time_text = fig.text(0.5, 0.96, '', ha='center', va='top', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(k):
        im0.set_data(UE[k])
        im1.set_data(UI[k])
        time_text.set_text(f"t = {t[k]:.1f}")
        return im0, im1, time_text

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=1000 / fps, blit=True)
    ani.save(outfile, writer='ffmpeg', fps=fps, dpi=150)
    plt.close(fig)
    print(f"Video saved to {outfile}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Time series at specified (i,j) locations
# ---------------------------------------------------------------------------
def plot_timeseries(t, UE, locations, outfile=None, title="E-cell time series"):
    """
    Plot UE(t) at a list of (i,j) grid locations.

    Returns
    -------
    traces : dict {(i,j): 1-D array}  (for further analysis, FFT, etc.)
    """
    traces = {}
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(locations), 1)))

    for (i, j), color in zip(locations, colors):
        ts = UE[:, i, j]
        traces[(i, j)] = ts
        ax.plot(t, ts, color=color, lw=1.2, label=f"({i},{j})")

    ax.set_xlabel("time")
    ax.set_ylabel("$u_E$")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, ncol=max(1, len(locations) // 8))
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=150)
        print(f"Time-series plot saved to {outfile}")

    return traces


# ---------------------------------------------------------------------------
# Demo see main code for definitions of parameters
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # XPP default parameters — origin is stable, system should return to zero
    params = dict(
        N=60,   # Increasing N and keeping sig the same increases the domain size
        # increasing N and increasing sig makes the system smoother
        aee=5.4,   # below first Hopf bifurcation (m=3) at aee~7.049
        aie=10.0, aei=10.0, aii=8.0,
        sige=4.0, sigi=8.0,
        te=0.5, ti=0.3,
        taue=3.0, taui=6.6,
        qex=0.2, qey=0.,rie=0, mx=6, my=6, A=50,g=0.1,r0=12,stim_seed=0,
        noise_e=0.00, noise_i=0.00, nz_i=0.2,
        dt=0.1, total=1700.0, trans=1200.0,
        nout=10,
        seed=42,use_noise_stim=False,
    )

    print("Running Wilson-Cowan 2D simulation...")
    t, UE, UI,ste_E = simulate_wc2d(**params)
    print(f"  Saved {len(t)} frames after transient.")
    print(f"  UE range: [{UE.min():.3f}, {UE.max():.3f}]")
    print(f"  Temporal std at (30,30): {UE[:,30,30].std():.4f}")

# without making the video, it is fast. Otherwise it takes some time
#    make_video(t, UE, UI, outfile="wc2d.mp4", fps=20, every=1)

    locs = [(0,0), (15,0), (30,0), (45,0), (30,15), (30,30), (30,45)]
    traces = plot_timeseries(t, UE, locations=locs, outfile="wc2d_timeseries.png")

    print("\nPeak-to-peak UE amplitudes:")
    for loc, ts in traces.items():
        print(f"  {loc}: {ts.max() - ts.min():.4f}")

    plt.show()
    
    fig, ax = plt.subplots()
    im = ax.imshow(UE[-1], origin='lower', cmap='RdBu_r')
    ax.set_title('E population (final frame)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()



    F = np.fft.fft2(ste_E)
    power = np.abs(np.fft.fftshift(F))**2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(ste_E, origin='lower', cmap='RdBu_r')
    axes[0].set_title('Filtered noise stimulus (E)')
    axes[1].imshow(np.log1p(power), origin='lower', cmap='hot')
    axes[1].set_title('Log power spectrum')
    for ax in axes:
        plt.colorbar(ax.images[0], ax=ax)
    plt.tight_layout()
    plt.show()
