"""
Interactive 31P MRSI viewer — opens in your web browser.

Usage:
    python scripts/viewer_31p.py [phantom|phantom2|dynamic]

Then open: http://localhost:8050
"""
import sys, os, glob
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODE = sys.argv[1] if len(sys.argv) > 1 else "phantom"

# ── Load data ──────────────────────────────────────────────────────────────────
if MODE == "phantom":
    out_dir = "./output_phantom"
    title   = "31P Phantom (80 rings, 12×12)"
    larmor  = 49.894571
    dwell_s = 360e-9
    N       = 12

elif MODE == "phantom2":
    out_dir = "./output_phantom2"
    title   = "31P Phantom2 (80 rings, 12×12)"
    larmor  = 49.894571
    dwell_s = 360e-9
    N       = 12

elif MODE == "dynamic":
    out_dir = "./output_dynamic_3T"
    title   = "31P Dynamic calf (52 reps, 10×10, 5 rings)"
    larmor  = 49.894571
    dwell_s = 360e-9
    N       = 10
else:
    sys.exit(f"Unknown mode: {MODE}. Use phantom / phantom2 / dynamic")

data_dir = os.path.join(out_dir, "data")

if MODE == "dynamic":
    files = sorted(glob.glob(os.path.join(data_dir, "iter3d_recon_spec_rep*.npy")))
    print(f"Loading {len(files)} dynamic reps...")
    reps  = [np.load(f).astype(np.complex128) for f in files]
    spec  = np.stack(reps, axis=0)   # (nrep, Nx, Ny, Nz, nspec)
    n_reps = spec.shape[0]
else:
    raw   = np.load(os.path.join(data_dir, "iter3d_recon_spec.npy")).astype(np.complex128)
    spec  = raw[np.newaxis, ...]      # (1, Nx, Ny, Nz, nspec)
    n_reps = 1

# spec shape: (nrep, Nx, Ny, Nz, nspec)
n_spec = spec.shape[-1]
FREQ   = np.fft.fftshift(np.fft.fftfreq(n_spec, d=dwell_s))
PPM    = -FREQ / larmor
ppm_mask = (PPM >= -20.0) & (PPM <= 20.0)
pcr_mask = (PPM >= -1.5)  & (PPM <= 1.5)

mag      = np.abs(spec)                                  # (nrep, Nx, Ny, Nz, nspec)
pcr_amp  = mag[:, :, :, 0, :][:, :, :, pcr_mask].max(-1)  # (nrep, Nx, Ny)
noise    = mag[:, :, :, 0, :][:, :, :, np.abs(PPM) > 20].std(-1)
snr_map  = np.where(noise > 0, pcr_amp / noise, 0.0)

print(f"Loaded: spec {spec.shape}, PPM range [{PPM.min():.1f}, {PPM.max():.1f}]")

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

rep_slider = html.Div([
    html.Label("Rep:", style={"color": "#ccc", "marginRight": "8px"}),
    dcc.Slider(id="rep-slider", min=0, max=n_reps-1, step=1, value=0,
               marks={i: str(i) for i in range(0, n_reps, max(1, n_reps//10))},
               tooltip={"placement": "bottom"}),
], style={"display": "block" if n_reps > 1 else "none", "marginBottom": "12px"})

app.layout = html.Div([
    html.H2(title, style={"color": "#eee", "textAlign": "center", "marginBottom": "4px"}),
    html.P("Click a voxel to see its spectrum   |   Toggle map: PCr amplitude / SNR",
           style={"color": "#aaa", "textAlign": "center", "marginBottom": "8px", "fontSize": "13px"}),
    rep_slider,
    html.Div([
        # Left: spatial map (clickable)
        html.Div([
            dcc.Graph(id="spatial-map", style={"height": "480px"},
                      config={"displayModeBar": False}),
            dcc.RadioItems(
                id="map-type",
                options=[{"label": " PCr amplitude", "value": "pcr"},
                         {"label": " SNR",           "value": "snr"}],
                value="pcr",
                inline=True,
                style={"color": "#ccc", "fontSize": "13px", "marginTop": "4px"}),
        ], style={"flex": "1", "minWidth": "300px"}),
        # Right: spectrum
        html.Div([
            dcc.Graph(id="spectrum", style={"height": "480px"},
                      config={"displayModeBar": False}),
            html.Div(id="voxel-label",
                     style={"color": "#aaa", "fontSize": "12px", "textAlign": "center"}),
        ], style={"flex": "1.4", "minWidth": "380px"}),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    dcc.Store(id="selected-voxel", data={"ix": N//2, "iy": N//2}),
], style={"backgroundColor": "#1e1e1e", "padding": "16px", "fontFamily": "sans-serif",
          "minHeight": "100vh"})


@app.callback(
    Output("spatial-map", "figure"),
    Input("map-type", "value"),
    Input("rep-slider", "value"),
    Input("selected-voxel", "data"),
)
def update_map(map_type, rep_idx, sel):
    rep_idx = rep_idx or 0
    data2d  = pcr_amp[rep_idx, :, :] if map_type == "pcr" else snr_map[rep_idx, :, :]
    label   = "PCr amplitude" if map_type == "pcr" else "SNR"

    # Display with 90° CCW: row = N-1-ix, col = iy
    disp = np.flipud(data2d)   # ix=0 at top → flip so ix=N-1 at top... wait
    # Convention: axes_z[N-1-ix, iy] → np.flipud(data)[row, col]
    # np.flipud flips along axis 0: row 0 shows ix=N-1, row N-1 shows ix=0
    disp = np.flipud(data2d)

    fig = go.Figure(go.Heatmap(
        z=disp, colorscale="Hot", zmin=0,
        colorbar=dict(title=label, tickfont=dict(color="#ccc"),
                      titlefont=dict(color="#ccc")),
    ))

    # Mark selected voxel
    ix0, iy0 = sel["ix"], sel["iy"]
    disp_row = (N - 1 - ix0)   # row in displayed image
    fig.add_trace(go.Scatter(
        x=[iy0], y=[disp_row], mode="markers",
        marker=dict(symbol="cross", size=12, color="cyan", line=dict(width=2)),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f"{label} map (90° CCW)  rep={rep_idx}", font=dict(color="#eee")),
        plot_bgcolor="#2a2a2a", paper_bgcolor="#1e1e1e",
        xaxis=dict(title="iy (READ)", tickfont=dict(color="#ccc"),
                   titlefont=dict(color="#ccc"), showgrid=False, zeroline=False),
        yaxis=dict(title="ix (PHASE, ↑=0)", tickfont=dict(color="#ccc"),
                   titlefont=dict(color="#ccc"), showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(t=40, b=40, l=50, r=10),
    )
    return fig


@app.callback(
    Output("selected-voxel", "data"),
    Input("spatial-map", "clickData"),
    State("selected-voxel", "data"),
)
def on_map_click(click_data, current):
    if click_data is None:
        return current
    pt   = click_data["points"][0]
    iy0  = int(round(pt["x"]))
    disp_row = int(round(pt["y"]))
    ix0  = (N - 1) - disp_row   # invert display transform
    ix0  = max(0, min(N-1, ix0))
    iy0  = max(0, min(N-1, iy0))
    return {"ix": ix0, "iy": iy0}


@app.callback(
    Output("spectrum", "figure"),
    Output("voxel-label", "children"),
    Input("selected-voxel", "data"),
    Input("rep-slider", "value"),
)
def update_spectrum(sel, rep_idx):
    rep_idx = rep_idx or 0
    ix0, iy0 = sel["ix"], sel["iy"]

    s = mag[rep_idx, ix0, iy0, 0, :]
    ppm = PPM

    # All voxel spectra (faint)
    fig = go.Figure()
    all_mag = mag[rep_idx, :, :, 0, :]  # (Nx, Ny, nspec)
    vmax    = float(all_mag.max())

    # Background: all other spectra in gray
    for ix in range(N):
        for iy in range(N):
            if ix == ix0 and iy == iy0:
                continue
            fig.add_trace(go.Scatter(
                x=ppm[ppm_mask], y=all_mag[ix, iy, ppm_mask],
                mode="lines", line=dict(color="rgba(150,150,150,0.15)", width=0.7),
                showlegend=False, hoverinfo="skip",
            ))

    # Highlighted voxel
    pcr_val = float(pcr_amp[rep_idx, ix0, iy0])
    snr_val = float(snr_map[rep_idx, ix0, iy0])
    fig.add_trace(go.Scatter(
        x=ppm[ppm_mask], y=s[ppm_mask],
        mode="lines", line=dict(color="#00ccff", width=2),
        name=f"[{ix0},{iy0}]  PCr={pcr_val:.0f}  SNR={snr_val:.1f}",
    ))

    # PCr region shading
    fig.add_vrect(x0=-1.5, x1=1.5, fillcolor="rgba(255,200,0,0.08)",
                  layer="below", line_width=0)

    fig.update_layout(
        title=dict(text=f"Spectrum at voxel [{ix0}, {iy0}]  (rep {rep_idx})",
                   font=dict(color="#eee")),
        xaxis=dict(title="Chemical shift (ppm)", autorange="reversed",
                   tickfont=dict(color="#ccc"), titlefont=dict(color="#ccc"),
                   showgrid=True, gridcolor="#333", zeroline=False),
        yaxis=dict(title="|magnitude|", range=[0, vmax * 1.05],
                   tickfont=dict(color="#ccc"), titlefont=dict(color="#ccc"),
                   showgrid=True, gridcolor="#333", zeroline=False),
        plot_bgcolor="#2a2a2a", paper_bgcolor="#1e1e1e",
        legend=dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=50, l=60, r=10),
    )
    label = f"Voxel [{ix0}, {iy0}]  |  PCr amp = {pcr_val:.1f}  |  SNR = {snr_val:.1f}  |  Rep {rep_idx}/{n_reps-1}"
    return fig, label


if __name__ == "__main__":
    print(f"\nOpen in browser: http://localhost:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
