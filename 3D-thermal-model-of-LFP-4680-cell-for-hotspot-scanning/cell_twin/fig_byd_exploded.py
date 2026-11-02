"""Paper 1.2, Figure 1: the BYD FC4680 LFP cell, exploded, and where its
thermal non-uniformity comes from.

    python fig_byd_exploded.py   ->  figures/fig_byd_exploded.{pdf,png}

(a) Exploded oblique view along the cell axis, cap (positive) end at the left.
    Diameters and axial lengths to scale; the gaps between parts are not.
    The four aluminium cathode tabs are drawn on the roll's cap face at their
    measured radii.
(b) The two roll end faces, true scale, seen along the axis: cathode tabs on
    the Al disk, anode tabs on the Cu disk, the disconnected anode tab hatched.
(c) True-scale half section (r, z) with the parts that make the field
    non-uniform, numbered as in the table of the plan.

Every dimension is read from byd_parts.py (Liu et al. 2025 teardown, tags
[M] measured, [D] derived, [E] estimated). What the teardown does not give
and is drawn here as an assumption [E]:
  * tab angles: cathode tabs 90 deg apart ("~90 deg" in [S1]); anode tabs
    120 deg apart; tab width 4 mm; tab free length 4 mm;
  * which end is positive: cathode (Al) at the cap, as in the Tesla cell;
  * the insulator disks sit between each collector disk and the can end, the
    arrangement the thermal network of byd_parts.py uses.
"""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, RegularPolygon, Wedge
from matplotlib.collections import PolyCollection

import byd_parts as B

MM = 10.0                      # cm -> mm
K = 0.33                       # oblique factor: end ellipse width / radius

STEEL, STEEL_D = "#b3b5b9", "#7f8186"
AL, AL_D = "#cfd4da", "#9aa1a9"
CU, CU_D = "#c47f4f", "#8c5330"
NI = "#9fa4a0"
PLASTIC, PLASTIC_D = "#e6c75a", "#b8982f"
ROLL, ROLL_D = "#ece6da", "#bfb6a3"
CATH = "#5b6f8c"
MANDREL = "#3a3a3a"
INK, MUTE = "#1f2328", "#5d6470"
ORANGE, GREEN = "#d0632f", "#2f9d6e"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.edgecolor": INK, "axes.linewidth": 0.6})

# ---------------------------------------------------------------- geometry
R_CAN = B.CAN_OD / 2 * MM                  # 23.15
WALL = B.CAN_WALL * MM                     # 0.418
H_CAN = B.CAN_H * MM                       # 80.2
R_ROLL = B.roll_radius_max() * MM          # 22.73, bore-filling roll [D]
R_CORE = B.core_radius() * MM              # 3.16 mandrel outer radius
H_ROLL = B.AN_H * MM                       # 71 (anode height; cathode 69)
H_CATH = B.CATH_H * MM
T_DISK = B.DISK_T * MM                     # 0.31
T_PL = B.PLASTIC_T * MM                    # 0.29
R_ALD = B.CATHODE_DISK_OD / 2 * MM         # 20 [E]
R_CUD = B.ANODE_DISK_OD / 2 * MM           # 22 [E]
TAB_W = B.TAB_W * MM                       # 4 [E]
TAB_L = 4.0                                # free tab length [E]
CATH_TAB_R = [r * MM for r in B.CATHODE_TAB_R]
AN_TAB_R = [r * MM for r in B.ANODE_TAB_R]
CATH_TAB_A = [90, 0, 270, 180]             # deg [E], 90 deg apart
# [S1] gives 0.4 mm for the innermost cathode tab, which is inside the
# mandrel (outer radius 3.16 mm). Drawn at the first wrap; to check in [S1].
R_FIRST = None
AN_TAB_A = [60, 180, 300]                  # deg [E], 120 deg apart
AN_DEAD = 0                                # index of the PET-covered tab [M]: the 9.0 mm one is assumed dead [E]
PITCH = B.winding_period() * MM            # 0.302 mm
N_TURNS = (R_ROLL - R_CORE) / PITCH


def rdraw(r):
    return r if r > R_CORE else R_CORE + 0.8


# --------------------------------------------------------- oblique drawing
def ell(xc, r, n=181, t0=0, t1=2 * np.pi):
    t = np.linspace(t0, t1, n)
    return np.c_[xc + K * r * np.sin(t), r * np.cos(t)]


def solid(ax, x0, x1, r, face, side, edge, z=1, r_in=0.0, lw=0.5, alpha=1.0):
    """Cylinder (or disk) from x0 to x1 (x0 < x1), seen from the left.
    Body = band between the two tangent lines, far (right) end drawn as the
    visible half ellipse, near (left) end as the full face."""
    # body: left half-ellipse at x0 (front half) .. right half-ellipse at x1
    tb = np.linspace(0, np.pi, 91)
    right = np.c_[x1 + K * r * np.sin(tb), r * np.cos(tb)]           # far end, outer half
    left = np.c_[x0 + K * r * np.sin(tb[::-1]), r * np.cos(tb[::-1])]
    body = np.vstack([right, left])
    # graded barrel: three bands
    ax.add_patch(Polygon(body, closed=True, fc=side, ec=edge, lw=lw, zorder=z, alpha=alpha))
    for frac, col_a in ((0.55, 0.10), (0.85, 0.18)):
        tt = np.linspace(np.pi * (1 - frac) / 2 + np.pi * frac, np.pi, 40)
        band = np.vstack([np.c_[x1 + K * r * np.sin(tt), r * np.cos(tt)],
                          np.c_[x0 + K * r * np.sin(tt[::-1]), r * np.cos(tt[::-1])]])
        ax.add_patch(Polygon(band, closed=True, fc="black", ec="none", alpha=col_a * alpha, zorder=z))
    # near face
    ax.add_patch(Polygon(ell(x0, r), closed=True, fc=face, ec=edge, lw=lw, zorder=z + 0.1, alpha=alpha))
    if r_in > 0:
        ax.add_patch(Polygon(ell(x0, r_in), closed=True, fc="white", ec=edge, lw=lw * 0.8, zorder=z + 0.2))


def face_xy(x, r, ang_deg):
    a = np.deg2rad(ang_deg)
    return x + K * r * np.sin(a), r * np.cos(a)


def tab(ax, x_face, r, ang, length, w, fc, ec, z):
    """A tab strip standing out of an end face, leftwards by `length`:
    tangential width w at radius r, angle ang (0 = top)."""
    a = np.deg2rad(ang)
    # tangential unit vector in (y, zdepth) plane: (-sin a, cos a) for (y, z) = (r cos a, r sin a)
    ty, tz = -np.sin(a), np.cos(a)
    yc, zc = r * np.cos(a), r * np.sin(a)
    p1 = (yc - ty * w / 2, zc - tz * w / 2)
    p2 = (yc + ty * w / 2, zc + tz * w / 2)

    def sc(x, y, zd):
        return (x + K * zd, y)
    quad = [sc(x_face, *p1), sc(x_face, *p2), sc(x_face - length, *p2), sc(x_face - length, *p1)]
    ax.add_patch(Polygon(quad, closed=True, fc=fc, ec=ec, lw=0.5, zorder=z))


def spiral_face(ax, x, r0, r1, step_turns, color, z):
    t = np.linspace(0, 2 * np.pi * (r1 - r0) / PITCH, 6000)
    r = r0 + PITCH * t / (2 * np.pi)
    sel = np.arange(0, len(t), 1)
    # draw only every `step_turns` turns as a thin line (true spiral, decimated)
    mask = ((t / (2 * np.pi)) % step_turns) < 1.0
    xs = x + K * r * np.sin(t)
    ys = r * np.cos(t)
    xs[~mask] = np.nan
    ax.plot(xs, ys, color=color, lw=0.35, zorder=z, solid_capstyle="butt")


def callout(ax, xy, text, xytext, color=INK, ha="center"):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=7.3, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=1, shrinkB=1), zorder=50)


def panel_a(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    g = 16.0     # exploded gap, mm (not to scale)
    # x positions, cap at left
    x_can0, x_can1 = 0.0, H_CAN
    x = x_can1 + g
    x_pl1 = (x, x + T_PL); x = x_pl1[1] + 0.6 * g
    x_al = (x, x + T_DISK); x = x_al[1] + 0.6 * g + TAB_L
    x_roll = (x, x + H_ROLL); x = x_roll[1] + 0.6 * g + TAB_L
    x_cu = (x, x + T_DISK); x = x_cu[1] + 0.6 * g
    x_pl2 = (x, x + T_PL); x = x_pl2[1] + g
    x_lid = (x, x + WALL * 2)
    # draw right to left (far to near)
    solid(ax, *x_lid, R_CAN, STEEL, STEEL_D, INK, z=2)
    solid(ax, *x_pl2, R_CAN - WALL, PLASTIC, PLASTIC_D, PLASTIC_D, z=3)
    solid(ax, *x_cu, R_CUD, CU, CU_D, CU_D, z=4)
    # anode tabs on the far (right) face of the roll: drawn as stubs between roll and Cu disk
    for i, (r, a) in enumerate(zip(AN_TAB_R, AN_TAB_A)):
        tab(ax, x_roll[1] + TAB_L, r, a, TAB_L, TAB_W, NI if i != AN_DEAD else "#e9e9e9",
            "#555" if i != AN_DEAD else "#999", z=4.5)
    # roll
    solid(ax, *x_roll, R_ROLL, ROLL, ROLL_D, ROLL_D, z=5)
    # tape bands
    for xb in (x_roll[0] + 6, x_roll[1] - 12):
        tb = np.linspace(0, np.pi, 91)
        band = np.vstack([np.c_[xb + 6 + K * (R_ROLL + .05) * np.sin(tb), (R_ROLL + .05) * np.cos(tb)],
                          np.c_[xb + K * (R_ROLL + .05) * np.sin(tb[::-1]), (R_ROLL + .05) * np.cos(tb[::-1])]])
        ax.add_patch(Polygon(band, closed=True, fc="#4c78b5", ec="#3a6197", lw=0.4, zorder=5.2, alpha=0.9))
    ax.add_patch(Polygon(ell(x_roll[0], R_ROLL), closed=True, fc=ROLL, ec=ROLL_D, lw=0.5, zorder=5.3))
    spiral_face(ax, x_roll[0], R_CORE, R_ROLL, 6, ROLL_D, z=5.4)
    ax.add_patch(Polygon(ell(x_roll[0], R_CORE), closed=True, fc=MANDREL, ec=MANDREL, lw=0.4, zorder=5.5))
    hexr = B.MANDREL_HEX_SIDE * MM
    hx = [x_roll[0] + K * hexr * np.sin(np.deg2rad(30 + 60 * k)) for k in range(6)]
    hy = [hexr * np.cos(np.deg2rad(30 + 60 * k)) for k in range(6)]
    ax.add_patch(Polygon(np.c_[hx, hy], closed=True, fc="white", ec=MANDREL, lw=0.4, zorder=5.6))
    # cathode tabs standing out of the near face
    for r, a in zip(CATH_TAB_R, CATH_TAB_A):
        tab(ax, x_roll[0], rdraw(r), a, TAB_L, TAB_W, AL, AL_D, z=5.8)
    solid(ax, *x_al, R_ALD, AL, AL_D, AL_D, z=6)
    # weld marks on Al disk near face
    for r, a in zip(CATH_TAB_R, CATH_TAB_A):
        px, py = face_xy(x_al[0], rdraw(r), a)
        ax.plot(px, py, marker="s", ms=2.2, color=AL_D, zorder=6.5)
    solid(ax, *x_pl1, R_CAN - WALL, PLASTIC, PLASTIC_D, PLASTIC_D, z=7)
    # can: closed cap end at the left, open at the right -> draw open ring on the right
    solid(ax, x_can0, x_can1, R_CAN, STEEL, STEEL_D, INK, z=8)
    # positive terminal boss on the cap face (hidden side) -> outline only
    term = ell(x_can0, 8.0)
    ax.plot(term[:, 0], term[:, 1], color=STEEL_D, lw=0.5, ls=(0, (3, 2)), zorder=9)

    # callouts (above: y > 30, below: y < -30)
    callout(ax, face_xy(x_can0 + 40, R_CAN, 0), f"steel can  ∅{2*R_CAN:.1f} × {H_CAN:.1f}, wall {WALL:.2f} [D]", (40, 36))
    callout(ax, (x_can0, -5), "+ terminal (cap)", (12, -36))
    callout(ax, face_xy(x_pl1[0], R_CAN - WALL, 20), f"insulator\n{T_PL:.2f}", (x_pl1[0] - 6, 40), PLASTIC_D)
    callout(ax, face_xy(x_al[0], R_ALD, 165), f"Al cathode disk\n∅{2*R_ALD:.0f} × {T_DISK:.2f} [E]", (x_al[0] - 10, -40), "#6f7882")
    callout(ax, face_xy(x_roll[0] - TAB_L, CATH_TAB_R[3], 180), "4 Al cathode tabs", (x_roll[0] - 4, -50), "#6f7882")
    callout(ax, face_xy(x_roll[0], R_CORE, 0), "plastic mandrel\n(hex, 2.5 mm side)", (x_roll[0] + 14, 40), MANDREL)
    callout(ax, (x_roll[0] + H_ROLL * 0.6, -R_ROLL), f"jelly roll  ∅{2*R_ROLL:.1f} × {H_ROLL:.0f};  {N_TURNS:.0f} turns,\n"
            f"pitch {PITCH*1000:.0f} µm, electrode 5.37 m", (x_roll[0] + H_ROLL * 0.52, -40))
    callout(ax, face_xy(x_roll[1] + TAB_L * 0.5, AN_TAB_R[2], 300), "3 Ni anode tabs\n(one disconnected)",
            (x_roll[1] - 14, 40), "#4f5552")
    callout(ax, face_xy(x_cu[0], R_CUD, 160), f"Cu anode disk\n∅{2*R_CUD:.0f} × {T_DISK:.2f} [E]", (x_cu[0] + 6, -48), CU_D)
    callout(ax, face_xy(x_pl2[0], R_CAN - WALL, 20), f"insulator\n{T_PL:.2f}", (x_pl2[0] + 12, 40), PLASTIC_D)
    callout(ax, face_xy(x_lid[0], R_CAN, 160), "base lid (−)", (x_lid[0] + 4, -40), STEEL_D)
    ax.text(-10, 49, "a", fontsize=11, weight="bold", va="bottom")
    ax.text(-2, 49, "BYD FC4680 (LFP), exploded along the axis; diameters and lengths to scale, gaps not",
            fontsize=8.5, va="bottom")
    ax.set_xlim(-12, x_lid[1] + 12)
    ax.set_ylim(-52, 54)


def end_view(ax, title, disk_r, disk_c, disk_e, tabs_r, tabs_a, tab_c, dead=None, label_c=INK):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), R_CAN, fc="none", ec=STEEL_D, lw=0.8))
    ax.add_patch(Circle((0, 0), R_ROLL, fc=ROLL, ec=ROLL_D, lw=0.5))
    t = np.linspace(0, 2 * np.pi * N_TURNS, 8000)
    r = R_CORE + PITCH * t / (2 * np.pi)
    m = ((t / (2 * np.pi)) % 6) < 1
    xs, ys = r * np.sin(t), r * np.cos(t)
    xs[~m] = np.nan
    ax.plot(xs, ys, color=ROLL_D, lw=0.3)
    ax.add_patch(Circle((0, 0), disk_r, fc=disk_c, ec=disk_e, lw=0.8, alpha=0.35))
    ax.add_patch(Circle((0, 0), R_CORE, fc=MANDREL, ec=MANDREL))
    ax.add_patch(RegularPolygon((0, 0), 6, radius=B.MANDREL_HEX_SIDE * MM, fc="white", ec=MANDREL, lw=0.4))
    for i, (rr, aa) in enumerate(zip(tabs_r, tabs_a)):
        rr_d = max(rr, 1.2)
        a = np.deg2rad(aa)
        cx, cy = rr_d * np.sin(a), rr_d * np.cos(a)
        rect = Rectangle((-TAB_W / 2, -0.5), TAB_W, 1.0, angle=0, fc=tab_c, ec="#333", lw=0.5)
        # rotate tab tangentially
        import matplotlib.transforms as mt
        tr = mt.Affine2D().rotate(-a).translate(cx, cy) + ax.transData
        rect.set_transform(tr)
        if dead is not None and i == dead:
            rect.set_facecolor("white")
            rect.set_hatch("/////")
            rect.set_edgecolor("#888")
        ax.add_patch(rect)
        ax.text(cx * 1.0 + 3.4 * np.sin(a), cy + 3.4 * np.cos(a), f"{rr:.1f}" + ("*" if rr < R_CORE else ""), ha="center", va="center", fontsize=6.5, color=label_c)
    ax.set_xlim(-26, 26)
    ax.set_ylim(-33, 31)
    ax.text(0, 27.5, title, ha="center", va="bottom", fontsize=7.4)


def panel_c(ax):
    """True-scale half section r (horizontal) x z (vertical), cap at top."""
    ax.set_aspect("equal")
    ax.axis("off")
    z_top = H_CAN
    # can wall and ends
    ax.add_patch(Rectangle((R_CAN - WALL, 0), WALL, H_CAN, fc=STEEL, ec=STEEL_D, lw=0.4))
    ax.add_patch(Rectangle((0, H_CAN - WALL), R_CAN, WALL, fc=STEEL, ec=STEEL_D, lw=0.4))
    ax.add_patch(Rectangle((0, 0), R_CAN, WALL, fc=STEEL, ec=STEEL_D, lw=0.4))
    ax.add_patch(Rectangle((0, H_CAN), 8.0, 1.0, fc=STEEL, ec=STEEL_D, lw=0.4))   # terminal boss [E]
    # stack from the top: insulator, Al disk, tab gap, roll, tab gap, Cu disk, insulator
    zc = H_CAN - WALL
    ax.add_patch(Rectangle((0, zc - T_PL), R_CAN - WALL, T_PL, fc=PLASTIC, ec=PLASTIC_D, lw=0.3)); zc -= T_PL
    ax.add_patch(Rectangle((0, zc - T_DISK), R_ALD, T_DISK, fc=AL, ec=AL_D, lw=0.3)); zc -= T_DISK
    gap_top = zc
    zb = WALL
    ax.add_patch(Rectangle((0, zb), R_CAN - WALL, T_PL, fc=PLASTIC, ec=PLASTIC_D, lw=0.3)); zb += T_PL
    ax.add_patch(Rectangle((0, zb), R_CUD, T_DISK, fc=CU, ec=CU_D, lw=0.3)); zb += T_DISK
    gap_bot = zb
    free = (gap_top - gap_bot) - H_ROLL
    z_roll0 = gap_bot + free / 2
    ax.add_patch(Rectangle((R_CORE, z_roll0), R_ROLL - R_CORE, H_ROLL, fc=ROLL, ec=ROLL_D, lw=0.3))
    ax.add_patch(Rectangle((R_CORE, z_roll0 + (H_ROLL - H_CATH) / 2), R_ROLL - R_CORE, H_CATH, fc=CATH, ec="none", alpha=0.10))
    ax.add_patch(Rectangle((R_CORE - 0.66, z_roll0 + H_ROLL / 2 - B.MANDREL_H * MM / 2), 0.66, B.MANDREL_H * MM, fc=MANDREL, ec=MANDREL))
    # tabs as vertical strips at their radii
    for r in CATH_TAB_R:
        rr = max(r, R_CORE + 0.4) if r < R_CORE else r
        ax.add_patch(Rectangle((rr - 0.35, z_roll0 + H_ROLL), 0.7, gap_top - (z_roll0 + H_ROLL), fc=AL_D, ec="none"))
    for i, r in enumerate(AN_TAB_R):
        c = "#cfcfcf" if i == AN_DEAD else "#5a5f5b"
        ax.add_patch(Rectangle((r - 0.35, gap_bot), 0.7, z_roll0 - gap_bot, fc=c, ec="none"))
    # foam cap (validation), dashed, on the base end
    ax.add_patch(Rectangle((0, -6), R_CAN + 1.5, 6, fc="none", ec=GREEN, lw=0.8, ls=(0, (3, 2))))
    ax.text(R_CAN / 2, -3, "foam cap (test)", ha="center", va="center", fontsize=6.5, color=GREEN)
    # numbered sources
    def num(n, x, y):
        ax.add_patch(Circle((x, y), 1.7, fc=ORANGE, ec="none", zorder=20))
        ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=6.5, weight="bold", zorder=21)
    num(1, 13.1, z_roll0 + H_ROLL + 4.5)
    num(2, 27.5, z_roll0 + H_ROLL - 1.5)
    num(3, 27.5, H_CAN + 2.5)
    num(4, R_CORE - 3, z_roll0 + H_ROLL / 2)
    num(5, AN_TAB_R[0], z_roll0 - 4.5)
    num(6, R_CAN + 2.6, H_CAN / 2)
    num(7, 11, z_roll0 + H_ROLL * 0.62)
    ax.annotate("", xy=(0, -8.5), xytext=(R_CAN, -8.5), arrowprops=dict(arrowstyle="<->", lw=0.5, color=MUTE))
    ax.text(R_CAN / 2, -10.5, "r, 0–23.15 mm", ha="center", fontsize=6.5, color=MUTE)
    ax.annotate("", xy=(-3, 0), xytext=(-3, H_CAN), arrowprops=dict(arrowstyle="<->", lw=0.5, color=MUTE))
    ax.text(-4.5, H_CAN / 2, "z, 80.2 mm", rotation=90, ha="center", va="center", fontsize=6.5, color=MUTE)
    ax.set_xlim(-7, 30)
    ax.set_ylim(-12, 85)
    ax.text(-7, 84.5, "c", fontsize=11, weight="bold", va="bottom")
    ax.text(-1, 84.5, "true-scale half section", fontsize=8, va="bottom")


SOURCES = [
    ("1", "Discrete tabs: current runs along the 5.37 m electrode to 4 Al / 2 live Ni tabs, so heat is concentrated near them."),
    ("2", "Roll-to-disk through tab welds only: 41–49 W/K (tabless NMC cell: 1,030–2,100 W/K)."),
    ("3", "Two insulator disks (0.29 mm) decouple both ends from the can."),
    ("4", "Plastic mandrel: no metallic core path; the hottest region is near the core."),
    ("5", "Disconnected anode tab (PET-covered): asymmetric collection at the base."),
    ("6", "Can wall: the axial path outside the roll; end asymmetry ≈ 10."),
    ("7", "A soft short: a local heat source anywhere in the roll; the subject of the paper."),
]


def main():
    fig = plt.figure(figsize=(7.4, 7.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], width_ratios=[1, 1, 1.05], hspace=0.05, wspace=0.08)
    panel_a(fig.add_subplot(gs[0, :]))
    axb1 = fig.add_subplot(gs[1, 0])
    end_view(axb1, "cap end: Al tabs on Al disk", R_ALD, AL, AL_D, CATH_TAB_R, CATH_TAB_A, AL)
    axb1.text(-27, 31, "b", fontsize=11, weight="bold")
    axb1.text(0, -29.5, "*reported 0.4 mm, inside the mandrel;\ndrawn at the first wrap (to check)", ha="center", va="top", fontsize=6.3, color=MUTE)
    axb2 = fig.add_subplot(gs[1, 1])
    end_view(axb2, "base end: Ni tabs on Cu disk", R_CUD, CU, CU_D, AN_TAB_R, AN_TAB_A, NI, dead=AN_DEAD)
    axb2.text(0, -29.5, "hatched: disconnected tab", ha="center", fontsize=6.5, color=MUTE)
    panel_c(fig.add_subplot(gs[1, 2]))
    # sources table under everything
    y = 0.035
    txt = "\n".join(f"{n}  {s}" for n, s in SOURCES)
    fig.text(0.06, 0.012, "Sources of thermal non-uniformity (numbers as in c)\n" + txt, fontsize=6.6, va="bottom",
             color=INK, linespacing=1.35)
    fig.text(0.98, 0.012, "Dimensions: byd_parts.py (Liu et al. 2025 teardown).\n[E] = estimated: tab angles, tab width,\n"
             "disk diameters, which end is positive,\ninsulator position.", fontsize=6.0, color=MUTE, ha="right", va="bottom")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"../figures/fig_byd_exploded.{ext}", dpi=220)
    print("ok", N_TURNS, R_ROLL)


if __name__ == "__main__":
    main()
