"""A MuJoCo scene as a world for the eyes: what the photoreceptors see is rendered
by MuJoCo instead of ray-cast analytically.

Needs the flybody extra (mujoco, dm_control): python tasks.py env --extras flybody

`MujocoWorld` has the same interface as the analytic `Corridor`
(`luminance_at(origin, directions)`), so the eyes, flyvis, the LPTC layer and the
controller are unchanged. Each call:

1. A rig of six cameras (a cube map: +-x, +-y, +-z, each a little over 90 deg)
   is moved to the viewpoint. The rig is a mocap body and stays aligned with the
   world axes, so its position is the only thing that changes.
2. The six faces are rendered unlit: scene lights are removed and every visible
   surface emits its texture, so a pixel is the texture's luminance (no shading,
   no highlights). The fly's own body is hidden, as in the analytic worlds.
3. Every viewing direction looks up its face and pixel. Faces are averaged into
   a pyramid (2x2 boxes) and each direction reads the level whose pixels are as
   wide as the spacing of the eye's samples, so one sample averages the patch of
   scene it stands for instead of aliasing it (the analogue of the prefiltered
   texture pyramid of `Corridor`).

Rendering uses MuJoCo's OpenGL renderer, on the GPU where there is one
(MUJOCO_GL picks the backend; egl or osmesa on a headless node).

`CorridorArena` builds the flyloop Corridor as MuJoCo geometry with the same
textures, so the two renderings of one scene can be compared
(tests/test_mujoco_world.py).

Units: MuJoCo is in cm; flyloop positions are in corridor widths,
pos_cm = centre_cm + pos_flyloop * scale_cm.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from dm_control import composer, mjcf

from flyloop.geometry import directions

from .corridor import Corridor

ARENA_GROUP = 0    # geom group of walls, floor and end caps
CEILING_GROUP = 2  # geom group of the ceiling, hidden in views from above
RIG = "eye_rig"
# cube faces: (forward, up) of each camera, world axes
FACES = (((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)),
         ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (-1, 0, 0)), ((0, 0, -1), (1, 0, 0)))
FACE_FOV_DEG = 96.0  # over 90 so that lookups near a face's edge stay inside it
_FORWARD = np.array([f for f, _ in FACES], float)
_UP = np.array([u for _, u in FACES], float)
_RIGHT = np.cross(_FORWARD, _UP)


def add_eye_rig(root) -> None:
    """Add the cube-map camera rig (a mocap body) to an MJCF model."""
    rig = root.worldbody.add("body", name=RIG, mocap="true")
    for k in range(len(FACES)):
        rig.add("camera", name=f"{RIG}_{k}", fovy=FACE_FOV_DEG,
                xyaxes=np.concatenate([_RIGHT[k], _UP[k]]))


def _write_png(img: np.ndarray, path: Path, upsample: int) -> None:
    import imageio.v3 as iio
    # MuJoCo puts the image's first row at texture v = 0, as Corridor's row 0
    g = np.clip(np.kron(img, np.ones((upsample, upsample))), 0, 1)
    rgb = (np.repeat(g[..., None], 3, axis=2) * 255).round().astype(np.uint8)
    iio.imwrite(path, rgb)


class CorridorArena(composer.Arena):
    """The flyloop Corridor as textured quads (cm): floor at z = 0, walls at
    y = +-W/2, ceiling at z = H; textures repeat every corridor.length. The ends
    (n_periods texture periods away) are closed by caps of the mean luminance,
    which is what the analytic corridor shows at its vanishing point. Carries
    the eye rig of `MujocoWorld`."""

    def _build(self, corridor: Corridor, scale_cm: float, texture_dir: Path,
               n_periods: int = 2, upsample: int = 8, name: str = "corridor"):
        super()._build(name=name)
        W, H = corridor.width * scale_cm, corridor.height * scale_cm
        L = corridor.length * scale_cm
        self.centre_cm = np.array([0.0, 0.0, H / 2])
        self.scale_cm = scale_cm
        x0, x1 = -n_periods * L, n_periods * L
        u0 = -n_periods  # texture periods along x
        root = self.mjcf_model
        # the fly model brings its own headlight; avoid conflicting values
        for k in ("ambient", "diffuse", "specular"):
            setattr(root.visual.headlight, k, None)
        root.asset.add("texture", name="sky", type="skybox", builtin="gradient",
                       rgb1=(0.55, 0.65, 0.8), rgb2=(0.1, 0.1, 0.15), width=64, height=64)
        texture_dir = Path(texture_dir)
        texture_dir.mkdir(parents=True, exist_ok=True)
        quads = {  # surface: corners, geom group
            "left": ([[x0, W / 2, 0], [x0, W / 2, H], [x1, W / 2, H], [x1, W / 2, 0]],
                     ARENA_GROUP),
            "right": ([[x0, -W / 2, 0], [x1, -W / 2, 0], [x1, -W / 2, H], [x0, -W / 2, H]],
                      ARENA_GROUP),
            "floor": ([[x0, -W / 2, 0], [x0, W / 2, 0], [x1, W / 2, 0], [x1, -W / 2, 0]],
                      ARENA_GROUP),
            "ceiling": ([[x0, -W / 2, H], [x1, -W / 2, H], [x1, W / 2, H], [x0, W / 2, H]],
                        CEILING_GROUP),
        }
        two_sided = [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]
        self._ground_geom = None
        for surface, (corners, group) in quads.items():
            if not corridor.present[surface]:
                continue
            png = texture_dir / f"{surface}.png"
            _write_png(corridor.pyramids[surface][0], png, upsample)
            c = np.array(corners, float)
            # u along x (periods); v across the surface (0..1), as Corridor._sample
            v_axis = 2 if surface in ("left", "right") else 1
            span = H if v_axis == 2 else W
            off = 0.0 if v_axis == 2 else W / 2
            uv = np.stack([u0 + (c[:, 0] - x0) / L, (c[:, v_axis] + off) / span], 1)
            root.asset.add("texture", name=f"tex_{surface}", type="2d", file=str(png))
            root.asset.add("material", name=f"mat_{surface}", texture=f"tex_{surface}",
                           emission=0.35, specular=0.0, shininess=0.0)
            root.asset.add("mesh", name=f"mesh_{surface}", vertex=c.ravel(),
                           texcoord=uv.ravel(), inertia="shell", face=two_sided)
            geom = root.worldbody.add("geom", name=surface, type="mesh",
                                      mesh=f"mesh_{surface}", material=f"mat_{surface}",
                                      group=group, contype=0, conaffinity=0)
            if surface == "floor":
                self._ground_geom = geom
        # textured like the walls, so that the unlit render shows exactly mean_lum
        _write_png(np.full((2, 2), corridor.mean_lum), texture_dir / "end.png", 1)
        root.asset.add("texture", name="tex_end", type="2d", file=str(texture_dir / "end.png"))
        root.asset.add("material", name="mat_end", texture="tex_end", emission=0.35,
                       specular=0.0, shininess=0.0)
        for end, x in (("near", x0), ("far", x1)):
            c = np.array([[x, -W / 2, 0], [x, W / 2, 0], [x, W / 2, H], [x, -W / 2, H]])
            root.asset.add("mesh", name=f"mesh_end_{end}", vertex=c.ravel(), inertia="shell",
                           texcoord=[0, 0, 1, 0, 1, 1, 0, 1], face=two_sided)
            root.worldbody.add("geom", name=f"end_{end}", type="mesh", mesh=f"mesh_end_{end}",
                               material="mat_end", group=ARENA_GROUP, contype=0, conaffinity=0)
        add_eye_rig(root)

    @property
    def ground_geoms(self):
        return () if self._ground_geom is None else (self._ground_geom,)


class MujocoWorld:
    """Luminance along rays, rendered by MuJoCo (see module docstring).

    Args:
        model, data: MuJoCo model and data holding the scene and the eye rig
            (`add_eye_rig`). Rendering reads the current state of `data`.
        centre_cm, scale_cm: Where flyloop's origin sits in the scene, and the
            length of one flyloop unit (see module docstring).
        viewpoint: Optional function returning the eye position in cm (for
            example the fly's head). If given, the `origin` passed to
            `luminance_at` is ignored: the scene is seen from where the fly is.
        face_px: Pixels along each cube face.
        sample_spacing_deg: Angular spacing of the eye's samples (flyvis column
            spacing 5.8 deg / n_sub), for prefiltering.
        groups: Geom groups the eyes see. The default leaves out the fly's body.
        near_cm: Nearest distance rendered. MuJoCo's default near plane is far
            too distant for an eye that flies close to a wall.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, centre_cm=(0.0, 0.0, 0.0),
                 scale_cm: float = 1.0, viewpoint: Callable[[], np.ndarray] | None = None,
                 face_px: int = 128, sample_spacing_deg: float = 5.8 / 3,
                 groups: tuple[int, ...] = (ARENA_GROUP, CEILING_GROUP),
                 near_cm: float = 0.01):
        self.model, self.data = model, data
        self.centre_cm, self.scale_cm = np.asarray(centre_cm, float), float(scale_cm)
        self.viewpoint, self.face_px = viewpoint, int(face_px)
        self.spacing = np.radians(sample_spacing_deg)
        cam = mujoco.mjtObj.mjOBJ_CAMERA
        self._cams = [mujoco.mj_name2id(model, cam, f"{RIG}_{k}") for k in range(len(FACES))]
        rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIG)
        if rig < 0 or min(self._cams) < 0:
            raise ValueError("the model has no eye rig; build it with add_eye_rig")
        self._mocap = int(model.body_mocapid[rig])
        self._znear = near_cm / model.stat.extent  # vis.map.znear is relative to extent
        self._half_tan = float(np.tan(np.radians(model.cam_fovy[self._cams[0]]) / 2))
        self._option = mujoco.MjvOption()
        for g in range(mujoco.mjNGROUP):
            self._option.geomgroup[g] = int(g in groups)
        self._renderer = None
        self._physics = None  # keeps a standalone scene alive (from_corridor)
        self.faces = np.zeros((len(FACES), self.face_px, self.face_px), np.float32)
        self.eye_cm = self.centre_cm.copy()  # viewpoint of the last render

    @classmethod
    def from_corridor(cls, corridor: Corridor, scale_cm: float = 5.0,
                      texture_dir: Path | str = "mujoco_textures", **kwargs) -> MujocoWorld:
        """The corridor alone (no fly), seen from the origin passed to
        `luminance_at`."""
        arena = CorridorArena(corridor, scale_cm, Path(texture_dir))
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        world = cls(physics.model.ptr, physics.data.ptr, arena.centre_cm, scale_cm, **kwargs)
        world._physics = physics
        return world

    # --- rendering ----------------------------------------------------------------

    def render_faces(self, eye_cm) -> np.ndarray:
        """Unlit luminance of the six cube faces seen from eye_cm, (6, n, n) in
        0..1, rows top to bottom."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.face_px, self.face_px)
        self.eye_cm = np.asarray(eye_cm, float).copy()
        self.data.mocap_pos[self._mocap] = self.eye_cm
        # camera poses only; the simulation state is not touched
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        r = self._renderer
        znear = self.model.vis.map.znear
        self.model.vis.map.znear = self._znear
        try:
            for k, cam in enumerate(self._cams):
                r.update_scene(self.data, cam, scene_option=self._option)
                scene = r.scene
                scene.nlight = 0
                for i in range(scene.ngeom):
                    scene.geoms[i].emission = 1.0
                self.faces[k] = r.render().mean(-1) / 255.0
        finally:
            self.model.vis.map.znear = znear  # other views keep their near plane
        return self.faces

    def luminance_at(self, origin, dirs: np.ndarray) -> np.ndarray:
        """Luminance along rays from origin (3,), flyloop units, in world
        directions dirs (..., 3). With a viewpoint, origin is ignored."""
        if self.viewpoint is not None:
            eye_cm = self.viewpoint()
        else:
            eye_cm = self.centre_cm + np.asarray(origin, float) * self.scale_cm
        faces = self.render_faces(eye_cm)
        d = np.asarray(dirs, float)
        shape = d.shape[:-1]
        d = d.reshape(-1, 3)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        ahead = d @ _FORWARD.T  # (n, 6): cosine to each face's axis
        face = ahead.argmax(1)
        m = ahead[np.arange(len(d)), face]
        n, t = self.face_px, self._half_tan
        # continuous pixel coordinates (pixel i covers i..i+1), x right, y down
        x = (np.einsum("ij,ij->i", d, _RIGHT[face]) / m / t + 1) / 2 * n
        y = (1 - np.einsum("ij,ij->i", d, _UP[face]) / m / t) / 2 * n
        # a pixel spans about (2 t / n) m^1.5 rad; pick the level matching the samples
        pixel = 2 * t / n * m ** 1.5
        n_levels = int(np.log2(n)) - 2
        level = np.clip(np.log2(np.maximum(self.spacing / pixel, 1.0)), 0, n_levels - 1)
        lo = np.floor(level).astype(int)
        hi = np.minimum(lo + 1, n_levels - 1)
        f = level - lo
        out = np.zeros(len(d), np.float32)
        img = faces
        for lv in range(n_levels):
            w = np.where(lo == lv, 1 - f, 0) + np.where(hi == lv, f, 0)
            sel = w > 0
            if sel.any():
                s = 2 ** lv
                out[sel] += w[sel] * _bilinear(img, face[sel], x[sel] / s, y[sel] / s)
            if lv + 1 < n_levels:
                k = img.shape[-1] // 2
                img = img[:, :2 * k, :2 * k].reshape(len(FACES), k, 2, k, 2).mean((2, 4))
        return out.reshape(shape)

    def luminance(self, azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        """Seen from the origin (or the viewpoint)."""
        return self.luminance_at(np.zeros(3), directions(azimuth_deg, elevation_deg))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _bilinear(faces: np.ndarray, face: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup in faces (6, n, n) at continuous pixel coordinates, clamped
    to the face."""
    n = faces.shape[-1]
    x, y = np.clip(x - 0.5, 0, n - 1), np.clip(y - 0.5, 0, n - 1)
    c0, r0 = np.floor(x).astype(int), np.floor(y).astype(int)
    c1, r1 = np.minimum(c0 + 1, n - 1), np.minimum(r0 + 1, n - 1)
    fc, fr = x - c0, y - r0
    top = faces[face, r0, c0] * (1 - fc) + faces[face, r0, c1] * fc
    bot = faces[face, r1, c0] * (1 - fc) + faces[face, r1, c1] * fc
    return top * (1 - fr) + bot * fr
