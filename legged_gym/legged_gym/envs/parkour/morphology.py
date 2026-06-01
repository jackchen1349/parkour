"""Morphology parameter management for Lite3-based quadruped.

Scaling factors xi = [xi0, xi1, xi2, xi3] for:
  xi0: front thigh (FL/FR THIGH link)
  xi1: front shank (FL/FR SHANK link)
  xi2: hind thigh  (HL/HR THIGH link)
  xi3: hind shank  (HL/HR SHANK link)

Modifies URDF by scaling link masses, inertias, and joint origins per Table I.
Also computes PD gain corrections via polynomial (paper Eq. 1).
"""

import torch
import numpy as np
from typing import Dict


class MorphologyManager:
    # Link name groups matching Lite3 URDF (1.urdf)
    FRONT_THIGH_LINKS = ['FL_THIGH', 'FR_THIGH']
    FRONT_SHANK_LINKS = ['FL_SHANK', 'FR_SHANK']
    HIND_THIGH_LINKS  = ['HL_THIGH', 'HR_THIGH']
    HIND_SHANK_LINKS  = ['HL_SHANK', 'HR_SHANK']

    # Joints needing PD correction (HipY + Knee = 8 joints)
    PD_CORRECTION_JOINTS = [
        'FL_HipY_joint', 'FR_HipY_joint', 'HL_HipY_joint', 'HR_HipY_joint',
        'FL_Knee_joint', 'FR_Knee_joint', 'HL_Knee_joint', 'HR_Knee_joint',
    ]

    # DOF order from Isaac Gym (alphabetical by default)
    # FL_HipX_joint, FL_HipY_joint, FL_Knee_joint,
    # FR_HipX_joint, FR_HipY_joint, FR_Knee_joint,
    # HL_HipX_joint, HL_HipY_joint, HL_Knee_joint,
    # HR_HipX_joint, HR_HipY_joint, HR_Knee_joint
    # PD correction applied to HipY [1,4,7,10] and Knee [2,5,8,11]
    PD_INDICES = [1, 2, 4, 5, 7, 8, 10, 11]

    def __init__(self, scaling_range=(0.6, 1.4), pd_coeffs=(0.0, 0.0, 1.0, 0.0)):
        self.scaling_range = scaling_range
        self.pd_coeffs = pd_coeffs
        self._cached_base_urdf = None

    def sample_morphologies(self, num_samples: int) -> torch.Tensor:
        lo, hi = self.scaling_range
        return torch.rand(num_samples, 4) * (hi - lo) + lo

    def get_default_morphology(self) -> torch.Tensor:
        return torch.ones(4)

    def build_urdf_string(self, xi: torch.Tensor) -> str:
        """Generate modified URDF with scaled parameters per Table I."""
        xi_list = xi.tolist() if isinstance(xi, torch.Tensor) else list(xi)

        link_params = {}
        for leg_links, xi_idx in [
            (self.FRONT_THIGH_LINKS, 0),
            (self.FRONT_SHANK_LINKS, 1),
            (self.HIND_THIGH_LINKS, 2),
            (self.HIND_SHANK_LINKS, 3),
        ]:
            s = xi_list[xi_idx]
            # URDF base values: thigh mass=0.86, cylinder length=0.20, radius=0.022
            #                   shank mass=0.153, cylinder length=0.21012, radius=0.015
            orig_mass = 0.86 if 'THIGH' in leg_links[0] else 0.153
            orig_len = 0.20 if "THIGH" in leg_links[0] else 0.21012
            b = 0.044 if "THIGH" in leg_links[0] else 0.030  # 2 * cylinder radius

            new_mass = orig_mass * s
            new_len = orig_len * s
            new_ixx = new_mass / 12.0 * (b**2 + new_len**2)
            new_iyy = new_mass / 12.0 * (b**2 + new_len**2)
            new_izz = new_mass * b**2 / 6.0

            for link_name in leg_links:
                link_params[link_name] = {
                    'mass': new_mass,
                    'ixx': new_ixx, 'iyy': new_iyy, 'izz': new_izz,
                    'length': new_len,
                    'scale': s,
                }

        # Joint origin scaling per Table I: z_knee * ξ_{0,2}, z_ankle * ξ_{1,3}
        # URDF: FL_Knee_joint origin z=-0.20, FL_Ankle origin z=-0.21012
        joint_origin_z = {}
        for prefix, xi_idx in [('FL', 0), ('FR', 0), ('HL', 2), ('HR', 2)]:
            joint_origin_z[f'{prefix}_Knee_joint'] = (-0.20) * xi_list[xi_idx]
        for prefix, xi_idx in [('FL', 1), ('FR', 1), ('HL', 3), ('HR', 3)]:
            joint_origin_z[f'{prefix}_Ankle'] = (-0.21012) * xi_list[xi_idx]

        urdf = self._load_base()
        urdf = self._apply_scaling(urdf, link_params, joint_origin_z)
        return urdf

    def compute_pd_corrections(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute PD correction for the 8 HipY+Knee joints.

        Polynomial: eta = a*xi^3 + b*xi^2 + c*xi + d   (paper Eq.1)
        Input:  xi can be (4,) or (N, 4)
        Returns: (12,) or (N, 12) — corrections for HipY/Knee at indices
                 [1,2,4,5,7,8,10,11]; HipX joints stay at 1.0.
        """
        a, b_coeff, c, d = self.pd_coeffs

        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        # (N, 4) → (N, 4) polynomial evaluation
        corrections_4 = a * xi**3 + b_coeff * xi**2 + c * xi + d

        # Build (N, 12) result — HipX=1.0, HipY/Knee from corrections_4
        N = xi.shape[0]
        result = torch.ones(N, 12, dtype=torch.float, device=xi.device)
        # DOF order: FL[HipX, HipY, Knee], FR[HipX, HipY, Knee],
        #             HL[HipX, HipY, Knee], HR[HipX, HipY, Knee]
        # Map xi[0]=front_thigh → FL_HipY[1]+FR_HipY[4]
        #     xi[1]=front_shank → FL_Knee[2]+FR_Knee[5]
        #     xi[2]=hind_thigh  → HL_HipY[7]+HR_HipY[10]
        #     xi[3]=hind_shank  → HL_Knee[8]+HR_Knee[11]
        result[:, 1] = corrections_4[:, 0]   # FL_HipY ← xi0
        result[:, 2] = corrections_4[:, 1]   # FL_Knee ← xi1
        result[:, 4] = corrections_4[:, 0]   # FR_HipY ← xi0
        result[:, 5] = corrections_4[:, 1]   # FR_Knee ← xi1
        result[:, 7] = corrections_4[:, 2]   # HL_HipY ← xi2
        result[:, 8] = corrections_4[:, 3]   # HL_Knee ← xi3
        result[:, 10] = corrections_4[:, 2]  # HR_HipY ← xi2
        result[:, 11] = corrections_4[:, 3]  # HR_Knee ← xi3

        if squeeze_out:
            result = result.squeeze(0)
        return result

    def _load_base(self) -> str:
        if self._cached_base_urdf is None:
            import os
            from legged_gym import LEGGED_GYM_ROOT_DIR
            path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'resources/robots/parkour_quadruped/urdf/parkour_quadruped.urdf'
            )
            with open(path, 'r') as f:
                self._cached_base_urdf = f.read()
        return self._cached_base_urdf

    def _apply_scaling(self, urdf: str, link_params: Dict, joint_z: Dict) -> str:
        """Apply Table I scaling to a URDF string.

        Per Table I:
          Inertial → Origin:  x=0, y=0, z = -l_i × ξ_i / 2
          Inertial → Mass:    m_i × ξ_i
          Inertial → Inertia: recalculated with scaled mass+length
          Visual → Origin:    x/y kept, z = -l_i × ξ_i / 2
          Visual → Geometry:  cylinder length = l_i × ξ_i
          Collision → Origin: x/y kept, z = -l_i × ξ_i / 2
          Collision → Box:    size z = l_i × ξ_i
          Knee/Ankle joint → Origin: z = z_orig × ξ_i
        """
        import re

        for link_name, params in link_params.items():
            s = params['scale']
            is_thigh = 'THIGH' in link_name
            orig_len = 0.20 if is_thigh else 0.21012
            new_len = orig_len * s

            link_pat = rf'(<link\s+name="{link_name}">.*?</link>)'
            m = re.search(link_pat, urdf, flags=re.DOTALL)
            if not m:
                continue
            link_block = m.group(1)
            orig_block = link_block

            # ---- Inertial: Mass ----
            link_block = re.sub(
                r'(<mass\s+value=")([^"]+)(")',
                rf'\g<1>{params["mass"]:.6f}\g<3>',
                link_block
            )
            # ---- Inertial: Inertia ----
            for axis in ('ixx', 'iyy', 'izz'):
                link_block = re.sub(
                    rf'(<inertia\s+{axis}=")([^"]+)(")',
                    rf'\g<1>{params[axis]:.6f}\g<3>',
                    link_block
                )

            # ---- Inertial Origin: x=0, y=0, z=-l_new/2 (Table I row 1) ----
            def set_inertial_origin(match):
                new_z = -new_len / 2.0
                return match.group(1) + f'0 0 {new_z:.6f}' + match.group(3)

            link_block = re.sub(
                r'(<inertial>\s*<origin\s+xyz=")([^"]+)(")',
                set_inertial_origin,
                link_block, flags=re.DOTALL
            )

            # ---- Non-inertial origins: scale z to center, keep x,y (Table I rows 4,6) ----
            # Match <origin> that is NOT inside <inertial>
            def scale_non_inertial_origin(match):
                xyz_val = match.group(2)
                parts = xyz_val.split()
                if len(parts) >= 2:
                    parts[2] = f'{-new_len / 2.0:.6f}'
                else:
                    # 1 or 2 components: pad with zeros then set z
                    while len(parts) < 3:
                        parts.append('0')
                    parts[2] = f'{-new_len / 2.0:.6f}'
                return match.group(1) + ' '.join(parts) + match.group(3)

            # Only match <origin> in <visual> or <collision> blocks (not inside <inertial>)
            link_block = re.sub(
                r'(<(?:visual|collision)>\s*<origin\s+xyz=")([^"]+)(")',
                scale_non_inertial_origin,
                link_block, flags=re.DOTALL
            )

            # ---- Visual: cylinder length (Table I row 5 visual geometry) ----
            link_block = re.sub(
                r'(<cylinder\s+length=")([^"]+)(")',
                lambda m, nl=new_len: m.group(1) + f'{nl:.6f}' + m.group(3),
                link_block
            )

            # ---- Collision: box size z (Table I row 7 collision geometry) ----
            def scale_box_size_z(match):
                size_val = match.group(2)
                parts = size_val.split()
                if len(parts) >= 3:
                    parts[2] = f'{new_len:.6f}'
                return match.group(1) + ' '.join(parts) + match.group(3)

            link_block = re.sub(
                r'(<box\s+size=")([^"]+)(")',
                scale_box_size_z,
                link_block
            )

            urdf = urdf.replace(orig_block, link_block)

        # ---- Joint origin z (Table I rows 8,9) ----
        for joint_name, new_z in joint_z.items():
            def make_replacer(jn, nz):
                def replacer(m):
                    xyz_val = m.group(2)
                    parts = xyz_val.split()
                    if len(parts) >= 3:
                        parts[2] = f'{nz:.6f}'
                    return m.group(1) + ' '.join(parts) + m.group(3)
                return replacer
            urdf = re.sub(
                rf'(<joint\s+name="{joint_name}".*?<origin\s+xyz=")([^"]+)(")',
                make_replacer(joint_name, new_z),
                urdf, flags=re.DOTALL
            )

        return urdf
