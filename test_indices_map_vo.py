import sys
import pickle

from utils_m2_factorize import get_indices_mapping_2_wvn_vo
from utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)

if __name__ == "__main__":
    

    bond_length     = sys.argv[1]
    filename        = f'h2o_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'

    with open(filename, 'rb') as f:
        (
        list_CSF, # CSF reference states
        list_list_ia_CSF, # W_ia indices for each CSF
        list_list_theta_CSF,
        list_sym_CSF_vec,
        list_UCSF_tz,
        tz_states, # W V |HF>
        somos_list,
        psi_GS_UCSF_smik,
        list_orb_rot,
        x_orbrot,
        Enuc,
        obt_spatial,
        tbt_spatial
        ) = pickle.load(f)
    if len(list_orb_rot) != 0:
      obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
    else:
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)

    Nqubits = obt.shape[0]
    Norb    = Nqubits // 2

    k = 7

    print(f"CSF: {list_CSF[k]}")
    print(f"W_ia indices: {list_list_ia_CSF[k]}")
    print(f"TZ state: {tz_states[k]}")

    mapping_dict = get_indices_mapping_2_wvn_vo(list_CSF[k], list_list_ia_CSF[k], Norb=Norb)

    print(f"Mapping Dict: {mapping_dict}")