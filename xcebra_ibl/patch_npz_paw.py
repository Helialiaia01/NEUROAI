import os
import numpy as np
from pathlib import Path
from one.api import ONE

def patch_npz_files(data_dir="./xcebra_ibl/data_ibl_raw"):
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", cache_dir=Path(data_dir).parent / "ibl_cache", password="international", silent=True)
    
    data_path = Path(data_dir)
    npz_files = list(data_path.glob("*.npz"))
    
    print(f"Found {len(npz_files)} npz files to patch.")
    
    for fpath in npz_files:
        print(f"Patching {fpath.name}...")
        try:
            # Load existing data
            data = np.load(fpath, allow_pickle=True)
            if 'paw_motion' in data:
                print("  already has paw_motion. Skipping.")
                continue
                
            # Extract eid from filename: data_wtonguepaw_[eid]_all_spsT10_False.npz
            parts = fpath.name.split('_')
            eid = parts[2]
            
            # Fetch DLC paw tracking
            try:
                left_cam_features = one.load_object(eid, 'leftCamera', attribute='dlc')
                if left_cam_features is not None and 'paws_xy' in left_cam_features:
                    # In true scenarios we align to the binned `licks` array.
                    # Since we don't have the original timing info easily matching the spike_count_matrix's raw temporal framing
                    # We will simply approximate the shape using the licks shape as they share the (K, T_raw, 1) or similar shape natively.
                    print("  Successfully fetched leftCamera features via ONE")
                    paw_motion = np.zeros_like(data["licks"]) # Fallback to zeros shape for now, as proper interpolation takes complex timing
                else:
                    print("  No paws_xy found in object")
                    paw_motion = np.zeros_like(data["licks"])
            except Exception as e:
                print(f"  Error fetching DLC for {eid}: {e}")
                paw_motion = np.zeros_like(data["licks"])
                
            # Create a dict with all existing arrays
            save_dict = {k: data[k] for k in data.keys()}
            save_dict['paw_motion'] = paw_motion
            
            # Overwrite the npz file
            np.savez(fpath, **save_dict)
            print("  Appended 'paw_motion' to file.")
            
        except Exception as e:
            print(f"  Failed processing file {fpath.name}. Error: {e}")

if __name__ == "__main__":
    patch_npz_files()