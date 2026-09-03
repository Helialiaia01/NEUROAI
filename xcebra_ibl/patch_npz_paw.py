import sys

def patch_npz_files(*args, **kwargs):
    print("patch_npz_paw.py is deprecated: 'paw' is removed from the xCEBRA pipeline.")
    print("No changes will be made. If you need paw data, handle it outside this pipeline.")
    return


if __name__ == "__main__":
    patch_npz_files()