import gdown

model_zoo = {"sparse_inst_r50vd_base": "1fjPFy35X2iJu3tYwVdAq4Bel82PfH5kx",
             "sparse_inst_r50_giam": "1pXU7Dsa1L7nUiLU9ULG2F6Pl5m5NEguL",
             "sparse_inst_r50_giam_soft": "1doterrG89SjmLxDyU8IhLYRGxVH69sR2",
             "sparse_inst_r50_giam_aug": "1MK8rO3qtA7vN9KVSBdp0VvZHCNq8-bvz",
             "sparse_inst_r50_dcn_giam_aug": "1qxdLRRHbIWEwRYn-NPPeCCk6fhBjc946",
             "sparse_inst_r50vd_giam_aug": "1dlamg7ych_BdWpPUCuiBXbwE0SXpsfGx",
             "sparse_inst_r50vd_dcn_giam_aug": "1clYPdCNrDNZLbmlAEJ7wjsrOLn1igOpT",
             "sparse_inst_r101_giam": "1EZZck-UNfom652iyDhdaGYbxS0MrO__z",
             "sparse_inst_r101_dcn_giam": "1shkFvyBmDlWRxl1ActD6VfZJTJYBGBjv",
             "sparse_inst_pvt_b1_giam": "13l9JgTz3sF6j3vSVHOOhAYJnCf-QuNe_",
             "sparse_inst_pvt_b2_li_giam": "1DFxQnFg_UL6kmMoNC4StUKo79RXVHyNF"}


def gdrive_download(file_id, dst_path):
    print("Downloading model")
    url = "https://drive.google.com/uc?id="
    gdown.download(url+file_id, dst_path, quiet=False)
    print("Model downloaded")


