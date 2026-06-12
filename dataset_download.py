import os
import json
from datasets import load_dataset
from PIL import Image
import shutil

# SRC_BLUR_ROOT = "/Users/anikashrivastava/Downloads/blur"

# DST_ROOT = "/Users/anikashrivastava/research/mae_mine"
# DST_IMAGENET_C = os.path.join(DST_ROOT, "Imagenet-C")

# SELECTED_CLASSES = {
#     'n02106662', 'n01855032', 'n02107312', 'n02105641',
#     'n02782093', 'n02788148', 'n02802426',
#     'n03788195', 'n04065272', 'n04273569'
# }

# os.makedirs(DST_IMAGENET_C, exist_ok=True)

# # Iterate over blur types (zoom_blur, gaussian_blur, ...)
# for blur_type in os.listdir(SRC_BLUR_ROOT):
#     blur_type_path = os.path.join(SRC_BLUR_ROOT, blur_type)
#     if not os.path.isdir(blur_type_path):
#         continue

#     print(f"Processing blur type: {blur_type}")

#     # Iterate over severity levels (1,2,3,4,5)
#     for severity in os.listdir(blur_type_path):
#         severity_path = os.path.join(blur_type_path, severity)
#         if not os.path.isdir(severity_path):
#             continue

#         print(f"  Severity: {severity}")

#         # Destination severity folder
#         dst_severity_path = os.path.join(
#             DST_IMAGENET_C, blur_type, severity
#         )
#         os.makedirs(dst_severity_path, exist_ok=True)

#         # Copy only selected classes
#         for cls in SELECTED_CLASSES:
#             src_cls_path = os.path.join(severity_path, cls)
#             dst_cls_path = os.path.join(dst_severity_path, cls)

#             if os.path.exists(src_cls_path):
#                 shutil.copytree(
#                     src_cls_path,
#                     dst_cls_path,
#                     dirs_exist_ok=True
#                 )
#             else:
#                 print(f"Missing class {cls} in {blur_type}/{severity}")

# print("\n Imagenet-C subset creation completed.")


SRC_NOISE_ROOT = "/Users/anikashrivastava/Downloads/noise"
DST_IMAGENET_C = "/Users/anikashrivastava/research/mae_mine/Imagenet-C"

SELECTED_CLASSES = {
    'n02106662', 'n01855032', 'n02107312', 'n02105641',
    'n02782093', 'n02788148', 'n02802426',
    'n03788195', 'n04065272', 'n04273569'
}

for corruption in os.listdir(SRC_NOISE_ROOT):
    corruption_path = os.path.join(SRC_NOISE_ROOT, corruption)
    if not os.path.isdir(corruption_path):
        continue

    print(f"Processing noise corruption: {corruption}")

    for severity in os.listdir(corruption_path):
        severity_path = os.path.join(corruption_path, severity)
        if not os.path.isdir(severity_path):
            continue

        print(f"  Severity: {severity}")

        # 🔥 FLAT: directly under Imagenet-C
        dst_severity_path = os.path.join(
            DST_IMAGENET_C, corruption, severity
        )
        os.makedirs(dst_severity_path, exist_ok=True)

        for cls in SELECTED_CLASSES:
            src_cls_path = os.path.join(severity_path, cls)
            dst_cls_path = os.path.join(dst_severity_path, cls)

            if os.path.exists(src_cls_path):
                shutil.copytree(
                    src_cls_path,
                    dst_cls_path,
                    dirs_exist_ok=True
                )
            else:
                print(f"    ⚠️ Missing {cls} in {corruption}/{severity}")

print("\n Noise corruptions copied directly under Imagenet-C/")
