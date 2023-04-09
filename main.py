# 视频转帧
import cv2
import os
# phosa需要的包
import argparse
import json
import logging
import os

import numpy as np
from PIL import Image

from phosa.bodymocap import get_bodymocap_predictor, process_mocap_predictions  # 人的重建
from phosa.constants import DEFAULT_LOSS_WEIGHTS, IMAGE_SIZE  # 物理限制
from phosa.global_opt import optimize_human_object, visualize_human_object  # 人、物优化函数
from phosa.pointrend import get_pointrend_predictor  # 物的分割？
from phosa.pose_optimization import find_optimal_poses
from phosa.utils import bbox_xy_to_wh

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s"
)


#  detectron2：分割
#  bodymocap：3D人体重建
#  neural_renderer：把3D投影到2D
def get_args():
    parser = argparse.ArgumentParser(description="Optimize object meshes w.r.t. human.")
    parser.add_argument(
        "--filename", default="input/000000038829.jpg", help="Path to image."
    )
    parser.add_argument("--output_dir", default="output", help="Output directory.")
    parser.add_argument("--class_name", default="bicycle", help="Name of class.")
    parser.add_argument("--mesh_index", type=int, default=0, help="Index of mesh ")
    parser.add_argument(
        "--lw_inter",
        type=float,
        default=None,
        help="Loss weight for coarse interaction loss. (None: default weight)",  # 交互的loss权重
    )
    parser.add_argument(
        "--lw_depth",
        type=float,
        default=None,
        help="Loss weight for ordinal depth loss. (None: default weight)",  # 深度loss
    )
    parser.add_argument(
        "--lw_inter_part",
        type=float,
        default=None,
        help="Loss weight for fine interaction loss. (None: default weight)",  # 细粒度的交互loss
    )
    parser.add_argument(
        "--lw_sil",
        type=float,
        default=None,
        help="Loss weight for mask loss. (None: default weight)",  # mask的loss
    )
    parser.add_argument(
        "--lw_collision",
        type=float,
        default=None,
        help="Loss weight for collision loss. (None: default weight)",  # 碰撞的loss
    )
    parser.add_argument(
        "--lw_scale",
        type=float,
        default=None,
        help="Loss weight for object scale loss. (None: default weight)",  # 物体缩放的loss
    )
    parser.add_argument(
        "--lw_scale_person",
        type=float,
        default=None,
        help="Loss weight for person scale loss. (None: default weight)",  # 人的缩放的loss
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="If added, saves computed metadata as filename.json.",
    )
    args = parser.parse_args()
    logger.info(f"Calling with args: {str(args)}")
    return args  # filename=input/000000038829.jpg、output_dir=output、class_name=bicycle、mesh_index=0、
    # lw_inter=none、lw_depth=none、lw_inter_part=none、lw_sil=none、lw_collision=none、lw_scale=none
    # lw_scale_person=none、save_metadata


def main(args):
    # weight好像只有某些特定的类 是不是不能随便找个视频试？
    loss_weights = DEFAULT_LOSS_WEIGHTS[args.class_name]  # 加载自行车的权重，字典
    # Update defaults based on commandline args.
    for loss_name in loss_weights.keys():
        loss_weight = getattr(args, loss_name)
        if hasattr(args, loss_name) and getattr(args, loss_name) is not None:
            loss_weights[loss_name] = loss_weight
            logger.info(f"Updated {loss_name} with {loss_weight}")

    # 准备输入
    video = cv2.VideoCapture("input/bicycle.mp4")
    video.read()
    video.read()
    video.read()
    video.read()
    video.read()
    video.read()
    video.read()
    video.read()
    success, frame = video.read()
    # frame 是N×3的ndarray的矩阵，rgb格式
    i = 0
    while success:
        out = 'output/' + str(i) + '.jpg'  # 这里写为jpg格式，下面不用考虑png了
        # 得到一帧之后马上处理 rather than得到每一帧后再处理 ：考虑到可视化的时候要实时看
        cv2.imwrite(out, frame)
        i = i + 1
        if i > 10:
            return
        success, frame = video.read()  # 前面已经存了，这里可以更新，放后面也行

        image = Image.open(out).convert("RGB")
        w, h = image.size
        r = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
        w = int(r * w)
        h = int(r * h)
        image = np.array(image.resize((w, h)))  # 输入是ndarray的图片

        segmenter = get_pointrend_predictor()  # 一个DefaultPredictor类的对象，用了models/model_final_3c3198.pkl
        instances = segmenter(image)["instances"]  # 得到人和物的各个分割实例？

        # Process Human Estimations. 处理2D人类分割
        is_person = instances.pred_classes == 0  # 应该是一个数组
        bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()  # 得到人的bounding box，应该是两个xy，N x 4形状
        masks_person = instances[is_person].pred_masks  # 得到人的mask ，N x H x W形状

        # 重建3D人
        human_predictor = get_bodymocap_predictor()  # 一个BodyMocap类的对象 使用了SMPL方法
        mocap_predictions = human_predictor.regress(
            image[..., ::-1], bbox_xy_to_wh(bboxes_person)  # 把bounding box转为(minX, minY, width, height)格式
        )  # 一个pred_output_list，元素是人的各种信息的字典，包括img_cropped，
        # pred_vertices_smpl，pred_vertices_img，pred_joints_img，pred_body_pose，pred_rotmat，pred_betas，pred_camera，
        # bbox_top_left，bbox_scale_ratio，faces，right_hand_joints_img_coord，left_hand_joints_img_coord

        # 人的参数 ，字典
        person_parameters = process_mocap_predictions(
            mocap_predictions=mocap_predictions, bboxes=bboxes_person, masks=masks_person
        )
        # 值都是torch.cuda.FloatTensor
        # bbox: Bounding boxes in xyxy format (N x 3).
        # cams: Weak perspective camera (N x 3). 是不是zxy？
        # masks: Bitmasks used for computing ordinal depth loss, cropped to image space (N x L x L).
        # local_cams: Weak perspective camera relative to the bounding boxes (N x 3).
        # faces和verts

        # 物体的参数，字典 主要耗时在这里,减少循环次数？
        object_parameters = find_optimal_poses(
            instances=instances, class_name=args.class_name, mesh_index=args.mesh_index
        )
        # rotations (N x 3 x 3): Top rotation matrices. 旋转矩阵
        # translations (N x 1 x 3): Top translations. 平移
        # target_masks (N x 256 x 256): Cropped occlusion-aware masks (for silhouette loss).
        # masks (N x 640 x 640): Object masks (for depth ordering loss).
        # K_roi (N x 3 x 3): Camera intrinsics corresponding to each object ROI crop.

        # 这个和上面差不多慢
        model = optimize_human_object(
            person_parameters=person_parameters,
            object_parameters=object_parameters,
            class_name=args.class_name,
            mesh_index=args.mesh_index,
            loss_weights=loss_weights,
        )  # PHOSA类的对象，用到了models/smpl_faces.npy
        frontal, top_down = visualize_human_object(model, image)  # 从正面 上面看

        os.makedirs(args.output_dir, exist_ok=True)
        file_name = os.path.basename(args.filename)
        ext = file_name[file_name.rfind("."):]
        # 正面
        frontal_path = 'output/' + str(i) + 'frontal.jpg'
        # frontal_path = os.path.join(args.output_dir, file_name)
        Image.fromarray(frontal).save(frontal_path)
        logger.info(f"Saved rendered image to {frontal_path}.")
        # 上面
        top_down_path = 'output/' + str(i) + 'top.jpg'
        # top_down_path = frontal_path.replace(ext, "_top" + ext)
        Image.fromarray(top_down).save(top_down_path)
        logger.info(f"Saved top-down image to {top_down_path}.")

        if args.save_metadata:
            json_path = frontal_path.replace(ext, ".json")
            metadata = model.get_parameters()
            with open(json_path, "w") as f:
                json.dump(metadata, json_path)
            logger.info(f"Saved metadata to {json_path}.")


if __name__ == "__main__":
    main(get_args())
