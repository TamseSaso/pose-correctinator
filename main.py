from pathlib import Path

import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    HRNetParser,
    GatherData,
    ImgDetectionsFilter,
)
from depthai_nodes.node.utils import generate_script_content

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode
import numpy as np

import socket

# Network alert configuration – update with your laptop’s IP and desired port
LAPTOP_IP = "192.168.178.37"  # replace with your laptop’s IP
LAPTOP_PORT = 5005
# UDP socket for sending posture alerts
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

DET_MODEL: str = "luxonis/yolov6-nano:r2-coco-512x288"
PADDING = 0.1

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if not args.fps_limit:
    args.fps_limit = 5 if platform == "RVC2" else 30
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # person detection model
    det_model_description = dai.NNModelDescription(DET_MODEL, platform=platform)
    det_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(det_model_description, useCached=True)
    )

    # pose estimation model
    rec_model_description = dai.NNModelDescription(args.model, platform=platform)
    rec_model_nn_archive = dai.NNArchive(
        dai.getModelFromZoo(rec_model_description, useCached=True)
    )

    # media/camera source
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(frame_type)
        replay.setLoop(True)
    else:
        cam = pipeline.create(dai.node.Camera).build()

        cam_video = cam.requestOutput(
            size=(720, 720), type=frame_type, fps=args.fps_limit
        )
        high_res_output = cam_video

    input_node = replay if args.media_path else cam

    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_model_nn_archive, fps=args.fps_limit
    )
    det_nn.input.setBlocking(False)
    det_nn.input.setMaxSize(1)

    # detection processing
    valid_labels = [
        det_model_nn_archive.getConfig().model.heads[0].metadata.classes.index("person")
    ]
    detections_filter = pipeline.create(ImgDetectionsFilter).build(
        det_nn.out, labels_to_keep=valid_labels
    )  # we only want to work with person detections

    script_node = pipeline.create(dai.node.Script)
    det_nn.out.link(script_node.inputs["det_in"])
    det_nn.passthrough.link(script_node.inputs["preview"])
    script_content = generate_script_content(
        resize_width=rec_model_nn_archive.getInputWidth(),
        resize_height=rec_model_nn_archive.getInputHeight(),
        padding=PADDING,
        valid_labels=valid_labels,
    )
    script_node.setScript(script_content)

    crop_node = pipeline.create(dai.node.ImageManip)
    crop_node.initialConfig.setOutputSize(
        rec_model_nn_archive.getInputWidth(), rec_model_nn_archive.getInputHeight()
    )
    crop_node.inputConfig.setWaitForMessage(True)

    script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
    script_node.outputs["manip_img"].link(crop_node.inputImage)

    rec_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out, rec_model_nn_archive
    )
    rec_nn.input.setBlocking(False)
    rec_nn.input.setMaxSize(1)
    parser: HRNetParser = rec_nn.getParser(0)
    parser.setScoreThreshold(
        0.0
    )  # to get all keypoints so we can draw skeleton. We will filter them later.

    # detections and recognitions sync
    gather_data_node = pipeline.create(GatherData).build(args.fps_limit)
    rec_nn.out.link(gather_data_node.input_data)
    detections_filter.out.link(gather_data_node.input_reference)

    # annotation
    skeleton_edges = (
        rec_model_nn_archive.getConfig()
        .model.heads[0]
        .metadata.extraParams["skeleton_edges"]
    )
    annotation_node = pipeline.create(AnnotationNode).build(
        gather_data_node.out,
        connection_pairs=skeleton_edges,
        valid_labels=valid_labels,
    )

    # visualization
    if high_res_output:
        visualizer.addTopic("Video", high_res_output)
    else:
        visualizer.addTopic("Video", det_nn.passthrough, "images")
    visualizer.addTopic("Detections", detections_filter.out, "images")
    visualizer.addTopic("Pose", annotation_node.out_pose_annotations, "images")

    print("Pipeline created.")

    # Host-side queue for pose data (v3 API: no explicit XLinkOut)
    pose_queue = gather_data_node.out.createOutputQueue(maxSize=1, blocking=False)
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    # Posture evaluation function: checks shoulder alignment angle
    def evaluate_posture(landmarks, angle_threshold=25.0, dy_threshold=0.055):
        left_shoulder = np.array(landmarks[5])
        right_shoulder = np.array(landmarks[6])
        # Compute horizontal (dx) and vertical (dy) offsets
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        # Check vertical shoulder alignment
        vertical_ok = abs(dy) < dy_threshold
        # Calculate angle of the shoulder line relative to horizontal
        angle = np.degrees(np.arctan2(dy, dx))
        # Debug info: print shoulder coordinates and computed angle
        print(f"Left shoulder: {left_shoulder}, Right shoulder: {right_shoulder}, Angle: {angle:.2f}°, Dy: {dy:.3f}")
        # Check horizontal alignment (near 0° or ±180°)
        horizontal_ok = abs(angle) < angle_threshold or abs(abs(angle) - 180) < angle_threshold
        return horizontal_ok and vertical_ok

    while pipeline.isRunning():
        # Try to get latest pose data
        pose_msg = pose_queue.tryGet()
        if pose_msg is not None and pose_msg.gathered:
            # extract first gathered Keypoints message
            kps_msg = pose_msg.gathered[0]
            # convert Keypoints to numpy array of shape (n_keypoints, 2)
            landmarks = np.array([[kp.x, kp.y] for kp in kps_msg.keypoints])
            if not evaluate_posture(landmarks):
                # Notify laptop of incorrect posture
                sock.sendto(b"bad_posture", (LAPTOP_IP, LAPTOP_PORT))
                print("Incorrect posture detected! Sent alert to laptop.")

        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
