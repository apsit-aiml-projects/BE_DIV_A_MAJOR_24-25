from ultralytics import YOLO 

model = YOLO('yolov8x')

model.predict('player1_hit_frames/frame_16.jpg', save=True)

# result = model.track('input_videos/input_video.mp4', save=True)
# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)export KMP_DUPLICATE_LIB_OK=TRUE
