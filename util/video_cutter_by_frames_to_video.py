import cv2

# define input and output files
input_file = "/media/dudy/FB31-3461/Downtest_ch0007_00010000262000000.mp4"
output_file = "shabak_short_Downtest_ch0007_00010000262000000.mp4"

# define start and end frames
start_frame = 0
end_frame = 800

# initialize video reader and writer objects
cap = cv2.VideoCapture(input_file)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# loop over frames and write selected frames to output
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count >= start_frame and frame_count <= end_frame:
        out.write(frame)
    if frame_count > end_frame:
        break
    frame_count += 1

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()
