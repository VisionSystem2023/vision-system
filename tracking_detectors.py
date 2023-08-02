import cv2
import numpy as np
from ultralytics import YOLO


new_orange_color_properties = [[np.array([3, 100, 80]), np.array([8, 255, 255])]]
new_yellow_color_properties = [[np.array([10, 100, 80]), np.array([25, 255, 255])]]

new_orange_jacket_color_properties = [[np.array([0, 100, 60]), np.array([10, 255, 255])], [np.array([175, 100, 60]), np.array([180, 255, 255])]]

pink_helmet_color_properties = [[np.array([160, 0, 150]), np.array([180, 60, 255])]]
yellow_and_orange_jacket_color_properties = [[np.array([3, 100, 0]), np.array([8, 255, 255])], [np.array([25, 100, 0]), np.array([35, 255, 255])]]


def color_finder(video_path, colors_to_detect_properties=new_orange_jacket_color_properties, minimum_radius_to_detect_object=25, show_result=False, show_mask=False, show_object=False, save_video=False):
	cap = cv2.VideoCapture(video_path)  # Capture the video

	ret, frame = cap.read()

	if save_video:
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Allow to save modified video
		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

	positions_pixel = []

	while True:
		ret, frame = cap.read()

		if not ret:  # When the video ends
			break

		image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(image, np.array([-1, -1, -1]), np.array([-1, -1, -1]))  # Create empty mask

		for color_properties in colors_to_detect_properties:  # To browse every color which has to be recognized
			lower_pixel = color_properties[0]
			higher_pixel = color_properties[1]

			mask = cv2.bitwise_or(mask, cv2.inRange(image, lower_pixel, higher_pixel))  # adding new color

		image = cv2.blur(image, (1, 1))
		# mask = cv2.erode(mask, None, iterations=3)
		# mask = cv2.dilate(mask, None, iterations=3)
		image2 = cv2.bitwise_and(frame, frame, mask=mask)

		elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		if len(elements) > 0:
			c = max(elements, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)

			if radius > minimum_radius_to_detect_object:
				positions_pixel.append([int(x), int(y)])
				cv2.circle(image2, (int(x), int(y)), int(radius), (0, 255, 255), 2)
				cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), 10)
				cv2.line(frame, (int(x), int(y)), (int(x) + 150, int(y)), (0, 255, 255), 2)
				cv2.putText(frame, "jacket", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
				cv2.putText(frame, "YES", (800, 500), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 255, 0), 3, cv2.LINE_AA)
			else:
				positions_pixel.append([None, None])
				cv2.putText(frame, "NO", (800, 500), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 3, cv2.LINE_AA)
		else:
			positions_pixel.append([None, None])
			cv2.putText(frame, "NO", (800, 500), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 3, cv2.LINE_AA)

		if show_result:
			cv2.imshow('Video', frame)
		if show_object:
			cv2.imshow('image2', image2)
		if show_mask:
			cv2.imshow('Mask', mask)

		if save_video:
			out.write(frame)



		if cv2.waitKey(1) & 0xFF == 27:  # If you press escape
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

	return positions_pixel


def yolo_body_tracker(video_path='videos/Georges.mp4', model_to_use="yolov8n.pt"):
	model = YOLO(model_to_use)
	results = model.track(source=video_path, save=True, show=False, save_txt=True, conf=0.2, classes=[0, 14, 15, 16, 17, 18, 19, 20, 21, 77])

	ID_list = [r.boxes.id for r in results]
	operator_IDs = determines_operator_IDs(ID_list)


	# print(results)
	# print(results[3].boxes.id)
	# print(results[3].boxes.xyxy)
	# print(results[5].boxes.id)
	# print(results[5].boxes.xyxy)


# yolo_body_tracker()


def find_ind_max(counter_list):
	ind_max = 0
	maxi = 0
	for ind, count in enumerate(counter_list):
		if count >= maxi:
			maxi = count
			ind_max = ind

	if maxi != 0:
		return ind_max
	else:
		return -1


def IDs_to_ban(ID_list, ID_operator):
	to_ban=[]
	for IDs in ID_list:
		if ID_operator in IDs:
			for ID in IDs:
				if ID != ID_operator and ID not in to_ban:
					to_ban.append(ID)
	return to_ban


def determines_operator_IDs(ID_list):
	counter_list = [0]*100
	for IDs in ID_list:
		for ID in IDs:
			counter_list[ID-1] += 1

	operator_IDs = []

	ind_max = find_ind_max(counter_list)
	while ind_max != -1:
		counter_list[ind_max] = 0
		operator_IDs.append(ind_max + 1)
		to_ban = IDs_to_ban(ID_list, ind_max+1)
		for ID in to_ban:
			counter_list[ID-1] = 0
		ind_max = find_ind_max(counter_list)

	return operator_IDs

# IDS=[[1,2],[1],[1,2],[1,3],[1,2,3],[2,3,4],[5],[6,7],[1,3]]
#
#
# print(determines_operator_IDs(IDS))