import cv2
import time
import imutils
import winsound  # For Windows beep
from datetime import datetime

# -----------------------------
# Initial Setup
# -----------------------------
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend to avoid MSMF errors
time.sleep(1)
firstFrame = None
area = 1500  

print("[INFO] Press 'q' to quit")
print("[INFO] Press '+' to increase sensitivity (detect smaller motion)")
print("[INFO] Press '-' to decrease sensitivity (ignore small motion)")

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame. Exiting...")
        break

    text = "Normal"
    img = imutils.resize(img, width=1000)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    # Auto-reset first frame every ~10 seconds to adapt to lighting
    if firstFrame is None or time.time() % 10 < 0.1:
        firstFrame = gaussianImg
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 30, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

        # Beep Sound Alert
        winsound.Beep(1000, 200)

        # Save Snapshot
        filename = f"motion_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        print(f"[INFO] Motion detected! Snapshot saved: {filename}")

    # Add Timestamp Overlay
    timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(img, timestamp, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display Status
    cv2.putText(img, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Motion Detection Camera", img)

    # Handle Keyboard Inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] Quitting...")
        break
    elif key == ord('+'):
        area = max(500, area - 500)
        print(f"[INFO] Increased sensitivity. New min area: {area}")
    elif key == ord('-'):
        area += 500
        print(f"[INFO] Decreased sensitivity. New min area: {area}")

cam.rele
