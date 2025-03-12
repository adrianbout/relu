import cv2
import os
import time

# Create a directory to save captured images if it doesn't exist
save_dir = "captured_images/usb"
os.makedirs(save_dir, exist_ok=True)

# Open the USB camera (usually device index 1)
cap_usb = cv2.VideoCapture(1)

if not cap_usb.isOpened():
    print("Error: Cannot open USB camera.")
    exit()

print("Capturing images from the USB camera every 2 seconds. Press 'q' to quit.")

while True:
    ret, frame = cap_usb.read()
    if not ret:
        print("Failed to capture frame from USB camera. Exiting...")
        break

    # Display the frame from the USB camera
    cv2.imshow("USB Camera", frame)

    # Create a unique filename using the current timestamp
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    usb_filename = os.path.join(save_dir, f"usb_img_{timestamp}.png")
    cv2.imwrite(usb_filename, frame)
    print(f"Image saved: {usb_filename}")

    # Wait for 2 seconds, also check if 'q' is pressed during this time
    start_time = time.time()
    while time.time() - start_time < 2:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap_usb.release()
            cv2.destroyAllWindows()
            exit()

# Release the USB camera and close any open windows
cap_usb.release()
cv2.destroyAllWindows()
