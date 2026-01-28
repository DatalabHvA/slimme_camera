from picamera2 import Picamera2
import os
import sys
from datetime import datetime
picam2 = Picamera2()
now = datetime.now()
timer = datetime.now().hour
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if timer < 8 or timer >= 20:
	sys.exit()

formatted = now.strftime("%d%m%Y_%H%M%S")
folder = "Pictures"
image_path = os.path.join(folder, str(formatted) + ".jpeg")

config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
picam2.capture_file(image_path)
picam2.stop()
