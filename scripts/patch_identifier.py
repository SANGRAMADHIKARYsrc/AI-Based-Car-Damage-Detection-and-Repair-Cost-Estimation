from pathlib import Path

path = Path("src/component/identifier.py")
text = path.read_text(encoding="utf-8")

old_text = """        # Check front vs rear for bumpers and lamps based on vertical/horizontal position
        if primary_comp in ["front_bumper", "rear_bumper"]:
            # If box is in lower middle, default to front_bumper unless specified
            pass"""

new_text = """        # Disambiguate front vs rear bumper using visual taillight cues or viewpoint
        if primary_comp in ["front_bumper", "rear_bumper"]:
            if viewpoint_hint == "rear":
                primary_comp = "rear_bumper"
            elif viewpoint_hint == "front":
                primary_comp = "front_bumper"
            elif image is not None:
                h, w = image.shape[:2]
                upper_h = max(1, int(yc * h))
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
                red2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
                red = (red1 | red2) > 0
                upper_pct = float(np.mean(red[:upper_h, :]) * 100)
                if upper_pct > 0.5:
                    primary_comp = "rear_bumper"
                else:
                    primary_comp = "front_bumper" """

text = text.replace("\r\n", "\n")
old_text = old_text.replace("\r\n", "\n")
new_text = new_text.replace("\r\n", "\n")

if old_text in text:
    text = text.replace(old_text, new_text)
    path.write_text(text, encoding="utf-8")
    print("SUCCESS")
else:
    print("NOT FOUND")
