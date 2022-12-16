from PIL import Image
import numpy as np

orig = Image.open("dd/00100001.png")
new = orig.convert("RGB")

r, g, b = new.split()
r = Image.fromarray(np.array(r)*0.2126)
g = Image.fromarray(np.array(g)*0.7152)
b = Image.fromarray(np.array(b)*0.0722)
new = Image.merge("RGB", (b,g,r))

# [r, g, b] = new.getpixel((x, y))
# r = int(r * 0.21)
# g = int(g * 0.71)
# b = int(b * 0.07)
# value = (r, g, b)
# new.putpixel((x, y), value)
new = new.save("ddd.png")
print("All work!")