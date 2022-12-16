from PIL import Image
import numpy as np

orig = Image.open("signatureExample(CEDAR 21)/original_21_1.png")
new = orig.convert("RGB")
r, g, b = new.split()

r = r.save("r.png")
g = g.save("g.png")
b = b.save("b.png")
print("All work!")