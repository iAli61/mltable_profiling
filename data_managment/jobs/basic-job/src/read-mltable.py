# ./src/read-mltable.py
import argparse
import mltable

from PIL import Image
from io import BytesIO
import mlflow
import matplotlib.image as mpimg

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='mltable artifact to read')
args = parser.parse_args()

# load mltable
tbl = mltable.load(args.input)

# show table
print(tbl.show())

df = tbl.to_pandas_dataframe()
# print(df['image_url'][1].__Dict__)
print(df['image_url'][1])

stream = df['image_url'][15]
# pritn all attributes of stream
print(dir(stream))

# Assuming 'stream' is your StreamInfo object
stream_info_file_obj = stream.open()

# Read the contents of the file and store them in a bytes-like object
bytes_data = stream_info_file_obj.read()

# Create a BytesIO object from the bytes data
imgby = BytesIO(bytes_data)
img = Image.open(imgby)

# img = mpimg.imread(.path())
# # img = mpimg.imread(df['image_url'].iloc(0).open())
mlflow.log_image(img, f"figure.png")

# int = 1
# with df.image_url[int].open() as f:

    
#     # convert stream to tiff
#     img = Image.open(f)

#     # show image
#     print(df.label[int])
#     # save image
#     mlflow.log_image(img, f"figure-{int}.png")