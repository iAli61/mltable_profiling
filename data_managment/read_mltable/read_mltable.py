import mltable
from mltable import DataType
# paths = [
#     # {"pattern": "wasbs://data@adls129873.blob.core.windows.net/pet-images/**/*.jpg"},
#     {"pattern": "data-science/experiment/bosch/data_managment/data/fridgeObjects/**/*.jpg"}
# ]
# tbl = mltable.from_paths(paths)
# tbl = tbl.extract_columns_from_partition_format("{account_url}/{container}/{folder}/{label}")
# tbl = tbl.keep_columns(["Path", "label"])

mltable_path = "../MLtableBuilder/mltable/"
# read mltable from folder
tbl = mltable.from_paths

# read the mltable from the jsonl file
# tbl = mltable.from_json_lines_files([{'file': "../MLtableBuilder/mltable/validation_annotations_absoult_path.jsonl"}], include_path_column=True)
tbl = mltable.from_json_lines_files([{'file': "azureml://subscriptions/f804f2da-c27b-45ac-bf80-16d4d331776d/resourcegroups/rg-mlopsv2clas-505prod/workspaces/mlw-mlopsv2clas-505prod/datastores/workspaceblobstore/paths/LocalUpload/96d7fe024cfc0ecfd6bbf9f9a6ac9edc/mltable/validation_annotations_absoult_path.jsonl"}], include_path_column=True)
# tbl = mltable.from_json_lines_files([{'file': "../MLtableBuilder/mltable/validation_annotations.jsonl"}], include_path_column=True)
# tbl = mltable.from_json_lines_files([{'file': "../data/training-mltable-folder/train_annotations.jsonl"}])
tbl = tbl.convert_column_types({"image_url": DataType.to_stream()})

df = tbl.to_pandas_dataframe()
print(df.head())

# open example image

from PIL import Image
import matplotlib.pyplot as plt

int = 16
print(df.iloc[int])
print(df.image_url[int])

with df.image_url[int].open() as f:

    
    # convert stream to tiff
    img = Image.open(f)

    # show image
    print(df.label[int])
    plt.imshow(img)
    # save image
    img.save(f"test-{int}.jpg")

