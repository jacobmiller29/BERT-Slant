### Use the trained slant model for inference: get the predicted ideology scores
### given the text of cable news broadcasts
# load in necessary packages
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import dask.dataframe as dd

# check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda = True
else:
    print("We will use the CPU.")
    usecuda = False

pandarallel.initialize()

# load in the cable news data
print("Loading in media data")
media_data = dd.read_csv("../raw-data/cable_news_data.txt",
    sep=";",
    header=None,
    names=["id", "link", "segment", "start", "end", "text"],
    usecols=["id", "text", "start", "end"]
    )
media_data = media_data.compute()
# drop missings
media_data = media_data.dropna(subset=["text"])
# convert text column to strings
media_data["text"] = media_data["text"].parallel_apply(lambda x: str(x))

# convert pandas dataframe into numpy array to speed up inference
media_segments = np.array(media_data["text"])
# split numpy array into 8 chunks (full array too large for memory)
media_segments = np.array_split(media_segments, 8)

print("Loading in ideology model")

# set up model arguments for inference
model_args = ClassificationArgs(
    sliding_window=True,
    eval_batch_size=2048
    )
# load in the trained slant model
model = ClassificationModel(
    "distilbert",
    "../models/slant/best-model/",
    use_cuda=usecuda,
    num_labels=1,
    args=model_args
)

# run the model on the cable news broadcasts
print("Running the model on the media segments")

# data too large for memory, need to get the predicted scores in batches
slant_scores = []
for batch in media_segments:
    # run inference
    preds = model.predict(batch)[1]
    # if text is too long, get an average over the sliding window results
    preds = [np.mean(x) for x in preds]
    slant_scores.append(preds)
# convert array to list
slant_scores = [item for sublist in slant_scores for item in sublist]

# create pandas column from the list
media_data["slant"] = slant_scores
# drop the text column (saves disk space)
media_data = media_data.drop(columns=["text"], axis=0)

# save the results
print("Saving the results")
media_data.to_csv("../output-data/media-slant-scores.csv")
