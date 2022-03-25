### WandB suggests that 3 epochs and a learning rate of 2e-05 is optimal for the slant model
### Train such a model
# import necessary packages
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score

# check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda=True
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    usecuda=False

# load in congressional speeches with politician ideology labels
print("Reading in congressional speeches")
cong_ideology_training = pd.read_csv("../raw-data/congressional_speeches_ideology.csv", encoding="ISO-8859-1")
# keep only speeches from the 111th to the 114th congresses (overlaps with sample period)
cong_ideology_training = cong_ideology_training[cong_ideology_training["congress"] >= 111]
# convert text to lower case
cong_ideology_training["speech"] = cong_ideology_training["speech"].apply(lambda x: x.lower())

# shuffle the data into 80% training and 20% test
print("Shuffling into training and test data")
cong_ideology_training = cong_ideology_training[["speech", "nominate_dim1"]]
cong_ideology_training = cong_ideology_training.rename(columns={"speech":"text", "nominate_dim1":"labels"})
cong_fine_tuning_data = shuffle(cong_ideology_training, random_state=42)
# normalize the ideology score (speeds up training)
cong_fine_tuning_data["labels"] = minmax_scale(cong_fine_tuning_data["labels"])
train_data,test_data = train_test_split(
    cong_fine_tuning_data,
    test_size=0.2,
    shuffle=False
    )

# distilBERT model arguments
model_args = ClassificationArgs(
    num_train_epochs=3,
    learning_rate = 2e-05,
    output_dir="../models/slant/",
    best_model_dir="../models/slant/best-model/",
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    evaluate_during_training_steps=-1,
    save_steps=-1,
    save_model_every_epoch=True,
    overwrite_output_dir=True,
    max_seq_length=512,
    sliding_window=True,
    train_batch_size=24,
    eval_batch_size=256,
    regression=True,
    do_lower_case=True,
    wandb_project="bert-slant"
    #use_early_stopping=True
    )

# create the model
model = ClassificationModel(
    "distilbert",
    "../models/pretraining/media-pretraining-bert/",
    args=model_args,
    use_cuda=usecuda,
    num_labels=1
)

print("Training the model")

# train the model
model.train_model(
train_data,
eval_df=test_data,
rsq=lambda truth, predictions: r2_score(
    truth, predictions
),
)

# evaluate the model
print("Evaluating model")
model.eval_model(test_data)
to_predict = list(test_data["text"])
preds = model.predict(to_predict)[1]
preds = [x[0][0] for x in preds]
test_data["predicted"] = preds
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
print("-----------------------------------")
print("DW1 Nominate Results: ")
print("Correlation: " + str(pearsonr(test_data["labels"], test_data["predicted"])))
print("R-Squared: " + str(r2_score(test_data["labels"], test_data["predicted"])))
eval_txt = open("../models/slant/slant_eval_results.txt", "a")
eval_txt.write("r:" + str(pearsonr(test_data["labels"], test_data["predicted"])))
eval_txt.write("r2:" + str(r2_score(test_data["labels"], test_data["predicted"])))
eval_txt.close()
print("Done!")
