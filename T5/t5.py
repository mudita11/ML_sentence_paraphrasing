import zipfile, requests, io, os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

zip_file_url = "https://datascience-models-ramsri.s3.amazonaws.com/t5_paraphraser.zip"
folder_path = zip_file_url.split("/")[-1].replace(".zip", "")

# download pretrained embedding from S3 and unzip in the current folder
if not os.path.exists(folder_path):
  r = requests.get(zip_file_url)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall('./t5_paraphraser')
else:
  print("Folder available:", folder_path)

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.coda.manuel_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('./t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model = model.to(device)

sentence = "Which course should I take to get in data science?"
text = "paraphrase: " + sentence + " </s>"
max_len = 256
encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
beam_outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks, do_sample=True, max_length=256, top_k=120, top_p=0.98, early_stopping=True, num_return_sequences=10)
print("\nOriginal Question ::")
print(sentence, "\n")
print("Paraphrased Question :: ")
final_outputs = []
for beam_output in beam_outputs:
  sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  if sent.lower() != sentence.lower() and sent not in final_outputs:
    final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
  print("{}: {}".format(i, final_output))
