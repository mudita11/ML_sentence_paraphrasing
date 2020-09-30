#import gpt_2_simple as gpt2
from gpt_2_simple import gpt_2
import os
import requests
import sys

flag_pretrain = sys.argv[2]
model_name = "124M" ##355M ##774M

# flag_pretrain to download the pre-trained embeddings
if flag_pretrain.lower() == "true":
    print("if",flag_pretrain, model_name)
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt_2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
else:
    print("else",flag_pretrain, model_name)
    file_name = "train_data.txt"
    if not os.path.isfile(file_name):
        print("No training file found. Training on shakespeare data.")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data = requests.get(url)
    
        with open(file_name, 'w') as f:
            f.write(data.text)
    
    # to fine-tune the model
    #sess = gpt_2.start_tf_sess()
    #gpt_2.finetune(sess, file_name, model_name=model_name, steps=10)   # steps is max number of training steps

sess = gpt_2.start_tf_sess()
gpt_2.load_gpt2(sess)

gpt_2.generate(sess, prefix = "# can you tell me how much you will charge for your service? # how much will i pay for it? # can you tell me the price? # what will you charge? # what is its cost? # i have a dispute but need to know how much it will cost before I instruct you? # i am buying a​ factory house​​ and want to know how much you charge? # i am selling my business and want to know how much you charge? # i have been given a ​contract how much will it cost? # will there be any additional fees to your fee? # what are the costs to your services if i am not entitled to legal aid? # is it free to amend a will? # how much do you charge for will amendments? # how much do you charge to prepare a visa? # what are the costs for your services if i am not entitled to legal aid? # what is the cost of making a will if i am not entitled to legal aid? # i need a cost estimate for legal proceedings. # how much will legal proceedings cost overall? # how much do you charge for a new lease of a commercial property? # how much will it cost to prepare a new will? # how much will it cost to get advice on a personal injury? # i need advice on a a dispute how much will it cost? # how much do you charge for equity release? # i need help with a construction dispute but need to know what you charge? # how much is an employment tribunal going to cost? # how much does it cost if i want to claim against​ my employer? # what is the cost of taking my employer to court? # i am involved in a dispute​ how much would this cost? # how much will my dispute claim cost me? # how much does it cost to make a will? # do you charge for amending a will? # how much is will? # is a ​visa expensive to make? # how much does your fee cost to me? # will it cost me in the end? # how much do you charge for damages? # how much do you charge for damage to the building? #")
## Question: what is the capital of Italy? Answer: Rome # Question: who is the president of America? Answer: Donald Trump. # Question: what is the largest country in the world? Answer:


