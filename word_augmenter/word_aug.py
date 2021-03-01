import xlsxwriter
import os
import nlpaug
import nlpaug.augmenter.word as naw
from itertools import product

def model_for_aug_type(aug_type):
    if aug_type == "random_aug":
        database_choice = None
    if aug_type == "synonym_replacement":
        database_choice = [
                            ('wordnet', 'wordnet')
                        ]
    if aug_type == "word_emb":
        database_choice = [
                            #('word2vec', 'word2vec', './GoogleNews-vectors-negative300'),
                            #('glove50d', 'glove', './glove.6B/glove.6B.50d.txt'),
                            #('glove100d', 'glove', './glove.6B/glove.6B.100d.txt'),
                            #('glove200d', 'glove', './glove.6B/glove.6B.200d.txt'),
                            #('glove300d', 'glove', './glove.6B/glove.6B.300d.txt'),
                            ('fasttext', 'fasttext', './wiki-news-300d-1M.vec')
                        ]
    if aug_type == "cont_word_emb":
        database_choice = [
                            ('be-un', 'bert-base-uncased'),
                            ('be-ba-ca', 'bert-base-cased'),
                            ('di-ba-un', 'distilbert-base-uncased'),
                            ('ro-ba', 'roberta-base'),
                            ('dr-ba', 'distilroberta-base'),
                            ('xl-ba-ca', 'xlnet-base-cased')
                        ]
    return database_choice
    
    
class DataAugmenter():
    def __init__(self, aug_type, stopwords, target_words, n_words):
        self.aug_type = aug_type
        self.database_choice = model_for_aug_type(self.aug_type)
        self.action_choices = ["substitute"]#, "insert"]
        self.action_choices_random_aug = ['substitute', 'swap', 'delete'] # 'crop' can only generate one sample # ‘substitute’ uses target_words argument
        self.aug_p_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.randomness_factor = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #temperature
        self.stopwords = stopwords
        self.target_words = target_words
        self.n_words = n_words
        self.workbook = xlsxwriter.Workbook('./'+aug_type+'.xlsx')
        
    def generate_data(self):
        method_name = 'fn_'+str(self.aug_type)
        call_method = getattr(self, method_name, lambda: "Invalid entry. Valid entries: random_aug, synonym_replacement, word_emb, cont_word_emb")
        call_method()
        
    def fn_random_aug(self):
        product_choices = list(product(self.action_choices_random_aug, self.aug_p_choices))
        for item in product_choices:
            aug = naw.RandomWordAug(name='RandomWord_Aug', action=item[0], aug_p=item[1], stopwords=self.stopwords, target_words=self.target_words)
            print("\nmodelname-action-words augmented: {}-{}-{}\n".format("random Augmenter", item[0], item[1]))
            augmented_text = aug.augment(text, n=self.n_words)
            print(augmented_text)
            
            self.write_excel(augmented_text, None, item[0], item[1])
        self.workbook.close()
                
    def fn_synonym_replacement(self):
        product_choices = list(product(self.database_choice, self.aug_p_choices))
        for item in product_choices:
            aug = naw.SynonymAug(aug_src=item[0][1], aug_p=item[1], stopwords=self.stopwords)
            print("\nmodelname-action-words augmented: {}-{}-{}\n".format(item[0][0], "substitute", item[1]))
            augmented_text = aug.augment(text, n=self.n_words)
            print(augmented_text, "\n")
            
            self.write_excel(augmented_text, item, item[1])
        self.workbook.close()
        
    def fn_word_emb(self):
        product_choices = list(product(self.database_choice, self.action_choices, self.aug_p_choices))
        for item in product_choices:
            aug = naw.WordEmbsAug(model_type=item[0][1], model_path=item[0][2], action=item[1], aug_p=item[2], stopwords=self.stopwords)
            print("\nmodelname-action-words augmented: {}-{}-{}\n".format(item[0][0], item[1], item[2]))
            augmented_text = aug.augment(text, n=self.n_words)
            print(augmented_text, "\n")
                    
            self.write_excel(augmented_text, item, item[1], item[2])
        self.workbook.close()
        
    def fn_cont_word_emb(self):
        product_choices = list(product(self.database_choice, self.action_choices, self.aug_p_choices, self.randomness_factor))
        for item in product_choices:
            print("item: ",item); print(item[0])
            aug = naw.ContextualWordEmbsAug(model_path=item[0][1], action=item[1], aug_p=item[2], temperature = item[3], stopwords=self.stopwords)
            print("\nmodelname-action-words augmented-randomness: {}-{}-{}-{}".format(item[0][0], item[1], item[2], item[3]))
            augmented_text = aug.augment(text, n=self.n_words)
            print(augmented_text, "\n")
        
            self.write_excel(augmented_text, item, item[1], item[2], item[3])
        self.workbook.close()
    
    def write_excel(self, augmented_text, item, choice1, choice2=None, choice3=None):
        if self.aug_type == "random_aug":
            worksheet_name = str(choice1)[:3]+'-'+str(choice2)
        else:
            worksheet_name = str(item[0][0])+'-'+str(choice1)[:3]+'-'+str(choice2)+'-'+str(choice3)
        worksheet = self.workbook.add_worksheet(worksheet_name)
        row = 0
        col = 0
        for sent in augmented_text:
            worksheet.write(row, col, sent)
            row += 1


text = 'What are the charges to your services?'
stopwords = None
target_words = None
n_words = 2

print("Original: {}\n".format(text))

aug_type = ["random_aug","synonym_replacement","word_emb","cont_word_emb"]

da = DataAugmenter(aug_type[0], stopwords, target_words, n_words)
da.generate_data()
            

