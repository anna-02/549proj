# Model parameters
from collections import Counter
from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
import torch 
import numpy 
from functools import reduce


class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]

class NamedEntityPipeline():
    def __init__(self,device): 
        self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        self.model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        self.device= device
        self.nlp = pipeline('ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    
    def extract_elts(self,text:list[str], allowed_cats = ['PER',"ORG",'LOC']): 

        results = self.nlp(text)
        # print(results)
         
        result_counts = [Counter([(elt['entity_group'],elt['word']) for elt in res if elt['entity_group'] in allowed_cats]) for res  in results]

        return results, result_counts

class DocFeatsExtractor(): 
    def __init__(self,
                 kw_model_name = "ml6team/keyphrase-generation-keybart-inspec",
                 device = None 
                 ): 
        if device is None: 
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu '
        
        # print('Getting models.')
        self.kp = KeyphraseGenerationPipeline(model=kw_model_name,device=device)
        self.ner = NamedEntityPipeline(device=device)
    
    def get_keys(self,text:list[str] | str): 
        # preprocess
        one_elt = type(text) is str
        if one_elt: 
            text = [text]
        
        # Extract 
        ners = self.ner.extract_elts(text=text)

        keyphrases = self.kp(text,do_sample=True,temperature=0.1,max_new_tokens=700)

        # print(ners,keyphrases)
        # Postprocess
        results = []

        for i in range(len(keyphrases)): 
            entities = list(ners[1][i].keys())

            results.append(
                keyphrases[i] + [ent[1] for ent in entities ]
            )
        
        # Return only one element if only one was passed in 
        if one_elt: 
            return results[0]
        return results 

    def __call__(self,*args, **kwargs):
        return self.get_keys(*args,**kwargs)




##### Process text sample (from wikipedia)


if __name__ == "__main__":
    device = 'mps' if torch.backends.mps.is_available() else 'cpu '
    
    model_name = "ml6team/keyphrase-generation-keybart-inspec"
    print('Getting models.')
    # generator = KeyphraseGenerationPipeline(model=model_name,device=device)
    # ner_getter = NamedEntityPipeline(device=device)
    extractor = DocFeatsExtractor()
    print('Running inference.')
    # Inference
    text = ["""
Keyphrase extraction is a technique in text analysis where you extract the
important keyphrases from a document. It is named after inventor Key Phrase. Thanks to these keyphrases humans can
understand the content of a text very quickly and easily without reading it
completely. Keyphrase extraction was first done primarily by human annotators,
who read the text in detail and then wrote down the most important keyphrases.
The disadvantage is that if you work with a lot of documents, this process
can take a lot of time. """.replace("\n", " "),
"""
Here is where Artificial Intelligence comes in. Currently, classical machine
learning methods, that use statistical and linguistic features, are widely used
for the extraction process. Now with deep learning, it is possible to capture
the semantic meaning of a text even better than these classical methods.
Classical methods look at the frequency, occurrence and order of words
in the text, whereas these neural approaches can capture long-term
semantic dependencies and context of words in a text.
    """.replace("\n", " ")]

    # keyphrases = generator(text,do_sample=True,temperature=0.1,max_new_tokens=700)
    # ners = ner_getter.extract_elts(text=text)

    print(extractor.get_keys(text))

    # print(keyphrases)
    # print(ners[0])
    # print(ners[1])