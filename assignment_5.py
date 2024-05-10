# -*- coding: utf-8 -*-
"""Assignment_5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dMemcTOXtCi74wsfNa7aBlnnVxsGQiOd
"""

! pip install git+https://github.com/huggingface/transformers -q

from transformers import pipeline,AutoTokenizer, AutoModelForQuestionAnswering

ckpt = "alasdevcenter/az-question-answering"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(ckpt)

qa = pipeline("question-answering", model=model, tokenizer=tokenizer)
whisper=pipeline("summarization",model="Falconsai/text_summarization")
sentiment=pipeline("sentiment-analysis",model="arpanghoshal/EmoRoBERTa")

su=input("Enter the text to summarize: ")

summary=whisper(su)

summary



senti=str(summary)
output=sentiment(senti)

output

context=str(su)
question=input("Enter the question: ")

qa({"context": context, "question": question})
