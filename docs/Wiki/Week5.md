# Week 5

- [] Get more training examples.
- [] Refine the StyleLM approach.
- [X] Explore detecting hyperbole in text as a possible indicator of plagiarism.

## Notes 

Draft 1 of Project Plan

Various approaches taken across each week to be considered as individual features.

  - Flesh-Kincaid_formula result to be taken as 1 feature
  - Code_present_in_text(binary) and Code_similarity result as two features 
  - Perplexity difference between original text and writer's text as one feature
  - Overall text similarity(percentage) as one feature
  - hyperbole in text as one feature

We build a dataset using these features with sufficient records. After preprocessing, we then feed this through machine learning models
to classify the degree of plagiarism each record falls into (low, moderate, high).


Structure of Final Product : 

 - A web app that allows a user to 
                   1. upload a pdf via the click of a button
                   2. paste text into a text box(along with source link)

 - After accepting the input, either the pdf text extraction script is run (OR) text is directly converted into above features which is then passed on to the model(pickle file uploaded). 

 - Display the output to the user ( along with summary statistics if possible. (EDA on output text.))