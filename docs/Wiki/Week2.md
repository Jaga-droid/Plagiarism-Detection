# Week 2

Tasks
- [x] Meet supervisor
- [ ] Project motivation


----

# Project Planning and Kickoff

Objective of Project: To detect potential instances of plagiarism in a document and provide insight

Goals: 
   1. Capture cases common tools like Turnitin would miss
   2. Attempt to compare 2 programs for their similarity to assess the level of collusion

Presenting Outcomes: 
  1. Plagiarism level detected( low, medium, high) displayed on the web application
  2. Present a report to the user to help understand segments of the document suspected to have been plagiarized.

Evaluation Metrics: 
  1. Model Accuracy
  2. Model Precision and Recall
  3. Perplexity
  4. Burstiness


## Turnitin : 

* Uses text matching to compare submitted papers against a vast database of academic and web sources. 
* Similarity **score** indicates the percentage of text that matches other sources.
* If a string of 3 words is matched with its repository it gets flagged as plagiarism. Word match length can be modified.


### Drawbacks of Turnitin : 
     
* Largely based on string matching algorithms which can be unreliable without considering the context in which the words were used. This can lead to false positives

* Not the best at detecting text written by tools such as conch.ai, meaning there are also false negatives.

### Programming Language Used : 
 
* Python

### Tools Required : 

* Streamlit( To deploy a shareable website for the AI detection tool)
