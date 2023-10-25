# Week 7

- [ ] Refine abstract and get feedback

- [X] Consolidate structure of Literature Review

- [X] Initiate structure on Methodologies 




Methodologies Blueprint :

 1. Data Acquisition
 2. Data Preprocessing
 3. Feature Engineering
 4. Model Building
 5. Model Evaluation
    5.1 Bias-Variance Trade-off
 6. Model Optimisation
 7. Model Validation and Deployment
----------------------------------------------
 1. Data Acquisition : 
   - Talk about where we obtain data from 
           a. Human responses
           b. Code samples from the web
           c. Sample text from past dissertations

2. Data Preprocessing :
   - Talk about steps taken to convert the data into a more readable format
   - In this project dataset is built from scratch
   - Data to be manually labelled(output)

3. Feature Engineering: 
   - Based on Data Extracted, new features(i.e. all of them) are built
   - Transformations, if necessary will be applied on the relevant variables

4. Model Building:
   - Apply the base model to entire dataset at first to establish a baseline model.

5. Model Evaluation:
   - Use proper evaluation metrics.
       - For classification, precision/recall , ROC_AUC score better.
    - Check bias and variance error to see extent of overfitting/underfitting. Talk about bias-variance trade-off and hence
    optimal model complexity

6. Model Optimisation: 
   - Perform hyperparameter tuning
   - Log each retuned model using the tool MLFlow.
   - Apply ensemble techniques(bagging/boosting) to reduce bias/variance error to the point where optimal model complexity is obtained.

7. Model Validation:
   - Test the model on unseen data and deploy it in the web app(streamlit).
