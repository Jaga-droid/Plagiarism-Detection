# Week 3

## Literature Review

1. Readability Formula

Citation : G. M. McClure, "Readability formulas: Useful or useless?," in IEEE Transactions on Professional Communication, vol. PC-30, no. 1, pp. 12-15, March 1987, doi: 10.1109/TPC.1987.6449109.

Link : https://ieeexplore.ieee.org/document/6449109

Gist : 

Reading grade level = 0.39 (words/sentence) + 11.8 (syllables/word) -15.59 


We determine readability of text using two aspects :

   1 . Sentence complexity ( Sentence length)
   2.  Vocabulary complexity ( number of syllables per word)


2. Capture the author's writing style 

Citation: Syed, B., Verma, G., Srinivasan, B. V., Natarajan, A., & Varma, V. (2019). Adapting Language Models for Non-Parallel Author-Stylized Rewriting arXiv:1909.09962

Link : https://arxiv.org/abs/1909.09962

Gist : 

. Pre-training a Transformer-based language model on a large dataset using the MLM(Masked Language Modelling) that acts as a base which is then cascaded into an encoder-decoder framework.

. Fine-tuning on author-specific corpus using denoising auto encoders loss to enable stylized rewriting.

. The idea is to use MLM to capture the style of the author and check it against their latest submission to see if it was human-generated or not.


3. Watermark : 

Citation :  John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. (2023) A Watermark for Large Language Models

Link : https://arxiv.org/pdf/2301.10226.pdf


Gist : 

. A watermark is a hidden pattern in text that is imperceptible to humans, while making the text algorithmically identifiable as synthetic.

. Describe a simple “hard” red list watermark that is easy to analyze and detect. 

. Apply language model to tokens to get Probability vector over the vocabulary and compute a hash of one token using which you partition the vocabulary into a "green list" and a "red list" of equal size. 

.The simplicity of this approach comes at thecost of poor generation quality on low entropy sequences.


4. Software Similarity : 

(a) Abstract Syntax Trees

Citation : Karnalim, Oscar & Simon,. (2020). Syntax Trees and Information Retrieval to Improve Code Similarity Detection. 48-55. 10.1145/3373165.3373171. 

Link : https://www.researchgate.net/publication/338785728_Syntax_Trees_and_Information_Retrieval_to_Improve_Code_Similarity_Detection

Gist : 

. Generate the syntax trees of program code files, extracts directly connected n-gram structure tokens from them, and performs the subsequent comparisons using an algorithm from information retrieval, cosine correlation in the vector space model. 

. Evaluation of the approach shows that consideration of the program structure  increases the recall and f-score  at the cost of execution time. 


(b) Graph based approach


Citation : Feng, Zhang & Li, Guofan & Liu, Cong & Song, Qian. (2020). Flowchart-Based Cross-Language Source Code Similarity Detection. Scientific Programming. 2020. 1-15. 10.1155/2020/8835310. 

Link : https://researchgate.net/publication/347732336_Flowchart-Based_Cross-Language_Source_Code_Similarity_Detection

Gist : 
 
. Uses Dependency graphs and Control flow graphs to determine similarities between 2 programs.
