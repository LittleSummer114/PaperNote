# PaperNote
a collection of my idea
Section 3.1 2ConCNN
Section 3.2 Candidate Generation
Section 3.3 Context Representation
Section 4 Experiment
In this section, we conduct experiment to verify the effectiveness of our proposed model. First, we compare our model with state-of-art end-to-end NN-based models to 
Demonstrate the effectiveness of the proposed method. What¡¯s more, in order to know the influence of the different components of our model, we further discuss the impacts of the various parts in the next experiment. Our experiment are conducted on four widely used datasets:MR,TREC,AG,Subj. we use accuracy as our metric to evaluate the effectiveness of our model followed by \cite{}.
The statistics of each dataset are listed in table 3.
Table 3 : statistics of the dataset
Datasets #class Trainning/Test Set Avg.Len Max.Len
MR:The Movie Review (MR) dataset contains 10662 sentences, each sentence is a positive or negative comment on movies. For this dataset, we randomly split 80% as the training set the remaining 20% as test set. In the process, we keep the category distribution balanced as \cite{} do.
TREC:This dataset is a question dataset, which classify sentences into 6 question types (whether the question is about person, location, numeric information, ect.) 
AG:This dataset is a news dataset, which include title and descriptions of AG¡¯s corpus of news, we only use the titles of each news in our experiment followed by .
Subj:This dataset contains 10000 sentences, each sentence is being classified into subjective and objective.

The details of our hyper parameters setting is shown in Table \cite{}. We use the word2vec to initialize the embedding of each word, and for unknown word, we will randomly initialize its embedding.

Section 4.2 Results
The experiment results on all the dataset are shown in Table \cite{}. According to the table \cite{}, we can see that our model significantly outperforms state of the art methods. Even without concept embedding, the basic CNN model can achieve good performance for the reason that the convolutional layer of CNN can capture like n-gram features from the text and transfer them to the next layer of CNN. What¡¯s more, the max-pooling layer of CNN adopted in our experiment can extra the most representative feature from the n-gram like features. In other words, the convolutional neural network can extract the most important feature from the text and thus it can represent the text effectively. And our concept embedding can bring richer semantic information into word representation. Hence, CNN can capture 
