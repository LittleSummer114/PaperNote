# PaperNote
a collection of my idea
When we obtain top-10 concepts which connected to an instance, we get a concept and a list of instances connected with it at the same time.   
In this section, we will introduce how to generate a list of candidate concepts for each word in a short text and corresponding representation for each concept. We first briefly introduce how to represent a concept in a knowledge base. When we obtain top-10 concepts which connected to an instance from Probase, we get a concept and a list of instances connected with it at the same time. Here we denote the concept vector as C= {<c1,w1 >,< c2,w2 >,...,<ck,wk>, where ci is a concept in Probase, and wi is a weight to represent the probability of ci as a co-occurrence neighbor of an instance. Similarily, we also define the instance vector as I = {<i1,w1>,<i2,w2>,...,<ik,wk>}, wehere ij is a instance in Probase, and wj is a weight to represent the probability of ij as a co-occurrence neighbor of a concept. Let's take 'apple' as an example, we can map it to a concept vector {<fruit,>,<>}. In the same way, for concept ¡®fruit¡¯, we can map it to an instance vector {}. Given the instance vector, we use the pre-trained word vectors obtained from Google word2vec to represent each instance and the concept embedding Vc ¡ÊRm is weighted average of embedding of each instance:
Vc= wivic /wi
3.2 Context representation
According to the previous step, we can get a list of candidate concepts and corresponding concept embedding for each word in a short text. As we have mentioned above, the meanings of a word varies cross contexts. What¡¯s more, the importance of each word in a short text is different. Therefore, in this section, we employ an attention-based neural network to dynamically obtain the representation of the text under the influence of each word in a short text. And then we will take text representation as the context for each word. First of all, Suppose the short text T is expressed as T=(x1,x2,¡­,xn), where xi denotes the ith word in the short text, and vi ¡ÊRm is an m-dimensional vector obtained from word2vec1 to represent xi.Then we introduce attention mechanism to give each word a weight ai, those words which are more representative/important in T will get higher weights. Finally, a sentence vector s will be formed based on weighted sum of representation of all words. Specifically, 
ai=exp(wi)/ k, 1-n exp(wi)   (5)
Wi=Wwvi+bw     (6)
S= 1-n, ai vi   (7)
Here, W¡ÊRmx1 is an intermediate matrix and b is the offset.

extra words that are more representative in T and they will get a higher weight ai, which represent their importance in T.

given a short text which is made up of a sequence of words, we obtain a vector from word2vec1 to represent each word 
