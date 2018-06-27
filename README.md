# sentiment-analysis-by-three-layer-neural-network

Data:
movie reviews
https://github.com/abromberg/sentiment_analysis_python/tree/master/polarityData
polarity of 1-gram words
http://sentistrength.wlv.ac.uk/documentation/language_changes.html

Three layer network :
--input layer - four nodes (the total polarity of positive words, the total polarity of neg words, the total number of pos words, the total number of neg words)
--hidden layer - the number of nodes in this layer regarding to the efficiency of network can be changed
--output layer - one node (which would be a number between 0 and 1)

results:
learning rate = 0.01 , num of hidden layer neurons = 3 --> accuracy on the training data = 90% , on the test dataset = 88%
