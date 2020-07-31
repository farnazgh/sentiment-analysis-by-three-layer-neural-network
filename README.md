# sentiment-analysis-by-three-layer-neural-network

Data:<br>
movie reviews <br>
https://github.com/abromberg/sentiment_analysis_python/tree/master/polarityData <br>
polarity of 1-gram words <br>
http://sentistrength.wlv.ac.uk/documentation/language_changes.html

Three layer network : <br>
• input layer : four nodes (the total polarity of positive words, the total polarity of neg words, the total number of pos words, the total number of neg words)<br>
• hidden layer : the number of nodes in this layer regarding to the efficiency of network can be changed <br>
• output layer : one node (which would be a number between 0 and 1)

Results:<br>
learning rate = 0.01 , num of hidden layer neurons = 3 --> accuracy on the training data = 90% , on the test dataset = 88%
