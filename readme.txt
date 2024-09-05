
To run the random forest go to the folder containing randomForest.py and modifiedDecisionTree.py and run the follow command:
python3 randomForest.py


This will run the main program in randomForest.py, which will call on modifiedDecisionTree.py
What you should see printed in the terminal is the average measurements for each evaluation metric for each ntree value printed for each dataset.
First the wine dataset is run (x-y coords printed, 4 graphs pop up one after the other). Then the vote dataset (x-y coords printed, 4 graphs pop up one after the other).
Each graph should follow each other one by one; as you close one graph, the next should pop up. Each graph is titled so you know what it represents.


You may modify minimal size for split, minimal gain, and maximal depth on lines 248-250 in randomForest.py
You may modify beta on line in 315(Wine dataset) and 401(Vote dataset) randomForest.py 