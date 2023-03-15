# Minecraft-Evaluation-model2

## all-test-data.zip
Please decompress 'all-test-data.zip' and put the folder under the same directory with 'parser_and_classifier_2.py'. 

The structure of <all-test-data> folder is as following: 

	/all-test-data 
	
		/trace
			(all worker trace json files here)
			(they start with 'trace_worker...')
	
		/survey
			(all manager survey json files here)
			(they start with 'surveyTrace_...')

## /models
Please put the 'models' folder under the same directory with 'parser_and_classifier_2.py'.

The 'models' folder includes 7 models:
* km_0.model to km_4.model is for kmeans clustering of feature[0] to feature[4];
* RF-origin.model is the random forest model trained by original data-label pairs;
* RF-1.model is the random forest model trained by optimized data-label pairs, the average accuracy in testing is usually 0.01-0.03 higher than RF-origin.model.
* RF-2.model: trained by 87 pilot data. max_depth=5, n_estimators=50; min_samples_leaf=4, max_features='log2'.

	
## parser_and_classifier_2.py
Predict by RF-1.model, generate json files.

To run the 'parser_and_classifier_2.py', use command:
	
	$ python your-path/parser_and_classifier_2.py <path to test data> <path to output json>
e.g. 
	$ python ./parser_and_classifier.py ./all-test-data ./results
	or:
	$ python ./parser_and_classifier.py ./all-test-data -

* \<path to test data\> is the father folder include trace and survey;
* \<path to output json\> is the father folder that 'features.json' and 'output.json' will be stored in. Use '-' if you want the output jsons under the same folder of test data.
	
	* 'output.json' includes only results predicted by RF-1.model
	* 'features.json' includes both prediction results of RF-1.model and input features that extracted from json trace.


## json2csv.py
Transfer from json to csv, aggregates ground truth.
To run the 'json2csv.py', use command:
	
	$ python ./json2csv.py <path to test data> <path to features.json> 
If there's only 1 parameter, <path to features.json> will be the same with <path to test data> (make sure where the features.json is)
e.g. 
	
	$ python ./json2csv.py ./all-test-data ./results
	or:
	$ python ./json2csv.py ./all-test-data

Then one file named 'fea_m_compare.csv' will be generated under <path to features.json>, which aggregates input features, prediction results of RF-1.model, and human manager survey (ground truth).


## train_RF2.py
Train a new Random Forests model (RF-2) by 20 features + results of RF-1.model
To run the 'train_RF2.py', use command:
	
	$ python ./train_RF2.py <path to fea_m_compare.csv>
e.g.
	
	$ python ./train_RF2.py ./results
'train_RF2.py' will generate a RF-2.model according to data in 'fea_m_compare.csv', and you can replace the current RF-2 model by the new generated model. Be aware that RF-2 models trained by the same data can be slightly different due to randomness setting. 

'train_RF2.py' will also generate a 'fea_m_ai2.csv' file that added columns of prediction results by RF-2.model

## test_RF2.py
Test RF-2 by 10-folds cross validation and shuffle-repeat 50 times.
To run the 'test_RF2.py', use command:
	
	$ python ./test_RF2.py <path to fea_m_compare.csv>
e.g.
	
	$ python ./test_RF2.py ./results
'test_RF2.py' will print test results and a png image of confusion matrix

	
## Others information:
* The formats of features are in the descriptions.pdf.
* 'RF_model2_test.ipynb' is the notebook of the training-testing process of RF-2
* 'RF_model2_sampling_test.ipynb' is the notebook of the training-testing process of RF-2 and do sampling after RF prediction.

The 'results' folder includes 3 csv files:
* 'eva_compare_all.csv': transfered from output.json, aggregates prediction results of RF-1.model, and human manager survey (ground truth).
* 'fea_m_compare.csv' transfered from features.json, aggregates input features, prediction results of RF-1.model, and human manager survey (ground truth). It is generated by data in all-test-data.zip.
* 'fea_m_all_ai2.csv' added columns of prediction results by one RF-2.model. If you re-generate RF-2.model, the corresponding 'fea_m_ai2.csv' might be different from this 'fea_m_all_ai2.csv'. So I use slightly different names to distinguish.
