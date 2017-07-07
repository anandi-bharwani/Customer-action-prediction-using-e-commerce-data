I # Customer-action-prediction-using-e-commerce-data

The ecommerce data-set contains the following information about 500 customers: 
1. Is visitor visiting site on mobile  or not? 
2. No. of products the consumer has viewed 
3. For how long a user has visited the site 
4. Is the user a returning or new user 
5. Time of day if visit. Time is divided into 4 catogories 

Goal is to create model which predicts the consumer action from the below 4 option: 
1. bounce, i.e, customer simply returns withot buying 
2. add to cart, i.e, cutomer adds products to cart but doesnt buy 
3. Customer begins to check-out but for some reason doesnt finish the transaction 
4. Customer finishes the checkout and buys the product. 

This has been solved in two ways:- 
1. Logistic Regression with softmax in logistic_softmax_train.py
2. Neural Network with one hidden layer in ann_train.py
