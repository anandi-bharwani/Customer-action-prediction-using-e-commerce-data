# Customer-action-prediction-using-e-commerce-data

The ecommerce data-set contains the following information about 500 customers: 
1. Is visitor visiting site on mobile  or not? 
2. No. of products the consumer has viewed 
3. For how long a user has visited the site 
4. Is the user a returning or new user 
5. Time of day of visit.

The goal of the project is to create model which predicts the consumer action and place them in of the 4 categories:- 
1. Bounce: Customer simply exits without buying. 
2. Add to cart: Cutomer adds products to cart but doesnt buy them.
3. Check-out: Customer begins to check-out but doesnt finish the transaction 
4. Customer finishes the checkout and buys the product. 

This has been solved in two ways:- 
1. Logistic Regression with softmax in logistic_softmax_train.py
2. Neural Network with one hidden layer in ann_train.py
