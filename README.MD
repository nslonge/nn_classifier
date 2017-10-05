## nn_classifier

Neural-network classifier for movie reviews, built using pytorch. Uses a DAN that averages word embeddings for each review.

# Architecture

doc: [w_1, ..., w_n] ->
embedding layer (EL) -> mean(EL(w_i)) ->
hidden layer (HL) -> tanh(HL) ->
output layer (OL) -> log_softmax(OL)

Uses a negative log liklihood loss function.
 
# Usage

python src/main.py [args]


Adapted from a tutorial by Adam Yala, https://github.com/yala/pytorch-example
