from neuralnets import RecurrentNet
import numpy as np

def to_one_hot (ind, arrlen):
    arr = np.zeros(arrlen)
    arr[ind] = 1
    return arr

test = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
test_len = len(test)

for i in range(8):
    test += test

letters = list(test)
unique_letters = set(letters)
num_unique = len(unique_letters)

letters_to_int = {}
int_to_letters = {}

for i, l in enumerate(unique_letters):
    letters_to_int[l] = i
    int_to_letters[i] = l
training_inputs = []
expected_outputs = []
training_set = []
for currletter, nextletter in zip(test[:-1], test[1:]):
    training_inputs.append(to_one_hot(letters_to_int[currletter], num_unique))
    expected_outputs.append(to_one_hot(letters_to_int[nextletter], num_unique))

    training_set.append((to_one_hot(letters_to_int[currletter], num_unique),
                         to_one_hot(letters_to_int[nextletter], num_unique)))

rnn = RecurrentNet(num_unique)
rnn.add("recurr", 40)
rnn.add("recurr", 30)
rnn.add("soft", num_unique)
print(rnn.feed_forward(training_inputs[0]))
rnn.forget_past()
rnn.stochastic_gradient_descent(epochs=100,
                                step_size=0.01,
                                mini_batch_size=test_len,
                                training_set=training_set)
print(rnn.feed_forward(training_inputs[0]))

