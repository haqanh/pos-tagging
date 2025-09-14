'''
Ha Quang Anh - A0258588E
Terri Tan Xin Li - A0257815W
Beverley Teo - A0256985H
'''

import math 

SMOOTHING_CONSTANT_LIST = [0.01, 0.1, 1, 10]
DELTA = SMOOTHING_CONSTANT_LIST[0]

# Implement the four functions below
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):

    '''
    Using MLE: P(word|tag) = count(tag, word) / count(tag)
    '''
    # Read in data
    train_data = open(in_output_probs_filename, encoding="utf-8").readlines()
    # Format of train data: word \t tag
    test_data = open(in_test_filename, encoding="utf-8").readlines()
    # Format of test data: word

    unique_words = {}
    mle = {}

    # Handling unseen words: tag is the tag with the highest probability
    max_proba = 0
    unseen_word_tag = "" 

    for x in train_data:
        if len(x.strip()) == 0:
            continue
        word, tag, proba = x.strip().split("\t")
        word = word.strip().lower()
        if word not in unique_words:
            unique_words[word] = 1
        if word not in mle:
             mle[word] = [tag, float(proba)]
        else:
            if float(proba) > mle[word][1]:
                mle[word] = [tag, float(proba)]
        if float(proba) > max_proba:
            unseen_word_tag = tag
            max_proba = float(proba)

    # Write to file
    output = []
    for line in test_data:
        # If empty strings:
        if len(line.strip()) == 0:
            continue
        token = line.strip().lower()
        if token in mle:
            output.append(mle[token][0])
        else:
            output.append(unseen_word_tag) # Have to handle smoothing!
    
    final_output = open(out_prediction_filename, "w", encoding="utf-8")

    for tag in output:
        final_output.write(tag + "\n")



def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    '''
    P(y = j | x = w) = P(x = w | y = j) * P(y = j) / P(x = w)
    -> we compute P(y = j) and P(x = w). P(x = w | y = j) is already calculated in in_output_probs_filename.
    '''

    # Read in data
    naive_proba = open(in_output_probs_filename, encoding="utf-8").readlines()
    train_data = open(in_train_filename, encoding="utf-8").readlines()
    test_data = open(in_test_filename, encoding="utf-8").readlines()

    words_proba = {} # this contains P(x = w)
    tags_proba = {} # this contains P(y = j)
    token_count = 0
    word_tag_proba = {} # this contains P(x = w | y = j)

    for x in train_data:
        if len(x.strip()) == 0:
            continue
        word, tag = x.strip().split("\t")
        word = word.strip().lower()
        if word not in words_proba:
            words_proba[word] = 0
        if tag not in tags_proba:
            tags_proba[tag] = 0
        words_proba[word] += 1
        tags_proba[tag] += 1
        token_count += 1

    for word in words_proba:
        words_proba[word] /= token_count
    
    for tag in tags_proba:
        tags_proba[tag] /= token_count

    for x in naive_proba:
        if len(x.strip()) == 0:
            continue
        word, tag , prob = x.strip().split("\t")
        word = word.strip().lower()
        tag = tag.strip()
        prob = float(prob)
        if (word, tag) not in word_tag_proba:
            word_tag_proba[(word, tag)] = prob

    for item, prob in word_tag_proba.items():
        word = item[0]
        tag = item[1]

        word_tag_proba[item] = prob * tags_proba[tag] / words_proba[word]    

    mle = {}

    # Handling unseen words: tag is the tag with the highest probability
    max_proba = 0
    unseen_word_tag = ""

    for item, prob in word_tag_proba.items():
        word = item[0]
        tag = item[1]
        if word not in mle:
            mle[word] = [tag, prob]
        else:
            if prob > mle[word][1]:
                mle[word] = [tag, prob]
        if prob > max_proba:
            unseen_word_tag = tag
            max_proba = prob
    
    # Write to file
    output = []
    for line in test_data:
        # If empty strings:
        if len(line.strip()) == 0:
            continue
        token = line.strip().lower()
        if token in mle:
            output.append(mle[token][0])
        else:
            output.append(unseen_word_tag) # Have to handle smoothing!
    
    final_output = open(out_prediction_filename, "w", encoding="utf-8")

    for tag in output:
        final_output.write(tag + "\n")  



def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename): 
    
    # Read in data
    naive_proba = open(in_output_probs_filename, encoding="utf-8").readlines()
    tags_data = open(in_tags_filename, encoding="utf-8").readlines()
    transition_data =  open(in_trans_probs_filename, encoding="utf-8").readlines()
    test_data = open(in_test_filename, encoding="utf-8").readlines()

    '''
    Where to get each information:

    - States: tags_data
    - Initial state: P(y0) = probability of each tag (possible improvement = how to know 
        which tag is the "start"?)
    - Transition: transition_data
    - Emit: naive_proba
    - Observation: test_data
    '''

    '''
    Test words: test_token
    '''

    test_token = []

    for x in test_data:
        if len(x.strip()) == 0:
            continue
        test_token.append(x.strip().lower())


    '''
    Emission: emit_proba
    '''

    emit_proba = {}
    for x in naive_proba:
        if len(x.strip()) == 0:
            continue
        word, tag, proba = x.strip().split("\t")
        word = word.strip().lower()
        tag = tag.strip()
        proba = float(proba)
        emit_proba[(word, tag)] = proba

    '''
    Tag Probability: tag_proba
    '''
    tags = {}
    tag_index = {}
    index_to_tag = {}
    index = 0
    for x in tags_data:
        if len(x.strip()) == 0:
            continue
        if x.strip() not in tags:
            tags[x.strip()] = 1
            tag_index[x.strip()] = index
            index_to_tag[index] = x.strip()
            index += 1

    tag_count = len(tags.keys())

    tag_proba = {}

    for tag, count in tags.items():
        tag_proba[tag] = count / tag_count


    '''
    Transition: trans_proba
    '''
    trans_proba = {}

    for line in transition_data:
        if len(line.strip()) == 0:
            continue
        prev_tag, current_tag, proba = line.strip().split("\t")
        prev_tag = prev_tag.strip()
        current_tag = current_tag.strip()
        trans_proba[(prev_tag, current_tag)] = float(proba)
        proba = float(proba)
    

    for tag in tags:
        trans_proba[("START", tag)] = 1 / tag_count # Assume uniform distribution for starting tag 

    
    '''
    Viterbi
    '''

    SMALL_CONSTANT = 1e-10  # Define a small constant

    ### Implement the Viterbi algorithm. As I faced the issue of underflow during calculation with multiplication of probabilities, I decided to use log probabilities instead to avoid underflow.
    viterbi = []

    for i in range(len(test_token)):
        viterbi.append({})
        for tag in tags:
            viterbi[i][tag] = {'prev': None, 'prob': -math.inf}

    for tag in tags: # Initialize the first (starting) state
        viterbi[0][tag]['prob'] = math.log(trans_proba[('START', tag)]) + math.log(emit_proba.get((test_token[0].strip().lower(), tag), SMALL_CONSTANT))
        viterbi[0][tag]['prev'] = 'START' 

    for i in range(1, len(test_token)):
        for nexttag in tags:
            max_prob = -math.inf
            prev_tag_selected = None
            for prevtag in tags:
                prob = viterbi[i-1][prevtag]['prob'] + math.log(trans_proba.get((prevtag, nexttag), SMALL_CONSTANT)) + math.log(emit_proba.get((test_token[i].strip().lower(), nexttag), SMALL_CONSTANT))
                if prob > max_prob:
                    max_prob = prob
                    prev_tag_selected = prevtag
                viterbi[i][nexttag]['prob'] = max_prob
                viterbi[i][nexttag]['prev'] = prev_tag_selected

    # Backtrack to get the best path
    best_path = []
    max_prob = -math.inf
    prev_tag = None
    

    for tag in tags:
        if viterbi[-1][tag]['prob'] > max_prob:
            max_prob = viterbi[len(test_token) - 1][tag]['prob']
            best_path = [tag]
            prev_tag = tag

    for i in range(len(test_token) - 2, -1, -1):
        best_path.insert(0, viterbi[i + 1][prev_tag]['prev'])
        prev_tag = viterbi[i + 1][prev_tag]['prev']

    final_output = open(out_predictions_filename, "w", encoding="utf-8")

    for tag in best_path:
        final_output.write(tag + "\n") 


def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    
    '''
    Improvements to the emission probabilities: handle common patterns in data.
    '''
    def improved_emission(token, tag, emit_proba, SMALL_CONSTANT = 1e-10): 
        proba = emit_proba.get((token, tag), SMALL_CONSTANT)
        if ('@' in token): # if token is a username
            if tag == '@':
                proba = 1
        if ('ing' in token) or ("'t" in token): # if token is a verb (commonly ends with -ing or -nt (e.g. isn't, don't))
            if tag == 'V':
                proba = 1
        if ('http' in token) or ('www' in token): # if token is a URL
            if tag == 'U':
                proba = 1
        if ('#' in token): # if token is a hashtag
            if tag == '#':
                proba = 1
        return proba


    # Read in data
    naive_proba = open(in_output_probs_filename, encoding="utf-8").readlines()
    tags_data = open(in_tags_filename, encoding="utf-8").readlines()
    transition_data =  open(in_trans_probs_filename, encoding="utf-8").readlines()
    test_data = open(in_test_filename, encoding="utf-8").readlines()
    

    '''
    Improvements: start_proba (implemented in 4a.py)
    '''
    start_proba = open('C:/Users/Quang Anh/OneDrive/NUS/Y2/S2/BT3102/Project/project_files/start_proba.txt', encoding="utf-8").readlines()
    starting_tag = {}
    for x in start_proba:
        tag, proba = x.strip().split("\t")
        tag = tag.strip()
        proba = float(proba)
        starting_tag[tag] = proba
    
    '''
    Where to get each information:

    - States: tags_data
    - Initial state: P(y0) = probability of each tag (possible improvement = how to know 
        which tag is the "start"?)
    - Transition: transition_data
    - Emit: naive_proba
    - Observation: test_data
    '''

    '''
    Test words: test_token
    '''

    test_token = []

    for x in test_data:
        if len(x.strip()) == 0:
            continue
        test_token.append(x.strip().lower())


    '''
    Emission: emit_proba
    '''

    emit_proba = {}
    for x in naive_proba:
        if len(x.strip()) == 0:
            continue
        word, tag, proba = x.strip().split("\t")
        word = word.strip().lower()
        tag = tag.strip()
        proba = float(proba)
        emit_proba[(word, tag)] = proba

    '''
    Tag Probability: tag_proba
    '''
    tags = {}
    for x in tags_data:
        if len(x.strip()) == 0:
            continue
        if x.strip() not in tags:
            tags[x.strip()] = 1

    tag_count = len(tags.keys())

    tag_proba = {}

    for tag, count in tags.items():
        tag_proba[tag] = count / tag_count


    '''
    Transition: trans_proba
    '''
    trans_proba = {}

    for line in transition_data:
        if len(line.strip()) == 0:
            continue
        prev_tag, current_tag, proba = line.strip().split("\t")
        prev_tag = prev_tag.strip()
        current_tag = current_tag.strip()
        trans_proba[(prev_tag, current_tag)] = float(proba)

    for tag in tags:
        trans_proba[("START", tag)] = starting_tag.get(tag, 1e-10) # Use improved starting tag probability

    
    '''
    Viterbi
    '''

    ### Implement the Viterbi algorithm. As I faced the issue of underflow during calculation with multiplication of probabilities, I decided to use log probabilities instead to avoid underflow.

    SMALL_CONSTANT = 1e-10  # Define a small constant in context of log probabilities

    ### Implement the Viterbi algorithm
    viterbi = []

    for i in range(len(test_token)):
        viterbi.append({})
        for tag in tags:
            viterbi[i][tag] = {'prev': None, 'prob': -math.inf}

    for tag in tags: # Initialize the first (starting) state
        viterbi[0][tag]['prob'] = math.log(trans_proba[('START', tag)]) + math.log(improved_emission(test_token[0].strip().lower(), tag, emit_proba))
        viterbi[0][tag]['prev'] = 'START'

    for i in range(1, len(test_token)):
        for nexttag in tags:
            max_prob = -math.inf
            prev_tag_selected = None
            for prevtag in tags:
                prob = viterbi[i-1][prevtag]['prob'] + math.log(trans_proba.get((prevtag, nexttag), SMALL_CONSTANT)) + math.log(improved_emission(test_token[i].strip().lower(), nexttag, emit_proba))
                if prob > max_prob:
                    max_prob = prob
                    prev_tag_selected = prevtag
                viterbi[i][nexttag]['prob'] = max_prob
                viterbi[i][nexttag]['prev'] = prev_tag_selected

    # Backtrack to get the best path
    best_path = []
    max_prob = -math.inf
    prev_tag = None 


    for tag in tags:
        if viterbi[-1][tag]['prob'] > max_prob:
            max_prob = viterbi[len(test_token) - 1][tag]['prob']
            best_path = [tag]
            prev_tag = tag

    for i in range(len(test_token) - 2, -1, -1):
        best_path.insert(0, viterbi[i + 1][prev_tag]['prev'])
        prev_tag = viterbi[i + 1][prev_tag]['prev']

    final_output = open(out_predictions_filename, "w", encoding="utf-8")

    for tag in best_path:
        final_output.write(tag + "\n")

    

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    print(f"Predicted tags: {len(predicted_tags)}")
    print(f"Ground truth tags: {len(ground_truth_tags)}")
    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = 'C:/Users/Quang Anh/OneDrive/NUS/Y2/S2/BT3102/Project/project_files' 

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'
    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()