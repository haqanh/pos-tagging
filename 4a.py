'''
Ha Quang Anh - A0258588E
Terri Tan Xin Li - A0257815W
Beverley Teo - A0256985H
'''

ddir = 'C:/Users/Quang Anh/OneDrive/NUS/Y2/S2/BT3102/Project/project_files' 
SMOOTHING_CONSTANT_LIST = [0.01, 0.1, 1, 10]
DELTA = SMOOTHING_CONSTANT_LIST[0]

'''
Improvements:
- Calculate the probability of transition from a prev_tag to all next_tag. For unseen transitions, use the remaining probability
(1 - sum of all seen transitions) / number of unseen transitions. (new_transition_proba)
- Calculate the probability of each tag being the starting tag. Each document (tweet) in the input data is separated by a blank line. So we can count the number of times each tag appears as the starting tag and divide by the total number of documents to get the probability of each tag being the starting tag. (start_proba)
'''

def new_transition_proba():

    # Read in data
    list_of_tags = open(f'{ddir}/twitter_tags.txt', encoding="utf8").readlines()
    train_data = open(f'{ddir}/twitter_train.txt', encoding="utf8").readlines()

    tag_frequency = {}
    unique_words = {}
    transition_proba = {}

    for tag in list_of_tags:
        tag_frequency[tag.strip()] = 0

    train_data_list = []
    
    # Count frequency of tags & unique words
    for x in train_data:
        if len(x.strip()) == 0:
            continue
        word, tag = x.strip().split("\t")
        word = word.strip().lower()
        train_data_list.append((word, tag))
        if word not in unique_words:
            unique_words[word] = 1
        
        tag_frequency[tag] += 1

    UNIQUE_WORD_COUNT = len(unique_words.keys())

    for i in range(1, len(train_data_list)):
        prev_tag = train_data_list[i-1][1]
        current_tag = train_data_list[i][1]
        if (prev_tag, current_tag) not in transition_proba:
            transition_proba[(prev_tag, current_tag)] = 0
        transition_proba[(prev_tag, current_tag)] += 1
    
    for pair, count in transition_proba.items():
        prev_tag = pair[0]
        current_tag = pair[1]
        transition_proba[pair] = (count + DELTA) / (tag_frequency[prev_tag] + DELTA * (UNIQUE_WORD_COUNT + 1))
    
    unseen_transitions = {} 

    # Calculate the probability of transition from a prev_tag to all next_tag. For unseen transitions, use the remaining probability
    # (1 - sum of all seen transitions) / number of unseen transitions

    for pair, proba in transition_proba.items():
        prev_tag = pair[0]
        next_tag = pair[1]
        prob = proba
        if prev_tag not in unseen_transitions:
            unseen_transitions[prev_tag] = [0, 0] # [num of unseen transitions, sum of seen transitions]
        unseen_transitions[prev_tag][0] += 1
        unseen_transitions[prev_tag][1] += prob
    
    for prev_tag, values in unseen_transitions.items():
        num_unseen_transitions = values[0]
        sum_seen_transitions = values[1]
        remaining_prob = 1 - sum_seen_transitions
        unseen_transitions[prev_tag] = remaining_prob / num_unseen_transitions
        
    ### Replacing unseen transitions with the calculated probability above

    for tag in list_of_tags:
        for tag2 in list_of_tags:
            tag = tag.strip()
            tag2 = tag2.strip()
            if (tag, tag2) not in transition_proba:
                transition_proba[(tag, tag2)] = unseen_transitions[tag]

    output = open(f'{ddir}/trans_probs2.txt',
              'w',
              encoding="utf8")
    
    for pair, proba in transition_proba.items():
        output.write(pair[0] + '\t' +
                     pair[1] + '\t' + 
                     str(proba) + '\n')


def start_proba():
    # Read in data
    list_of_tags = open(f'{ddir}/twitter_tags.txt', encoding="utf8").readlines()
    train_data = open(f'{ddir}/twitter_train.txt', encoding="utf8").readlines()

    tag_frequency = []
    start_proba = {}

    for tag in list_of_tags:
        tag_frequency.append(tag.strip())

    num_sentences = 0

    for i in range(1, len(train_data)-1):
        if len(train_data[i].strip()) == 0:
            start_word, start_tag = train_data[i+1].strip().split("\t")
            if start_tag not in start_proba:
                start_proba[start_tag] = 0
            start_proba[start_tag] += 1
            num_sentences += 1

    for tag, count in start_proba.items():
        start_proba[tag] = count / num_sentences

    output = open(f'{ddir}/start_proba.txt',
              'w',
              encoding="utf8")
    
    for tag, proba in start_proba.items():
        output.write(tag + '\t' + 
                     str(proba) + '\n')


new_transition_proba()
start_proba()



    

