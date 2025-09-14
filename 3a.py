'''
Ha Quang Anh - A0258588E
Terri Tan Xin Li - A0257815W
Beverley Teo - A0256985H
'''

ddir = 'C:/Users/Quang Anh/OneDrive/NUS/Y2/S2/BT3102/Project/project_files' 
SMOOTHING_CONSTANT_LIST = [0.01, 0.1, 1, 10]
DELTA = SMOOTHING_CONSTANT_LIST[0]

def transition():
    # Read in data
    list_of_tags = open(f'{ddir}/twitter_tags.txt', encoding="utf8").readlines()
    train_data = open(f'{ddir}/twitter_train.txt', encoding="utf8").readlines()

    tag_frequency = {}
    unique_words = {}
    transition_proba = {}

    # Count frequency of tags & unique words
    for tag in list_of_tags:
        tag_frequency[tag.strip()] = 0

    train_data_list = []
    
    for x in train_data:
        if len(x.strip()) == 0:
            continue
        word, tag = x.strip().split("\t")
        word = word.strip().lower() # lowercase all words for consistency
        train_data_list.append((word, tag))
        if word not in unique_words:
            unique_words[word] = 1

        tag_frequency[tag] += 1

    UNIQUE_WORD_COUNT = len(unique_words.keys())

    # Transition probabilities
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

    output = open(f'{ddir}/trans_probs.txt',
              'w',
              encoding="utf8")
    
    # Write to file
    for pair, proba in transition_proba.items():
        output.write(pair[0] + '\t' +
                     pair[1] + '\t' + 
                     str(proba) + '\n')

transition()


    
