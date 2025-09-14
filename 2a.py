'''
Ha Quang Anh - A0258588E
Terri Tan Xin Li - A0257815W
Beverley Teo - A0256985H
'''

ddir = 'C:/Users/Quang Anh/OneDrive/NUS/Y2/S2/BT3102/Project/project_files' 
SMOOTHING_CONSTANT_LIST = [0.01, 0.1, 1, 10]
DELTA = SMOOTHING_CONSTANT_LIST[0]

def mle():
    # Read in data
    list_of_tags = open(f'{ddir}/twitter_tags.txt', encoding="utf8").readlines()
    train_data = open(f'{ddir}/twitter_train.txt', encoding="utf8").readlines()

    tag_frequency = {}
    unique_words = {}

    # Count frequency of tags & unique words
    for tag in list_of_tags:
        tag_frequency[tag.strip()] = 0
    
    for x in train_data:
        if len(x.strip()) == 0:
            continue
        word, tag = x.strip().split("\t")
        word = word.strip().lower() # lowercase all words for consistency
        if word not in unique_words:
            unique_words[word] = 1
        
        tag_frequency[tag] += 1
    
    UNIQUE_WORD_COUNT = len(unique_words.keys())


    # Create tag dictionary to count frequency of tags
    # Create word dictionary for later calculations of probability
    token_tag_proba = {}

    for line in train_data:
        if len(line.strip()) == 0:
            continue
        token, tag = line.strip().split("\t")
        token = token.strip().lower()
        tag = tag.strip()
        if (token, tag) not in token_tag_proba:
            token_tag_proba[(token, tag)] = 0
        token_tag_proba[(token, tag)] += 1

    # Calculate highest probability of word, tag pairs
    for pair, count in token_tag_proba.items():
        token = pair[0]
        tag = pair[1]
        current_prob = (count + DELTA) / (tag_frequency[tag] + DELTA * (UNIQUE_WORD_COUNT + 1))
        token_tag_proba[(token, tag)] = current_prob
 
    output = open(f'{ddir}/naive_output_probs.txt',
              'w',
              encoding="utf8")
    
    # Write to file
    for pair, proba in token_tag_proba.items():
        output.write(pair[0] + '\t' +
                     pair[1] + '\t' + 
                     str(proba) + '\n')

mle()