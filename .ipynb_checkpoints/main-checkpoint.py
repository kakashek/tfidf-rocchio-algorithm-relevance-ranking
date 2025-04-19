
# Define function to get synonyms
def get_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms


# Define function parse_docs to parse data collection
def parse_docs(folder_number, path, stop_ws):
    folder_number = str(folder_number)
    doc_list = []
    folder_name = "Dataset"+folder_number
    inputpath = os.path.join(path, folder_name)
    files = os.listdir(inputpath) 
    os.chdir(inputpath)
    for doc in files:
        if doc.endswith(".xml"):
            myfile = open(doc)
            start_end = False
            file_ = myfile.readlines()
            word_count = 0 
            document = {}
            term_list = {}
            doc_id = 0
            for line in file_: 
                line = line.strip()
                if(start_end == False):
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                doc_id = part.split("=")[1].split("\"")[1]
                                break  
                    if line.startswith("<title>"):
                        query_list = {}
                        line = line.strip()
                        line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))             
                    if line.startswith("<text>"):
                        start_end = True  
                elif line.startswith("</text>"):
                    break
                else:
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                    line = line.replace("\\s+", " ")
                    for term in line.split():
                        word_count += 1 
                        term = stem(term.lower())
                        if len(term) > 2 and term not in stop_words: 
                            try:
                                term_list[term] += 1
                            except KeyError:
                                term_list[term] = 1
            doc_list.append({doc_id: term_list})
            myfile.close()
    return doc_list
    

# Define function to parse query from Queries.txt
def parse_query(inputpath, stop_words):
    query_list = {}
    with open(os.path.join(inputpath, 'Queries.txt'), 'r') as f:       
        query_lines = f.readlines()
        query = {}
        query_num = 0
        for i in range(len(query_lines)):
            if query_lines[i].startswith('<num>'):
                query_num = query_lines[i].split(':')[1].strip(' R').strip('\n').strip(' ')
            if query_lines[i].startswith('<title>'):
                query_lines[i] = query_lines[i].replace('<title>', '')
                for term in query_lines[i].split():
                    term = term.lower().translate(str.maketrans('', '', string.punctuation))
                    if len(term) > 2 and term not in stop_words:
                        try:
                            query[stem(term)] += 1.5
                        except KeyError:
                            query[stem(term)] = 1.5
                        # Find synonyms for title query
                        synonyms = get_synonyms(term)
                        for synonym in synonyms:
                            if stem(synonym) not in query and '_' not in stem(synonym) and synonym not in stop_words:
                                query[stem(synonym)] = 0.5
            if query_lines[i].startswith('<desc>'):
                j = 1
                while query_lines[i+j].startswith('<narr>') != True:
                    for term in query_lines[i+j].split():
                        term = term.lower().translate(str.maketrans('', '', string.punctuation))
                        if len(term) > 2 and term not in stop_words:
                            try:
                                query[stem(term)] += 0.75
                            except KeyError:
                                query[stem(term)] = 0.75
                    j += 1
            if query_lines[i].startswith('</Query>'):
                query_list.update({query_num: query})
                query = {}
                query_num = 0
    return query_list


# Define model functions
def rocchhio_tfidf_model(folder_number, coll, query_list):
    
    # Calculate document frequency
    document_frequency = {}
    folder_number = str(folder_number)
    for doc in coll:
        for terms in doc.values():
            for term in terms:
                if term not in document_frequency:
                    document_frequency[term] = 1
                else:
                    document_frequency[term] += 1

                    
    # Calculate td*idf score for each term for each document
    document_list = {}
    for doc in coll:
        tfidf_list = {}
        tfidf_total = 0
        for terms in doc.values():
            for term, value in terms.items():
                tf = math.log(value, 10) + 1
                idf = math.log(len(coll) / document_frequency[term], 10)
                tfidf_value = tf * idf
                tfidf_list[term] = tfidf_value
                tfidf_total += tfidf_value ** 2
        for term in tfidf_list:
            tfidf_list[term] /= math.sqrt(tfidf_total)
        document_list[list(doc.keys())[0]] = tfidf_list

        
    # Retrieve query terms 
    try:
        query_terms = query_list[folder_number]
    except KeyError:
        print("Incorrect folder number")
  
    
    # Calculate initial document score for each document in the folder
    document_score = {}
    for doc_id, tfidf_list in document_list.items():
        score = 0
        for term in query_terms:
            if term in tfidf_list:
                term_score = tfidf_list[term] * query_terms[term]
                score += term_score
        document_score[doc_id] = score
    document_score_sorted = dict(sorted(document_score.items(), key=lambda x: x[1], reverse=True))

    
    # Append the top 10 highest ranking into a list of relevant and non-relevant documents with pseudo-relevance feedback
    relevant_docs = {}
    non_relevant_docs = {}
    i = 0
    for doc in document_score_sorted:
        if i < 10:
            relevant_docs[doc] = document_score_sorted[doc]
            i += 1
        else:
            non_relevant_docs[doc] = document_score_sorted[doc]

            
    # Calculate new document score using Rocchio's algorithm
    alpha = 1
    beta = 2
    gamma = 0.5
    updated_document_score = {}
    for doc_id, tfidf_list in document_list.items():
        score = alpha * document_score[doc_id]
        if doc_id in relevant_docs:
            score += beta * sum(tfidf_list.get(term, 0) for term in query_terms.keys())
        elif doc_id in non_relevant_docs:
            score -= gamma * sum(tfidf_list.get(term, 0) for term in query_terms.keys())
        updated_document_score[doc_id] = score
    updated_document_score_sorted = dict(sorted(updated_document_score.items(), key=lambda x: x[1], reverse=True))
    return updated_document_score_sorted


# Define function to calculate the benchmark results for the model
def test_results(folder_number, updated_document_score_sorted, feedback_path):
    os.chdir(feedback_path)
    file_name = 'Dataset'+str(folder_number)+'.txt'
    benFile = open(file_name)
    file_ = benFile.readlines()
    ben={}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        ben[lineList[1]]=float(lineList[2])
    benFile.close()
    
    # Calculate average precision
    ri = 0
    map1 = 0.0
    R = len([id for (id, v) in ben.items() if v > 0])
    i = 0
    for (key, value) in updated_document_score_sorted.items():
        i += 1
        if ben[key] > 0:
            ri += 1
            pi = float(ri) / float(i)
            recall = float(ri) / float(R)
            map1 += pi
    map1 /= float(ri)
    
    # Calculate precision@12
    ri = 0
    map2 = 0.0
    R = len([id for (id, v) in ben.items() if v > 0])
    i = 0
    for (key, value) in updated_document_score_sorted.items():
        i += 1
        if ben[key] > 0:
            ri += 1
            pi = float(ri) / float(i)
            recall = float(ri) / float(R)
            map2 += pi
        if i == 12:
            break
    map2 /= float(ri)
    
    # Calculate DCG12
    ri = 0
    dcg = 0.0
    R = len([id for (id, v) in ben.items() if v > 0])
    i = 0
    for (key, value) in updated_document_score_sorted.items():
        i += 1
        if ben[key] > 0:
            if i == 1:
                dcg = 1
            else:     
                dcg += ben[key]/math.log2(int(i))
        if i == 12:
            break
    return map1, map2, dcg


# --------------------------------- Main --------------------------------- #
if __name__ == '__main__':
    # Import libraries
    import glob, os
    import string
    import sys
    import math
    from stemming.porter2 import stem
    import nltk
    from nltk.corpus import wordnet
    nltk.download('wordnet')


    # Read stop words file
    stopwords_f = open(os.path.join(sys.path[0], "common-english-words.txt"), "r")
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()

    # Write input path
    data_path = os.path.join(sys.path[0], "DataSets")
    feedback_path = os.path.join(sys.path[0], "Feedback")
    output_path = os.path.join(sys.path[0], "Output")
    path = os.path.join(sys.path[0])

    # Parse query
    query_list = parse_query(path, stop_words)

    # Define output files
    output_file_names = ["Rocchio_rankings.dat", "Rocchio_results.dat"]
    os.chdir(output_path)
    for file in output_file_names:
        try:
            os.remove(file)
        except Exception:
            pass 


    # Run the model through each document and query and output results
    document_score = {}
    average_precision_list = {}
    precision12_list = {}
    dcg_list = {}
    average_precision_array = []
    precision12_array = []
    dcg_array = []

    for i in range (101, 151):
        document_list = parse_docs(i, data_path, stop_words)
        updated_document_score_sorted = rocchhio_tfidf_model(i, document_list, query_list)

        # Remove old document rankings
        file_name = "Rocchio_R"+str(i)+"Rankings.dat"
        try:
            os.remove(file_name)
        except Exception:
            pass 

        # Output document rankings
        with open(os.path.join(output_path, file_name), "w") as f:
            f.write(f"Query{i} (DocID Weight):\n")
            count = 0
            for key, value in updated_document_score_sorted.items():
                count += 1
                f.write(f"{key} {value}\n")
                if count == 12:
                    break
            f.write("\n")

        # Save test results
        average_precision, precision12, dcg = test_results(i, updated_document_score_sorted, feedback_path)
        average_precision_list.update({i: average_precision})
        precision12_list.update({i: precision12})
        dcg_list.update({i: dcg})

    # Remove test results
    try:
        os.remove("Rocchio_results.txt")
    except Exception:
        pass   

    # Output test results                
    with open(os.path.join(output_path, "Rocchio_results.dat"), "w") as f:
        f.write(f"Average Precision:\n")
        sum_average_precision = 0
        for key, value in average_precision_list.items():
            f.write(f"R{key}: {value}\n")
            average_precision_array.append(value)
            sum_average_precision += value
        f.write(f"MAP: {sum_average_precision/len(average_precision_list)}\n")
        f.write("\n")

        f.write(f"Precision@12:\n")
        sum_precision_12 = 0
        for key, value in precision12_list.items():
            f.write(f"R{key}: {value}\n")
            precision12_array.append(value)
            sum_precision_12 += value
        f.write(f"Average: {sum_precision_12/len(precision12_list)}\n")
        f.write("\n")

        f.write(f"DCG12:\n")
        sum_dcg = 0
        for key, value in dcg_list.items():
            f.write(f"R{key}: {value}\n")
            dcg_array.append(value)
            sum_dcg += value
        f.write(f"Average: {sum_dcg/len(dcg_list)}\n")
        f.write("\n")
        print(average_precision_array)

    # Final averages
    average_map = sum(average_precision_array) / len(average_precision_array)
    average_p12 = sum(precision12_array) / len(precision12_array)
    average_dcg = sum(dcg_array) / len(dcg_array)

    print("\n=== Final Evaluation Averages Across All Queries ===")
    print(f"Mean Average Precision (MAP): {average_map:.4f}")
    print(f"Average Precision@12: {average_p12:.4f}")
    print(f"Average DCG@12: {average_dcg:.4f}")






