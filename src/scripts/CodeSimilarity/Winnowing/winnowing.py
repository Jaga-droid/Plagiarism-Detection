

from hashlib import sha1


def hash_fun(text):
    hs = sha1(text.encode("utf-8"))
    hs = hs.hexdigest()[-4:]
    hs = int(hs, 16)
    return hs



def kgrams(text, n):
  text = list(text)
  return zip(*[text[i:] for i in range(n)])



def do_hashing(kgrams):
    hashlist = []
    for i,kg in enumerate(list(kgrams)):
        ngram_text = "".join(kg)
        hashvalue = hash_fun(ngram_text)
        hashlist.append((hashvalue, i))
    return hashlist



def sl_window(hashes, n):
    return zip(*[hashes[i:] for i in range(n)])



def get_min(windows):
    result = []
    prev_min = ()
    for w in windows:
        
        min_h = min(w, key=lambda x: (x[0], -x[1])) 

       
        if min_h != prev_min:
            result.append(min_h)
        prev_min = min_h
    return result

# winnowing
def winnowing(text, size_k, window_size):
    hashes = (do_hashing(kgrams(text,size_k)))
    return set(get_min(sl_window(hashes, window_size)))


def intersection(lst1, lst2): 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return len(lst3) 


def winnowing_similarity(text_a, text_b, size_k = 5, window_size = 4):
    # Get fingerprints using winnowing
    w1 = winnowing(text_a, size_k, window_size)
    w2 = winnowing(text_b, size_k, window_size)

    # Do use list instead of set to also consider number of occurece of copied content
    hash_list_a = [x[0] for x in w1]
    hash_list_b = [x[0] for x in w2]

    intersect = intersection(hash_list_a, hash_list_b) \
                + intersection(hash_list_b, hash_list_a)
    
    union = len(hash_list_a) + len(hash_list_b)

    return (intersect / union)